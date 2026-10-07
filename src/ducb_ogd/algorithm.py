"""DUCB-OGD on synthetic multi-source quadratics (arXiv:2610.06329).

No LLM. Losses are deterministic quadratics, so the paper's stochastic
mini-batch is the identity here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def schedule(k: int, t: int):
    """beta and eta from Theorem 2. K sources, horizon T."""
    k = jnp.asarray(k, dtype=jnp.float64)
    t = jnp.asarray(t, dtype=jnp.float64)
    beta = 1.0 - jnp.power(k, -0.5) * jnp.power(t, -0.5)
    eta = jnp.power(k, -0.25) * jnp.power(t, -0.75)
    return beta, eta


def horizon_factor(beta):
    """H(beta) = 1 / (1 - beta)."""
    return 1.0 / (1.0 - beta)


def ema_update(mu, n_eff, source, loss, beta):
    """One discounted-EMA step (Eq. 7).

    N_t^k = beta * N_{t-1}^k + 1{k_t = k}
    mu is unchanged for sources that were not selected.
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    n_eff = jnp.asarray(n_eff, dtype=jnp.float64)
    source = jnp.asarray(source, dtype=jnp.int32)
    loss = jnp.asarray(loss, dtype=jnp.float64)
    beta = jnp.asarray(beta, dtype=jnp.float64)
    n_new = beta * n_eff
    n_new = n_new.at[source].add(1.0)
    updated = (beta * n_eff[source] * mu[source] + loss) / n_new[source]
    mu_new = mu.at[source].set(updated)
    return mu_new, n_new


def radius(n_eff, bound, alpha, beta):
    """r^k = 2 C sqrt(2 alpha log H(beta) / N^k)."""
    h = horizon_factor(beta)
    return 2.0 * bound * jnp.sqrt(2.0 * alpha * jnp.log(h) / n_eff)


def ducb_index(mu, radii):
    return mu + radii


def select_source(mu, radii, step, n_sources):
    """First K steps visit each source once. Then argmax of the index."""
    step = int(step)
    n_sources = int(n_sources)
    if step < n_sources:
        return step
    return int(jnp.argmax(ducb_index(mu, radii)))


def ogd_step(theta, grad, eta):
    """theta <- theta - eta * grad. Unprojected."""
    return jnp.asarray(theta, dtype=jnp.float64) - eta * jnp.asarray(grad, dtype=jnp.float64)


def dema_update(ema1, ema2, value, beta):
    """Double-EMA: 2 * EMA(x) - EMA(EMA(x)). Returns (dema, ema1, ema2)."""
    ema1 = beta * ema1 + (1.0 - beta) * value
    ema2 = beta * ema2 + (1.0 - beta) * ema1
    return 2.0 * ema1 - ema2, ema1, ema2


def enforced_exploration(one_hot, beta, step, n_sources):
    """p <- (1 - beta^t) p + beta^t / K. step is 1-indexed in the paper's beta^t.

    At t = 0, beta^0 = 1, so the mixture is uniform.
    """
    mix = beta ** step
    return (1.0 - mix) * one_hot + mix / n_sources


def sampling_probabilities(mu, radii, bound, n_eff, n_nodes=32):
    """P(k = argmax_j (mu_j + phi_j)), phi ~ N(2 r, C^2 / (2 N)).

    One-dimensional Gauss-Hermite quadrature on each source, times the product
    of the other CDFs. Probabilities are renormalized so they sum to 1.
    """
    from numpy.polynomial.hermite import hermgauss

    mu = jnp.asarray(mu, dtype=jnp.float64)
    radii = jnp.asarray(radii, dtype=jnp.float64)
    n_eff = jnp.asarray(n_eff, dtype=jnp.float64)
    nodes, weights = hermgauss(n_nodes)
    nodes = jnp.asarray(nodes, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    # hermgauss integrates exp(-x^2) f(x). For N(m, s^2), x = (z - m) / (s sqrt(2)).
    mean = mu + 2.0 * radii
    var = (bound ** 2) / (2.0 * jnp.maximum(n_eff, 1e-12))
    std = jnp.sqrt(var)
    k = mu.shape[0]

    def cdf_others(z, i):
        # product of CDFs of the other Gaussians at z
        other = jnp.arange(k) != i
        # Phi((z - mean) / std)
        z_std = (z - mean) / std
        phi = 0.5 * (1.0 + jax.scipy.special.erf(z_std / jnp.sqrt(2.0)))
        phi = jnp.where(other, phi, 1.0)
        return jnp.prod(phi)

    def mass(i):
        scale = std[i] * jnp.sqrt(2.0)
        zs = mean[i] + scale * nodes
        dens = jax.vmap(lambda z: cdf_others(z, i))(zs)
        # int f(z) gaussian = (1/sqrt(pi)) sum w_i f(m + s*sqrt(2)*x_i)
        return jnp.sum(weights * dens) / jnp.sqrt(jnp.pi)

    probs = jax.vmap(mass)(jnp.arange(k))
    probs = jnp.maximum(probs, 0.0)
    return probs / jnp.sum(probs)


def quadratic_loss_and_grad(theta, center, bias):
    diff = theta - center
    loss = 0.5 * jnp.sum(diff ** 2) + bias
    return loss, diff


def run_ducb_ogd(centers, biases, horizon, theta0, bound, alpha=1.0, dema=False, explore=False):
    """Deterministic DUCB-OGD. centers (K, d), biases (K,)."""
    centers = jnp.asarray(centers, dtype=jnp.float64)
    biases = jnp.asarray(biases, dtype=jnp.float64)
    theta = jnp.asarray(theta0, dtype=jnp.float64)
    k, _d = centers.shape
    beta, eta = schedule(k, horizon)
    mu = jnp.zeros(k, dtype=jnp.float64)
    n_eff = jnp.zeros(k, dtype=jnp.float64)
    ema1 = jnp.zeros(k, dtype=jnp.float64)
    ema2 = jnp.zeros(k, dtype=jnp.float64)
    chosen = []
    thetas = []
    for step in range(int(horizon)):
        if step < k:
            source = step
        else:
            rad = radius(jnp.maximum(n_eff, 1e-12), bound, alpha, beta)
            index = ducb_index(mu, rad)
            if explore:
                one = jnp.zeros(k).at[int(jnp.argmax(index))].set(1.0)
                probs = enforced_exploration(one, beta, step, k)
                source = int(jnp.argmax(probs))
            else:
                source = int(jnp.argmax(index))
        loss, grad = quadratic_loss_and_grad(theta, centers[source], biases[source])
        if dema:
            tracked, ema1_s, ema2_s = dema_update(ema1[source], ema2[source], loss, beta)
            ema1 = ema1.at[source].set(ema1_s)
            ema2 = ema2.at[source].set(ema2_s)
            observed = tracked
        else:
            observed = loss
        mu, n_eff = ema_update(mu, n_eff, source, observed, beta)
        theta = ogd_step(theta, grad, eta)
        chosen.append(source)
        thetas.append(theta)
    losses = jax.vmap(lambda c, b: quadratic_loss_and_grad(theta, c, b)[0])(centers, biases)
    return theta, jnp.asarray(chosen), losses
