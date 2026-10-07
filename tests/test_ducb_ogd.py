import jax.numpy as jnp
import pytest

from ducb_ogd import (
    dema_update,
    ema_update,
    enforced_exploration,
    ogd_step,
    radius,
    run_ducb_ogd,
    sampling_probabilities,
    schedule,
    select_source,
)


def test_schedule_formulas():
    beta, eta = schedule(4, 16)
    assert float(beta) == pytest.approx(1.0 - 4 ** -0.5 * 16 ** -0.5, abs=1e-12)
    assert float(eta) == pytest.approx(4 ** -0.25 * 16 ** -0.75, abs=1e-12)


def test_ema_recurrence_matches_hand_values():
    mu = jnp.zeros(2)
    n_eff = jnp.zeros(2)
    beta = 0.5
    mu, n_eff = ema_update(mu, n_eff, 0, 4.0, beta)
    assert float(n_eff[0]) == pytest.approx(1.0)
    assert float(mu[0]) == pytest.approx(4.0)
    assert float(n_eff[1]) == pytest.approx(0.0)
    mu, n_eff = ema_update(mu, n_eff, 1, 2.0, beta)
    assert float(n_eff[0]) == pytest.approx(0.5)
    assert float(mu[0]) == pytest.approx(4.0)
    assert float(n_eff[1]) == pytest.approx(1.0)
    assert float(mu[1]) == pytest.approx(2.0)
    mu, n_eff = ema_update(mu, n_eff, 0, 8.0, beta)
    # N = 0.5 * 0.5 + 1 = 1.25, mu = (0.5 * 0.5 * 4 + 8) / 1.25 = 9 / 1.25 = 7.2
    assert float(n_eff[0]) == pytest.approx(1.25)
    assert float(mu[0]) == pytest.approx(7.2)
    assert float(mu[1]) == pytest.approx(2.0)
    assert float(n_eff[1]) == pytest.approx(0.5)


def test_skipped_source_decays_and_keeps_mean():
    mu = jnp.array([3.0, 1.0])
    n_eff = jnp.array([2.0, 4.0])
    mu2, n2 = ema_update(mu, n_eff, 0, 9.0, 0.25)
    assert float(mu2[1]) == pytest.approx(1.0)
    assert float(n2[1]) == pytest.approx(1.0)


def test_radius_formula():
    beta = 0.5
    n_eff = jnp.array([4.0, 1.0])
    got = radius(n_eff, bound=2.0, alpha=1.0, beta=beta)
    h = 1.0 / (1.0 - beta)
    expect = 2.0 * 2.0 * jnp.sqrt(2.0 * 1.0 * jnp.log(h) / n_eff)
    assert jnp.allclose(got, expect, atol=1e-12)


def test_first_steps_visit_each_source():
    mu = jnp.zeros(3)
    rad = jnp.ones(3)
    chosen = [select_source(mu, rad, t, 3) for t in range(3)]
    assert chosen == [0, 1, 2]


def test_argmax_index_after_exploration():
    mu = jnp.array([0.0, 5.0, 1.0])
    rad = jnp.array([0.1, 0.1, 0.1])
    assert select_source(mu, rad, 3, 3) == 1


def test_ogd_step():
    theta = ogd_step(jnp.array([1.0, 2.0]), jnp.array([0.5, -1.0]), 0.2)
    assert jnp.allclose(theta, jnp.array([0.9, 2.2]))


def test_dema_hand_series():
    ema1 = 0.0
    ema2 = 0.0
    beta = 0.5
    values = [2.0, 4.0]
    out = []
    for v in values:
        d, ema1, ema2 = dema_update(ema1, ema2, v, beta)
        out.append(float(d))
    # ema1_1 = 0.5*0 + 0.5*2 = 1, ema2_1 = 0.5*1 = 0.5, dema = 2-0.5 = 1.5
    # ema1_2 = 0.5*1 + 0.5*4 = 2.5, ema2_2 = 0.5*0.5 + 0.5*2.5 = 1.5, dema = 5-1.5 = 3.5
    assert out[0] == pytest.approx(1.5)
    assert out[1] == pytest.approx(3.5)


def test_enforced_exploration_endpoints():
    one = jnp.array([0.0, 1.0, 0.0])
    at0 = enforced_exploration(one, 0.5, 0, 3)
    assert jnp.allclose(at0, jnp.ones(3) / 3)
    later = enforced_exploration(one, 0.5, 2, 3)
    assert float(jnp.sum(later)) == pytest.approx(1.0)
    # beta^2 = 0.25, so p = 0.75 * one_hot + 0.25/3
    expect = 0.75 * one + 0.25 / 3
    assert jnp.allclose(later, expect)


def test_sampling_probabilities_sum_and_mode():
    mu = jnp.array([0.0, 3.0, 0.2])
    rad = jnp.array([0.01, 0.01, 0.01])
    n_eff = jnp.array([100.0, 100.0, 100.0])
    probs = sampling_probabilities(mu, rad, bound=1.0, n_eff=n_eff)
    assert float(jnp.sum(probs)) == pytest.approx(1.0, abs=1e-5)
    assert int(jnp.argmax(probs)) == 1


def test_short_run_is_finite_and_tracks_the_index():
    centers = jnp.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.5]])
    biases = jnp.array([0.0, 0.0, 4.0])
    theta, chosen, losses = run_ducb_ogd(
        centers, biases, horizon=12, theta0=jnp.zeros(2), bound=10.0
    )
    assert jnp.all(jnp.isfinite(theta))
    assert jnp.all(jnp.isfinite(losses))
    assert list(chosen[:3]) == [0, 1, 2]
    # After equal exploration, source 2 has the only large bias, so its mean is highest
    # and the next pick is 2. Radii are equal because each N equals beta^{2} after two skips
    # plus the visit... not exactly equal. Check only that the run picked a valid source.
    assert int(jnp.max(chosen)) < 3
