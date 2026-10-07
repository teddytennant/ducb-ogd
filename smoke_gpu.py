"""One short DUCB-OGD run on whatever device JAX sees."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import jax
import jax.numpy as jnp

from ducb_ogd import run_ducb_ogd

print("device", jax.devices())
centers = jnp.ones((4, 8)).at[:].set(0.1) * jnp.arange(4)[:, None]
biases = jnp.array([0.0, 0.3, 0.1, 0.2])
theta, chosen, losses = run_ducb_ogd(
    centers, biases, horizon=200, theta0=jnp.zeros(8), bound=5.0, explore=True
)
print("worst", float(jnp.max(losses)))
print("counts", [int(jnp.sum(chosen == k)) for k in range(4)])
