"""Short synthetic DUCB-OGD run. No LLM."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import jax.numpy as jnp

from ducb_ogd import run_ducb_ogd

centers = jnp.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.4, -0.2, 0.1, 0.0, 0.3, -0.1, 0.2, 0.0],
        [-0.3, 0.5, 0.0, 0.2, -0.4, 0.1, 0.0, 0.3],
        [0.2, 0.2, -0.5, 0.1, 0.0, 0.4, -0.2, 0.1],
    ]
)
biases = jnp.array([0.0, 0.2, 0.5, 0.1])
theta, chosen, losses = run_ducb_ogd(
    centers, biases, horizon=200, theta0=jnp.zeros(8), bound=5.0, explore=True
)
print("final worst-source loss", float(jnp.max(losses)))
print("counts", [int(jnp.sum(chosen == k)) for k in range(4)])
