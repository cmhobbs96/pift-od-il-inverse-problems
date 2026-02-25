from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.sgld import sgld_sample


def test_sgld_runs_and_shapes() -> None:
    key = jax.random.PRNGKey(0)

    def grad_and_metrics(theta):
        grad = theta.copy()
        h = 0.5 * jnp.sum(theta**2)
        return grad, {"hamiltonian": h, "likelihood": h, "physics": 0.0}

    chain, traces, meta = sgld_sample(
        theta0=jnp.zeros((3,), dtype=jnp.float64),
        grad_and_metrics_fn=grad_and_metrics,
        n_steps=50,
        step_size0=1e-2,
        decay=0.0,
        key=key,
    )

    assert chain.shape == (50, 3)
    assert traces["hamiltonian"].shape == (50,)
    assert np.isfinite(chain).all()
    assert meta["stopped"] is False
