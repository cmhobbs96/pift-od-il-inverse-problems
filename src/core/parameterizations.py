"""Field parameterizations for 1D examples."""

from __future__ import annotations

import jax.numpy as jnp


class SineBasisField:
    """Sine basis field on [0, 1] satisfying zero Dirichlet boundaries."""

    def __init__(self, n_modes: int) -> None:
        if n_modes < 1:
            raise ValueError("n_modes must be >= 1")
        self.n_modes = int(n_modes)
        self._k = jnp.arange(1, self.n_modes + 1, dtype=jnp.float64)

    def design_matrix(self, x):
        x = jnp.asarray(x, dtype=jnp.float64)
        return jnp.sin(jnp.pi * x[:, None] * self._k[None, :])

    def neg_second_derivative_matrix(self, x):
        basis = self.design_matrix(x)
        weights = (jnp.pi * self._k) ** 2
        return basis * weights[None, :]

    def eval(self, x, theta):
        return self.design_matrix(x) @ jnp.asarray(theta, dtype=jnp.float64)

    def neg_second_derivative(self, x, theta):
        return self.neg_second_derivative_matrix(x) @ jnp.asarray(theta, dtype=jnp.float64)
