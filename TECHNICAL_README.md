# Technical README

## 1) Repository Structure (modeled after `PredictiveScienceLab/pift-paper-2023`)

This project is currently minimal. The structure below is the recommended technical layout to follow the `pift-paper-2023` style (`examples/`, `src/`, `tests/`, `makefile`) while fitting this repository.

```text
pift-od-il-inverse-problems/
├── README.md
├── TECHNICAL_README.md
├── LICENSE
├── Alberts - Physics-informed information field theory for modeling physical systems with uncertainty quantification.pdf
├── Karkakov - Solving inverse problems in physics by optimizing a discrete loss_ Fast and accurate learning without neural networks.pdf
├── makefile
├── pyproject.toml
├── uv.lock
├── examples/
│   ├── 01_forward_poisson_basis.py
│   ├── 02_forward_heat_basis.py
│   ├── 03_inverse_beta_nested_sgld.py
│   └── 04_odil_baseline.py
├── src/
│   └── pift/
│       ├── __init__.py
│       ├── operators.py
│       ├── energies.py
│       ├── likelihoods.py
│       ├── parameterizations.py
│       ├── sgld.py
│       ├── nested_sgld.py
│       ├── diagnostics.py
│       └── plotting.py
├── tests/
│   ├── test_energies.py
│   ├── test_likelihoods.py
│   ├── test_sgld.py
│   └── test_inverse_gradients.py
├── configs/
│   ├── forward_poisson.yaml
│   ├── forward_heat.yaml
│   └── inverse_beta.yaml
├── scripts/
│   ├── run_forward.sh
│   ├── run_inverse.sh
│   └── run_all.sh
└── outputs/
    ├── figures/
    ├── traces/
    └── tables/
```

### Current Readable `src` Layout

The active codebase is now organized by concern so contributors can review files quickly:

```text
src/
├── core/         # numerical primitives (energies, likelihoods, basis, SGLD, FD solver)
├── models/       # model-specific packages (pift, bayesian_pinns, monte_carlo, odil)
├── pipelines/    # experiment-level orchestration and shared pipeline helpers
├── utils/        # diagnostics, plotting, validators, helper transforms
├── run_manager.py
├── storage.py
└── run_types.py
```

See [src/STRUCTURE.md](src/STRUCTURE.md) for the detailed map.

## 2) Core PIFT Formulation

Following Alberts and Bilionis (JCP 2023), define a posterior directly over fields:

$$
p(\phi \mid d) \propto p(d \mid \phi)\,p(\phi).
$$

With measurement misfit $\Omega$ and physics energy $U[\phi]$:

$$
p(d \mid \phi) \propto \exp\left(-\Omega(d, R\phi)\right),
$$

$$
p(\phi) \propto \exp\left(-\beta U[\phi]\right),
$$

$$
H[\phi \mid d] = \Omega(d, R\phi) + \beta U[\phi].
$$

The field prior is physics-informed and $\beta$ controls trust in physics vs data (model-form uncertainty handling).

## 3) Physics Energy Choices

Two common choices for the energy functional:

1. Variational/energy form (when PDE has known energy).
2. Integrated residual form:

$$
U[\phi] = \int_{\Omega} \| \mathcal{F}[\phi](x) \|^2\,dx,
$$

with boundary-condition penalties embedded in either the parameterization or $U[\phi]$.

## 4) Field Parameterization

Use a finite parameterization for computation:

$$
\phi(x) \approx \hat{\phi}(x;\theta),
$$

typically:

1. Basis-first path: truncated Fourier/spectral basis with boundary embedding.
2. Neural path: MLP parameterization for nonlinearity/flexibility.

This induces a parameter posterior:

$$
p(\theta \mid d) \propto \exp\left(-H(\theta \mid d)\right).
$$

## 5) SGLD for Forward PIFT

For forward problems, sample $\theta$ via stochastic gradient Langevin dynamics:

$$
\theta_{t+1}
= \theta_t
- \alpha_t \widehat{\nabla_\theta H(\theta_t \mid d)}
+ \sqrt{2\alpha_t}\,\xi_t,\quad \xi_t \sim \mathcal{N}(0,I).
$$

Use stochastic quadrature points in space to estimate gradients of $U[\hat{\phi}]$ and data minibatches for $\Omega$ when needed.

## 6) Nested SGLD for Inverse Problems

When unknown physics parameters $\lambda$ affect the partition function, use nested sampling (inner field loops, outer parameter loop). Key identity:

$$
\nabla_\lambda H(\lambda \mid d)
=
\mathbb{E}_{p(\phi \mid d,\lambda)}\!\left[\nabla_\lambda H[\phi \mid \lambda]\right]
-
\mathbb{E}_{p(\phi \mid \lambda)}\!\left[\nabla_\lambda H[\phi \mid \lambda]\right].
$$

Outer update:

$$
\lambda_{t+1}
= \lambda_t
- \eta_t \widehat{\nabla_\lambda H(\lambda_t \mid d)}
+ \sqrt{2\eta_t}\,\zeta_t,\quad \zeta_t \sim \mathcal{N}(0,I).
$$

## 7) Mapping Theory to Code

Suggested module responsibilities:

1. `src/core/`: PDE operators, energies, likelihoods, parameterizations, solvers, SGLD.
2. `src/models/<model>/`: model-specific `physics.py` and `pipeline.py`.
3. `src/pipelines/`: cross-model phase runners and shared runtime helpers.
4. `src/utils/`: diagnostics, plotting, data/config validation, helper transforms.
5. `src/run_manager.py`: run lifecycle and orchestration.
6. `src/storage.py`: SQLite persistence and run history queries.

## 8) Validation Protocol

Each example should report:

1. Posterior mean and variance of $\phi(x)$.
2. Posterior predictive intervals at observation points.
3. Reference-solver error (L2 and max norm).
4. SGLD diagnostics (trace, running mean, acceptance proxy/mixing proxy).
5. If inverse: parameter posterior summaries and sensitivity to initialization.

## 9) Reproducibility

1. Fix RNG seeds and store them in config files.
2. Keep all hyperparameters in `configs/*.yaml`.
3. Save raw chains and diagnostics in `outputs/traces/`.
4. Version-lock dependencies with `pyproject.toml` + `uv.lock`.
