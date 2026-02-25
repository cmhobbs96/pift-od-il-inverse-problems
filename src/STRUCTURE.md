# `src` Structure

```text
src/
├── core/                 # Shared numerical primitives and solvers
│   ├── energies.py
│   ├── likelihoods.py
│   ├── parameterizations.py
│   ├── reference_solver.py
│   └── sgld.py
├── models/               # Model-specific code
│   ├── pift/
│   │   ├── physics.py
│   │   ├── pipeline.py
│   │   └── __init__.py
│   ├── bayesian_pinns/
│   │   ├── physics.py
│   │   ├── pipeline.py
│   │   └── __init__.py
│   ├── monte_carlo/
│   │   ├── physics.py
│   │   ├── pipeline.py
│   │   └── __init__.py
│   └── odil/
│       ├── physics.py
│       ├── pipeline.py
│       └── __init__.py
├── pipelines/            # Cross-model orchestration by phase
│   ├── common.py
│   ├── phase_a.py
│   └── __init__.py
├── utils/                # Shared diagnostics/helpers/plotting/validation
│   ├── diagnostics.py
│   ├── diagnostics_runtime.py
│   ├── helpers.py
│   ├── plotting.py
│   ├── validators.py
│   └── __init__.py
├── run_manager.py        # Global run lifecycle management
├── storage.py            # Global SQLite persistence
└── run_types.py          # Shared run enums/types
```

There is no global `src/pift` tree anymore. `pift` is only a model under `src/models/pift`.
