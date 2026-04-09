# Examples

Reproduces all experiments from [Alberts & Bilionis (2023)](https://doi.org/10.1016/j.jcp.2023.112100)
plus novel ODIL warm-start contributions from [Karnakov et al. (2024)](https://doi.org/10.1093/pnasnexus/pgae005).

Each notebook is **self-contained** — installs from GitHub, runs on Colab GPU, no local setup needed.

| # | Notebook | Description | ~Time | Colab |
|---|----------|-------------|-------|-------|
| 1 | [Forward Poisson 1D](01_forward_poisson_1d.ipynb) | PIFT SGLD + MC comparison | 1 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmhobbs96/pift-od-il-inverse-problems/blob/main/examples/01_forward_poisson_1d.ipynb) |
| 2 | [Forward Poisson 2D](02_forward_poisson_2d.ipynb) | 2D extension with SineBasis2D | 1 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmhobbs96/pift-od-il-inverse-problems/blob/main/examples/02_forward_poisson_2d.ipynb) |
| 3 | [ODIL Solver](03_odil_solver.ipynb) | Gauss-Newton + L-BFGS MAP estimation | 30 sec | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmhobbs96/pift-od-il-inverse-problems/blob/main/examples/03_odil_solver.ipynb) |
| 4 | [ODIL Warm-Start](04_odil_warmstart.ipynb) | Cold / FD / ODIL initialization ablation | 1 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmhobbs96/pift-od-il-inverse-problems/blob/main/examples/04_odil_warmstart.ipynb) |
| 5 | [Accuracy vs Wall Time](05_accuracy_vs_walltime.ipynb) | ODIL vs PIFT vs MC benchmark | 1 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmhobbs96/pift-od-il-inverse-problems/blob/main/examples/05_accuracy_vs_walltime.ipynb) |
| 6 | [β Sensitivity](06_beta_sensitivity.ipynb) | Paper Example 1 — variance collapse | 3 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmhobbs96/pift-od-il-inverse-problems/blob/main/examples/06_beta_sensitivity.ipynb) |
| 7 | [Model-Form Uncertainty](07_model_form_uncertainty.ipynb) | Paper Example 2 — nested SGLD | 15–30 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmhobbs96/pift-od-il-inverse-problems/blob/main/examples/07_model_form_uncertainty.ipynb) |
| 8 | [Inverse Parameters](08_inverse_parameters.ipynb) | Paper Example 3a — joint (D, κ) | 30–60 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmhobbs96/pift-od-il-inverse-problems/blob/main/examples/08_inverse_parameters.ipynb) |
| 9 | [Inverse Source](09_inverse_source.ipynb) | Paper Example 3b — KLE source ID | 40–80 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmhobbs96/pift-od-il-inverse-problems/blob/main/examples/09_inverse_source.ipynb) |
| 10 | [Allen–Cahn Bimodal](10_allen_cahn_bimodal.ipynb) | Paper Example 4 — 2D bimodal posterior | 3 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmhobbs96/pift-od-il-inverse-problems/blob/main/examples/10_allen_cahn_bimodal.ipynb) |

## Prerequisites

- **GPU recommended** — Runtime → Change runtime type → T4/A100
- Each notebook installs the package via `pip install git+https://github.com/cmhobbs96/pift-od-il-inverse-problems.git`
- Notebooks 7–9 use nested SGLD and take longer; start with notebooks 1–6 for a quick tour
