# PIFT + ODIL Research Project Plan
## From Current State → Publication-Ready Paper & Research GUI

---

## Current State

- 1D forward Poisson PIFT working: L2 = 0.027, coverage = 100%, 2000 samples
- Classical Monte Carlo baseline working: L2 = 0.040, coverage = 71.4%
- Diagonal preconditioner implemented with condition number clipping
- FD warm-start initialization working
- PySide6 GUI with SQLite persistence, multi-method execution, diagnostics JSON
- ODIL baseline broken (diverges at step 61, learning rate issue)
- BayesPINN unstable at current hyperparameters
- Nested SGLD (inverse problems) not yet implemented

## End State

- All 4 paper examples replicated (Examples 1–4 from Alberts & Bilionis 2023)
- ODIL properly implemented per Karnakov et al. 2024
- ODIL warm-start for PIFT demonstrated and benchmarked
- Novel contribution: ODIL-warm-started PIFT combining MAP speed with Bayesian UQ
- Publication-ready paper targeting JCP or SIAM Journal on Scientific Computing
- Best-in-class research GUI covering all methods, examples, and analysis tools

---

## Repository Structure

```
pift-research/
│
├── core/                              # Pure numerical modules, no GUI deps
│   ├── __init__.py
│   ├── energies.py                    # 1D Poisson residual energy (existing)
│   ├── energies_nonlinear.py          # Module 1.3 — nonlinear PDE energy
│   ├── energies_2d.py                 # Module 2.1 — 2D field energies
│   ├── energies_timedependent.py      # Module 2.4 — parabolic PDE energy
│   ├── likelihoods.py                 # Gaussian NLL (existing)
│   ├── parameterizations.py           # SineBasisField (existing)
│   ├── parameterizations_nonlinear.py # Module 1.3 — mixed Fourier + BC embedding
│   ├── parameterizations_2d.py        # Module 2.1 — 2D sine basis
│   ├── parameterizations_mlp.py       # Module 2.3 — MLP field parameterization
│   ├── sgld.py                        # SGLD sampler with preconditioning (existing)
│   ├── nested_sgld.py                 # Module 1.2 — Algorithm 3 from paper
│   ├── odil.py                        # Module 2.6 — ODIL proper (Newton + L-BFGS)
│   ├── kle.py                         # Module 1.6 — KLE via Nyström approximation
│   ├── reference_solver.py            # FD reference solver (existing)
│   └── hmcecs.py                      # Module 1.7 — HMCECS via NumPyro wrapper
│
├── phases/                            # Per-example pipeline runners
│   ├── __init__.py
│   ├── common.py                      # Shared: forcing, observations, emit (existing)
│   ├── phase_a.py                     # Forward 1D Poisson — PIFT + MC (existing)
│   ├── phase_b_beta_sweep.py          # Module 1.1 — Example 1
│   ├── phase_b_model_form.py          # Module 1.4 — Example 2
│   ├── phase_c_inverse_params.py      # Module 1.5 — Example 3a
│   ├── phase_c_inverse_source.py      # Module 1.6 — Example 3b
│   ├── phase_d_allen_cahn.py          # Module 1.7 — Example 4
│   ├── phase_e_2d_poisson.py          # Module 2.1 — 2D forward
│   └── phase_f_heat_timedep.py        # Module 2.4 — time-dependent
│
├── benchmarks/                        # Module 2.8 — ODIL vs PIFT timing
│   ├── __init__.py
│   ├── bench_odil_vs_pift.py          # Accuracy vs runtime comparison
│   ├── bench_warmstart.py             # Cold-start vs ODIL-warm-start PIFT
│   └── bench_gpu_cpu.py               # Module 2.5 — GPU acceleration
│
├── frontend/                          # PySide6 GUI
│   ├── app.py                         # Main application (existing, refactored)
│   ├── widgets/
│   │   ├── method_panel.py            # Method selector + config
│   │   ├── diagnostics_panel.py       # Module 4.3 — diagnostics dashboard
│   │   ├── comparison_panel.py        # Module 4.4 — multi-run comparison
│   │   ├── data_panel.py              # Module 4.5 — data + problem config
│   │   ├── export_panel.py            # Module 4.6 — publication export
│   │   └── field_2d_widget.py         # 2D heatmap viewer
│   └── plots/
│       ├── overlay.py                 # Multi-method posterior overlay
│       ├── beta_sweep.py              # Beta sensitivity multi-panel
│       ├── joint_posterior.py         # 2D joint parameter posterior contour
│       └── mode_separation.py         # GMM mode separation visualization
│
├── utils/
│   ├── __init__.py
│   ├── diagnostics.py                 # ESS, coverage, autocorr (existing)
│   ├── diagnostics_runtime.py         # RuntimeGuard (existing)
│   ├── plotting.py                    # plot_field_summary, plot_diagnostics (existing)
│   └── export.py                      # Module 4.6 — LaTeX/PDF export utilities
│
├── storage/
│   ├── __init__.py
│   ├── storage.py                     # SQLite run persistence (existing)
│   └── run_manager.py                 # Run lifecycle (existing)
│
├── config/
│   ├── defaults.yaml                  # Single source of truth for all hyperparams
│   ├── presets/
│   │   ├── example1_beta_sweep.yaml
│   │   ├── example2_model_form.yaml
│   │   ├── example3a_inverse.yaml
│   │   ├── example3b_source.yaml
│   │   └── example4_allen_cahn.yaml
│   └── paper_replication.yaml         # Exact hyperparams to reproduce paper figures
│
├── paper/                             # Publication manuscript
│   ├── main.tex
│   ├── sections/
│   │   ├── abstract.tex
│   │   ├── introduction.tex
│   │   ├── background.tex
│   │   ├── methodology.tex
│   │   ├── experiments.tex
│   │   └── conclusion.tex
│   ├── figures/                       # All generated figures (PDF vector)
│   ├── tables/                        # Generated LaTeX tables
│   ├── references.bib
│   └── generate_figures.py            # Reproducible figure generation script
│
├── tests/
│   ├── test_energies.py               # (existing, 7+ passing)
│   ├── test_sgld.py
│   ├── test_nested_sgld.py            # Module 1.2
│   ├── test_odil.py                   # Module 2.6
│   ├── test_kle.py                    # Module 1.6
│   └── test_parameterizations.py
│
├── notebooks/
│   ├── 01_phase_a_workbench.ipynb     # (existing)
│   ├── 02_prior_visualization.ipynb   # Prior log-prob visualization
│   ├── 03_odil_vs_pift.ipynb          # Benchmark comparison
│   └── 04_paper_figures.ipynb         # Reproduce all paper figures
│
├── run_types.py                       # RunStatus, RunResult dataclass (existing)
├── requirements.txt
├── README.md
└── PLAN.md                            # This file
```

---

## Phase 1 — Complete Paper Replication
*Target: all 4 Alberts & Bilionis (2023) examples working*

---

### Module 1.1 — Beta Sensitivity Sweep (Example 1)
**File:** `phases/phase_b_beta_sweep.py`
**Goal:** Reproduce Fig. 1 from paper — posterior variance collapses as β → ∞.

- Extend current 1D pipeline to run at β ∈ {1, 10, 100, 1000}
- Use 1D heat equation with non-zero Dirichlet BCs: φ(0)=1, φ(1)=0.1
- Mixed Fourier basis with BC embedding (Eq. 35–36 in paper)
- Plot posterior mean ± credible band for each β as a multi-panel figure
- Verify: covariance → 0 as β → ∞ matches Klein-Gordon analytical result

**Deliverable:** `phase_b_beta_sweep.py`, multi-panel Fig. 1 reproduction
**Effort:** 1–2 days
**Dependencies:** None (pure forward extension of existing pipeline)

---

### Module 1.2 — Nested SGLD Core
**File:** `core/nested_sgld.py`
**Goal:** Implement Algorithm 3 from the paper. Load-bearing for Examples 2, 3a, 3b.

- Outer SGLD loop over parameters λ with Robbins-Monro step schedule
- Inner loop: T=10 prior SGLD steps + T̃=1 posterior SGLD step per outer iteration
- Gradient estimator: E[∇λH[φ|λ] | d,λ] − E[∇λH[φ|λ] | λ] + ∇λH(λ)
- Warm-start: 1M prior + posterior field sampling steps at λ=λ₁ before outer loop
- Jeffrey's prior support: sample log(λ) for positive parameters
- Diagnostics: outer chain trace, λ posterior histogram, R-hat across restarts
- Full unit test suite: verify gradient identity from Appendix B

**Deliverable:** `core/nested_sgld.py`, `tests/test_nested_sgld.py`
**Effort:** 3–5 days
**Dependencies:** None (standalone module)

---

### Module 1.3 — Nonlinear PDE Energy (Shared Infrastructure)
**File:** `core/energies_nonlinear.py`, `core/parameterizations_nonlinear.py`
**Goal:** Energy functional for Dφ'' − κφ³ = f used in Examples 2, 3a, 3b.

- Energy: U[φ] = ∫₀¹ (½D(φ')² + ¼κφ⁴ + φf) dx
- Verify stationarity matches PDE via variational calculus (Appendix C)
- BC parameterization: φ̂(x;θ) = (1−x)φ(0) + xφ(1) + (1−x)x·ψ̂(x;θ)
- Mixed Fourier basis ψ̂ with K=20 terms (cos + sin, Eq. 36)
- Unit test: FD solver and energy minimizer agree on ground truth field

**Deliverable:** Two new core modules, unit tests
**Effort:** 1–2 days
**Dependencies:** None

---

### Module 1.4 — Model-Form Uncertainty (Example 2)
**File:** `phases/phase_b_model_form.py`
**Goal:** Reproduce Fig. 2 — inferred β posterior shifts with physics correctness γ.

- Ground truth: D=0.1, κ=1, f=cos(4x), 40 observations, σ=0.01
- Two experiments: source term error and energy functional error
- Six values of γ ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0} for each experiment
- Nested SGLD with Jeffrey's prior on β, sampling λ=log(β)
- Plot posterior β distributions as violin plots vs γ for both experiments

**Deliverable:** `phase_b_model_form.py`, Fig. 2 reproduction
**Effort:** 2–3 days
**Dependencies:** Modules 1.2, 1.3

---

### Module 1.5 — Inverse Parameter Identification (Example 3a)
**File:** `phases/phase_c_inverse_params.py`
**Goal:** Reproduce Fig. 3 — joint posterior over D and κ, fitted prior predictive.

- Infer log(D) and log(κ) jointly via nested SGLD
- Jeffrey's priors, β=10⁵ (strong physics trust)
- SGLD learning rates: α₀=0.1 for λ, α̂₀=10 for posterior field
- Four-panel output: SGLD trace, joint D-κ contour, fitted prior predictive, posterior predictive

**Deliverable:** `phase_c_inverse_params.py`, Fig. 3 reproduction
**Effort:** 1–2 days
**Dependencies:** Modules 1.2, 1.3

---

### Module 1.6 — Source Term Identification (Example 3b)
**File:** `phases/phase_c_inverse_source.py`, `core/kle.py`
**Goal:** Reproduce Fig. 4 — simultaneously infer D, κ, and f.

- KLE of f with squared-exponential kernel C(x,x') = exp(−(x−x')²/0.18)
- 10-term truncation via Nyström approximation on fine grid
- Infer λ = [log(D), log(κ), z₁,...,z₁₀] jointly
- Plot: joint D-κ posterior, recovered source term ± credible band, predictives

**Deliverable:** `core/kle.py`, `phase_c_inverse_source.py`, Fig. 4 reproduction
**Effort:** 3–4 days
**Dependencies:** Module 1.5

---

### Module 1.7 — 2D Allen-Cahn Bimodal Posterior (Example 4)
**File:** `phases/phase_d_allen_cahn.py`, `core/hmcecs.py`
**Goal:** Reproduce Figs. 5–8 — bimodal posterior, mode separation, per-mode predictions.

- 2D Allen-Cahn energy Uε[φ] on [−1,1]² with ε=0.01
- 2D real Fourier basis, 9 total terms
- Switch from SGLD to HMCECS via NumPyro (paper uses NumPyro for this example)
- Observations on 3 boundaries only (15 points each, σ²=0.01²)
- Gaussian mixture model to separate posterior modes
- Four-panel output per mode: ground truth, median, mode 1, mode 2

**Deliverable:** `core/hmcecs.py`, `phase_d_allen_cahn.py`, Figs. 5–8 reproduction
**Effort:** 5–7 days
**Dependencies:** None (independent from Modules 1.2–1.6)

---

## Phase 2 — Novel Contributions (ODIL Integration + Expansion)
*Target: publishable novel results beyond the Alberts & Bilionis paper*

---

### Module 2.1 — 2D Forward Poisson
**File:** `core/parameterizations_2d.py`, `phases/phase_e_2d_poisson.py`
**Goal:** Extend working 1D PIFT pipeline to 2D.

- 2D sine basis on [0,1]² with automatic zero Dirichlet BCs
- 2D stochastic quadrature via uniform domain sampling
- Test on 2D Poisson with known analytical solution
- Beta sweep in 2D to confirm variance collapse extends to higher dimensions

**Effort:** 3–4 days
**Dependencies:** None

---

### Module 2.2 — Adaptive Beta Inference (Forward Problems)
**File:** `core/sgld.py` (extension)
**Goal:** Jointly infer β in forward problems without nested SGLD.

- Treat β as latent with Jeffrey's prior p(β) ∝ 1/β; sample log(β)
- Augment forward SGLD to jointly sample (θ, β) — valid since β doesn't affect partition function in forward case
- Compare calibration vs fixed-β PIFT across noise levels

**Effort:** 2–3 days
**Dependencies:** None

---

### Module 2.3 — MLP Field Parameterization
**File:** `core/parameterizations_mlp.py`
**Goal:** Replace Fourier basis with a small MLP to handle non-smooth fields.

- Shallow MLP (2–3 layers, tanh activations) as drop-in for SineBasisField
- Compare PIFT posterior: Fourier vs MLP on smooth and discontinuous test fields
- Demonstrate Gibbs phenomenon failure and MLP recovery
- Addresses open problem stated in Alberts & Bilionis conclusions section

**Effort:** 3–5 days
**Dependencies:** None

---

### Module 2.4 — Time-Dependent Problems
**File:** `core/energies_timedependent.py`, `phases/phase_f_heat_timedep.py`
**Goal:** Extend PIFT to parabolic PDEs (∂φ/∂t = Lφ + f).

- Space-time energy functional for heat equation
- Method of lines discretization in time
- Test on 1D heat equation with time-varying source, known analytical solution
- Addresses second open problem from Alberts & Bilionis conclusions

**Effort:** 5–7 days
**Dependencies:** None

---

### Module 2.5 — GPU Acceleration & Scalability
**File:** `benchmarks/bench_gpu_cpu.py`
**Goal:** Full JAX JIT pipeline and GPU benchmarking.

- Complete JIT compilation of inner SGLD loop (currently partial)
- GPU execution via JAX device placement — benchmark on Colab A100
- Vectorized spatial mini-batching for stochastic quadrature
- Profile nested SGLD wall time vs accuracy as function of T and T̃
- Runtime comparison table: CPU vs GPU, JIT vs no-JIT

**Effort:** 3–4 days
**Dependencies:** None (independent optimization)

---

### Module 2.6 — ODIL Proper Implementation ⭐ KEY MODULE
**File:** `core/odil.py`
**Goal:** Implement ODIL correctly per Karnakov et al. (2024), fixing the broken baseline.

- Grid-based FD discretization of PDE residual (distinct from PIFT's functional energy)
- Gauss-Newton iteration with sparse Jacobian via JAX automatic differentiation
- L-BFGS-B fallback for nonlinear problems
- Multigrid decomposition for accelerated convergence
- Loss function: L(u,θ) = Σᵢ (1/Nc) Σc (F^(i)_c[u,θ])²
- Sparse Jacobian calculation via shift operator approach (Section "Calculating sparse Jacobians")
- Test on same 1D Poisson problem: should converge in ~25 Newton iterations vs 40k SGLD steps
- Fixes current GUI ODIL failure

**Deliverable:** `core/odil.py`, `tests/test_odil.py`
**Effort:** 3–4 days
**Dependencies:** None

---

### Module 2.7 — ODIL Warm-Start for PIFT ⭐ NOVEL CONTRIBUTION
**File:** `phases/phase_a.py` (extension), `benchmarks/bench_warmstart.py`
**Goal:** Use ODIL MAP solution as theta₀ for PIFT SGLD.

- Run ODIL to convergence (~seconds on CPU)
- Project ODIL grid solution onto sine basis via least-squares
- Use projected theta as PIFT initialization instead of FD reference projection
- Compare: cold-start vs FD-warm-start vs ODIL-warm-start PIFT
  - Metrics: burn-in length, effective sample size, final L2, wall time to L2 < 0.05
- Show ODIL warm-start reduces required burn-in steps by ~50-80%
- This is the key methodological contribution: MAP-initialized Bayesian posterior

**Deliverable:** Updated `phase_a.py`, `bench_warmstart.py`, benchmark figure
**Effort:** 1–2 days
**Dependencies:** Module 2.6

---

### Module 2.8 — ODIL vs PIFT Computational Benchmark ⭐ KEY FIGURE
**File:** `benchmarks/bench_odil_vs_pift.py`
**Goal:** Reproduce spirit of Karnakov Fig. 1 in the PIFT setting — the paper's central comparison figure.

- Fix accuracy target: L2 < 0.05 on 1D Poisson
- Measure wall time: ODIL (Newton) vs ODIL (L-BFGS) vs PIFT (SGLD) vs BayesPINN vs MC
- Plot: accuracy vs wall time on log-log axes
- Annotate: ODIL reaches MAP in seconds; PIFT takes minutes but gives full posterior
- Show: ODIL-warm PIFT reaches same accuracy as cold PIFT 2-3x faster
- This figure makes the PIFT/ODIL tradeoff concrete and publishable

**Deliverable:** `bench_odil_vs_pift.py`, key comparison figure for paper
**Effort:** 2 days
**Dependencies:** Modules 2.6, 2.7

---

## Phase 3 — Publication-Ready Paper

---

### Module 3.1 — Novel Contribution Framing

**Central claim:**
> ODIL and PIFT occupy complementary positions in physics-informed inference. ODIL provides fast MAP estimation via discrete grid optimization. PIFT provides principled Bayesian posteriors via functional priors and SGLD. ODIL-warm-started PIFT achieves the posterior quality of pure PIFT with significantly reduced burn-in, combining the computational efficiency of ODIL with the uncertainty quantification of PIFT.

**Target venue:** Journal of Computational Physics (same as Alberts & Bilionis) or SIAM Journal on Scientific Computing

---

### Module 3.2 — Manuscript Structure
**Directory:** `paper/`

```
1. Abstract
2. Introduction
   - Physics-informed inference landscape
   - PIFT and ODIL: complementary approaches
   - Contribution summary
3. Background
   3.1 Information Field Theory
   3.2 PIFT: functional priors and SGLD
   3.3 ODIL: discrete loss optimization
4. Methodology
   4.1 PIFT forward problems (review)
   4.2 Nested SGLD for inverse problems (review)
   4.3 ODIL implementation
   4.4 ODIL warm-start for PIFT (novel)
   4.5 Adaptive beta inference (novel)
5. Numerical Experiments
   5.1 1D Poisson: PIFT vs MC vs ODIL (existing result)
   5.2 Beta sensitivity (Example 1 replication)
   5.3 Model-form uncertainty (Example 2 replication)
   5.4 Inverse parameter identification (Examples 3a/3b replication)
   5.5 Bimodal posterior recovery (Example 4 replication)
   5.6 ODIL vs PIFT benchmark (novel)
   5.7 ODIL warm-start ablation (novel)
   5.8 2D extension (novel)
6. Discussion
7. Conclusion
Appendix A: Proofs (Appendix B,C from Alberts & Bilionis)
Appendix B: Hyperparameter tables
Appendix C: Reproducibility (config + seeds)
```

---

### Module 3.3 — Publication Figure Suite
**Directory:** `paper/figures/`

| Figure | Content | Source module |
|---|---|---|
| Fig. 1 | PIFT vs MC vs ODIL on 1D Poisson | Phase A |
| Fig. 2 | Beta sensitivity collapse (Example 1) | Module 1.1 |
| Fig. 3 | Model-form uncertainty β posterior (Example 2) | Module 1.4 |
| Fig. 4 | Joint D-κ posterior (Example 3a) | Module 1.5 |
| Fig. 5 | Source term identification (Example 3b) | Module 1.6 |
| Fig. 6 | Bimodal posterior + mode separation (Example 4) | Module 1.7 |
| Fig. 7 | ODIL vs PIFT: accuracy vs wall time | Module 2.8 |
| Fig. 8 | ODIL warm-start ablation | Module 2.7 |
| Fig. 9 | 2D Poisson posterior | Module 2.1 |

All figures: LaTeX fonts, vector PDF, colorblind-safe palette (viridis/cividis).

---

## Phase 4 — Best-in-Class Research GUI

---

### Module 4.1 — Architecture Refactor
- Strip all hyperparameters from `UI_DEFAULT_CONFIG` and `STABLE_PRESETS`
- Single source of truth: `config/defaults.yaml`
- Runner modules return pure `RunResult` dataclasses, zero GUI imports
- Config round-trips through YAML: GUI reads → user edits → writes back → runner uses

---

### Module 4.2 — Full Example Coverage
- Add all paper examples to method selector
- Example 1: beta sweep multi-panel output
- Examples 2–3: nested SGLD with outer chain diagnostics
- Example 4: 2D field heatmap with Gaussian mixture mode selector
- ODIL proper: convergence history + MAP field visualization

---

### Module 4.3 — Diagnostics Dashboard
- Live Hamiltonian trace and gradient norm during sampling
- ESS per mode (from full autocorrelation function)
- R-hat across multiple independent chains
- Credible interval empirical coverage vs nominal level
- Preconditioner weight bar chart with condition number
- Export as LaTeX table or CSV

---

### Module 4.4 — Comparison & Benchmarking Panel
- Pin any run as baseline; compare all subsequent runs against it
- Auto-generate comparison table: L2, coverage, ESS, runtime, n_samples
- Beta sensitivity overlay: run multiple β values, overlay posteriors automatically
- ODIL vs PIFT: side-by-side MAP vs posterior visualization

---

### Module 4.5 — Data & Problem Configuration
- Load arbitrary CSV observation data
- Define custom forcing f(x) via text field with sympy parsing
- Custom Dirichlet/Neumann boundary conditions
- Save/load problem configurations as named YAML presets
- GPU/CPU toggle with JAX device placement

---

### Module 4.6 — Publication Export
- Export any plot as PDF vector or 300dpi PNG at configurable size
- Auto-generate LaTeX figure block with correct caption and label
- One-click Methods paragraph generation from current config
- Full reproducibility export: config YAML + random seed + library versions + git hash

---

## Timeline

| Phase | Modules | Effort | Can Parallelize With |
|---|---|---|---|
| 1.1 Beta sweep | 1.1 | 1–2 days | 1.2, 1.3 |
| 1.2 Nested SGLD | 1.2 | 3–5 days | 1.1, 1.3, 2.6 |
| 1.3 Nonlinear energy | 1.3 | 1–2 days | 1.2 |
| 1.4 Example 2 | 1.4 | 2–3 days | After 1.2+1.3 |
| 1.5 Example 3a | 1.5 | 1–2 days | After 1.4 |
| 1.6 Example 3b | 1.6 | 3–4 days | After 1.5 |
| 1.7 Example 4 | 1.7 | 5–7 days | Parallel to 1.4–1.6 |
| 2.1–2.5 Extensions | 2.1–2.5 | 3–4 weeks | After Phase 1 |
| **2.6 ODIL proper** | 2.6 | 3–4 days | Parallel to Phase 1 |
| **2.7 Warm-start** | 2.7 | 1–2 days | After 2.6 |
| **2.8 Benchmark** | 2.8 | 2 days | After 2.7 |
| 3 Paper | 3.1–3.3 | 2–3 weeks | After Phase 2 |
| 4 GUI | 4.1–4.6 | 3–4 weeks | Parallel throughout |
| **Total** | | **~12–14 weeks** | |

---

## Critical Path

```
Module 1.2 (Nested SGLD)
    → 1.4 (Example 2)
        → 1.5 (Example 3a)
            → 1.6 (Example 3b)

Module 2.6 (ODIL proper)          ← can start immediately
    → 2.7 (Warm-start)
        → 2.8 (Benchmark figure)

Both paths converge → Phase 3 (Paper)
```

Module 1.7 (Allen-Cahn/Example 4) and all of Phase 4 (GUI) are independent and can proceed in parallel with the critical path at any time.

---

## Key References

- Alberts, A. & Bilionis, I. (2023). Physics-informed information field theory for modeling physical systems with uncertainty quantification. *Journal of Computational Physics*, 486, 112100.
- Karnakov, P., Litvinov, S. & Koumoutsakos, P. (2024). Solving inverse problems in physics by optimizing a discrete loss: Fast and accurate learning without neural networks. *PNAS Nexus*, 3(1).
- Welling, M. & Teh, Y.W. (2011). Bayesian learning via stochastic gradient Langevin dynamics. *ICML*.
- Yang, L., Meng, X. & Karniadakis, G.E. (2021). B-PINNs: Bayesian physics-informed neural networks. *Journal of Computational Physics*, 425, 109913.