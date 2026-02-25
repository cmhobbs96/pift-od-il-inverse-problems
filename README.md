# Project Scope — Physics-Informed Information Field Theory (PIFT) with Optional Inverse Problems Extension (ODIL)

Technical companion: [TECHNICAL_README.md](TECHNICAL_README.md)
Source layout guide: [src/STRUCTURE.md](src/STRUCTURE.md)
Global code lives directly under `src/` (`core/`, `models/`, `pipelines/`, `utils/`).

Frontend:
- Active GUI: `frontend/app.py` (PySide6)

## 1. Repository
**Repo name:** `pift-od-il-inverse-problems`  
**Description:** Reproducible implementation of Physics-Informed Information Field Theory (PIFT) for uncertainty-aware field inference from sparse/noisy measurements, with an optional ODIL-style inverse-problems baseline and comparisons on accuracy, cost, and uncertainty calibration.

---

## 2. Problem Statement
Scientific ML and inverse problems frequently require inferring an unknown **continuous field** (state) from sparse/noisy measurements while incorporating **PDE constraints**. Many popular approaches (PINNs, Bayesian PINNs, GP surrogates) can be:
- discretization/parameterization dependent,
- prone to uncertainty collapse under strong physics enforcement,
- weak at representing ill-posedness/multimodality,
- missing a systematic mechanism to model **model-form uncertainty** (incorrect/incomplete physics).

This project implements and documents **Physics-Informed Information Field Theory (PIFT)** as a discretization-independent Bayesian approach for posterior inference over fields, with an optional baseline extension using **Optimization-based Discrete Loss** (ODIL) methods for fast deterministic inverse solves.

---

## 3. Primary Reference
Alberts, A., & Bilionis, I. (2023). *Physics-informed information field theory for modeling physical systems with uncertainty quantification*. Journal of Computational Physics, 486, 112100.

**Optional extension reference:**  
*Solving inverse problems in physics by optimizing a discrete loss: Fast and accurate learning without neural networks* (ODIL-style approach).

---

## 4. Objectives

### 4.1 Primary Objective (PIFT)
Implement a reproducible PIFT pipeline that:
1. Defines a **physics-informed functional prior** over a field $\phi$:  
   $$
   p(\phi) \propto \exp\left(-\beta U[\phi]\right),
   $$
   where $U[\phi]$ encodes PDE consistency (energy functional or integrated squared residual).
2. Defines a measurement likelihood $p(d \mid \phi)$ using a measurement operator $R$ and noise model.
3. Samples from the posterior $p(\phi \mid d)$ via a parameterization $\phi(\cdot;\theta)$ and **SGLD**.
4. Produces posterior summaries: mean, variance, and posterior predictive intervals.
5. Validates against a reference solver and Monte Carlo checks using synthetic data.

### 4.2 Secondary Objective (Optional: Inverse Problems within PIFT)
Extend to inverse problems where unknown physical parameters affect the partition function, requiring **nested SGLD**:
- Inner loop samples fields under fixed parameter(s),
- Outer loop updates parameter(s) using expectation-difference gradient identities.

Targets include:
- inference of $\beta$ (model-form uncertainty),
- inference of PDE parameters (e.g., diffusivity $\kappa$, damping $D$, forcing coefficients).

### 4.3 Optional Extension (ODIL Baseline)
Implement an ODIL-style baseline:
- Discretize the PDE and unknown field on a grid,
- Define a discrete residual operator,
- Optimize a discrete loss (gradient-based / Gauss–Newton) to recover a point estimate.

Compare ODIL vs PIFT on:
- reconstruction accuracy,
- wall-clock cost,
- uncertainty quantification (PIFT only),
- robustness under physics mismatch.

---

## 5. Scope Boundaries

### In Scope
- 1D (primary) PDE/ODE examples for tractability and clear diagnostics:
  - Example candidates: heat equation, Poisson-type equation, simple nonlinear ODE/PDE.
- Synthetic data generation with controlled Gaussian noise.
- Forward posterior sampling via SGLD.
- Posterior summaries and calibration diagnostics.
- Reference solver implementation for verification (finite difference, spectral, or shooting in 1D).
- Optional nested SGLD for parameter inference.
- Optional ODIL baseline + comparison report.

### Out of Scope (for this phase)
- Large-scale 2D/3D PDEs with production-level performance.
- Full HMC/NUTS pipelines unless needed for a focused comparison.
- Real-world experimental datasets (unless a small dataset is trivially integrable).
- Extensive architecture search (e.g., multiple NN backbones); emphasis is on method correctness and reproducibility.

---

## 6. Method Overview (Technical)

### 6.1 Posterior Definition
Given measurements $d$, measurement operator $R$, likelihood term $\Omega$, and physics energy $U$:
- Likelihood:
  $$
  p(d \mid \phi) \propto \exp\left(-\Omega(d, R\phi)\right)
  $$
- Physics-informed prior:
  $$
  p(\phi) \propto \exp\left(-\beta U[\phi]\right)
  $$
- Posterior:
  $$
  p(\phi \mid d) \propto p(d \mid \phi)\,p(\phi)
  $$
- Information Hamiltonian:
  $$
  H[\phi \mid d] = \Omega(d, R\phi) + \beta U[\phi]
  $$

### 6.2 Numerical Representation
Parameterize the field:
- Basis approach (recommended first): truncated Fourier basis + boundary embedding  
- NN approach (optional): $\phi(x;\theta)$ via MLP

Approximate integrals in $U[\phi]$ via stochastic quadrature:
- sample spatial points $x \sim q(x)$,
- compute unbiased gradient estimates.

### 6.3 Sampling (Forward Problems)
Use **Stochastic Gradient Langevin Dynamics (SGLD)** on $\theta$:
- stochastic gradient step on $\nabla_\theta H$,
- injected Gaussian noise consistent with the SGLD update rule,
- step-size scheduling + diagnostics.

### 6.4 Inverse Problems (Optional)
For unknown physics parameters $\lambda$ that change the partition function:
- implement nested SGLD (inner field sampling, outer parameter updates),
- begin with $\beta$ inference as the simplest case.

---

## 7. Implementation Plan

### Phase A — PIFT Forward Problem (Required)
1. Select 1D PDE/ODE example and define:
   - domain, boundary conditions,
   - measurement operator $R$,
   - synthetic ground truth and noise model.
2. Implement:
   - physics energy functional $U[\phi]$,
   - likelihood $\Omega(d, R\phi)$,
   - field parameterization $\phi(\cdot;\theta)$.
3. Implement SGLD sampling loop + stochastic quadrature.
4. Produce:
   - posterior mean/variance over $\phi(x)$,
   - posterior predictive intervals at measurement points,
   - plots and diagnostics.

### Phase B — Validation + Diagnostics (Required)
- Reference solver (finite difference / spectral / shooting) to compute ground truth.
- Convergence and mixing checks:
  - trace plots of key coefficients,
  - autocorrelation / effective sample proxies,
  - calibration: coverage of credible intervals.

### Phase C — Inverse Problems (Optional)
- Implement nested SGLD for $\beta$ (model-form uncertainty) and/or PDE parameters.
- Stress test under physics mismatch (incorrect PDE term or wrong boundary condition prior).

### Phase D — ODIL Baseline (Optional)
- Implement discrete grid-based residual loss optimization.
- Compare against PIFT: cost vs accuracy vs uncertainty.

---

## 8. Tools, Libraries, and Compute

### Stack
- **Python**: NumPy, SciPy, Matplotlib
- **Optional acceleration**: JAX (JIT + autodiff), NumPyro (if exploring HMC variants)
- **Reproducibility**: pinned dependencies + seed control

### Compute
- Laptop is sufficient for 1D forward sampling and diagnostics.
- Optional GPU is helpful for longer chains, nested SGLD, or larger parameterizations (JAX path).

---

## 9. Evaluation Criteria

### Correctness
- Matches reference solutions in posterior mean (within tolerance).
- Posterior predictive intervals correctly reflect observation noise.

### Uncertainty Quality
- Calibration checks: empirical coverage of credible intervals.
- Stability across discretizations/parameterizations (basis resolution sensitivity).

### Efficiency
- Wall-clock timings for forward sampling.
- If ODIL implemented: compare solve time vs accuracy.

### Robustness
- Ill-posed settings (sparse measurements): posterior broadening and/or multimodality.
- Physics mismatch: learned $\beta$ decreases (less trust in physics), posterior shifts toward data.

---

## 10. Deliverables

### Required Deliverables
1. Reproducible PIFT forward problem implementation (script/notebook).
2. Figures:
   - posterior mean/variance over the field,
   - posterior predictive vs observed data,
   - trace/diagnostic plots.
3. Short technical write-up summarizing:
   - setup, priors/likelihood, sampling method,
   - validation results and key observations.

### Optional Deliverables
4. Nested SGLD inverse parameter inference ($\beta$ and/or PDE parameters) with posterior summaries.
5. ODIL baseline implementation + comparison table/report.

---

## 11. Risks and Mitigations
- **SGLD tuning / poor mixing:** start with unimodal examples; use conservative step-size schedules; add diagnostics early.
- **High variance stochastic quadrature:** mini-batch spatial sampling; increase samples per step; test control variates if needed.
- **Nested SGLD cost:** start with single-parameter inference ($\beta$); keep inner chains short with diagnostics.
- **Validation complexity:** implement a simple reference solver early to prevent silent failures.

---

## 12. Definition of Done
The project is complete when:
- A forward PIFT pipeline runs end-to-end on a 1D example, producing posterior samples and calibrated uncertainty summaries, validated against a reference solution.
- (Optional) At least one inverse-parameter experiment or ODIL baseline comparison is implemented and documented with quantitative results.


## 13. Expansion from original paper
There remain a number of open problems regarding the method. First, analytic representations of the functional priors 
and posteriors were able to be derived for quadratic operators. Ideally, this could be generalized to other operators. Doing 
so requires perturbation approximations of the path integrals that appear or through the rich theory of Feynman diagrams 
[69]. The application of Feynman diagrams to classical IFT has been explored in [70] for  Gaussian random fields.Because the model is able to detect when the representation of the physics is incorrect, this contributes to the problem 
of  quantifying  model-form  uncertainty.  However,  the  exact  nature  of  this  contribution  is  not  completely  understood,  and 
further analytical and empirical studies could prove to be insightful. The theory presented here is developed only for de-
terministic problems without dynamics, and it would be beneficial to study time-dependent problems with stochastic state 
transitions. Because the overall goal of this paper is to introduce the theory, problems coming from real physical systems 
were not presented, and this leaves gaps for real-world applications to be explored. As Fourier series in 1D or 2D was the 
primary choice of surrogate functions in this paper, it may be useful to study how other choices perform. There are certain 
applications for which the Fourier series is a bad choice. For example, in situations for which the field has discontinuities 
or “jumps”, the Fourier series will fail to accurately capture the discontinuities due to the notorious and well-documented 
Gibbs phenomenon. In this type of situation a different surrogate which can capture discontinuities must be used, such as 
a deep neural network with an appropriate activation function. Finally, the approach used here to numerically approximate 
the  field  posteriors  was  primarily  stochastic  gradient  Langevin  dynamics,  and  it  may  be  interesting  to  develop  different 
approaches based on other schemes such as a variational inference approach [30].24
