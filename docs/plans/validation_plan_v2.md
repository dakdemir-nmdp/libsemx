# Comprehensive Validation Plan (v2)

## Goal
Expand validation coverage to include simple-to-complex models, various covariance structures, estimation methods (ML/REML), and comparison with `lm`, `glm`, `lme4`, `nlme`, `lavaan`, and `sommer`.

## Strategy
Use simulated data where possible to have ground truth, and compare `libsemx` estimates with established R packages. Validate both parameter recovery and parity with external software; trust simulation truth over any single comparator when discrepancies appear.

## Test Cases

### 1. Simple Fixed Effects Models (vs `lm`, `glm`)
- [ ] **Linear Regression**: $y = \beta_0 + \beta_1 x + \epsilon$. Compare coefficients, SEs, LogLik with `lm()`.
- [ ] **Logistic Regression**: $logit(p) = \beta_0 + \beta_1 x$. Compare with `glm(family=binomial)`.
- [ ] **Poisson Regression**: $log(\lambda) = \beta_0 + \beta_1 x$. Compare with `glm(family=poisson)`.

### 2. Linear Mixed Models (vs `lme4`, `nlme`)
- [ ] **Random Intercept** (ML & REML): $y_{ij} = \beta_0 + \beta_1 x_{ij} + u_j + \epsilon_{ij}$; parity on coefficients, variance components, log-likelihoods.
- [ ] **Random Slope** (unstructured vs diagonal): $y_{ij} = \beta_0 + \beta_1 x_{ij} + u_{0j} + u_{1j} x_{ij} + \epsilon_{ij}$.
- [ ] **Random-only models**: no fixed predictors beyond intercept to probe REML behavior.
- [ ] **Crossed/nested factors**: small crossed example to ensure grouping handling matches `lme4::lmer` and `nlme::lme`.

### 3. GLMMs and Non-Gaussian Mixed Models (vs `glm`, `glmer`)
- [ ] **Binomial Random Intercept**: logit model with cluster-level intercept; align fixed effects, random-effect variance, log-likelihood.
- [ ] **Poisson Random Intercept**: compare to `glmer` (or `glmmTMB` if needed for stability).
- [ ] **Negative Binomial**: simulate over-dispersion; ensure dispersion/shape is recovered and matches Laplace path.
- [ ] **Mixed-type random effects**: intercept + slope for a non-Gaussian outcome (binomial/poisson).

### 4. Covariance Structures (vs `nlme`, `sommer`)
- [ ] **AR(1) Residuals vs `nlme::corAR1`**: longitudinal data with known $\rho$; compare estimated $\rho$, variance, and log-likelihood under ML/REML.
- [ ] **Compound Symmetry / Toeplitz**: parity on parameterization and scaling; check both random-effect and residual parameterizations.
- [ ] **Factor-Analytic**: small FA(1) block to verify loadings/uniquenesses against `nlme`.
- [ ] **Heterogeneous variances**: group-wise residuals (e.g., `varIdent` analogue).

### 5. SEM & Multi-Outcome Models (vs `lavaan`, `nlme`)
- [ ] **Path Analysis**: $y_1 \leftarrow x$, $y_2 \leftarrow y_1 + x$ (Gaussian).
- [ ] **CFA**: Latent variable measuring multiple indicators (Gaussian).
- [ ] **SEM + Mixed Outcomes**: mixed Gaussian/Binomial path model; compare SEM mode vs non-SEM mode for the same structure.
- [ ] **Bivariate/Multivariate LMM**: multi-outcome random effects (Gaussian + non-Gaussian) vs `nlme` where applicable.

### 6. Genomic Selection / GxE (vs `sommer`)
- [ ] **GBLUP**: $y = X\beta + Zu + \epsilon$, $Var(u) = K \sigma^2_u$ (ML & REML); compare variance components, BLUPs, and likelihoods.
- [ ] **Multi-kernel / RKHS**: weighted kernels, simplex/softmax weights; check scaling and gradients.
- [ ] **G×E / Kronecker**: simulate small multi-environment trait with Kronecker kernel; compare to `sommer::mmer`.
- [ ] **Sensitivity to precomputed kernels**: validate parity between `markers`-derived GRM and precomputed matrices.

### 7. Validation Methodology (truth-first)
- [ ] Use simulated datasets with known $\beta$, variance components, and correlation parameters; assert recovery within tolerances before cross-software comparison.
- [ ] For each scenario, capture ML vs REML fits separately and ensure all software uses the same likelihood definition.
- [ ] Record cases where comparator packages disagree with simulated truth; flag for further investigation instead of auto-accepting comparator output.

## Implementation Plan
1. Create/extend `validation/comprehensive_validation.R` with modular helpers for simulation, fitting (`libsemx`, `lm`/`glm`, `lme4`, `nlme`, `sommer`), and comparison.
2. Implement simulation functions for each model type and covariance structure (support ML/REML switches).
3. Run `libsemx` and comparator models, capturing parameters, SEs, variance components, and log-likelihoods.
4. Report differences against both simulated truth and comparator outputs; add assertions/tolerance bands and log any systematic drifts.
