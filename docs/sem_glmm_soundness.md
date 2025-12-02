# SEM–GLMM theoretical review

This note summarizes a quick audit of the theoretical correctness and modeling coverage of the SEM and GLMM machinery in the current `libsemx` core.

## Strengths
- Covariance parameterization is PSD-aware (log transforms for variances, Cholesky-based unstructured factors) and enforced before likelihood evaluation (`cpp/src/libsemx/model_objective.cpp:24-72`, `cpp/src/libsemx/covariance_structure.cpp:1-260`).
- Gaussian mixed-model likelihood uses the standard REML-adjusted closed form with explicit `V = Z G Zᵀ + R` construction and Cholesky solves (`cpp/src/libsemx/likelihood_driver.cpp:842-1001`), matching the usual linear mixed model theory.
- Laplace approximation for non-Gaussian GLMMs follows the textbook form `ℓ ≈ ℓ(y|û) + ℓ(û) - ½ log|H(û)| + ½ q log 2π` with Newton iterations on the mode and a log-det from the negative Hessian (`cpp/src/libsemx/likelihood_driver.cpp:361-462`, `cpp/src/libsemx/likelihood_driver.cpp:980-1001`).
- SEM gradients for the current CFA-like path are wired analytically and checked against finite differences (`cpp/tests/sem_gradient_tests.cpp`), reducing numerical noise relative to finite-difference optimization.

## Key gaps and risks

### SEM path (stacked mixed-model trick)
- Measurement-only: once latent variables are detected, the SEM objective hard-codes zero linear predictors for `_stacked_y`, so all structural regressions/means (observed→observed, latent→latent, intercepts) are ignored (`cpp/src/libsemx/model_objective.cpp:90-124`, `cpp/src/libsemx/model_objective.cpp:144-168`). The fitted model collapses to a CFA with zero means regardless of specified regressions.
- Latent covariance structure is forced to independent factors: each latent gets its own 1×1 diagonal covariance, and only variance edges of the form `η ~~ η` are mapped to parameters. Cross-latent covariances (e.g., `η1 ~~ η2`) are never materialized (`cpp/src/libsemx/model_objective.cpp:334-360`), so correlated factors or structural disturbance covariances are silently dropped.
- Residual covariances between indicators are ignored. The SEM stacking only reads edges where `source == target` when filling dispersions, so paths like `y1 ~~ y2` never influence the likelihood (`cpp/src/libsemx/model_objective.cpp:399-425`).
- No mean structure or intercept handling: the stacked outcome is centered at zero, and there is no way to estimate item intercepts or latent means. This deviates from standard SEM/FIML practice and can bias loadings/variances when means are non-zero.
- Multivariate outcomes beyond the measurement model are not supported: only variables targeted by edges are stacked as outcomes (`cpp/src/libsemx/model_objective.cpp:276-293`), and any regressions on those outcomes are lost because the SEM-specific branch bypasses `build_prediction_workspaces`.

### GLMM path (Laplace and Gaussian)
- Single-response assumption: the mixed-model branches pick the first observed target and ignore additional outcomes (`cpp/src/libsemx/likelihood_driver.cpp:761-799`), so multivariate GLMMs are unsupported even though the IR permits multiple observed variables.
- Dispersion and threshold parameters are not optimized under Laplace. Gradients for dispersions are only produced in the Gaussian branch (`cpp/src/libsemx/likelihood_driver.cpp:1258-1310`), while the non-Gaussian Laplace branch omits them entirely, leaving NB/probit thresholds or overdispersion fixed at their initial values.
- `EstimationMethod` is effectively unused in optimization: the gradient explicitly discards the method flag (`cpp/src/libsemx/likelihood_driver.cpp:1015-1017`), and `ModelObjective` always evaluates ML, so REML requests can be inconsistent between value and gradient.
- Laplace mode solver has no damping/line search and returns `-∞` on any Cholesky failure after up to 30 Newton steps (`cpp/src/libsemx/likelihood_driver.cpp:411-462`). This makes the approximation brittle for large random-effects variances or near-separation and can derail outer optimization without diagnostics.
- Extra-model parameters (e.g., ordinal thresholds in `extra_params`) are not part of the parameter catalog, so they cannot be estimated through the standard `fit()` path.

## Recommendations
- Extend the SEM branch to honor the full structural model: propagate regression edges into `_stacked_y` predictors, carry intercepts/means, and map cross-latent and residual covariances into the random-effects block or a multivariate normal likelihood instead of assuming independence.
- Add explicit handling for latent–latent covariances (and potentially FA-style covariance structures) when building the SEM `ModelIR`, so correlated factors are identified rather than silently dropped.
- Implement dispersion/threshold parameter handling for non-Gaussian GLMMs (both in the likelihood and gradient) and expose them through the parameter catalog so NB size, ordinal cutpoints, or overdispersion can be estimated.
- Make `EstimationMethod` plumbed through `ModelObjective::value`/`gradient`, or remove the flag to avoid mismatched ML/REML behavior; if REML is kept, provide gradient support for it. (I want to keep both reml and ML)
- Harden the Laplace solver with step-halving/damping and surfaced diagnostics instead of returning `-∞`, so outer optimizers can backtrack rather than failing silently.
