# Plan: Front-End Integration Coverage (2025-12-15)

Blueprint §§8–10 call for parity between the ergonomic formula APIs and the
existing deterministic fixtures (Laplace, Kronecker, survival). To close the
remaining TODO bullets we will:

1. **Laplace GLMM via Python `Model`** – Reuse the binomial random-intercept
   payload from `python/semx/tests/test_laplace_gradients.py`, but build the IR
   through the formula parser (``eta =~`` + ``y ~ x + g``). Assert
   `LikelihoodDriver.fit` matches the baseline log-likelihood/gradient values.
2. **Kronecker covariance smoke via Python** – Serialize the Kronecker +
   diagonal multi-kernel fixture (blueprint §7.2) with `Model`, attach
   `kinds={'env': 'grouping'}` for the environment factors, and confirm the
   resulting IR produces the same parameter ordering and Laplace convergence as
   `test_kronecker_laplace_gradient`.
3. **Survival + competing risks via R `semx_model`** – Mirror
   `docs/examples/survival_cif.md` by constructing lavaan-style equations in R
   and piping them through `LikelihoodDriver$evaluate_model_loglik` to validate
   CIF aggregation stays deterministic.
4. **Shared helper scaffolding** – Introduce small builders (one per language)
   that translate parsed models into the existing deterministic data payloads so
   later fixtures (Gaussian REML, ordinal Laplace) can plug in with minimal
   boilerplate.
