# Deterministic Survival + CIF Fixture

Blueprint §7 introduces the parametric survival families (lognormal/loglogistic alongside Weibull/exponential) and the requirement to accumulate cause-specific contributions into a single CIF-aware objective. This document records the deterministic payload exercised in Catch2, Python, and R tests so that future documentation and examples stay synchronized.

## Scenario

- Two observed survival variables share the same event times but use different AFT families.
- `cause_lognormal` follows the lognormal AFT parameterization with dispersion interpreted as the log-scale standard deviation.
- `cause_loglogistic` follows the log-logistic AFT parameterization with dispersion representing the gamma/shape parameter.
- Status/censoring indicators follow the usual `1 = event`, `0 = right-censored` convention.

| Observation | Time | cause_lognormal η | cause_lognormal σ | status | cause_loglogistic η | cause_loglogistic γ | status |
|-------------|------|-------------------|-------------------|--------|---------------------|---------------------|--------|
| 1           | 1.3  | 0.20              | 0.80              | 1      | -0.30               | 1.20                | 0      |
| 2           | 2.0  | -0.15             | 1.05              | 0      | 0.35                | 1.00                | 1      |
| 3           | 3.6  | 0.25              | 0.90              | 0      | 0.05                | 1.10                | 0      |

These values align with the analytic helper functions `lognormal_loglik` and `loglogistic_loglik` that appear in both the Python and R binding tests. Any change to the fixture should update this table and the shared helpers together.

## Minimal Python Example

```python
from semx import LikelihoodDriver, ModelIRBuilder, VariableKind

builder = ModelIRBuilder()
builder.add_variable("cause_lognormal", VariableKind.Observed, "lognormal")
builder.add_variable("cause_loglogistic", VariableKind.Observed, "loglogistic")
model = builder.build()

times = [1.3, 2.0, 3.6]
data = {"cause_lognormal": times, "cause_loglogistic": times}
linear_predictors = {
    "cause_lognormal": [0.2, -0.15, 0.25],
    "cause_loglogistic": [-0.3, 0.35, 0.05],
}
dispersions = {
    "cause_lognormal": [0.8, 1.05, 0.9],
    "cause_loglogistic": [1.2, 1.0, 1.1],
}
status = {
    "cause_lognormal": [1.0, 0.0, 0.0],
    "cause_loglogistic": [0.0, 1.0, 0.0],
}

driver = LikelihoodDriver()
loglik = driver.evaluate_model_loglik(
    model,
    data,
    linear_predictors,
    dispersions,
    status=status,
)
print(f"log-likelihood: {loglik:.10f}")
```

## Minimal R Example

```r
library(semx)

builder <- new(ModelIRBuilder)
builder$add_variable("cause_lognormal", 0L, "lognormal")
builder$add_variable("cause_loglogistic", 0L, "loglogistic")
model <- builder$build()

times <- c(1.3, 2.0, 3.6)
data <- list(cause_lognormal = times, cause_loglogistic = times)
linear_predictors <- list(
  cause_lognormal = c(0.2, -0.15, 0.25),
  cause_loglogistic = c(-0.3, 0.35, 0.05)
)
dispersions <- list(
  cause_lognormal = c(0.8, 1.05, 0.9),
  cause_loglogistic = c(1.2, 1.0, 1.1)
)
status <- list(
  cause_lognormal = c(1, 0, 0),
  cause_loglogistic = c(0, 1, 0)
)

driver <- new(LikelihoodDriver)
loglik <- driver$evaluate_model_loglik_full(
  model,
  data,
  linear_predictors,
  dispersions,
  list(),
  status,
  list(),
  NULL,
  0L
)
print(loglik)
```

Both snippets are covered by automated tests (`cpp/tests/survival_likelihood_tests.cpp`, `python/semx/tests/test_likelihood_driver.py`, and `Rpkg/semx/tests/testthat/test-bindings.R`), making this a reliable reference point for documentation or future regression checks.
