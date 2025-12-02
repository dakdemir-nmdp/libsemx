# Formula Front-Ends

Blueprint §§8–10 describe thin bindings that translate ergonomic formulas into
the shared `ModelIR`. The examples below build the same latent-variable model
from Python and R, demonstrating how both front-ends stay aligned with the C++
parameter registry.

## Model specification

```text
equations:
  eta =~ y1 + y2
  y1 ~ x1 + x2
  y2 ~ eta + g
  y1 ~~ y2
families:
  y1, y2 -> gaussian
kinds:
  g -> grouping
```

## Python

```python
from semx import LikelihoodDriver, Model

spec = Model(
    equations=[
        "eta =~ y1 + y2",
        "y1 ~ x1 + x2",
        "y2 ~ eta + g",
        "y1 ~~ y2",
    ],
    families={"y1": "gaussian", "y2": "gaussian"},
    kinds={"g": "grouping"},
)

driver = LikelihoodDriver()
model_ir = spec.to_ir()
loglik = driver.evaluate_model_loglik(
    model_ir,
    data={"y1": [1.0, 0.4], "y2": [0.7, -0.2]},
    linear_predictors={},
    dispersions={},
)
```

## R

```r
library(semx)

spec <- semx_model(
  equations = c(
    "eta =~ y1 + y2",
    "y1 ~ x1 + x2",
    "y2 ~ eta + g",
    "y1 ~~ y2"
  ),
  families = c(y1 = "gaussian", y2 = "gaussian"),
  kinds = c(g = "grouping")
)

ldr <- LikelihoodDriver()
loglik <- ldr$evaluate_model_loglik(
  spec$ir,
  data = list(y1 = c(1.0, 0.4), y2 = c(0.7, -0.2)),
  linear_predictors = list(),
  dispersions = list()
)
```

Both snippets emit identical IR payloads, ensuring downstream tests (Laplace,
Kronecker, survival) continue to exercise the same deterministic fixtures across
languages.

## Testing

Run the parser-focused suites directly from the project root:

```bash
PYTHONPATH=build uv run pytest python/semx/tests -k model_api
uv run R -q -e "testthat::test_dir('Rpkg/semx/tests/testthat', filter = 'bindings')"
```

These commands are wired into CI so regressions in either front-end are caught
before the C++ stack is exercised.

## Mixed Models (Random Effects)

The front-ends also support specifying random effects and covariance structures directly.

### Python

```python
model = Model(
    equations=["y ~ 1 + cluster"],
    families={"y": "binomial"},
    kinds={"cluster": "grouping"},
    covariances=[
        {"name": "G_diag", "structure": "diagonal", "dimension": 1}
    ],
    random_effects=[
        {"name": "u_cluster", "variables": ["cluster"], "covariance": "G_diag"}
    ]
)
```

### R

```r
model <- semx_model(
  equations = c("y ~ 1 + cluster"),
  families = c(y = "binomial"),
  kinds = c(cluster = "grouping"),
  covariances = list(
    list(name = "G_diag", structure = "diagonal", dimension = 1)
  ),
  random_effects = list(
    list(name = "u_cluster", variables = c("cluster"), covariance = "G_diag")
  )
)
```

