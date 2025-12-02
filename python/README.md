# semx: Python Front-End for libsemx

Python package providing a user-friendly interface to the libsemx unified SEM/mixed-model engine.

## Installation

This package is part of the libsemx project. Install in development mode:

```bash
cd python
uv pip install -e .
```

For testing:

```bash
uv pip install -e ".[test]"
```

## Usage

### Formula-based model builder

Blueprint §§8–10 call for a thin, ergonomic front-end that turns equations
into the shared ``ModelIR`` payload. The :class:`semx.Model` helper mirrors
lavaan syntax (``=~``, ``~``, ``~~``) and keeps the bindings synchronized with
the C++ registry:

```python
from semx import LikelihoodDriver, Model

spec = Model(
	equations=[
		"eta =~ y1 + y2 + y3",
		"y1 ~ x1 + x2",
		"y2 ~ eta + g",
		"y1 ~~ y2",
	],
	families={"y1": "gaussian", "y2": "gaussian", "y3": "gaussian"},
	kinds={"g": "grouping"},
)

driver = LikelihoodDriver()
loglik = driver.evaluate_model_loglik_full(
	spec.to_ir(),
	data={"y1": [1.2, 0.7], "y2": [0.1, -0.3], "y3": [0.4, 0.8]},
	linear_predictors={},
	dispersions={},
)
```

The cached ``variables`` and ``edges`` properties mirror the deterministic test
fixtures so Python examples stay aligned with the C++ parameter catalog.

### Survival + Competing Risks quickstart

Blueprint §7 describes the survival families (Weibull, exponential, lognormal, loglogistic) plus CIF aggregation. The Python bindings already expose these families through the low-level IR builder:

```python
from semx import LikelihoodDriver, ModelIRBuilder, VariableKind

builder = ModelIRBuilder()
builder.add_variable("cause_lognormal", VariableKind.Observed, "lognormal")
builder.add_variable("cause_loglogistic", VariableKind.Observed, "loglogistic")
model = builder.build()

times = [1.3, 2.0, 3.6]
data = {
	"cause_lognormal": times,
	"cause_loglogistic": times,
}
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
print(f"log-likelihood: {loglik:.6f}")
```

The payload above matches the deterministic fixture documented in `docs/examples/survival_cif.md` and mirrors the Catch2/Python/R unit tests so regressions surface quickly.

## Testing

Run tests with:

```bash
uv run pytest
```

## Development

- Follow the main project conventions
- Tests mirror C++ fixtures for consistency
- Use `uv` for all Python operations