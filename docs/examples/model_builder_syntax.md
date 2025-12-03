# Model Builder Syntax Guide

Reference for the Python `semx.Model` and R `semx_model` front-ends. Both emit
the same `ModelIR` and share the exact syntax and option names.

## Core arguments
- `equations`: lavaan/lme4-style strings. Blank lines ignored.
- `families`: named map of observed outcomes to likelihoods. Required for every
  observed variable that appears on the left of `~` or `=~` (except `_intercept`
  which is auto-registered when needed).
- `kinds` (optional): force variable kinds (`observed`, `latent`, `grouping`).
- `covariances` (optional): list of covariance specs (`name`, `structure`,
  `dimension`). Dimension = number of random-effect coefficients per level.
- `random_effects` (optional): list of random-effect specs (`name`,
  `variables`, `covariance`). `variables = [group, design1, design2, ...]`.
- `genomic` (optional): map `covariance_id -> {markers, structure?, center?,
  normalize?, precomputed?}`. Registers the covariance and supplies fixed
  kernels to the driver.

## Equation syntax
- **Loadings (`=~`)**: `eta =~ y1 + y2 + y3`. First loading auto-fixed to 1.0.
- **Regressions (`~`)**: `y ~ x1 + x2`. Predictors `1` or `0` toggle fixed
  intercepts but are not stored as edges.
- **Covariances (`~~`)**: `y1 ~~ y2` (order-insensitive).
- **Survival**: `Surv(time, status) ~ x1 + x2`. `time` must declare a survival
  family; `status` column is passed automatically during `fit()`.
- **Mixed-model shorthand**: `(1 | g)` (random intercept), `(slope | g)`,
  `(s1 + s2 | g)`, `(0 + slope | g)` (no intercept). These add random effects
  with an *unstructured* covariance of dimension `q = #design terms [+1 if
  intercept]`.

## Random effects (explicit list form)
Use when you want to pick a specific covariance structure or wire genomic
covariances:

```python
Model(
    equations=["y ~ x"],
    families={"y": "gaussian", "x": "gaussian"},
    covariances=[{"name": "G", "structure": "diagonal", "dimension": 1}],
    random_effects=[{"name": "u_g", "variables": ["g"], "covariance": "G"}],
    kinds={"g": "grouping"},
)
```

Fields:
- `name`: identifier for the random effect block.
- `variables`: `[group, design1, design2, ...]`. Group is treated as
  `VariableKind.Grouping`; designs are observed.
- `covariance`: id matching a `covariances` entry (or genomic id).

## Covariance structures
Normalized identifiers (case-insensitive aliases in parentheses):
- `unstructured` (default for `(… | g)`): free lower triangle.
- `diagonal`: independent coefficients.
- `scaled_fixed`: single scale parameter times a fixed matrix (supply via
  `fixed_covariance_data`).
- `grm` / `genomic`: positive-definite kernel from markers (auto-supplied via
  `genomic` argument).
- `multi_kernel`, `multi_kernel_simplex`: weighted sum of fixed kernels.
- `compound_symmetry` / `cs`: σ² on diagonal plus common covariance.
- `ar1`: ρ^|i-j| with marginal variance.
- `toeplitz`: banded covariance with free lags.
- `fa{k}`: factor-analytic rank-`k` (e.g., `fa1`). Requires `0 < k < dim`.
- `kronecker`: Kronecker product structure (dimension is the block size q).

`dimension` always refers to the number of coefficients per level, not the
number of groups. For random slopes with intercept: `dimension = q`.

## Outcome families
Exact strings accepted by the core factory:
- `gaussian`
- `binomial`
- `poisson`
- `negative_binomial` / `nbinom`
- `weibull`, `exponential`, `lognormal`, `loglogistic` (AFT survival)
- `ordinal` / `probit` (threshold model)

Dispersion conventions (internally named `psi_*`):
- Gaussian: residual variance per outcome.
- Binomial/Poisson: fixed to 1 (no dispersion parameter).
- Negative binomial: dispersion/shape.
- Survival families: scale/shape per outcome (exponential fixes shape = 1).
- Ordinal: ordered thresholds plus latent variance fixed to 1.

## Data conventions
- Provide numeric matrices/vectors. Factor strings should be encoded to integers
  starting at 0 for grouping variables.
- Observed variables must appear in `families`; grouping variables can be set via
  `kinds` or inferred from `(… | g)` terms.
- `_intercept` is injected automatically when random slopes use an intercept; no
  family entry is needed.
- For survival models include both `time` and `status` columns in the data
  passed to `fit()`.
- Fixed/genomic covariances: `Model.fixed_covariance_data()` returns the kernel
  payload to reuse in `LikelihoodDriver.fit`.

## Examples
- **Latent CFA + regression**:
  ```python
  Model(
      equations=[
          "eta =~ y1 + y2 + y3",
          "y1 ~ x1 + x2",
          "y2 ~ eta",
          "y1 ~~ y2",
      ],
      families={k: "gaussian" for k in ["y1", "y2", "y3", "x1", "x2"]},
  )
  ```
- **Survival with competing risks**:
  ```python
  Model(
      equations=[
          "Surv(t_weibull, status_weibull) ~ x",
          "Surv(t_lognormal, status_lognormal) ~ x",
      ],
      families={
          "t_weibull": "weibull",
          "t_lognormal": "lognormal",
          "x": "gaussian",
      },
  )
  ```
- **G×E with genomic kernel**: see `data/README.md` “mdp_traits / mdp_numeric”.
