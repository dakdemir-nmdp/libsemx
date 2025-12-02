# Data for libsemx demonstrations

## sleepstudy (lme4)
- Source: Rdatasets mirror of `lme4::sleepstudy` – https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/lme4/sleepstudy.csv
- What it captures: Reaction time (ms) for 18 subjects measured across 10 days of sleep deprivation; classic linear mixed model example with subject-level random intercepts and slopes.
- File: `data/sleepstudy.csv` (180 rows, 4 columns)
- Columns:
  - `rownames`: row index from Rdatasets (drop for modeling)
  - `Reaction`: average reaction time in milliseconds
  - `Days`: number of days of restricted sleep (0–9)
  - `Subject`: subject identifier (character/factor)

## bfi (psych)
- Source: Rdatasets mirror of `psych::bfi` – https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/psych/bfi.csv
- What it captures: 25 Likert-scale personality items measuring the Big Five traits plus demographics; suited for factor analysis or SEM with latent variables.
- File: `data/bfi.csv` (2,800 rows, 29 columns)
- Columns:
  - `rownames`: row index from Rdatasets (drop for modeling)
  - `A1`–`A5`: Agreeableness items (1–6 Likert)
  - `C1`–`C5`: Conscientiousness items (1–6 Likert)
  - `E1`–`E5`: Extraversion items (1–6 Likert)
  - `N1`–`N5`: Neuroticism items (1–6 Likert)
  - `O1`–`O5`: Openness items (1–6 Likert)
  - `gender`: 1 = male, 2 = female
  - `education`: ordinal schooling code 1–5 (higher = more schooling)
  - `age`: age in years
- Notes: Blank fields represent missing responses. Many items are reverse-keyed in the original documentation; account for that if computing scale scores.

## ovarian_survival (survival)
- Source: Rdatasets mirror of `survival::ovarian` – https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/ovarian.csv
- What it captures: Relapse-free survival in an ovarian cancer trial; useful for survival analysis or latent frailty modeling.
- File: `data/ovarian_survival.csv` (26 rows, 7 columns)
- Columns:
  - `rownames`: row index from Rdatasets (drop for modeling)
  - `futime`: follow-up time in days
  - `fustat`: event indicator (1 = death, 0 = censored)
  - `age`: age at diagnosis (years)
  - `resid.ds`: residual disease present? (1 = no, 2 = yes)
  - `rx`: treatment group (1 or 2)
  - `ecog.ps`: ECOG performance status

## pbc (mixed outcomes: survival + binary + continuous)
- Source: Rdatasets mirror of `survival::pbc` – https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/pbc.csv
- What it captures: Primary biliary cirrhosis trial with survival time, event indicator, and several binary/ordinal clinical findings plus continuous labs; good for multi-outcome modeling across different likelihoods.
- File: `data/pbc.csv` (418 rows, 21 columns)
- Columns (selected):
  - `rownames`: row index from Rdatasets (drop for modeling)
  - `time`: follow-up time in days; `status`: event (0 = censored, 1 = transplant, 2 = death)
  - `trt`: treatment arm; `sex`
  - Binary/ordinal: `ascites`, `hepato`, `spiders`, `edema`, `stage`
  - Continuous: `age`, `bili`, `chol`, `albumin`, `copper`, `alk.phos`, `ast`, `trig`, `platelet`, `protime`
- Notes: Status includes transplant as a distinct code; recode to a binary event or treat as competing risks depending on the analysis goal.

## mdp_traits / mdp_numeric (maize genomic selection)
- Source: GAPIT tutorial data for the Maize Diversity Panel – https://zzlab.net/GAPIT/data/
- What it captures: Phenotypes and SNP markers for maize lines; suitable for genomic selection with GBLUP, multi-kernel, or Bayesian regression.
- Files:
  - `data/mdp_traits.csv` (302 rows, 4 columns): phenotypes with columns `Taxa`, `EarHT` (ear height, cm), `dpoll` (days to pollen shed), `EarDia` (ear diameter, mm); `NaN` indicates missing.
  - `data/mdp_numeric.csv` (302 rows, 4,000+ columns): marker matrix coded 0/1/2 (homozygous ref / heterozygous / homozygous alt); first column `taxa` matches `Taxa` in traits.
- Notes: Some markers/traits contain missing values (`NaN`); impute or drop as needed before modeling. The high-dimensional marker matrix pairs well with additive or multi-kernel covariance structures.

## Ready-to-run examples
Assumes libsemx is built and the Python/R front-ends are installed in editable mode (`cd python && uv pip install -e .`, `R CMD INSTALL Rpkg/semx`). Paths are relative to the repo root.

### sleepstudy: random intercept + slope LMM
```python
import pandas as pd
from semx import Model

df = pd.read_csv("data/sleepstudy.csv").drop(columns=["rownames"])
df["Subject"] = pd.factorize(df["Subject"], sort=True)[0]

model = Model(
    equations=["Reaction ~ Days + (Days | Subject)"],
    families={"Reaction": "gaussian", "Days": "gaussian"},
    kinds={"Subject": "grouping"},
)

fit = model.fit(df[["Reaction", "Days", "Subject"]])
print(fit.summary().head())
```
```r
library(semx)
df <- read.csv("data/sleepstudy.csv")
df$Subject <- as.integer(factor(df$Subject)) - 1

mod <- semx_model(
  equations = c("Reaction ~ Days + (Days | Subject)"),
  families = c(Reaction = "gaussian", Days = "gaussian"),
  kinds = c(Subject = "grouping")
)
fit <- semx_fit(mod, df[, c("Reaction", "Days", "Subject")])
print(head(summary(fit)))
```

### bfi: Big Five CFA (latent factors)
```python
import pandas as pd
from semx import Model

df = pd.read_csv("data/bfi.csv").drop(columns=["rownames"])

eqs = [
    "Agree =~ A1 + A2 + A3 + A4 + A5",
    "Consc =~ C1 + C2 + C3 + C4 + C5",
    "Extra =~ E1 + E2 + E3 + E4 + E5",
    "Neuro =~ N1 + N2 + N3 + N4 + N5",
    "Open  =~ O1 + O2 + O3 + O4 + O5",
]
items = [f"{p}{i}" for p in ["A", "C", "E", "N", "O"] for i in range(1, 6)]
families = {col: "gaussian" for col in items}

model = Model(equations=eqs, families=families)
fit = model.fit(df)
print(fit.fit_indices)
```
```r
library(semx)
df <- read.csv("data/bfi.csv")
items <- unlist(lapply(c("A","C","E","N","O"), function(p) paste0(p, 1:5)))

mod <- semx_model(
  equations = c(
    "Agree =~ A1 + A2 + A3 + A4 + A5",
    "Consc =~ C1 + C2 + C3 + C4 + C5",
    "Extra =~ E1 + E2 + E3 + E4 + E5",
    "Neuro =~ N1 + N2 + N3 + N4 + N5",
    "Open  =~ O1 + O2 + O3 + O4 + O5"
  ),
  families = stats::setNames(rep("gaussian", length(items)), items)
)
fit <- semx_fit(mod, df)
print(head(summary(fit)))
```

### ovarian_survival: parametric survival (Weibull)
```python
import pandas as pd
from semx import Model

df = pd.read_csv("data/ovarian_survival.csv").drop(columns=["rownames"])

model = Model(
    equations=["Surv(futime, fustat) ~ age + rx + resid.ds"],
    families={"futime": "weibull", "age": "gaussian", "rx": "gaussian", "resid.ds": "gaussian"},
)

fit = model.fit(df)
print(fit.summary().head())
```
```r
library(semx)
df <- read.csv("data/ovarian_survival.csv")

mod <- semx_model(
  equations = c("Surv(futime, fustat) ~ age + rx + resid.ds"),
  families = c(futime = "weibull", age = "gaussian", rx = "gaussian", resid.ds = "gaussian")
)
fit <- semx_fit(mod, df)
print(head(summary(fit)))
# Survival curve at 1/2/3 years
print(semx_predict_survival(fit, df, times = c(365, 730, 1095), outcome = "futime"))
```

### pbc: mixed outcomes (survival + binary)
```python
import pandas as pd
from semx import Model

df = pd.read_csv("data/pbc.csv")
df["status_event"] = (df["status"] > 0).astype(int)  # treat transplant/death as events

model = Model(
    equations=[
        "Surv(time, status_event) ~ trt + age + bili + albumin",
        "ascites ~ trt + age",
    ],
    families={
        "time": "weibull",
        "trt": "gaussian",
        "age": "gaussian",
        "bili": "gaussian",
        "albumin": "gaussian",
        "ascites": "binomial",
    },
)

cols = ["time", "status_event", "trt", "age", "bili", "albumin", "ascites"]
fit = model.fit(df[cols])
print(fit.summary().loc[["beta_trt_on_time", "beta_age_on_time"]])
```
```r
library(semx)
df <- read.csv("data/pbc.csv")
df$status_event <- as.integer(df$status > 0)

mod <- semx_model(
  equations = c(
    "Surv(time, status_event) ~ trt + age + bili + albumin",
    "ascites ~ trt + age"
  ),
  families = c(
    time = "weibull", trt = "gaussian", age = "gaussian",
    bili = "gaussian", albumin = "gaussian", ascites = "binomial"
  )
)
fit <- semx_fit(mod, df[, c("time", "status_event", "trt", "age", "bili", "albumin", "ascites")])
print(head(summary(fit)))
```

### mdp_traits / mdp_numeric: genomic selection with GRM
```python
import pandas as pd
from semx import Model
from semx.genomic import cv_genomic_prediction

traits = pd.read_csv("data/mdp_traits.csv")
markers = pd.read_csv("data/mdp_numeric.csv")

# Align marker rows to the trait order and encode taxa as 0..n-1
markers = markers.rename(columns={"taxa": "Taxa"}).set_index("Taxa")
traits = traits[traits["Taxa"].isin(markers.index)].reset_index(drop=True)
markers = markers.loc[traits["Taxa"]]
traits["taxa_code"] = range(len(traits))
traits["_intercept"] = 1.0
marker_matrix = markers.to_numpy(dtype=float)

model = Model(
    equations=["EarHT ~ 1"],
    families={"EarHT": "gaussian", "_intercept": "gaussian"},
    covariances=[{"name": "K_taxa", "structure": "grm", "dimension": 1}],
    genomic={"K_taxa": {"markers": marker_matrix}},
    random_effects=[{"name": "u_taxa", "variables": ["taxa_code", "_intercept"], "covariance": "K_taxa"}],
)

metrics = cv_genomic_prediction(model, traits[["EarHT", "taxa_code", "_intercept"]], outcome="EarHT", folds=3)
print(metrics)
```
```r
library(semx)
traits <- read.csv("data/mdp_traits.csv")
markers_df <- read.csv("data/mdp_numeric.csv")
rownames(markers_df) <- markers_df$taxa
markers_df$taxa <- NULL

traits <- traits[traits$Taxa %in% rownames(markers_df), ]
markers_mat <- as.matrix(markers_df[traits$Taxa, ])
traits$Taxa_code <- seq_len(nrow(traits)) - 1
traits$`_intercept` <- 1

mod <- semx_model(
  equations = c("EarHT ~ 1"),
  families = c(EarHT = "gaussian", `_intercept` = "gaussian"),
  kinds = c(Taxa_code = "grouping"),
  covariances = list(list(name = "K_taxa", structure = "grm", dimension = 1L)),
  genomic = list(K_taxa = list(markers = markers_mat)),
  random_effects = list(list(name = "u_taxa", variables = c("Taxa_code", "_intercept"), covariance = "K_taxa"))
)

fit <- semx_fit(mod, traits[, c("EarHT", "Taxa_code", "_intercept")])
print(head(summary(fit)))
```
