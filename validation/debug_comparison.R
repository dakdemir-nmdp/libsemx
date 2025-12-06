
if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools", repos = "https://cloud.r-project.org")
if (!requireNamespace("lme4", quietly = TRUE)) install.packages("lme4", repos = "https://cloud.r-project.org")
if (!requireNamespace("lavaan", quietly = TRUE)) install.packages("lavaan", repos = "https://cloud.r-project.org")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr", repos = "https://cloud.r-project.org")

library(lme4)
library(lavaan)
library(dplyr)

# Load semx from source
devtools::load_all("Rpkg/semx")

cat("\n======================================================\n")
cat("Part 1: Linear Mixed Model (GLMM) Comparison\n")
cat("======================================================\n")

data(sleepstudy, package = "lme4")
fm1 <- lmer(Reaction ~ Days + (Days | Subject), sleepstudy, REML = FALSE) # Use ML for comparison

# Prepare data: explicit intercept column needed for low-level interface
sleepstudy$`_intercept` <- 1

model_glmm <- semx_model(
  equations = c(
    "Reaction ~ _intercept + Days",
    "Reaction ~~ Reaction"
  ),
  families = list(Reaction = "gaussian"),
  random_effects = list(
    list(
      name = "subject_re",
      variables = c("Subject", "_intercept", "Days"),
      covariance = "re_cov"
    )
  ),
  covariances = list(
    list(name = "re_cov", structure = "unstructured", dimension = 2)
  )
)

# Fit using LBFGS
fit_glmm <- semx_fit(model_glmm, sleepstudy, optimizer_name = "lbfgs")
summ_glmm <- summary(fit_glmm)

# Extract lme4 values
fe_lme4 <- fixef(fm1)
vc_lme4 <- VarCorr(fm1)
sigma_lme4 <- sigma(fm1)

re_var_int_lme4 <- vc_lme4$Subject[1,1]
re_cov_lme4 <- vc_lme4$Subject[1,2]
re_var_slope_lme4 <- vc_lme4$Subject[2,2]
resid_var_lme4 <- sigma_lme4^2

# Extract semx values
params_semx <- summ_glmm$parameters

# Helper to safely get parameter
get_param <- function(df, pattern) {
  idx <- grep(pattern, rownames(df))
  if (length(idx) > 0) df[idx[1], "Estimate"] else NA
}

fe_semx <- c(
  Intercept = get_param(params_semx, "beta_Reaction_on__intercept"),
  Days = get_param(params_semx, "beta_Reaction_on_Days")
)

fe_comp <- data.frame(
  Parameter = c("Intercept", "Days"),
  lme4 = as.numeric(fe_lme4),
  semx = as.numeric(fe_semx)
)
fe_comp$Diff <- fe_comp$lme4 - fe_comp$semx
print(fe_comp)


# Variance Components
# semx parameters are log-Cholesky diagonal or Cholesky off-diagonal
# L11 = exp(re_cov_0)
# L21 = re_cov_1
# L22 = exp(re_cov_2)
# Var(Int) = L11^2
# Cov = L11 * L21
# Var(Slope) = L21^2 + L22^2

L11 <- exp(get_param(params_semx, "re_cov_0"))
L21 <- get_param(params_semx, "re_cov_1")
L22 <- exp(get_param(params_semx, "re_cov_2"))

re_var_int_semx <- L11^2
re_cov_semx <- L11 * L21
re_var_slope_semx <- L21^2 + L22^2

# Residual variance is exp(psi) because psi is log-variance
resid_var_semx <- exp(get_param(params_semx, "psi_Reaction_Reaction"))

vc_comp <- data.frame(
  Component = c("Subject (Intercept)", "Subject (Covariance)", "Subject (Slope)", "Residual"),
  lme4 = c(re_var_int_lme4, re_cov_lme4, re_var_slope_lme4, resid_var_lme4),
  semx = c(re_var_int_semx, re_cov_semx, re_var_slope_semx, resid_var_semx)
)
vc_comp$Diff <- vc_comp$lme4 - vc_comp$semx
print(vc_comp)

# Likelihood
ll_comp <- data.frame(
  Metric = "Log-Likelihood",
  lme4 = as.numeric(logLik(fm1)),
  semx = summ_glmm$fit_indices$loglik
)
print(ll_comp)


cat("\n======================================================\n")
cat("Part 2: Structural Equation Model (CFA) Comparison\n")
cat("======================================================\n")

data(HolzingerSwineford1939)
HS.model <- ' visual  =~ x1 + x2 + x3
              textual =~ x4 + x5 + x6
              speed   =~ x7 + x8 + x9 '

fit_lav <- cfa(HS.model, data = HolzingerSwineford1939, std.lv = FALSE)

model_sem <- semx_model(
  equations = c(
    "visual  =~ x1 + x2 + x3",
    "textual =~ x4 + x5 + x6",
    "speed   =~ x7 + x8 + x9",
    "visual ~~ textual",
    "visual ~~ speed",
    "textual ~~ speed",
    # Add intercepts to match lavaan default
    "x1 ~ 1", "x2 ~ 1", "x3 ~ 1",
    "x4 ~ 1", "x5 ~ 1", "x6 ~ 1",
    "x7 ~ 1", "x8 ~ 1", "x9 ~ 1"
  ),
  families = list(
    x1="gaussian", x2="gaussian", x3="gaussian",
    x4="gaussian", x5="gaussian", x6="gaussian",
    x7="gaussian", x8="gaussian", x9="gaussian"
  )
)

# Fit SEMX model
options <- new(OptimizationOptions)
options$max_iterations <- 1000
options$tolerance <- 1e-6
options$learning_rate <- 1e-3 # Small learning rate for GD

cat("\nFitting SEMX model (GD)...\n")
fit_semx <- semx_fit(model_sem, HolzingerSwineford1939, options, optimizer_name = "gd")
summ_semx <- summary(fit_semx)

# Extract lavaan estimates
pe_lav <- parameterEstimates(fit_lav)
pe_lav_loadings <- pe_lav[pe_lav$op == "=~", ]

# Extract semx estimates
params_semx <- summ_semx$parameters

lav_est <- c()
# Check Data Variance
print("Data Variances:")
print(apply(HolzingerSwineford1939[, paste0("x", 1:9)], 2, var))

# Compare Loadings
print("Parameter Comparison (Loadings):")
params_lavaan_full <- parameterEstimates(fit_lav)
print(head(params_lavaan_full[params_lavaan_full$op == "=~",], 10))

# Compare Variances and Covariances
print("Lavaan Variances and Covariances:")
print(params_lavaan_full[params_lavaan_full$op == "~~",])

# Print all SEMX parameters for manual check
print("All SEMX Parameters:")
print(params_semx)


# Fit Indices
fit_lav_meas <- fitMeasures(fit_lav)
fi_comp <- data.frame(
  Index = c("CFI", "TLI", "RMSEA", "SRMR", "AIC", "BIC"),
  lavaan = c(fit_lav_meas["cfi"], fit_lav_meas["tli"], fit_lav_meas["rmsea"], fit_lav_meas["srmr"], fit_lav_meas["aic"], fit_lav_meas["bic"]),
  semx = c(summ_semx$fit_indices$cfi, summ_semx$fit_indices$tli, summ_semx$fit_indices$rmsea, summ_semx$fit_indices$srmr, summ_semx$fit_indices$aic, summ_semx$fit_indices$bic)
)
fi_comp$Diff <- fi_comp$lavaan - fi_comp$semx
print(fi_comp)

# Likelihood Comparison
ll_lav <- logLik(fit_lav)
ll_semx <- summ_semx$fit_indices$loglik
cat("\nLog-Likelihood Comparison:\n")
cat("Lavaan:", ll_lav, "\n")
cat("SEMX:  ", ll_semx, "\n")
cat("Diff:  ", as.numeric(ll_lav) - ll_semx, "\n")
