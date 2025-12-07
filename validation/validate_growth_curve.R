# Validation of Latent Growth Curve Model against lavaan

library(semx)
library(lavaan)
library(testthat)

# 1. Simulate or Load Data
# We use lavaan's Demo.growth dataset if available, or simulate similar data.
if (requireNamespace("lavaan", quietly = TRUE)) {
  data(Demo.growth, package = "lavaan")
  df <- Demo.growth
} else {
  # Simulate simple growth curve data
  set.seed(1234)
  N <- 400
  i <- rnorm(N, mean = 10, sd = 2)
  s <- rnorm(N, mean = 2, sd = 1)
  t1 <- i + 0*s + rnorm(N, 0, 1)
  t2 <- i + 1*s + rnorm(N, 0, 1)
  t3 <- i + 2*s + rnorm(N, 0, 1)
  t4 <- i + 3*s + rnorm(N, 0, 1)
  df <- data.frame(t1, t2, t3, t4)
}

cat("Data loaded. N =", nrow(df), "\n")

# 2. Fit with lavaan
# Standard Linear Growth Curve Model
lav_model <- '
  # intercept and slope with fixed coefficients
  i =~ 1*t1 + 1*t2 + 1*t3 + 1*t4
  s =~ 0*t1 + 1*t2 + 2*t3 + 3*t4
  
  # regressions
  # (none)
  
  # residual variances
  t1 ~~ t1
  t2 ~~ t2
  t3 ~~ t3
  t4 ~~ t4
  
  # latent variances and covariance
  i ~~ i
  s ~~ s
  i ~~ s
  
  # latent means
  i ~ 1
  s ~ 1
  
  # observed intercepts fixed to 0
  t1 ~ 0*1
  t2 ~ 0*1
  t3 ~ 0*1
  t4 ~ 0*1
'

cat("Fitting lavaan model...\n")
fit_lav <- lavaan(lav_model, data = df, meanstructure = TRUE)
summary(fit_lav)
loglik_lav <- logLik(fit_lav)
cat("Lavaan LogLik:", loglik_lav, "\n")

# 3. Fit with semx
cat("Fitting semx model...\n")

# In semx, we define latent variables via `latent()` or implicitly in equations.
# We need to specify fixed loadings.
# semx syntax supports `factor =~ load*ind`

semx_mod <- semx_model(
  equations = c(
    # Measurement model (Growth factors)
    "i =~ 1*t1 + 1*t2 + 1*t3 + 1*t4",
    "s =~ 0*t1 + 1*t2 + 2*t3 + 3*t4",
    
    # Latent means (intercepts)
    "i ~ 1",
    "s ~ 1",
    
    # Latent covariance
    "i ~~ s",
    
    # Observed intercepts fixed to 0
    "t1 ~ 0",
    "t2 ~ 0",
    "t3 ~ 0",
    "t4 ~ 0"
  ),
  families = c(t1="gaussian", t2="gaussian", t3="gaussian", t4="gaussian")
)

fit_semx <- semx_fit(semx_mod, df)
summary(fit_semx)

loglik_semx <- fit_semx$fit_result$loglik
cat("SEMX LogLik:", loglik_semx, "\n")

# 4. Compare Results
diff_loglik <- abs(loglik_lav - loglik_semx)
cat("LogLik Difference:", diff_loglik, "\n")

test_that("Growth Curve LogLik matches lavaan", {
  expect_lt(diff_loglik, 0.1)
})

# Compare Parameters
# Extract lavaan estimates
pe_lav <- parameterEstimates(fit_lav)
# Filter relevant parameters
pe_lav_subset <- pe_lav[pe_lav$op %in% c("=~", "~~", "~1"), ]

# Extract semx estimates
# We need to map names.
# semx names:
# Loadings: lambda_t1_on_i (fixed), etc.
# Means: alpha_i_on__intercept, etc.
# Covariances: psi_i_i, psi_s_s, psi_i_s
# Residuals: psi_t1_t1, etc.

# Helper to get semx param
get_semx_est <- function(name) {
  idx <- which(fit_semx$fit_result$parameter_names == name)
  if (length(idx) == 0) return(NA)
  fit_semx$fit_result$parameters[idx]
}

# Compare Latent Means
mean_i_lav <- pe_lav[pe_lav$lhs == "i" & pe_lav$op == "~1", "est"]
mean_s_lav <- pe_lav[pe_lav$lhs == "s" & pe_lav$op == "~1", "est"]

mean_i_semx <- get_semx_est("alpha_i_on__intercept")
mean_s_semx <- get_semx_est("alpha_s_on__intercept")

cat("Mean i: Lavaan =", mean_i_lav, " SEMX =", mean_i_semx, "\n")
cat("Mean s: Lavaan =", mean_s_lav, " SEMX =", mean_s_semx, "\n")

test_that("Latent Means match", {
  expect_equal(mean_i_semx, mean_i_lav, tolerance = 1e-3)
  expect_equal(mean_s_semx, mean_s_lav, tolerance = 1e-3)
})

# Compare Latent Covariances
var_i_lav <- pe_lav[pe_lav$lhs == "i" & pe_lav$op == "~~" & pe_lav$rhs == "i", "est"]
var_s_lav <- pe_lav[pe_lav$lhs == "s" & pe_lav$op == "~~" & pe_lav$rhs == "s", "est"]
cov_is_lav <- pe_lav[pe_lav$lhs == "i" & pe_lav$op == "~~" & pe_lav$rhs == "s", "est"]

# Note: semx might use Cholesky parameterization internally, but `parameters` should be on the requested scale 
# IF we implemented the back-transform for reporting.
# Wait, `fit_result$parameters` are the raw optimized parameters.
# For Unstructured covariance, these are Cholesky factors if `ExplicitCovariance` is NOT used.
# But for SEM, we usually use `ExplicitCovariance` (elements of Sigma) or we map them.
# In `validate_mixed_sem.R`, we saw `psi_...` parameters.
# If `semx` uses `ExplicitCovariance` for latents, then `psi_i_i` is the variance.
# Let's check the parameter names in the output.

var_i_semx <- get_semx_est("psi_i_i")
var_s_semx <- get_semx_est("psi_s_s")
cov_is_semx <- get_semx_est("psi_i_s") # or psi_s_i

if (is.na(cov_is_semx)) cov_is_semx <- get_semx_est("psi_s_i")

cat("Var i: Lavaan =", var_i_lav, " SEMX =", var_i_semx, "\n")
cat("Var s: Lavaan =", var_s_lav, " SEMX =", var_s_semx, "\n")
cat("Cov is: Lavaan =", cov_is_lav, " SEMX =", cov_is_semx, "\n")

test_that("Latent Covariances match", {
  expect_equal(var_i_semx, var_i_lav, tolerance = 1e-3)
  expect_equal(var_s_semx, var_s_lav, tolerance = 1e-3)
  expect_equal(cov_is_semx, cov_is_lav, tolerance = 1e-3)
})

# Compare Residual Variances
resid_t1_lav <- pe_lav[pe_lav$lhs == "t1" & pe_lav$op == "~~" & pe_lav$rhs == "t1", "est"]
resid_t1_semx <- get_semx_est("psi_t1_t1")

cat("Resid t1: Lavaan =", resid_t1_lav, " SEMX =", resid_t1_semx, "\n")

test_that("Residual Variances match", {
  expect_equal(resid_t1_semx, resid_t1_lav, tolerance = 1e-3)
})
