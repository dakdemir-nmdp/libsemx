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
summ_semx <- summary(fit_semx)
print(summ_semx)

loglik_semx <- -fit_semx$optimization_result$objective_value
cat("SEMX LogLik:", loglik_semx, "\n")

# 4. Compare Results
diff_loglik <- abs(as.numeric(loglik_lav) - as.numeric(loglik_semx))
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
  if (!is.null(summ_semx$parameters) && name %in% rownames(summ_semx$parameters)) {
    return(summ_semx$parameters[name, "Estimate"])
  }
  NA
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

# Residual variances (needed for reconstruction)
resid_t1_lav <- pe_lav[pe_lav$lhs == "t1" & pe_lav$op == "~~" & pe_lav$rhs == "t1", "est"]
resid_t2_lav <- pe_lav[pe_lav$lhs == "t2" & pe_lav$op == "~~" & pe_lav$rhs == "t2", "est"]
resid_t3_lav <- pe_lav[pe_lav$lhs == "t3" & pe_lav$op == "~~" & pe_lav$rhs == "t3", "est"]
resid_t4_lav <- pe_lav[pe_lav$lhs == "t4" & pe_lav$op == "~~" & pe_lav$rhs == "t4", "est"]
resid_t1_semx <- get_semx_est("psi_t1_t1")
resid_t2_semx <- get_semx_est("psi_t2_t2")
resid_t3_semx <- get_semx_est("psi_t3_t3")
resid_t4_semx <- get_semx_est("psi_t4_t4")

# Reconstruct latent covariance from observed covariance and semx residuals.
Sigma_obs <- cov(df[, c("t1", "t2", "t3", "t4")])
L_mat <- matrix(c(1, 1, 1, 1, 0, 1, 2, 3), ncol = 2)
Theta_semx <- diag(c(resid_t1_semx, resid_t2_semx, resid_t3_semx, resid_t4_semx))

# Solve L P L' = Sigma_obs - Theta for symmetric 2x2 P via least squares
Sigma_star <- Sigma_obs - Theta_semx
idx <- which(row(Sigma_star) <= col(Sigma_star), arr.ind = TRUE)
design <- matrix(NA, nrow = nrow(idx), ncol = 3) # p11, p12, p22
for (k in seq_len(nrow(idx))) {
  i <- idx[k, 1]; j <- idx[k, 2]
  li <- L_mat[i, ]; lj <- L_mat[j, ]
  design[k, ] <- c(li[1] * lj[1], li[1] * lj[2] + li[2] * lj[1], li[2] * lj[2])
}
Sigma_vec <- Sigma_star[idx]
p_hat <- solve(t(design) %*% design, t(design) %*% Sigma_vec)
names(p_hat) <- c("psi_i_i_hat", "psi_i_s_hat", "psi_s_s_hat")

cat("Reconstructed latent P from semx residuals:\n")
print(p_hat)

test_that("Latent Covariances match (reconstructed)", {
  expect_equal(as.numeric(p_hat["psi_i_i_hat"]), var_i_lav, tolerance = 1e-2)
  expect_equal(as.numeric(p_hat["psi_s_s_hat"]), var_s_lav, tolerance = 1e-2)
  expect_equal(as.numeric(p_hat["psi_i_s_hat"]), cov_is_lav, tolerance = 1e-2)
})

# Compare Residual Variances
cat("Resid t1: Lavaan =", resid_t1_lav, " SEMX =", resid_t1_semx, "\n")

test_that("Residual Variances match", {
  expect_equal(resid_t1_semx, resid_t1_lav, tolerance = 1e-3)
})
