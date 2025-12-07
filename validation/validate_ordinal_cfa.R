# Validation of Ordinal CFA against lavaan

library(semx)
library(lavaan)
library(testthat)

# 1. Simulate Data
# Model: f1 =~ u1 + u2 + u3 + u4
# u1-u4 are ordered with 3 levels (0, 1, 2).
# Underlying: u* = lambda*f + e, e ~ N(0, 1)
# Thresholds: tau1, tau2
cat("--- Simulating Ordinal Data ---\n")
set.seed(1234)
N <- 100
f1 <- rnorm(N)

# Loadings = 0.7
lambda <- 0.7
# Residual variance = 1 - 0.7^2 = 0.51
resid_sd <- sqrt(1 - lambda^2)

u1_star <- lambda*f1 + rnorm(N, 0, resid_sd)
u2_star <- lambda*f1 + rnorm(N, 0, resid_sd)
u3_star <- lambda*f1 + rnorm(N, 0, resid_sd)
u4_star <- lambda*f1 + rnorm(N, 0, resid_sd)

# Thresholds: -0.5, 0.5
# P(u=0) = P(u* < -0.5)
# P(u=1) = P(-0.5 < u* < 0.5)
# P(u=2) = P(u* > 0.5)
cut_points <- c(-0.5, 0.5)

cut_ordinal <- function(x, cuts) {
  as.numeric(cut(x, c(-Inf, cuts, Inf))) - 1
}

u1 <- cut_ordinal(u1_star, cut_points)
u2 <- cut_ordinal(u2_star, cut_points)
u3 <- cut_ordinal(u3_star, cut_points)
u4 <- cut_ordinal(u4_star, cut_points)

df <- data.frame(u1=ordered(u1), u2=ordered(u2), u3=ordered(u3), u4=ordered(u4))

cat("Data summary:\n")
print(summary(df))

# 2. Fit with lavaan
cat("\n--- Fitting with lavaan ---\n")
lav_model <- '
  f1 =~ u1 + u2 + u3 + u4
'
# std.lv=TRUE fixes factor variance to 1.0, allowing all loadings to be estimated.
# parameterization="theta" (default) or "delta"? 
# semx uses probit link, which corresponds to theta parameterization (residual variances = 1) or delta?
# For categorical, lavaan defaults to delta parameterization (marginal variances = 1) if estimator=WLSMV.
# But if we use estimator="ML", lavaan treats them as continuous unless we specify ordered.
# If we specify ordered, lavaan uses WLSMV by default.
# semx uses Marginal Maximum Likelihood (MML) via Laplace approximation or integration.
# This is equivalent to estimator="MML" in some contexts, or just GLMM.
# In GLMM (semx), we typically fix residual variance of latent response to 1 (Probit) or pi^2/3 (Logit).
# lavaan's "theta" parameterization fixes residual variances to 1 (for probit).
# So we should try to match that.

fit_lav <- cfa(lav_model, data=df, ordered=names(df), std.lv=TRUE, parameterization="theta", estimator="WLSMV") 
# Note: WLSMV is limited information. MML is full information.
# They should converge to similar estimates.
# Ideally we would use estimator="PML" (Pairwise ML) or numerical integration if lavaan supported it well for this.
# Let's stick with WLSMV as the "gold standard" for SEM, but acknowledge small differences vs MML.

print(summary(fit_lav))

# Extract lavaan parameters
pe <- parameterEstimates(fit_lav)
lav_loadings <- pe[pe$op == "=~", c("lhs", "rhs", "est")]
lav_thresholds <- pe[pe$op == "|", c("lhs", "rhs", "est")]

print("Lavaan Loadings:")
print(lav_loadings)
print("Lavaan Thresholds:")
print(lav_thresholds)

# 3. Fit with semx
cat("\n--- Fitting with semx (Free Loadings) ---\n")
semx_mod <- semx_model(
  equations = c(
    "f1 =~ u1 + u2 + u3 + u4",
    "f1 ~~ 1*f1" # Fix factor variance to 1
  ),
  families = c(u1="ordinal", u2="ordinal", u3="ordinal", u4="ordinal")
)

# Manually unfix the first loading (u1)
for (i in seq_along(semx_mod$edges)) {
  e <- semx_mod$edges[[i]]
  if (e$kind == 0 && e$source == "f1" && e$target == "u1") {
    semx_mod$edges[[i]]$parameter_id <- "lambda_u1_on_f1"
    semx_mod$edges[[i]]$fixed_value <- NULL
    break
  }
}

# Set initial values
if (is.null(semx_mod$parameters)) semx_mod$parameters <- list()
semx_mod$parameters[["lambda_u1_on_f1"]] <- 0.7
semx_mod$parameters[["lambda_u2_on_f1"]] <- 0.7
semx_mod$parameters[["lambda_u3_on_f1"]] <- 0.7
semx_mod$parameters[["lambda_u4_on_f1"]] <- 0.7

fit_semx <- semx_fit(semx_mod, df)
cat("semx_fit returned.\n")
print(summary(fit_semx))

# 3b. Fit with semx (Fixed Loadings = 0.7)
# cat("\n--- Fitting with semx (Fixed Loadings = 0.7) ---\n")
# semx_mod_fixed <- semx_model(
#   equations = c(
#     "f1 =~ 0.7*u1 + 0.7*u2 + 0.7*u3 + 0.7*u4",
#     "f1 ~~ 1*f1"
#   ),
#   families = c(u1="ordinal", u2="ordinal", u3="ordinal", u4="ordinal")
# )
# fit_semx_fixed <- semx_fit(semx_mod_fixed, df)
# print(summary(fit_semx_fixed))

# 3c. Fit with semx (Fixed Loadings = 0.0)
# cat("\n--- Fitting with semx (Fixed Loadings = 0.0) ---\n")
# semx_mod_zero <- semx_model(
#   equations = c(
#     "f1 =~ 0*u1 + 0*u2 + 0*u3 + 0*u4",
#     "f1 ~~ 1*f1"
#   ),
#   families = c(u1="ordinal", u2="ordinal", u3="ordinal", u4="ordinal")
# )
# fit_semx_zero <- semx_fit(semx_mod_zero, df)
# print(summary(fit_semx_zero))

cat("\n--- LL Comparison ---\n")
cat("Free (Converged to 0?): ", fit_semx$optimization_result$objective_value, "\n")
# cat("Fixed 0.7: ", fit_semx_fixed$optimization_result$objective_value, "\n")
# cat("Fixed 0.0: ", fit_semx_zero$optimization_result$objective_value, "\n")

# 4. Compare
cat("\n--- Comparison ---\n")

# Loadings
semx_params <- fit_semx$optimization_result$parameters
print("Semx Parameters:")
print(semx_params)

# Helper to get semx param
get_semx_param <- function(name) {
  if (name %in% names(semx_params)) return(semx_params[[name]])
  return(NA)
}

# Compare Loadings
# semx names: lambda_u1_on_f1, etc.
for (i in 1:4) {
  u_name <- paste0("u", i)
  lav_est <- lav_loadings[lav_loadings$rhs == u_name, "est"]
  semx_name <- paste0("lambda_", u_name, "_on_f1") # Note: _on_f1 suffix
  semx_est <- get_semx_param(semx_name)
  
  cat(sprintf("Loading %s: Lavaan=%.3f, Semx=%.3f, Diff=%.3f\n", 
              u_name, lav_est, semx_est, abs(lav_est - semx_est)))
  
  # Expect reasonable agreement (e.g. < 0.5) given different estimators (WLSMV vs MML) and small N
  expect_true(abs(lav_est - semx_est) < 0.5)
}

# Compare Thresholds
# semx names: u1_threshold_1, u1_threshold_2
# lavaan names: u1|t1, u1|t2
for (i in 1:4) {
  u_name <- paste0("u", i)
  
  # Threshold 1
  lav_t1 <- lav_thresholds[lav_thresholds$lhs == u_name & lav_thresholds$rhs == "t1", "est"]
  semx_t1 <- get_semx_param(paste0(u_name, "_threshold_1"))
  
  cat(sprintf("Threshold %s|t1: Lavaan=%.3f, Semx=%.3f\n", u_name, lav_t1, semx_t1))
  expect_true(abs(lav_t1 - semx_t1) < 0.5)
  
  # Threshold 2
  lav_t2 <- lav_thresholds[lav_thresholds$lhs == u_name & lav_thresholds$rhs == "t2", "est"]
  semx_t2 <- get_semx_param(paste0(u_name, "_threshold_2"))
  
  cat(sprintf("Threshold %s|t2: Lavaan=%.3f, Semx=%.3f\n", u_name, lav_t2, semx_t2))
  expect_true(abs(lav_t2 - semx_t2) < 0.5)
}

cat("\nValidation Complete!\n")
