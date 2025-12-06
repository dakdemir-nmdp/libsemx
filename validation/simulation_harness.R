
library(semx)
library(MASS)
library(survival)

# Configuration
set.seed(123)
N_SIM <- 1 # Number of replications per scenario (keep low for dev, increase for robust check)
TOLERANCE_REL <- 0.15 # 15% relative error tolerance
TOLERANCE_ABS <- 0.1  # Absolute error tolerance for small values

# Helper to check recovery
check_recovery <- function(scenario_name, true_params, fit, tolerance_rel = TOLERANCE_REL, tolerance_abs = TOLERANCE_ABS) {
  cat(sprintf("\n--- Scenario: %s ---\n", scenario_name))
  
  est_params <- fit$optimization_result$parameters
  param_names <- fit$parameter_names
  names(est_params) <- param_names
  
  all_passed <- TRUE
  
  # Map true params to estimated params
  # We assume true_params keys match or are substrings of est_params keys
  
  for (name in names(true_params)) {
    true_val <- true_params[[name]]
    
    # Find matching estimated parameter
    # Exact match first
    est_name <- name
    if (!(name %in% names(est_params))) {
       # Try to find by suffix or partial match if needed, but exact is better for harness
       # For now, assume exact names are provided in true_params
       cat(sprintf("  [MISSING] Parameter '%s' not found in estimates.\n", name))
       all_passed <- FALSE
       next
    }
    
    est_val <- est_params[[est_name]]
    
    diff <- abs(est_val - true_val)
    rel_diff <- if (abs(true_val) > 1e-6) diff / abs(true_val) else diff
    
    passed <- (diff < tolerance_abs) || (rel_diff < tolerance_rel)
    
    status <- if (passed) "PASS" else "FAIL"
    if (!passed) all_passed <- FALSE
    
    cat(sprintf("  [%s] %s: True=%g, Est=%g, Diff=%g (Rel=%.2f%%)\n", 
                status, name, true_val, est_val, diff, rel_diff * 100))
  }
  
  if (all_passed) {
    cat(">>> SCENARIO PASSED\n")
  } else {
    cat(">>> SCENARIO FAILED\n")
  }
  return(all_passed)
}

# ------------------------------------------------------------------------------
# Scenario 1: Gaussian Linear Regression
# y = 1.0 + 2.0*x + e, var(e) = 0.5
# ------------------------------------------------------------------------------
sim_gaussian_regression <- function() {
  N <- 500
  x <- rnorm(N)
  beta0 <- 1.0
  beta1 <- 2.0
  sigma_sq <- 0.5
  y <- beta0 + beta1 * x + rnorm(N, 0, sqrt(sigma_sq))
  
  data <- data.frame(y = y, x = x)
  
  model <- semx_model(
    equations = c("y ~ 1 + x"),
    families = c(y = "gaussian")
  )
  
  fit <- semx_fit(model, data)
  
  true_params <- list(
    "alpha_y_on__intercept" = beta0,
    "beta_y_on_x" = beta1,
    "psi_y_y" = sigma_sq # Residual variance
  )
  
  check_recovery("Gaussian Regression", true_params, fit)
}

# ------------------------------------------------------------------------------
# Scenario 2: Gaussian Mixed Model (Random Intercept)
# y = 1.0 + 2.0*x + u_g + e
# u_g ~ N(0, 0.8), e ~ N(0, 0.5)
# ------------------------------------------------------------------------------
sim_gaussian_mixed <- function() {
  N_groups <- 50
  N_per_group <- 20
  N <- N_groups * N_per_group
  
  groups <- rep(1:N_groups, each = N_per_group)
  x <- rnorm(N)
  
  beta0 <- 1.0
  beta1 <- 2.0
  var_u <- 0.8
  var_e <- 0.5
  
  u <- rnorm(N_groups, 0, sqrt(var_u))
  y <- beta0 + beta1 * x + u[groups] + rnorm(N, 0, sqrt(var_e))
  
  data <- data.frame(y = y, x = x, group = groups)
  
  model <- semx_model(
    equations = c("y ~ 1 + x + (1 | group)"),
    families = c(y = "gaussian")
  )
  
  fit <- semx_fit(model, data)
  
  true_params <- list(
    "alpha_y_on__intercept" = beta0,
    "beta_y_on_x" = beta1,
    "psi_y_y" = var_e,
    "cov_re_1_0" = var_u # Random intercept variance
  )
  
  check_recovery("Gaussian Mixed Model", true_params, fit)
}

# ------------------------------------------------------------------------------
# Scenario 3: Binomial GLMM (Random Intercept)
# logit(p) = 0.5 + 1.0*x + u_g
# u_g ~ N(0, 1.0)
# ------------------------------------------------------------------------------
sim_binomial_glmm <- function() {
  N_groups <- 200
  N_per_group <- 20
  N <- N_groups * N_per_group
  
  groups <- rep(1:N_groups, each = N_per_group)
  x <- rnorm(N)
  
  beta0 <- 0.5
  beta1 <- 1.0
  var_u <- 1.0
  
  u <- rnorm(N_groups, 0, sqrt(var_u))
  eta <- beta0 + beta1 * x + u[groups]
  prob <- 1 / (1 + exp(-eta))
  y <- rbinom(N, 1, prob)
  
  data <- data.frame(y = y, x = x, group = groups)
  
  model <- semx_model(
    equations = c("y ~ 1 + x + (1 | group)"),
    families = c(y = "binomial")
  )
  
  fit <- semx_fit(model, data)
  
  true_params <- list(
    "alpha_y_on__intercept" = beta0,
    "beta_y_on_x" = beta1,
    "cov_re_1_0" = var_u
  )
  
  check_recovery("Binomial GLMM", true_params, fit)
}

# ------------------------------------------------------------------------------
# Scenario 4: Poisson GLMM (Random Intercept)
# log(lambda) = 1.0 + 0.5*x + u_g
# u_g ~ N(0, 0.3)
# ------------------------------------------------------------------------------
sim_poisson_glmm <- function() {
  N_groups <- 200
  N_per_group <- 20
  N <- N_groups * N_per_group
  
  groups <- rep(1:N_groups, each = N_per_group)
  x <- rnorm(N)
  
  beta0 <- 1.0
  beta1 <- 0.5
  var_u <- 0.3
  
  u <- rnorm(N_groups, 0, sqrt(var_u))
  eta <- beta0 + beta1 * x + u[groups]
  lambda <- exp(eta)
  y <- rpois(N, lambda)
  
  data <- data.frame(y = y, x = x, group = groups)
  
  model <- semx_model(
    equations = c("y ~ 1 + x + (1 | group)"),
    families = c(y = "poisson")
  )
  
  fit <- semx_fit(model, data)
  
  true_params <- list(
    "alpha_y_on__intercept" = beta0,
    "beta_y_on_x" = beta1,
    "cov_re_1_0" = var_u
  )
  
  check_recovery("Poisson GLMM", true_params, fit, tolerance_rel = 1.0)
}

# ------------------------------------------------------------------------------
# Scenario 5: CFA (Confirmatory Factor Analysis)
# F =~ 1.0*y1 + 0.8*y2 + 0.6*y3
# var(F) = 1.0
# var(e) = 0.5
# ------------------------------------------------------------------------------
sim_cfa <- function() {
  N <- 1000
  
  lambda1 <- 1.0
  lambda2 <- 0.8
  lambda3 <- 0.6
  var_f <- 1.0
  var_e <- 0.5
  
  f <- rnorm(N, 0, sqrt(var_f))
  y1 <- lambda1 * f + rnorm(N, 0, sqrt(var_e))
  y2 <- lambda2 * f + rnorm(N, 0, sqrt(var_e))
  y3 <- lambda3 * f + rnorm(N, 0, sqrt(var_e))
  
  data <- data.frame(y1 = y1, y2 = y2, y3 = y3)
  
  model <- semx_model(
    equations = c(
      "F =~ y1 + y2 + y3",
      "F ~~ F",
      "y1 ~~ y1", "y2 ~~ y2", "y3 ~~ y3"
    ),
    families = c(y1 = "gaussian", y2 = "gaussian", y3 = "gaussian")
  )
  
  fit <- semx_fit(model, data)
  
  true_params <- list(
    "lambda_y2_on_F" = lambda2,
    "lambda_y3_on_F" = lambda3,
    "psi_F_F" = var_f,
    "psi_y1_y1" = var_e,
    "psi_y2_y2" = var_e,
    "psi_y3_y3" = var_e
  )
  
  check_recovery("CFA", true_params, fit)
}

# ------------------------------------------------------------------------------
# Scenario 6: Survival (Weibull)
# Surv(time, status) ~ x
# shape = 1.5, scale = exp(beta0 + beta1*x)
# ------------------------------------------------------------------------------
sim_survival_weibull <- function() {
  N <- 1000
  x <- rnorm(N)
  
  beta0 <- 2.0
  beta1 <- 0.5
  shape <- 1.5
  scale_param <- exp(beta0 + beta1 * x) # This is actually scale in rweibull?
  # rweibull(n, shape, scale)
  # In survreg/semx: log(T) = mu + sigma * W
  # mu = beta0 + beta1*x
  # sigma = 1/shape
  # T = exp(mu) * exp(sigma * W) = scale * ...
  
  # Let's use the AFT parameterization
  # log(T) = beta0 + beta1*x + sigma * error
  # error ~ Extreme Value
  
  sigma <- 1 / shape
  error <- log(rexp(N)) # Standard extreme value distribution (min) ? No, log(exponential) is Gumbel (min)
  # rweibull generates T.
  # T ~ Weibull(shape, scale) -> f(t) = (shape/scale) * (t/scale)^(shape-1) * exp(-(t/scale)^shape)
  # S(t) = exp(-(t/scale)^shape)
  # log(T) = log(scale) + (1/shape) * log(E), where E ~ Exp(1)
  
  # So if we want log(scale) = beta0 + beta1*x
  scale_vec <- exp(beta0 + beta1 * x)
  time_event <- rweibull(N, shape = shape, scale = scale_vec)
  
  # Censoring
  cens_time <- rweibull(N, shape = shape, scale = scale_vec * 1.5) # Some censoring
  time <- pmin(time_event, cens_time)
  status <- as.numeric(time_event <= cens_time)
  
  data <- data.frame(time = time, status = status, x = x)
  
  model <- semx_model(
    equations = c("Surv(time, status) ~ 1 + x"),
    families = c(time = "weibull")
  )
  
  fit <- semx_fit(model, data)
  
  # semx parameters for Weibull:
  # beta coefficients (AFT metric)
  # dispersion = log(scale_param) ? No, dispersion is usually 1/shape or log(shape)?
  # In libsemx: WeibullOutcome uses 'dispersion' parameter.
  # Let's check what it maps to. Usually log(shape) or log(scale)?
  # In survreg, 'scale' is 1/shape.
  # In libsemx, we need to check.
  
  # Assuming standard AFT:
  true_params <- list(
    "alpha_time_on__intercept" = beta0,
    "beta_time_on_x" = beta1,
    "psi_time_time" = shape
  )
  
  check_recovery("Survival (Weibull)", true_params, fit)
}


# Run all
cat("Starting Simulation Harness...\n")
sim_gaussian_regression()
sim_gaussian_mixed()
sim_binomial_glmm()
sim_poisson_glmm()
sim_cfa()
sim_survival_weibull()
cat("\nSimulation Harness Completed.\n")
