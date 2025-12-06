library(semx)
library(numDeriv)

# Setup model
set.seed(456)
n_subj <- 10
t_levels <- 4
subj <- factor(rep(seq_len(n_subj), each = t_levels))
time <- rep(seq_len(t_levels) - 1, times = n_subj)
toep_params <- c(1.0, 0.6, 0.3, 0.1) 
V <- matrix(0, t_levels, t_levels)
for (i in 1:t_levels) {
  for (j in 1:t_levels) {
    lag <- abs(i - j)
    V[i, j] <- toep_params[lag + 1]
  }
}
L <- t(chol(V))
y <- numeric(length(subj))
for (i in seq_len(n_subj)) {
  idx <- which(subj == levels(subj)[i])
  y[idx] <- 1.5 - 0.2 * time[idx] + L %*% rnorm(t_levels)
}
df_toep <- data.frame(y = y, time = time, subj = subj)
time_cols <- paste0("time_", seq_len(t_levels))
for (lvl in seq_len(t_levels)) {
  df_toep[[time_cols[lvl]]] <- as.numeric(df_toep$time == (lvl - 1))
}
covs <- list(list(name = "G_toep", structure = "toeplitz", dimension = length(time_cols)))
random_effects <- list(list(name = "toep_by_subj", variables = c("subj", time_cols), covariance = "G_toep"))

model <- semx_model(
  equations = c("y ~ 1 + time", "y ~~ 0.001 * y"),
  families = c(y = "gaussian"),
  covariances = covs,
  random_effects = random_effects
)

# Prepare data for driver
data_map <- as.list(df_toep)
data_map[["_intercept"]] <- rep(1.0, nrow(df_toep))

# Driver
driver <- new(LikelihoodDriver)
options <- new(OptimizationOptions)

# Initial parameters
# We need to know parameter names to construct the vector
# semx doesn't expose parameter names easily without fitting?
# We can fit for 0 iterations
fit <- semx_fit(model, df_toep, options = options)
param_names <- fit$parameter_names
initial_params <- fit$optimization_result$parameters
names(initial_params) <- param_names

cat("Initial parameters:\n")
print(initial_params)

# Function to evaluate loglik
eval_loglik <- function(p) {
  # Compute linear predictor for y
  # y ~ 1 + time
  # alpha_y_on__intercept * 1 + beta_y_on_time * time
  # Note: p might be named vector.
  
  alpha <- if ("alpha_y_on__intercept" %in% names(p)) p[["alpha_y_on__intercept"]] else 0
  beta <- if ("beta_y_on_time" %in% names(p)) p[["beta_y_on_time"]] else 0
  
  lp_y <- alpha * data_map[["_intercept"]] + beta * data_map[["time"]]
  linear_predictors <- list(y = lp_y)
  
  # Dispersion
  # Fixed to 0.001
  dispersions <- list(y = rep(0.001, nrow(df_toep)))
  
  # Covariance parameters
  covariance_parameters <- list()
  for (name in names(p)) {
    val <- p[[name]]
    if (startsWith(name, "G_toep_")) {
       parts <- strsplit(name, "_")[[1]]
       cov_id <- paste(parts[1:2], collapse="_")
       if (is.null(covariance_parameters[[cov_id]])) {
         covariance_parameters[[cov_id]] <- numeric()
       }
       idx <- as.integer(parts[3])
       covariance_parameters[[cov_id]][idx + 1] <- val
    }
  }
  
  # Convert to list for Rcpp
  for (n in names(linear_predictors)) linear_predictors[[n]] <- as.numeric(linear_predictors[[n]])
  for (n in names(dispersions)) dispersions[[n]] <- as.numeric(dispersions[[n]])
  
  ll <- driver$evaluate_model_loglik_full(
    model$ir, 
    data_map, 
    linear_predictors, 
    dispersions, 
    covariance_parameters, 
    list(), 
    list(), 
    list(), 
    0L 
  )
  return(ll)
}

# Evaluate gradient
eval_grad <- function(p) {
  # Same setup
  alpha <- if ("alpha_y_on__intercept" %in% names(p)) p[["alpha_y_on__intercept"]] else 0
  beta <- if ("beta_y_on_time" %in% names(p)) p[["beta_y_on_time"]] else 0
  
  lp_y <- alpha * data_map[["_intercept"]] + beta * data_map[["time"]]
  linear_predictors <- list(y = lp_y)
  dispersions <- list(y = rep(0.001, nrow(df_toep)))
  
  covariance_parameters <- list()
  for (name in names(p)) {
    val <- p[[name]]
    if (startsWith(name, "G_toep_")) {
       parts <- strsplit(name, "_")[[1]]
       cov_id <- paste(parts[1:2], collapse="_")
       if (is.null(covariance_parameters[[cov_id]])) {
         covariance_parameters[[cov_id]] <- numeric()
       }
       idx <- as.integer(parts[3])
       covariance_parameters[[cov_id]][idx + 1] <- val
    }
  }
  
  for (n in names(linear_predictors)) linear_predictors[[n]] <- as.numeric(linear_predictors[[n]])
  for (n in names(dispersions)) dispersions[[n]] <- as.numeric(dispersions[[n]])
  
  # We need to map data parameters (fixed effects) to gradients.
  # evaluate_model_gradient takes data_param_mappings.
  # But the R binding for evaluate_model_gradient does NOT expose data_param_mappings!
  # It only exposes up to 'method'.
  
  # Wait, check r_bindings.cpp again.
  # LikelihoodDriver_evaluate_model_gradient takes:
  # driver, model, data, linear_predictors, dispersions, covariance_parameters, status, extra_params, fixed_covariance_data, method.
  # It does NOT take mappings.
  
  # This means evaluate_model_gradient via R binding will NOT return gradients for fixed effects (alpha, beta) 
  # because it doesn't know which data/linear_predictor corresponds to which parameter!
  
  # It WILL return gradients for covariance parameters because they are passed explicitly.
  
  g_list <- driver$evaluate_model_gradient(
    model$ir, 
    data_map, 
    linear_predictors, 
    dispersions, 
    covariance_parameters, 
    list(), 
    list(), 
    list(), 
    0L
  )
  
  # Flatten to vector matching p
  g_vec <- numeric(length(p))
  names(g_vec) <- names(p)
  for (n in names(p)) {
    if (!is.null(g_list[[n]])) {
      g_vec[n] <- g_list[[n]]
    }
  }
  
  # For fixed effects, we can't get gradients from the driver via this binding.
  # We can only check covariance gradients.
  return(g_vec)
}

# Test point: True parameters
p0 <- initial_params
# Set G_toep parameters to true values
# Variance = 1.0 -> log(1) = 0? No, variance is just parameter[0].
# Wait, is variance parameterized as log?
# In ToeplitzCovariance::fill_covariance: const double variance = parameters[0];
# It is NOT log-transformed in the class.
# But validate_parameters checks > 0.
# The optimizer is unconstrained.
# If we use LBFGS, it might push it negative.
# But we saw G_toep_0 going to -1.17.
# If G_toep_0 is variance, it must be positive.
# Ah, maybe semx R package applies a transform?
# Let's check semx R package parameter mapping.
# But check_gradients.R passes parameters DIRECTLY to driver.
# If driver expects raw variance, and we pass negative, it throws.
# But in the output of check_gradients.R:
# G_toep_0 = -0.693147.
# This is log(0.5).
# So it seems semx uses log-variance?
# Let's check semx/R/semx.R or C++ bindings.

# In C++, ToeplitzCovariance uses parameters[0] as variance.
# It checks if > 0.
# If we pass -0.69, it throws.
# So how did check_gradients.R run?
# "Iter (V&G) 1 Params: ... G_toep_0=-0.693147"
# This output comes from the driver?
# No, it comes from the R script printing it?
# No, "Iter (V&G)" looks like C++ output or R callback.
# It's likely the R script's `eval_loglik` function is NOT printing this.
# Wait, check_gradients.R does NOT have a loop printing "Iter (V&G)".
# It calls `semx_fit` first!
# The output "Iter (V&G)" comes from `semx_fit`.
# So `semx_fit` is using a parameterization where variance can be negative?
# Or `semx_fit` wraps the parameters?

# If semx_fit uses a wrapper (e.g. log transform), then the parameters passed to C++ are transformed.
# But `LikelihoodDriver` takes `covariance_parameters`.
# Does `LikelihoodDriver` apply transforms?
# No, it passes them to `CovarianceStructure::fill_covariance`.
# `ToeplitzCovariance::fill_covariance` takes `parameters[0]` as variance.

# So if `semx_fit` prints negative values, but C++ requires positive, there must be a layer in between.
# `semx_fit` uses `LikelihoodDriver`.
# Maybe `semx_model` defines the covariance as "log_toeplitz"? No.

# Let's look at `check_gradients.R` again.
# It calls `semx_fit`.
# Then it extracts `initial_params`.
# Then it defines `eval_loglik`.
# Inside `eval_loglik`, it constructs `covariance_parameters`.
# It takes `p[[name]]`.
# If `p` comes from `semx_fit` result, and `semx_fit` uses log-variance, then `p` has log-variance.
# But `eval_loglik` passes `p` directly to `driver`.
# If `driver` expects variance, and we pass log-variance (negative), `driver` should throw.
# But `check_gradients.R` ran successfully (at least the gradient check part).
# The gradient check used `p0["G_toep_0"] <- 0.5`.
# 0.5 is positive. So it worked.
# But `semx_fit` output showed negative values.
# This implies `semx_fit` IS using a transform.

# Where is this transform?
# Maybe in `semx_model` construction?
# Or `LikelihoodDriver` has a wrapper?
# Or `CovarianceStructure` has a wrapper?

# I see `ScaledFixedCovariance`, `MultiKernelCovariance`.
# Maybe `semx` wraps `Toeplitz` in something?
# In `debug_toeplitz.R`: `structure = "toeplitz"`.

# Let's check `cpp/src/libsemx/covariance_structure.cpp` factory method.
# `CovarianceStructure::create`.

# I'll read the factory method.


# Analytic gradient
grad_analytic <- eval_grad(p0)
cat("Analytic gradient:\n")
print(grad_analytic)

# Numeric gradient
grad_numeric <- grad(eval_loglik, p0)
cat("Numeric gradient:\n")
print(grad_numeric)

# Compare
diff <- grad_analytic - grad_numeric
cat("Difference:\n")
print(diff)
cat("Max diff:", max(abs(diff)), "\n")
