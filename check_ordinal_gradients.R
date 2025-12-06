library(semx)
library(numDeriv)

cat("--- Checking Ordinal Gradients ---\n")

# 1. Generate Data
set.seed(123)
N <- 100
F_true <- rnorm(N, 0, 1)
y_star <- 1.0 * F_true + rnorm(N, 0, 1)
y <- cut(y_star, breaks = c(-Inf, 0, Inf), labels = FALSE) - 1
# y <- as.ordered(y)
df <- data.frame(y = y)

families <- c(y = "ordinal")

# 2. Define Model with FIXED loading to test threshold gradient
# We fix loading to 0.5
model_fixed <- semx_model(
    c("F =~ 0.5*y", "F ~~ 1*F", "y ~~ 1*y", "y ~ 0*1"),
    families = families,
    data = df
)

# 3. Setup Driver
driver <- new(LikelihoodDriver)

# Prepare data map
data_map <- as.list(df)
if (isTRUE(model_fixed$uses_intercept_column)) {
    data_map[["_intercept"]] <- rep(1.0, nrow(df))
}

# Initial parameters
param_ids <- model_fixed$ir$parameter_ids()
start_val <- c(y_threshold_1 = 0.0)

cat("Initial parameters:\n")
print(start_val)

# Inspect variables
cat("Variables:\n")
for (v in model_fixed$ir$variables) {
    cat("Name:", v$name, "Kind:", v$kind, "Family:", v$family, "\n")
}



# Helper to prepare arguments
prepare_args <- function(p) {
  # Fixed LP (alpha=0)
  lp_y <- rep(0.0, nrow(df))
  lp_int <- rep(0.0, nrow(df))
  
  linear_predictors <- list(y = lp_y, `_intercept` = lp_int)
  dispersions <- list(y = rep(1.0, nrow(df)), `_intercept` = rep(1.0, nrow(df)))
  covariance_parameters <- list()
  
  # Thresholds
  extra_params <- list()
  if ("y_threshold_1" %in% names(p)) {
      extra_params[["y"]] <- c(p[["y_threshold_1"]])
  }
  
  status <- list()
  
  list(
      linear_predictors = linear_predictors,
      dispersions = dispersions,
      covariance_parameters = covariance_parameters,
      extra_params = extra_params,
      status = status
  )
}

# Objective Function
obj_func_thresholds <- function(tau) {
    p <- c(y_threshold_1 = as.numeric(tau))
    # cat("DEBUG: p inside obj_func:\n")
    # print(p)
    args <- prepare_args(p)
    
    # cat("DEBUG: extra_params names:", paste(names(args$extra_params), collapse=", "), "\n")
    
    driver$evaluate_model_loglik_full(
        model_fixed$ir,
        data_map,
        args$linear_predictors,
        args$dispersions,
        args$covariance_parameters,
        args$status,
        args$extra_params,
        list(), # fixed_cov
        0L # ML
    )
}

# Analytic Gradient
args <- prepare_args(start_val)
extra_param_mappings <- list(y = c("y_threshold_1"))

grads <- driver$evaluate_model_gradient(
    model_fixed$ir,
    data_map,
    args$linear_predictors,
    args$dispersions,
    args$covariance_parameters,
    args$status,
    args$extra_params,
    list(),
    0L, # ML
    extra_param_mappings
)

cat("Analytic Gradient for Threshold:\n")
print(names(grads))
print(grads$y_threshold_1)

# Numeric Gradient
grad_num <- grad(obj_func_thresholds, start_val["y_threshold_1"])
cat("Numeric Gradient for Threshold:\n")
print(grad_num)

diff <- grads$y_threshold_1 - grad_num
cat("Difference:", diff, "\n")

if (abs(diff) < 1e-4) {
    cat("SUCCESS: Threshold gradient matches.\n")
} else {
    cat("FAILURE: Threshold gradient mismatch.\n")
}
