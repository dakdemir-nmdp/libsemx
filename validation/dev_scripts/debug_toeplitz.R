library(semx)

cat("\n--- Test 2: Toeplitz Covariance vs Simulation ---\n")
set.seed(456)
n_subj <- 100
t_levels <- 4
subj <- factor(rep(seq_len(n_subj), each = t_levels))
time <- rep(seq_len(t_levels) - 1, times = n_subj)

# Toeplitz parameters: diagonal (lag 0), lag 1, lag 2, lag 3
toep_params <- c(1.0, 0.6, 0.3, 0.1) 
# Construct V
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

model_semx_toep <- semx_model(
  equations = c("y ~ 1 + time", "y ~~ 0.001 * y"), # Fix residual to small positive
  families = c(y = "gaussian"),
  covariances = covs,
  random_effects = random_effects
)

# Set initial values close to truth
# Variance ~ 1.0 -> log(1) = 0
# k1 ~ 0.6 -> atanh(0.6) = 0.693
# k2 ~ -0.1 -> atanh(-0.1) = -0.1
# k3 ~ 0 -> 0
param_ids <- model_semx_toep$ir$parameter_ids()
init_params <- vector("list", length(param_ids))
names(init_params) <- param_ids

init_params[["G_toep_0"]] <- 1.0
init_params[["G_toep_1"]] <- 0.693
init_params[["G_toep_2"]] <- -0.1
init_params[["G_toep_3"]] <- 0.0
init_params[["alpha_y_on__intercept"]] <- 1.5
init_params[["beta_y_on_time"]] <- -0.2
model_semx_toep$parameters <- init_params

# Inspect edges
# print("Edges:")
# for (e in model_semx_toep$edges) {
#   if (e$kind == 2) { # Covariance
#      cat(sprintf("Cov: %s <-> %s, param: %s\n", e$source, e$target, e$parameter_id))
#   }
# }

fit_semx_toep <- semx_fit(model_semx_toep, df_toep, estimation_method = "REML")
print(summary(fit_semx_toep))

# Check likelihood at true parameters
driver <- new(LikelihoodDriver)
options <- new(OptimizationOptions)

# Prepare data map
data_map <- as.list(df_toep)
data_map[["_intercept"]] <- rep(1.0, nrow(df_toep))

# Linear predictors
lp_y <- 1.5 * data_map[["_intercept"]] - 0.2 * data_map[["time"]]
linear_predictors <- list(y = lp_y)

# Dispersions
dispersions <- list(y = rep(0.001, nrow(df_toep)))

# Covariance parameters
# Variance = 1.0
# Kappas: 0.6, -0.09375, -0.064
# Transformed: arctanh(0.6), arctanh(-0.09375), arctanh(-0.064)
k1 <- atanh(0.6)
k2 <- atanh(-0.09375)
k3 <- atanh(-0.064)

cov_params <- list(G_toep = c(1.0, k1, k2, k3))

ll_true <- driver$evaluate_model_loglik_full(
  model_semx_toep$ir, 
  data_map, 
  linear_predictors, 
  dispersions, 
  cov_params, 
  list(), 
  list(), 
  list(), 
  1L # REML
)

cat(sprintf("\nLog-likelihood at true parameters: %f\n", ll_true))

# Check gradient at true parameters
grad_true <- driver$evaluate_model_gradient(
  model_semx_toep$ir, 
  data_map, 
  linear_predictors, 
  dispersions, 
  cov_params, 
  list(), 
  list(), 
  list(), 
  1L # REML
)
print("Gradient at true parameters:")
print(grad_true)

# Finite Difference Check for G_toep_1
delta <- 1e-4
cov_params_plus <- cov_params
cov_params_plus$G_toep[2] <- cov_params_plus$G_toep[2] + delta # Perturb k1 (index 2 in vector)

ll_plus <- driver$evaluate_model_loglik_full(
  model_semx_toep$ir, 
  data_map, 
  linear_predictors, 
  dispersions, 
  cov_params_plus, 
  list(), 
  list(), 
  list(), 
  1L
)

fd_grad <- (ll_plus - ll_true) / delta
cat(sprintf("\nFinite Difference Gradient for G_toep_1: %f\n", fd_grad))
cat(sprintf("Analytic Gradient for G_toep_1: %f\n", grad_true$G_toep_1))

cat(sprintf("Log-likelihood at converged parameters: %f\n", fit_semx_toep$optimization_result$objective_value * -1)) # objective is NLL

