library(semx)
library(nlme)
library(testthat)
library(MASS)

set.seed(123)

# --- Helper to simulate LME data (Random Intercept + AR(1) Residual) ---
sim_lme_ar1 <- function(n_groups, n_time, rho, sigma_e, sigma_b, beta) {
  N <- n_groups * n_time
  time <- rep(1:n_time, n_groups)
  group <- rep(1:n_groups, each = n_time)
  x <- rnorm(N)
  
  # Random Intercepts
  b <- rnorm(n_groups, 0, sigma_b)
  
  # AR(1) Errors
  Sigma_e <- matrix(0, n_time, n_time)
  for (i in 1:n_time) {
    for (j in 1:n_time) {
      Sigma_e[i, j] <- sigma_e^2 * rho^abs(i - j)
    }
  }
  
  e <- mvrnorm(n_groups, mu = rep(0, n_time), Sigma = Sigma_e)
  e_vec <- as.vector(t(e))
  
  y <- 1 + beta * x + b[group] + e_vec
  
  df <- data.frame(y = y, x = x, time = factor(time), group = factor(group))
  
  # Create dummy columns for time (for semx)
  for(t in 1:n_time) {
    df[[paste0("t", t)]] <- as.numeric(df$time == t)
  }
  
  df
}

# --- 1. AR(1) Validation ---
cat("\n--- AR(1) Validation (vs lme) ---\n")
n_groups <- 200
n_time <- 5
rho_true <- 0.7
sigma_e_true <- 1.5
sigma_b_true <- 1.0
beta_true <- 0.5

df_ar1 <- sim_lme_ar1(n_groups, n_time, rho_true, sigma_e_true, sigma_b_true, beta_true)

# nlme::lme
# Random Intercept + AR(1) correlation on residuals
fit_lme <- lme(y ~ x, random = ~ 1 | group, data = df_ar1, 
               correlation = corAR1(form = ~ 1 | group), method = "ML")
ll_lme <- as.numeric(logLik(fit_lme))
coef_lme <- fixef(fit_lme)
# Extract parameters
# Phi (AR1 correlation)
phi_lme <- as.numeric(coef(fit_lme$modelStruct$corStruct, unconstrained = FALSE))
# Sigma (Residual SD)
sigma_e_lme <- fit_lme$sigma
# Random Intercept SD
vc <- as.numeric(VarCorr(fit_lme)[1, "StdDev"])
sigma_b_lme <- as.numeric(vc)

cat(sprintf("LME (ML): LL=%.4f, Beta=%.4f, Phi=%.4f, Sigma_e=%.4f, Sigma_b=%.4f\n", 
            ll_lme, coef_lme["x"], phi_lme, sigma_e_lme, sigma_b_lme))

# SEMX
time_cols <- paste0("t", 1:n_time)

model_semx_ar1 <- semx_model(
  equations = "y ~ 1 + x + (1 | group)",
  families = c(y = "gaussian"),
  covariances = list(
    list(name = "G_ar1", structure = "ar1", dimension = n_time)
  ),
  random_effects = list(
    list(name = "re_ar1", variables = c("group", time_cols), covariance = "G_ar1")
  )
)

fit_semx_ar1 <- semx_fit(model_semx_ar1, df_ar1)
ll_semx_ar1 <- -fit_semx_ar1$optimization_result$objective_value

cat(sprintf("LogLik: LME = %.4f, SEMX = %.4f\n", ll_lme, ll_semx_ar1))

# Check parameters
params <- fit_semx_ar1$optimization_result$parameters
names(params) <- fit_semx_ar1$parameter_names

# Transform parameters
# G_ar1_0: Variance of AR1 process (Natural scale)
# G_ar1_1: Tanh^-1(Rho)
# cov_re_1_0: Variance of Random Intercept (Natural scale)

sigma_ar1_semx <- sqrt(params["G_ar1_0"])
rho_semx <- tanh(params["G_ar1_1"])
sigma_b_semx <- params["cov_re_1_0"] # Unstructured 1x1 param is Cholesky factor (SD)

cat(sprintf("SEMX: Sigma_ar1=%.4f, Rho=%.4f, Sigma_b=%.4f\n", 
            sigma_ar1_semx, rho_semx, sigma_b_semx)
)

expect_equal(ll_semx_ar1, ll_lme, tolerance = 1e-2)
expect_equal(as.numeric(sigma_ar1_semx), as.numeric(sigma_e_lme), tolerance = 1e-2)
expect_equal(as.numeric(rho_semx), as.numeric(phi_lme), tolerance = 1e-2)
expect_equal(as.numeric(sigma_b_semx), as.numeric(sigma_b_lme), tolerance = 1e-2)

# --- Helper to simulate LME data (Random Intercept + CS Residual) ---
sim_lme_cs <- function(n_subjects = 200, n_times = 5, sigma_e = 1.0, sigma_b = 0.0, rho = 0.5, beta = 0.5) {
  N <- n_subjects * n_times
  time <- rep(1:n_times, n_subjects)
  id <- rep(1:n_subjects, each = n_times)
  x <- rnorm(N)
  
  # Random Intercepts
  b <- rnorm(n_subjects, 0, sigma_b)
  
  # CS Errors
  # Diagonal: sigma_e^2
  # Off-diagonal: sigma_e^2 * rho
  Sigma_e <- matrix(rho * sigma_e^2, n_times, n_times)
  diag(Sigma_e) <- sigma_e^2
  
  e <- MASS::mvrnorm(n_subjects, mu = rep(0, n_times), Sigma = Sigma_e)
  e_vec <- as.vector(t(e))
  
  y <- 1 + beta * x + b[id] + e_vec
  
  df <- data.frame(y = y, x = x, time = factor(time), id = factor(id))
  
  # Create dummy columns for time (for semx)
  for(t in 1:n_times) {
    df[[paste0("t", t)]] <- as.numeric(df$time == t)
  }
  
  df
}

# --- 2. Compound Symmetry (CS) Validation ---
cat("\n--- CS Validation (vs gls) ---\n")

# Simulate data for CS (No Random Intercept to ensure identifiability)
# We use GLS to fit a marginal model with CS structure.
# LME parameterization: Sigma^2 * ((1-rho)I + rho*J)
# SEMX parameterization: Unique + Shared
# Mapping:
#   Sigma^2 = Unique + Shared
#   Rho = Shared / (Unique + Shared)
#   Unique = Sigma^2 * (1 - Rho)
#   Shared = Sigma^2 * Rho

df_cs <- sim_lme_cs(n_subjects = 200, n_times = 5, sigma_e = 1.0, sigma_b = 0.0, rho = 0.5, beta = 0.5)

# Fit LME (GLS for marginal model)
library(nlme)
gls_fit_cs <- gls(y ~ x, correlation = corCompSymm(form = ~ 1 | id), data = df_cs, method = "ML")
gls_summ_cs <- summary(gls_fit_cs)

# Extract GLS parameters
beta_gls_cs <- coef(gls_fit_cs)["x"]
# Extract Rho
# corCompSymm stores rho in unconstrained form?
# getVarCov(gls_fit_cs) gives the matrix.
V_gls <- getVarCov(gls_fit_cs)
sigma_sq_gls <- V_gls[1,1]
cov_gls <- V_gls[1,2]
rho_gls <- cov_gls / sigma_sq_gls
sigma_gls <- sqrt(sigma_sq_gls)

cat(sprintf("GLS (ML): Beta=%.4f, Sigma=%.4f, Rho=%.4f\n", 
            beta_gls_cs, sigma_gls, rho_gls))

# Fit SEMX (CS only, no random intercept)
time_cols <- paste0("t", 1:5)
model_semx_cs <- semx_model(
  equations = c("y ~ 1 + x", "y ~~ 0.0001*y"),
  families = c(y = "gaussian"),
  covariances = list(
    list(name = "G_cs", structure = "compound_symmetry", dimension = 5)
  ),
  random_effects = list(
    list(name = "re_cs", variables = c("id", time_cols), covariance = "G_cs")
  )
)

# Provide better initial values to avoid local optima
model_semx_cs$parameters <- list(
  G_cs_0 = 0.5,
  G_cs_1 = 0.5
)

fit_semx_cs <- semx_fit(model_semx_cs, df_cs)

# Extract parameters
params_cs <- fit_semx_cs$optimization_result$parameters
names(params_cs) <- fit_semx_cs$parameter_names

print("Raw CS Parameters:")
print(params_cs)


# CS parameters: Unique (0), Shared (1)
# semx_fit returns constrained parameters (variances)
unique_var_semx <- params_cs["G_cs_0"]
shared_var_semx <- params_cs["G_cs_1"]

sigma_sq_semx <- unique_var_semx + shared_var_semx
sigma_semx <- sqrt(sigma_sq_semx)
rho_semx <- shared_var_semx / sigma_sq_semx

cat(sprintf("LogLik: GLS = %.4f, SEMX = %.4f\n", logLik(gls_fit_cs), -fit_semx_cs$optimization_result$objective_value))
cat(sprintf("SEMX: Unique=%.4f, Shared=%.4f -> Sigma=%.4f, Rho=%.4f\n", 
            unique_var_semx, shared_var_semx, sigma_semx, rho_semx))

# Assertions
tol <- 0.05
diffs <- c(
  abs(as.numeric(sigma_semx) - as.numeric(sigma_gls)),
  abs(as.numeric(rho_semx) - as.numeric(rho_gls))
)

if (any(diffs > tol)) {
  cat("Error: SEMX parameters do not match GLS parameters.\n")
  cat(sprintf("Sigma diff: %.4f\n", diffs[1]))
  cat(sprintf("Rho diff: %.4f\n", diffs[2]))
  stop("Validation failed")
} else {
  cat("CS Validation Passed!\n")
}





# SEMX
# Model: y = 1 + x + b + u_ar1 + e_semx
# b ~ N(0, sigma_b^2) (Random Intercept)
# u_ar1 ~ N(0, Sigma_AR1) (Random Vector on time dummies)
# e_semx ~ N(0, sigma_resid^2) (Residual)
# We expect sigma_resid^2 -> 0

time_cols <- paste0("t", 1:n_time)

# Covariances
covs <- list(
  list(name = "G_ar1", structure = "ar1", dimension = n_time),
  list(name = "G_int", structure = "diagonal", dimension = 1) # For random intercept
)

# Random Effects
# 1. Random Intercept: variables = "group" (and implicit intercept? No, need to specify design)
# If we use formula "y ~ ... + (1 | group)", semx handles it.
# But we want to mix formula and manual random effects.
# Let's use manual for both to be safe and explicit.
# Random Intercept: variables = c("group", "_intercept")? Or just "group" if intercept is implicit?
# semx_model's add_random_effect uses c(group, "_intercept") if intercept=TRUE.
# So we should pass c("group", "_intercept")?
# But "_intercept" is a special variable.
# Actually, we can use formula for the random intercept and manual for AR(1).
# Formula: "y ~ 1 + x + (1 | group)"
# This creates a random effect named "re_group_1" (or similar) with unstructured covariance.
# We can override its covariance structure if we knew the name.
# But "unstructured" 1x1 is same as "diagonal" 1x1. So default is fine.

# For AR(1), we add it manually.
# Note: semx_model doesn't link manual random effects to the outcome automatically?
# We suspected this.
# If so, we might need to add it to the equation as a latent variable?
# "y ~ ... + u_ar1"
# But u_ar1 is a vector?
# No, u_ar1 is a random effect definition.
# If semx doesn't support adding random effects to outcome via 'random_effects' list,
# we must use the formula syntax or the latent variable syntax.
# But formula syntax doesn't support "list of variables" easily.
# Wait, if we use "y ~ ... + (0 + t1 + t2 + t3 + t4 + t5 | group)", semx parses it.
# It creates a random effect with design variables t1...t5.
# And it assigns a covariance.
# The covariance name will be generated.
# BUT, we can't specify "structure = ar1" for it easily.
# UNLESS we rely on the order of covariances?
# Or we modify the IR after creation?
# Or we use the 'covariances' argument to define a covariance with the *expected* name?
# The name is usually "cov_re_1", "cov_re_2" etc. based on order.
# If we have one random effect in formula, it's "cov_re_1".
# So we can pass `covariances = list(list(name = "cov_re_1", structure = "ar1", dimension = n_time))`?
# Let's try this hack.
# We put the AR(1) term FIRST in the formula to ensure it's re_1?
# Or we only have one random effect in formula (the AR1 one), and add the intercept as another term?
# "y ~ 1 + x + (1 | group) + (0 + t1 + t2 + t3 + t4 + t5 | group)"
# This has 2 random effects.
# re_1: (1 | group). Dim 1.
# re_2: (0 + t1... | group). Dim 5.
# So we define covariance for "cov_re_2" as AR(1).

# We use formula for fixed effects and random intercept.
# We use manual definition for AR(1) random effect to control covariance structure.
# Note: semx applies all defined random effects to the model (likely to all outcomes or via implicit logic).

model_semx_ar1 <- semx_model(
  equations = "y ~ 1 + x + (1 | group)",
  families = c(y = "gaussian"),
  covariances = list(
    list(name = "G_ar1", structure = "ar1", dimension = n_time)
  ),
  random_effects = list(
    list(name = "re_ar1", variables = c("group", time_cols), covariance = "G_ar1")
  )
)

fit_semx_ar1 <- semx_fit(model_semx_ar1, df_ar1)
ll_semx_ar1 <- -fit_semx_ar1$optimization_result$objective_value

cat(sprintf("LogLik: LME = %.4f, SEMX = %.4f\n", ll_lme, ll_semx_ar1))

# Check parameters
print(fit_semx_ar1$parameters)


# --- 4. Toeplitz Validation ---
cat("\n--- Toeplitz Validation (vs Simulation) ---\n")

# Simulate Toeplitz data
# T=4
# Sigma = Toeplitz(1.0, 0.6, 0.3, 0.1)
n_subj <- 200
t_levels <- 4
toep_params <- c(1.0, 0.6, 0.3, 0.1)

V <- matrix(0, t_levels, t_levels)
for (i in 1:t_levels) {
  for (j in 1:t_levels) {
    lag <- abs(i - j)
    V[i, j] <- toep_params[lag + 1]
  }
}

subj <- factor(rep(seq_len(n_subj), each = t_levels))
time <- rep(seq_len(t_levels) - 1, times = n_subj)
L <- t(chol(V))

y <- numeric(length(subj))
for (i in seq_len(n_subj)) {
  idx <- which(subj == levels(subj)[i])
  y[idx] <- 1.5 - 0.2 * time[idx] + L %*% rnorm(t_levels)
}

df_toep <- data.frame(y = y, time = time, subj = subj)
time_cols <- paste0("t", 1:t_levels)
for (lvl in seq_len(t_levels)) {
  df_toep[[time_cols[lvl]]] <- as.numeric(df_toep$time == (lvl - 1))
}

# Fit SEMX
model_semx_toep <- semx_model(
  equations = c("y ~ 1 + time", "y ~~ 0.0001*y"),
  families = c(y = "gaussian"),
  covariances = list(
    list(name = "G_toep", structure = "toeplitz", dimension = t_levels)
  ),
  random_effects = list(
    list(name = "re_toep", variables = c("subj", time_cols), covariance = "G_toep")
  )
)

fit_semx_toep <- semx_fit(model_semx_toep, df_toep)
params_toep <- fit_semx_toep$optimization_result$parameters
names(params_toep) <- fit_semx_toep$parameter_names

print("Estimated Toeplitz Parameters:")
print(params_toep)

# Reconstruct Toeplitz
# G_toep_0: Log Variance
# G_toep_1..3: PACF (Tanh^-1)

var_est <- params_toep["G_toep_0"]
kappas_est <- numeric(t_levels - 1)
for (i in 1:(t_levels - 1)) {
  val <- params_toep[paste0("G_toep_", i)]
  kappas_est[i] <- tanh(val) 
}

# Levinson-Durbin recursion to get ACF from PACF (kappas)
reconstruct_toeplitz <- function(variance, kappas) {
  dim <- length(kappas) + 1
  acov <- numeric(dim)
  acov[1] <- 1.0
  
  # phi matrix (1-based indexing for R)
  phi <- matrix(0, dim, dim)
  
  for (k in 1:(dim - 1)) {
    kappa <- kappas[k]
    # phi[k][k] = kappa
    phi[k + 1, k + 1] <- kappa
    
    if (k > 1) {
      for (j in 1:(k - 1)) {
         # phi[k][j] = phi[k-1][j] - kappa * phi[k-1][k-j]
         val <- phi[k, j + 1] - kappa * phi[k, k - j + 1]
         phi[k + 1, j + 1] <- val
      }
    }
    
    r_k <- 0.0
    for (j in 1:k) {
       # r_k += phi[k][j] * acov[k-j]
       r_k <- r_k + phi[k + 1, j + 1] * acov[k - j + 1]
    }
    acov[k + 1] <- r_k
  }
  
  return(variance * acov)
}

est_toep <- reconstruct_toeplitz(var_est, kappas_est)

cat("True Toeplitz:", toep_params, "\n")
cat("Est Toeplitz:", est_toep, "\n")

expect_equal(est_toep, toep_params, tolerance = 0.2)
cat("Toeplitz Validation Passed!\n")

cat("\n--- Factor Analytic (FA1) Validation (vs Simulation) ---\n")

# Simulate FA(1) Data
# Sigma = Lambda * Lambda' + Psi
# Lambda (4x1), Psi (4x4 diagonal)

sim_fa1 <- function(n_subjects = 500, loadings, uniques) {
  dim <- length(uniques)
  Lambda <- matrix(loadings, ncol = 1)
  Psi <- diag(uniques)
  Sigma <- Lambda %*% t(Lambda) + Psi
  
  L <- t(chol(Sigma))
  
  y <- numeric(n_subjects * dim)
  time <- numeric(n_subjects * dim)
  subj <- numeric(n_subjects * dim)
  
  idx <- 1
  for (i in 1:n_subjects) {
    u <- L %*% rnorm(dim)
    for (j in 1:dim) {
      y[idx] <- 1.0 + 0.5 * (j - 1) + u[j]
      time[idx] <- j - 1
      subj[idx] <- i
      idx <- idx + 1
    }
  }
  
  return(data.frame(y = y, time = time, subj = subj))
}

true_loadings <- c(0.8, 0.6, 0.4, 0.2)
true_uniques <- c(0.2, 0.3, 0.4, 0.5)
n_dim <- 4

set.seed(123)
df_fa <- sim_fa1(n_subjects = 2000, loadings = true_loadings, uniques = true_uniques)

# Prepare data for SEMX
time_cols_fa <- paste0("t", 1:n_dim)
for (lvl in seq_len(n_dim)) {
  df_fa[[time_cols_fa[lvl]]] <- as.numeric(df_fa$time == (lvl - 1))
}

# Fit SEMX
model_semx_fa <- semx_model(
  equations = c("y ~ 1 + time", "y ~~ 0.0001*y"),
  families = c(y = "gaussian"),
  covariances = list(
    list(name = "G_fa", structure = "fa(1)", dimension = n_dim)
  ),
  random_effects = list(
    list(name = "re_fa", variables = c("subj", time_cols_fa), covariance = "G_fa")
  )
)

# Initialize with reasonable values to help convergence
# Loadings first (dim * rank), then Uniques (dim)
init_params <- list()
for(i in 0:(n_dim-1)) {
    init_params[[paste0("G_fa_", i)]] <- 0.5 # Loadings
}
for(i in n_dim:(2*n_dim-1)) {
    init_params[[paste0("G_fa_", i)]] <- 0.5 # Uniques
}
model_semx_fa$parameters <- init_params

fit_semx_fa <- semx_fit(model_semx_fa, df_fa)
params_fa <- fit_semx_fa$optimization_result$parameters
names(params_fa) <- fit_semx_fa$parameter_names

print("Estimated FA Parameters:")
print(params_fa)

# Extract Estimated Parameters
est_loadings <- numeric(n_dim)
est_uniques <- numeric(n_dim)

# Parameters are stored: Loadings (dim*rank) then Uniques (dim)
# For Rank 1: Loadings are indices 0 to dim-1
# Uniques are indices dim to 2*dim-1

for (i in 1:n_dim) {
  est_loadings[i] <- params_fa[paste0("G_fa_", i - 1)]
  est_uniques[i] <- params_fa[paste0("G_fa_", n_dim + i - 1)]
}

cat("True Loadings:", true_loadings, "\n")
cat("Est Loadings:", est_loadings, "\n")
cat("True Uniques:", true_uniques, "\n")
cat("Est Uniques:", est_uniques, "\n")

# Check Loadings (allow sign flip)
# If first loading is negative, flip all
if (sign(est_loadings[1]) != sign(true_loadings[1])) {
    est_loadings <- -est_loadings
}

expect_equal(est_loadings, true_loadings, tolerance = 0.1)
expect_equal(est_uniques, true_uniques, tolerance = 0.1)

cat("Factor Analytic Validation Passed!\n")
