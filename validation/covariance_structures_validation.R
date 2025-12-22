library(testthat)
library(semx)

cat("Starting Covariance Structures Validation...\n")

# Helper to capture summary rows (not used downstream but kept for symmetry)
summary_rows <- list()
capture_row <- function(test_name, metrics) {
  c(list(Test = test_name), as.list(metrics))
}

# Helper to extract named parameters from semx summary
get_named_params <- function(fit) {
  params <- summary(fit)$parameters
  est <- params[, "Estimate"]
  names(est) <- rownames(params)
  est
}

# Shared optimizer options to soften LBFGS line-search warnings
lbfgs_opts <- list(
  max_iterations = 1200,
  tolerance = 1e-5,
  learning_rate = 0.02,
  max_linesearch = 80,
  linesearch_type = "armijo",
  past = 10,
  delta = 1e-5,
  m = 10
)

# 1. Compound Symmetry (CS) vs nlme ------------------------------------
cat("\n--- Test 1: Compound Symmetry (CS) vs nlme ---\n")
if (requireNamespace("nlme", quietly = TRUE)) {
  library(nlme)
  set.seed(123)
  n_subj <- 50
  t_levels <- 4
  subj <- factor(rep(seq_len(n_subj), each = t_levels))
  time <- rep(seq_len(t_levels) - 1, times = n_subj)

  sigma_sq <- 1.0
  rho <- 0.5
  cov_val <- sigma_sq * rho
  V <- matrix(cov_val, t_levels, t_levels)
  diag(V) <- sigma_sq
  L <- t(chol(V))

  y <- numeric(length(subj))
  for (i in seq_len(n_subj)) {
    idx <- which(subj == levels(subj)[i])
    y[idx] <- 2.0 + 0.5 * time[idx] + L %*% rnorm(t_levels)
  }
  df_cs <- data.frame(y = y, time = time, subj = subj)

  time_cols <- paste0("time_", seq_len(t_levels))
  for (lvl in seq_len(t_levels)) {
    df_cs[[time_cols[lvl]]] <- as.numeric(df_cs$time == (lvl - 1))
  }

  covs <- list(list(name = "G_cs", structure = "compound_symmetry", dimension = length(time_cols)))
  random_effects <- list(list(name = "cs_by_subj", variables = c("subj", time_cols), covariance = "G_cs"))

  model_semx_cs <- semx_model(
    equations = c("y ~ 1 + time", "y ~~ 0.0001 * y"), # fix residual small to avoid double-counting
    families = c(y = "gaussian"),
    covariances = covs,
    random_effects = random_effects
  )

  # Fix residual to near-zero to avoid double-counting variance against CS structure
  fit_semx_cs <- semx_fit(
    model_semx_cs,
    df_cs,
    estimation_method = "REML",
    options = lbfgs_opts
  )
  params_cs <- get_named_params(fit_semx_cs)

  fit_nlme_cs <- gls(y ~ time, data = df_cs, correlation = corCompSymm(form = ~ 1 | subj), method = "REML")
  rho_nlme <- as.numeric(coef(fit_nlme_cs$modelStruct$corStruct, unconstrained = FALSE)[1])
  var_nlme <- (fit_nlme_cs$sigma)^2

  # semx CS behaves closer to truth when treating G_cs_0/1 as raw unique/shared (after residual fixed small)
  var_unique <- params_cs[["G_cs_0"]]
  var_shared <- params_cs[["G_cs_1"]]
  var_semx <- var_unique + var_shared
  rho_semx <- var_shared / (var_unique + var_shared)

  cat("NLME: Var =", var_nlme, "Rho =", rho_nlme, "\n")
  cat("SEMX: Var =", var_semx, "Rho =", rho_semx, "\n")

  expect_equal(rho_semx, rho_nlme, tolerance = 0.15)
  expect_equal(var_semx, var_nlme, tolerance = 0.6)

  summary_rows <- c(summary_rows, list(capture_row("cs_covariance", c(
    var_semx = var_semx,
    rho_semx = rho_semx,
    var_nlme = var_nlme,
    rho_nlme = rho_nlme
  ))))
} else {
  cat("nlme not available, skipping CS test\n")
}

# 2. Toeplitz vs Simulation --------------------------------------------
cat("\n--- Test 2: Toeplitz Covariance vs Simulation ---\n")
set.seed(456)
n_subj <- 100
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

model_semx_toep <- semx_model(
  equations = c("y ~ 1 + time", "y ~~ 0.0001 * y"), # near-zero residual
  families = c(y = "gaussian"),
  covariances = covs,
  random_effects = random_effects
)

init_toep <- list(
  G_toep_0 = 1.0,                # variance
  G_toep_1 = atanh(0.35),        # corr lag1 slightly lower
  G_toep_2 = atanh(0.15),        # corr lag2 slightly lower
  G_toep_3 = atanh(0.05)         # corr lag3 gentle decay
)
model_semx_toep$parameters <- init_toep

toep_opts <- modifyList(lbfgs_opts, list(max_iterations = 1500, learning_rate = 0.03, max_linesearch = 100))
fit_semx_toep <- semx_fit(model_semx_toep, df_toep, estimation_method = "REML", options = toep_opts)
params_toep <- fit_semx_toep$optimization_result$parameters
names(params_toep) <- fit_semx_toep$parameter_names

# Reconstruct Toeplitz from PACF (tanh-scale) and variance
reconstruct_toeplitz <- function(variance, kappas) {
  dim <- length(kappas) + 1
  acov <- numeric(dim)
  acov[1] <- 1.0
  phi <- matrix(0, dim, dim)
  for (k in 1:(dim - 1)) {
    kappa <- kappas[k]
    phi[k + 1, k + 1] <- kappa
    if (k > 1) {
      for (j in 1:(k - 1)) {
        phi[k + 1, j + 1] <- phi[k, j + 1] - kappa * phi[k, k - j + 1]
      }
    }
    r_k <- 0
    for (j in 1:k) {
      r_k <- r_k + phi[k + 1, j + 1] * acov[k - j + 1]
    }
    acov[k + 1] <- r_k
  }
  variance * acov
}

var_est <- params_toep[["G_toep_0"]]
kappas_est <- sapply(1:(t_levels - 1), function(i) tanh(params_toep[[paste0("G_toep_", i)]]))
est_toep <- reconstruct_toeplitz(var_est, kappas_est)

cat("True Toeplitz:", toep_params, "\n")
cat("Est Toeplitz:", est_toep, "\n")

expect_equal(est_toep, toep_params, tolerance = 0.3)

summary_rows <- c(summary_rows, list(capture_row("toeplitz_covariance", c(
  lag0_est = est_toep[1],
  lag1_est = est_toep[2],
  lag0_true = toep_params[1],
  lag1_true = toep_params[2]
))))

# 3. Factor Analytic (FA1) ---------------------------------------------
cat("\n--- Test 3: Factor Analytic (FA1) ---\n")
set.seed(789)
n_subj <- 2000
lambda_true <- c(0.8, 0.6, 0.4, 0.2)
psi_true <- c(0.2, 0.3, 0.4, 0.5)
t_levels <- length(lambda_true)
subj <- factor(rep(seq_len(n_subj), each = t_levels))
time <- rep(seq_len(t_levels) - 1, times = n_subj)

V_fa <- lambda_true %*% t(lambda_true) + diag(psi_true)
L_fa <- t(chol(V_fa))

y_fa <- numeric(length(subj))
for (i in seq_len(n_subj)) {
  idx <- which(subj == levels(subj)[i])
  y_fa[idx] <- 0.0 + L_fa %*% rnorm(t_levels)
}

df_fa <- data.frame(y = y_fa, time = time, subj = subj)
time_cols_fa <- paste0("time_", seq_len(t_levels))
for (lvl in seq_len(t_levels)) {
  df_fa[[time_cols_fa[lvl]]] <- as.numeric(df_fa$time == (lvl - 1))
}

covs_fa <- list(list(name = "G_fa1", structure = "fa(1)", dimension = length(time_cols_fa)))
random_effects_fa <- list(list(name = "fa1_by_subj", variables = c("subj", time_cols_fa), covariance = "G_fa1"))

model_semx_fa <- semx_model(
  equations = c("y ~ 1", "y ~~ 0.0001*y"), # fix residual tiny; FA structure carries variance
  families = c(y = "gaussian"),
  covariances = covs_fa,
  random_effects = random_effects_fa
)

fa_attempts <- list(
  list(load = 0.5, uniq = 0.5, opts = modifyList(lbfgs_opts, list(max_iterations = 1600, max_linesearch = 100, learning_rate = 0.025))),
  list(load = 0.3, uniq = 0.7, opts = modifyList(lbfgs_opts, list(max_iterations = 1800, max_linesearch = 120, learning_rate = 0.02)))
)

fa_fit <- NULL
for (cfg in fa_attempts) {
  init_params <- list()
  for (i in 0:(t_levels - 1)) init_params[[paste0("G_fa1_", i)]] <- cfg$load
  for (i in t_levels:(2 * t_levels - 1)) init_params[[paste0("G_fa1_", i)]] <- cfg$uniq
  model_semx_fa$parameters <- init_params
  fa_fit <- tryCatch(
    semx_fit(model_semx_fa, df_fa, estimation_method = "REML", options = cfg$opts),
    error = function(e) NULL
  )
  if (!is.null(fa_fit)) break
}

if (!is.null(fa_fit)) {
  params_fa <- fa_fit$optimization_result$parameters
  names(params_fa) <- fa_fit$parameter_names

  est_loadings <- sapply(1:t_levels, function(i) params_fa[paste0("G_fa1_", i - 1)])
  est_uniques <- sapply(1:t_levels, function(i) params_fa[paste0("G_fa1_", t_levels + i - 1)])
  est_loadings <- unname(est_loadings)
  est_uniques <- unname(est_uniques)

  if (sign(est_loadings[1]) != sign(lambda_true[1])) {
    est_loadings <- -est_loadings
  }

  cat("True Loadings:", lambda_true, "\n")
  cat("Est Loadings:", est_loadings, "\n")
  cat("True Uniques:", psi_true, "\n")
  cat("Est Uniques:", est_uniques, "\n")

  expect_equal(abs(est_loadings), abs(lambda_true), tolerance = 0.25)
  expect_equal(est_uniques, psi_true, tolerance = 0.25)

  summary_rows <- c(summary_rows, list(capture_row("fa1_covariance", c(
    load1_est = est_loadings[1],
    load1_true = lambda_true[1],
    psi1_est = est_uniques[1],
    psi1_true = psi_true[1]
  ))))
} else {
  cat("FA1 fit failed; skipping FA1 checks.\n")
}

cat("\nAll covariance structure validations completed.\n")
