#!/usr/bin/env Rscript

# Simulation study: compound symmetry (CS) recovery
# Compare semx vs nlme against known truth under several parameter mappings.

library(semx)
library(nlme)

set.seed(2025)

n_reps <- 20
n_subj <- 50
t_levels <- 4
beta0 <- 2.0
beta1 <- 0.5
sigma_true <- 1.0      # residual SD
rho_true <- 0.5        # CS correlation

# Simulate one dataset with CS structure
simulate_cs <- function() {
  subj <- factor(rep(seq_len(n_subj), each = t_levels))
  time <- rep(seq_len(t_levels) - 1, times = n_subj)
  cov_val <- sigma_true^2 * rho_true
  V <- matrix(cov_val, t_levels, t_levels)
  diag(V) <- sigma_true^2
  L <- t(chol(V))

  y <- numeric(length(subj))
  for (i in seq_len(n_subj)) {
    idx <- which(subj == levels(subj)[i])
    y[idx] <- beta0 + beta1 * time[idx] + L %*% rnorm(t_levels)
  }
  data.frame(y = y, time = time, subj = subj)
}

# Fit helpers
fit_nlme_cs <- function(df) {
  gls(y ~ time, data = df, correlation = corCompSymm(form = ~ 1 | subj), method = "REML")
}

fit_semx_cs <- function(df) {
  time_cols <- paste0("time_", seq_len(t_levels))
  for (lvl in seq_len(t_levels)) {
    df[[time_cols[lvl]]] <- as.numeric(df$time == (lvl - 1))
  }
  model_semx_cs <- semx_model(
    equations = c("y ~ 1 + time", "y ~~ 0.0001 * y"), # fix residual near-zero
    families = c(y = "gaussian"),
    covariances = list(list(name = "G_cs", structure = "compound_symmetry", dimension = length(time_cols))),
    random_effects = list(list(name = "cs_by_subj", variables = c("subj", time_cols), covariance = "G_cs"))
  )
  semx_fit(model_semx_cs, df, estimation_method = "REML")
}

extract_semx_mappings <- function(fit) {
  params <- fit$optimization_result$parameters
  names(params) <- fit$parameter_names
  out <- list()

  g0 <- params[["G_cs_0"]]
  g1 <- params[["G_cs_1"]]
  psi <- if ("psi_y_y" %in% names(params)) params[["psi_y_y"]] else NA_real_

  # Mapping A: assume g0 = log(var), g1 = atanh(rho)
  out$var_log <- exp(g0)
  out$rho_log <- tanh(g1)

  # Mapping B: assume g0/g1 are raw unique/shared variances
  out$var_raw <- g0 + g1
  out$rho_raw <- g1 / (g0 + g1)

  # Mapping C: assume both on log scale and include residual if present
  u <- exp(g0)
  s <- exp(g1)
  r <- if (is.na(psi)) 0 else exp(psi)
  out$var_full <- u + s + r
  out$rho_full <- s / (u + s)

  out
}

results <- lapply(seq_len(n_reps), function(rep_idx) {
  df <- simulate_cs()

  gls_fit <- fit_nlme_cs(df)
  gls_sigma <- gls_fit$sigma
  gls_rho <- as.numeric(coef(gls_fit$modelStruct$corStruct, unconstrained = FALSE)[1])

  semx_fit_cs <- fit_semx_cs(df)
  semx_map <- extract_semx_mappings(semx_fit_cs)

  data.frame(
    rep = rep_idx,
    sigma_true = sigma_true,
    rho_true = rho_true,
    sigma_gls = gls_sigma,
    rho_gls = gls_rho,
    sigma_semx_log = sqrt(semx_map$var_log),
    rho_semx_log = semx_map$rho_log,
    sigma_semx_raw = sqrt(semx_map$var_raw),
    rho_semx_raw = semx_map$rho_raw,
    sigma_semx_full = sqrt(semx_map$var_full),
    rho_semx_full = semx_map$rho_full
  )
})

df_res <- do.call(rbind, results)

rmse <- function(x, truth) sqrt(mean((x - truth)^2))

summary_df <- data.frame(
  metric = c("sigma_rmse_gls", "rho_rmse_gls",
             "sigma_rmse_semx_log", "rho_rmse_semx_log",
             "sigma_rmse_semx_raw", "rho_rmse_semx_raw",
             "sigma_rmse_semx_full", "rho_rmse_semx_full"),
  value = c(
    rmse(df_res$sigma_gls, sigma_true),
    rmse(df_res$rho_gls, rho_true),
    rmse(df_res$sigma_semx_log, sigma_true),
    rmse(df_res$rho_semx_log, rho_true),
    rmse(df_res$sigma_semx_raw, sigma_true),
    rmse(df_res$rho_semx_raw, rho_true),
    rmse(df_res$sigma_semx_full, sigma_true),
    rmse(df_res$rho_semx_full, rho_true)
  )
)

cat("\n--- RMSE vs truth over", n_reps, "replicates ---\n")
print(summary_df)

cat("\n--- Means of estimates ---\n")
means <- data.frame(
  metric = c("sigma_gls", "rho_gls",
             "sigma_semx_log", "rho_semx_log",
             "sigma_semx_raw", "rho_semx_raw",
             "sigma_semx_full", "rho_semx_full"),
  mean_est = c(
    mean(df_res$sigma_gls),
    mean(df_res$rho_gls),
    mean(df_res$sigma_semx_log),
    mean(df_res$rho_semx_log),
    mean(df_res$sigma_semx_raw),
    mean(df_res$rho_semx_raw),
    mean(df_res$sigma_semx_full),
    mean(df_res$rho_semx_full)
  )
)
print(means)

cat("\nNote: three semx mappings are reported:\n",
    "- log: G_cs_0 log(var), G_cs_1 atanh(rho)\n",
    "- raw: G_cs_0 unique variance, G_cs_1 shared variance\n",
    "- full: exp() of both and adds residual if present\n")
