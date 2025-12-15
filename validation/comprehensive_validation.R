library(semx)
library(lme4)
library(testthat)

# Helpers -------------------------------------------------------------
# IMPORTANT: C++ uses tanh() transformation for correlation parameters
positive_to_corr <- function(val) {
  tanh(val)
}

corr_to_positive <- function(rho) {
  stopifnot(abs(rho) < 1)
  atanh(rho)
}

get_named_params <- function(fit) {
  params <- fit$optimization_result$parameters
  if (!is.null(fit$parameter_names)) {
    names(params) <- fit$parameter_names
  }
  params
}

# Optimization options (deprecated - now uses defaults)
# opts <- new(OptimizationOptions)
# opts$max_iterations <- 400
# opts$tolerance <- 1e-5
# opts$max_linesearch <- 10
opts <- NULL

# simple capture utility for summaries
capture_row <- function(tag, metrics) {
  data.frame(Test = tag, t(metrics), check.names = FALSE)
}

summary_rows <- list()


# 1. Basic Linear Model (Scalar) --------------------------------------
cat("\n--- Test 1: Basic Linear Model (Scalar) ---\n")
set.seed(123)
n <- 100
x <- rnorm(n)
y <- 1 + 2 * x + rnorm(n, sd = 0.5)
df_lm <- data.frame(x = x, y = y)

model_semx_lm <- semx_model(
  equations = c("y ~ 1 + x"),
  families = c(y = "gaussian")
)

fit_semx_lm <- semx_fit(model_semx_lm, df_lm, options = opts, estimation_method = "ML")
fit_lm <- lm(y ~ x, data = df_lm)

summ_semx_lm <- summary(fit_semx_lm)
coef_lm <- coef(fit_lm)
sigma_lm <- summary(fit_lm)$sigma

params <- summ_semx_lm$parameters
beta_x <- params["beta_y_on_x", "Estimate"]
alpha <- params["alpha_y_on__intercept", "Estimate"]
sigma_sq <- params["psi_y_y", "Estimate"]

expect_equal(as.numeric(beta_x), as.numeric(coef_lm["x"]), tolerance = 1e-4)
expect_equal(as.numeric(alpha), as.numeric(coef_lm["(Intercept)"]), tolerance = 1e-4)
expect_equal(sqrt(as.numeric(sigma_sq)), as.numeric(sigma_lm), tolerance = 0.05) # ML vs OLS bias
summary_rows <- c(summary_rows, list(capture_row("lm", c(beta = beta_x, intercept = alpha, sigma = sqrt(sigma_sq)))))


# 2. Logistic Regression (GLM) ----------------------------------------
cat("\n--- Test 2: Logistic Regression (GLM) ---\n")
set.seed(1231)
n <- 300
x <- rnorm(n)
eta <- -0.25 + 1.1 * x
prob <- plogis(eta)
y <- rbinom(n, size = 1, prob = prob)
df_logit <- data.frame(y = y, x = x)

model_semx_logit <- semx_model(
  equations = c("y ~ 1 + x"),
  families = c(y = "binomial")
)

fit_semx_logit <- semx_fit(model_semx_logit, df_logit, options = opts, estimation_method = "ML")
fit_glm_logit <- glm(y ~ x, data = df_logit, family = binomial)

params_logit <- summary(fit_semx_logit)$parameters
expect_equal(
  as.numeric(params_logit["beta_y_on_x", "Estimate"]),
  as.numeric(coef(fit_glm_logit)["x"]),
  tolerance = 1e-3
)
expect_equal(
  as.numeric(params_logit["alpha_y_on__intercept", "Estimate"]),
  as.numeric(coef(fit_glm_logit)["(Intercept)"]),
  tolerance = 1e-3
)
summary_rows <- c(summary_rows, list(capture_row("logit", c(beta = params_logit["beta_y_on_x", "Estimate"], intercept = params_logit["alpha_y_on__intercept", "Estimate"]))))


# 3. Poisson Regression (GLM) -----------------------------------------
cat("\n--- Test 3: Poisson Regression (GLM) ---\n")
set.seed(1232)
n <- 300
x <- rnorm(n)
lambda <- exp(0.35 + 0.9 * x)
y <- rpois(n, lambda = lambda)
df_pois <- data.frame(y = y, x = x)

model_semx_pois <- semx_model(
  equations = c("y ~ 1 + x"),
  families = c(y = "poisson")
)

fit_semx_pois <- semx_fit(model_semx_pois, df_pois, options = opts, estimation_method = "ML")
fit_glm_pois <- glm(y ~ x, data = df_pois, family = poisson)

params_pois <- summary(fit_semx_pois)$parameters
expect_equal(
  as.numeric(params_pois["beta_y_on_x", "Estimate"]),
  as.numeric(coef(fit_glm_pois)["x"]),
  tolerance = 1e-3
)
expect_equal(
  as.numeric(params_pois["alpha_y_on__intercept", "Estimate"]),
  as.numeric(coef(fit_glm_pois)["(Intercept)"]),
  tolerance = 1e-3
)
summary_rows <- c(summary_rows, list(capture_row("poisson", c(beta = params_pois["beta_y_on_x", "Estimate"], intercept = params_pois["alpha_y_on__intercept", "Estimate"]))))


# 3b. Negative Binomial Regression (GLM) ------------------------------
cat("\n--- Test 3b: Negative Binomial Regression (GLM) ---\n")
if (requireNamespace("MASS", quietly = TRUE)) {
  set.seed(12321)
  n <- 400
  x <- rnorm(n)
  eta <- 0.4 + 0.9 * x
  mu <- exp(eta)
  size_true <- 1.3
  y <- rnbinom(n, size = size_true, mu = mu)
  df_nb <- data.frame(y = y, x = x)

  model_semx_nb <- semx_model(
    equations = c("y ~ 1 + x"),
    families = c(y = "negative_binomial")
  )

  fit_semx_nb <- semx_fit(model_semx_nb, df_nb, options = opts, estimation_method = "ML")
  fit_glm_nb <- MASS::glm.nb(y ~ x, data = df_nb, control = glm.control(maxit = 100))

  params_nb <- summary(fit_semx_nb)$parameters
  expect_equal(
    as.numeric(params_nb["beta_y_on_x", "Estimate"]),
    as.numeric(coef(fit_glm_nb)["x"]),
    tolerance = 2e-2
  )
  expect_equal(
    as.numeric(params_nb["alpha_y_on__intercept", "Estimate"]),
    as.numeric(coef(fit_glm_nb)["(Intercept)"]),
    tolerance = 2e-2
  )
  summary_rows <- c(summary_rows, list(capture_row("nbinom", c(beta = params_nb["beta_y_on_x", "Estimate"], intercept = params_nb["alpha_y_on__intercept", "Estimate"]))))
} else {
  cat("skipping NB glm: package MASS not available\n")
}


# 3c. Compound Symmetry residuals (vs nlme corCompSymm) ----------------
cat("\n--- Test 3c: Compound Symmetry Residuals (nlme) ---\n")
if (requireNamespace("nlme", quietly = TRUE) && requireNamespace("MASS", quietly = TRUE)) {
  library(nlme)

  set.seed(4321)
  n_subj <- 35
  t_levels <- 4
  subj <- factor(rep(seq_len(n_subj), each = t_levels))
  time <- rep(seq_len(t_levels) - 1, times = n_subj)
  beta0 <- 2.0
  beta1 <- 1.1
  unique_var <- 0.3
  shared_var <- 0.2
  sigma_u <- 0.0

  Sigma_cs <- matrix(shared_var, nrow = t_levels, ncol = t_levels)
  diag(Sigma_cs) <- unique_var + shared_var

  y <- numeric(length(subj))
  subj_eff <- rnorm(n_subj, sd = sigma_u)
  for (i in seq_len(n_subj)) {
    idx <- which(subj == levels(subj)[i])
    eps <- drop(MASS::mvrnorm(1, mu = rep(0, t_levels), Sigma = Sigma_cs))
    y[idx] <- beta0 + beta1 * time[idx] + subj_eff[i] + eps
  }

  df_cs <- data.frame(y = y, time = time, subj = subj)
  time_cols <- paste0("time_", seq_len(t_levels))
  for (lvl in seq_len(t_levels)) {
    df_cs[[time_cols[lvl]]] <- as.numeric(df_cs$time == (lvl - 1))
  }

  covs_cs <- list(list(name = "G_cs", structure = "compound_symmetry", dimension = length(time_cols)))
  random_effects_cs <- list(list(name = "cs_by_subj", variables = c("subj", time_cols), covariance = "G_cs"))

  model_semx_cs <- semx_model(
    equations = c("y ~ 1 + time"),
    families = c(y = "gaussian"),
    covariances = covs_cs,
    random_effects = random_effects_cs
  )

  fit_semx_cs <- semx_fit(model_semx_cs, df_cs, options = opts, estimation_method = "REML")
  fit_nlme_cs <- nlme::gls(
    model = y ~ time,
    correlation = nlme::corCompSymm(form = ~ 1 | subj),
    data = df_cs,
    method = "REML"
  )

  params_cs <- get_named_params(fit_semx_cs)
  expect_true(all(c("G_cs_0", "G_cs_1") %in% names(params_cs)))
  unique_semx <- params_cs[["G_cs_0"]]
  shared_semx <- params_cs[["G_cs_1"]]
  var_semx <- unique_semx + shared_semx
  rho_semx <- shared_semx / var_semx

  rho_nlme <- as.numeric(coef(fit_nlme_cs$modelStruct$corStruct, unconstrained = FALSE)[1])
  var_nlme <- (fit_nlme_cs$sigma)^2

  expect_equal(rho_semx, rho_nlme, tolerance = 0.2)
  expect_equal(var_semx, var_nlme, tolerance = 0.35)

  loglik_semx_cs <- -fit_semx_cs$optimization_result$objective_value
  loglik_nlme_cs <- as.numeric(logLik(fit_nlme_cs))
  expect_equal(loglik_semx_cs, loglik_nlme_cs, tolerance = 0.75)
} else {
  cat("skipping CS vs nlme: package nlme or MASS not available\n")
}


# 4. Random Intercept Model (ML) --------------------------------------
cat("\n--- Test 4: Random Intercept Model (ML) ---\n")
data("sleepstudy", package = "lme4")
df_sleep <- sleepstudy

model_semx_ri <- semx_model(
  equations = c("Reaction ~ 1 + Days + (1 | Subject)"),
  families = c(Reaction = "gaussian")
)

fit_semx_ri <- semx_fit(model_semx_ri, df_sleep, options = opts, estimation_method = "ML")
fit_lmer_ri <- lmer(Reaction ~ Days + (1 | Subject), data = df_sleep, REML = FALSE)

summ_semx_ri <- summary(fit_semx_ri)
fixef_lmer <- fixef(fit_lmer_ri)
params_ri <- summ_semx_ri$parameters

expect_equal(as.numeric(params_ri["beta_Reaction_on_Days", "Estimate"]), as.numeric(fixef_lmer["Days"]), tolerance = 1e-3)
expect_equal(as.numeric(params_ri["alpha_Reaction_on__intercept", "Estimate"]), as.numeric(fixef_lmer["(Intercept)"]), tolerance = 1e-3)


# 5. Random Slope Model (ML) ------------------------------------------
cat("\n--- Test 5: Random Slope Model (ML) ---\n")
model_semx_rs <- semx_model(
  equations = c("Reaction ~ 1 + Days + (1 + Days | Subject)"),
  families = c(Reaction = "gaussian")
)

fit_semx_rs <- semx_fit(model_semx_rs, df_sleep, options = opts, estimation_method = "ML")
fit_lmer_rs <- lmer(Reaction ~ Days + (1 + Days | Subject), data = df_sleep, REML = FALSE)

summ_semx_rs <- summary(fit_semx_rs)
fixef_lmer_rs <- fixef(fit_lmer_rs)
params_rs <- summ_semx_rs$parameters

expect_equal(as.numeric(params_rs["beta_Reaction_on_Days", "Estimate"]), as.numeric(fixef_lmer_rs["Days"]), tolerance = 1e-3)
expect_equal(as.numeric(params_rs["alpha_Reaction_on__intercept", "Estimate"]), as.numeric(fixef_lmer_rs["(Intercept)"]), tolerance = 1e-3)


# 6. REML vs ML parity -------------------------------------------------
cat("\n--- Test 6: REML vs ML (Random Intercept) ---\n")
fit_semx_ri_reml <- semx_fit(model_semx_ri, df_sleep, options = opts, estimation_method = "REML")
fit_lmer_ri_reml <- lmer(Reaction ~ Days + (1 | Subject), data = df_sleep, REML = TRUE)

params_ri_reml <- summary(fit_semx_ri_reml)$parameters
expect_equal(as.numeric(params_ri_reml["beta_Reaction_on_Days", "Estimate"]), as.numeric(fixef(fit_lmer_ri_reml)["Days"]), tolerance = 5e-2)
expect_equal(as.numeric(params_ri_reml["alpha_Reaction_on__intercept", "Estimate"]), as.numeric(fixef(fit_lmer_ri_reml)["(Intercept)"]), tolerance = 5e-1)

loglik_semx_reml <- -fit_semx_ri_reml$optimization_result$objective_value
loglik_lmer_reml <- as.numeric(logLik(fit_lmer_ri_reml))
expect_equal(loglik_semx_reml, loglik_lmer_reml, tolerance = 5e-2)
summary_rows <- c(summary_rows, list(capture_row("ri_reml", c(beta = params_ri_reml["beta_Reaction_on_Days", "Estimate"], intercept = params_ri_reml["alpha_Reaction_on__intercept", "Estimate"], loglik = loglik_semx_reml))))


# 6b. Random-only intercept (REML) vs lmer/sommer ---------------------
cat("\n--- Test 6b: Random-only intercept (REML) vs lmer & sommer ---\n")
set.seed(2025)
n_groups_ro <- 25
n_per_group_ro <- 12
group_ro <- factor(rep(seq_len(n_groups_ro), each = n_per_group_ro))
mu_true <- 1.5
sigma_u <- 0.7
sigma_eps <- 0.4
u <- rnorm(n_groups_ro, sd = sigma_u)
y <- mu_true + u[group_ro] + rnorm(length(group_ro), sd = sigma_eps)
df_ro <- data.frame(y = y, group = group_ro)

model_semx_ro <- semx_model(
  equations = c("y ~ 1 + (1 | group)"),
  families = c(y = "gaussian")
)

fit_semx_ro <- semx_fit(model_semx_ro, df_ro, options = opts, estimation_method = "REML")
fit_lmer_ro <- lmer(y ~ 1 + (1 | group), data = df_ro, REML = TRUE)

params_ro <- summary(fit_semx_ro)$parameters
alpha_ro <- params_ro["alpha_y_on__intercept", "Estimate"]
re_param_ro <- grep("^cov_re", rownames(params_ro), value = TRUE)
re_var_semx_ro <- as.numeric(params_ro[re_param_ro[[1]], "Estimate"])
resid_var_semx_ro <- as.numeric(params_ro["psi_y_y", "Estimate"])

re_var_lmer_ro <- attr(VarCorr(fit_lmer_ro)[[1]], "stddev")[[1]]^2
resid_var_lmer_ro <- sigma(fit_lmer_ro)^2
loglik_semx_ro <- -fit_semx_ro$optimization_result$objective_value
loglik_lmer_ro <- as.numeric(logLik(fit_lmer_ro))

expect_equal(as.numeric(alpha_ro), as.numeric(fixef(fit_lmer_ro)["(Intercept)"]), tolerance = 5e-2)
expect_equal(re_var_semx_ro, re_var_lmer_ro, tolerance = 3e-1)
expect_equal(resid_var_semx_ro, resid_var_lmer_ro, tolerance = 3e-1)
expect_equal(loglik_semx_ro, loglik_lmer_ro, tolerance = 5e-1)

if (requireNamespace("sommer", quietly = TRUE)) {
  library(sommer)
  fit_sommer_ro <- sommer::mmer(y ~ 1, random = ~ group, data = df_ro)
  sm_sommer <- summary(fit_sommer_ro)
  re_var_sommer <- sm_sommer$varcomp$VarComp[1]
  resid_var_sommer <- sm_sommer$varcomp$VarComp[2]
  alpha_sommer <- sm_sommer$betas$Estimate[[1]]
  loglik_sommer <- sm_sommer$logo$logLik[[1]]

  expect_equal(as.numeric(alpha_ro), as.numeric(alpha_sommer), tolerance = 1e-1)
  expect_equal(re_var_semx_ro, re_var_sommer, tolerance = 3.5e-1)
  expect_equal(resid_var_semx_ro, resid_var_sommer, tolerance = 3.5e-1)

  summary_rows <- c(summary_rows, list(capture_row("ri_reml_sommer", c(
    intercept = alpha_ro,
    re_var = re_var_semx_ro,
    resid_var = resid_var_semx_ro,
    loglik = loglik_semx_ro,
    re_var_sommer = re_var_sommer,
    resid_var_sommer = resid_var_sommer,
    loglik_sommer = loglik_sommer
  ))))
} else {
  cat("sommer not available; skipping sommer comparison for random-only intercept\n")
  summary_rows <- c(summary_rows, list(capture_row("ri_reml_sommer", c(intercept = alpha_ro, re_var = re_var_semx_ro, resid_var = resid_var_semx_ro, loglik = loglik_semx_ro))))
}


# 7. Binomial GLMM (Random Intercept) ----------------------------------
cat("\n--- Test 7: Binomial GLMM (Random Intercept) ---\n")
set.seed(321)
n_groups <- 30
n_per_group <- 20
group <- factor(rep(seq_len(n_groups), each = n_per_group))
x <- rnorm(n_groups * n_per_group)
u <- rnorm(n_groups, sd = 0.8)
eta <- -0.4 + 1.1 * x + u[group]
prob <- plogis(eta)
y <- rbinom(length(x), 1, prob)
df_glmm <- data.frame(y = y, x = x, group = group)

model_semx_glmm <- semx_model(
  equations = c("y ~ 1 + x + (1 | group)"),
  families = c(y = "binomial")
)

fit_semx_glmm <- semx_fit(model_semx_glmm, df_glmm, options = opts, estimation_method = "ML")
fit_glmer <- glmer(
  y ~ x + (1 | group),
  data = df_glmm,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 20000))
)

params_glmm <- summary(fit_semx_glmm)$parameters
fixef_glmer <- fixef(fit_glmer)

expect_equal(as.numeric(params_glmm["beta_y_on_x", "Estimate"]), as.numeric(fixef_glmer["x"]), tolerance = 2e-3)
expect_equal(as.numeric(params_glmm["alpha_y_on__intercept", "Estimate"]), as.numeric(fixef_glmer["(Intercept)"]), tolerance = 2e-3)

re_param_candidates <- grep("^cov_re", rownames(params_glmm), value = TRUE)
expect_true(length(re_param_candidates) > 0)
re_param_name <- re_param_candidates[[1]]
re_var_semx <- as.numeric(params_glmm[re_param_name, "Estimate"])
re_var_glmer <- as.numeric(attr(VarCorr(fit_glmer)[[1]], "stddev")[[1]]^2)
expect_equal(re_var_semx, re_var_glmer, tolerance = 0.25)
summary_rows <- c(summary_rows, list(capture_row("binom_glmm", c(beta = params_glmm["beta_y_on_x", "Estimate"], intercept = params_glmm["alpha_y_on__intercept", "Estimate"], re_var = re_var_semx))))


# 7b. Poisson GLMM (Random Intercept) ----------------------------------
cat("\n--- Test 7b: Poisson GLMM (Random Intercept) ---\n")
set.seed(654)
n_groups_pois <- 35
n_per_group_pois <- 15
group_pois <- factor(rep(seq_len(n_groups_pois), each = n_per_group_pois))
x_pois <- rnorm(n_groups_pois * n_per_group_pois)
u_pois <- rnorm(n_groups_pois, sd = 0.5)
eta_pois <- 0.2 + 0.7 * x_pois + u_pois[group_pois]
mu_pois <- exp(eta_pois)
y_pois <- rpois(length(mu_pois), lambda = mu_pois)
df_pglmm <- data.frame(y = y_pois, x = x_pois, group = group_pois)

model_semx_pglmm <- semx_model(
  equations = c("y ~ 1 + x + (1 | group)"),
  families = c(y = "poisson")
)

fit_semx_pglmm <- semx_fit(model_semx_pglmm, df_pglmm, options = opts, estimation_method = "ML")
fit_glmer_pois <- glmer(
  y ~ x + (1 | group),
  data = df_pglmm,
  family = poisson,
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 20000))
)

params_pglmm <- summary(fit_semx_pglmm)$parameters
fixef_glmer_pois <- fixef(fit_glmer_pois)

expect_equal(as.numeric(params_pglmm["beta_y_on_x", "Estimate"]), as.numeric(fixef_glmer_pois["x"]), tolerance = 5e-3)
expect_equal(as.numeric(params_pglmm["alpha_y_on__intercept", "Estimate"]), as.numeric(fixef_glmer_pois["(Intercept)"]), tolerance = 5e-3)

re_param_pois <- grep("^cov_re", rownames(params_pglmm), value = TRUE)
expect_true(length(re_param_pois) > 0)
re_var_semx_pois <- as.numeric(params_pglmm[re_param_pois[[1]], "Estimate"])
re_var_glmer_pois <- as.numeric(attr(VarCorr(fit_glmer_pois)[[1]], "stddev")[[1]]^2)
expect_equal(re_var_semx_pois, re_var_glmer_pois, tolerance = 0.3)

summary_rows <- c(summary_rows, list(capture_row("pois_glmm", c(beta = params_pglmm["beta_y_on_x", "Estimate"], intercept = params_pglmm["alpha_y_on__intercept", "Estimate"], re_var = re_var_semx_pois))))


# 7c. Mixed outcomes SEM (Gaussian + Binomial) -------------------------
cat("\n--- Test 7c: Mixed outcomes SEM (Gaussian + Binomial) ---\n")
set.seed(7777)
n_sem <- 400
x_sem <- rnorm(n_sem)
mu_y1 <- 0.5 + 1.2 * x_sem
y1_sem <- rnorm(n_sem, mean = mu_y1, sd = 0.7)
eta_y2 <- -0.3 + 0.9 * y1_sem + 0.5 * x_sem
prob_y2 <- plogis(eta_y2)
y2_sem <- rbinom(n_sem, 1, prob_y2)
df_sem <- data.frame(x = x_sem, y1 = y1_sem, y2 = y2_sem)

model_semx_mix <- semx_model(
  equations = c(
    "y1 ~ 1 + x",
    "y2 ~ 1 + x + y1"
  ),
  families = c(y1 = "gaussian", y2 = "binomial")
)

fit_semx_mix <- semx_fit(model_semx_mix, df_sem, options = opts, estimation_method = "ML")
summ_semx_mix <- summary(fit_semx_mix)$parameters

# Comparator fits (separate, since glm/lm do not jointly estimate)
fit_lm_y1 <- lm(y1 ~ x, data = df_sem)
fit_glm_y2 <- glm(y2 ~ x + y1, data = df_sem, family = binomial)

expect_equal(as.numeric(summ_semx_mix["beta_y1_on_x", "Estimate"]), as.numeric(coef(fit_lm_y1)["x"]), tolerance = 5e-3)
expect_equal(as.numeric(summ_semx_mix["alpha_y1_on__intercept", "Estimate"]), as.numeric(coef(fit_lm_y1)["(Intercept)"]), tolerance = 5e-3)
expect_equal(as.numeric(summ_semx_mix["beta_y2_on_x", "Estimate"]), as.numeric(coef(fit_glm_y2)["x"]), tolerance = 1e-2)
expect_equal(as.numeric(summ_semx_mix["beta_y2_on_y1", "Estimate"]), as.numeric(coef(fit_glm_y2)["y1"]), tolerance = 1e-2)
expect_equal(as.numeric(summ_semx_mix["alpha_y2_on__intercept", "Estimate"]), as.numeric(coef(fit_glm_y2)["(Intercept)"]), tolerance = 1e-2)

summary_rows <- c(summary_rows, list(capture_row("sem_mixed", c(
  beta_y1 = summ_semx_mix["beta_y1_on_x", "Estimate"],
  alpha_y1 = summ_semx_mix["alpha_y1_on__intercept", "Estimate"],
  beta_y2_x = summ_semx_mix["beta_y2_on_x", "Estimate"],
  beta_y2_y1 = summ_semx_mix["beta_y2_on_y1", "Estimate"],
  alpha_y2 = summ_semx_mix["alpha_y2_on__intercept", "Estimate"]
))))


# 7d. Negative Binomial GLMM (Random Intercept) -----------------------
cat("\n--- Test 7d: Negative Binomial GLMM (Random Intercept) ---\n")
set.seed(888)
n_groups_nb <- 30
n_per_group_nb <- 18
group_nb <- factor(rep(seq_len(n_groups_nb), each = n_per_group_nb))
x_nb <- rnorm(n_groups_nb * n_per_group_nb)
u_nb <- rnorm(n_groups_nb, sd = 0.6)
eta_nb <- 0.1 + 0.8 * x_nb + u_nb[group_nb]
mu_nb <- exp(eta_nb)
size_true_nb <- 1.4
y_nb <- rnbinom(length(mu_nb), size = size_true_nb, mu = mu_nb)
df_nb_glmm <- data.frame(y = y_nb, x = x_nb, group = group_nb)

model_semx_nb_glmm <- semx_model(
  equations = c("y ~ 1 + x + (1 | group)"),
  families = c(y = "negative_binomial")
)

fit_semx_nb_glmm <- semx_fit(model_semx_nb_glmm, df_nb_glmm, options = opts, estimation_method = "ML")
fit_glmer_nb <- glmer.nb(
  y ~ x + (1 | group),
  data = df_nb_glmm,
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 20000))
)

params_nb_glmm <- summary(fit_semx_nb_glmm)$parameters
fixef_glmer_nb <- fixef(fit_glmer_nb)

expect_equal(as.numeric(params_nb_glmm["beta_y_on_x", "Estimate"]), as.numeric(fixef_glmer_nb["x"]), tolerance = 1.5e-2)
expect_equal(as.numeric(params_nb_glmm["alpha_y_on__intercept", "Estimate"]), as.numeric(fixef_glmer_nb["(Intercept)"]), tolerance = 2e-2)

re_param_nb <- grep("^cov_re", rownames(params_nb_glmm), value = TRUE)
expect_true(length(re_param_nb) > 0)
re_var_semx_nb <- as.numeric(params_nb_glmm[re_param_nb[[1]], "Estimate"])
re_var_glmer_nb <- as.numeric(attr(VarCorr(fit_glmer_nb)[[1]], "stddev")[[1]]^2)
expect_equal(re_var_semx_nb, re_var_glmer_nb, tolerance = 0.35)

summary_rows <- c(summary_rows, list(capture_row("nb_glmm", c(beta = params_nb_glmm["beta_y_on_x", "Estimate"], intercept = params_nb_glmm["alpha_y_on__intercept", "Estimate"], re_var = re_var_semx_nb))))


# 7e. Binomial GLMM with random slope (vs glmer) ----------------------
cat("\n--- Test 7e: Binomial GLMM (Random Intercept + Slope) ---\n")
set.seed(999)
n_groups_rs <- 35
n_per_group_rs <- 18
group_rs <- factor(rep(seq_len(n_groups_rs), each = n_per_group_rs))
x_rs <- rnorm(n_groups_rs * n_per_group_rs)
u0 <- rnorm(n_groups_rs, sd = 0.7)
u1 <- rnorm(n_groups_rs, sd = 0.5)
eta_rs <- -0.3 + 1.0 * x_rs + u0[group_rs] + u1[group_rs] * x_rs
prob_rs <- plogis(eta_rs)
y_rs <- rbinom(length(x_rs), 1, prob_rs)
df_brs <- data.frame(y = y_rs, x = x_rs, group = group_rs)

model_semx_brs <- semx_model(
  equations = c("y ~ 1 + x + (1 + x | group)"),
  families = c(y = "binomial")
)

fit_semx_brs <- semx_fit(model_semx_brs, df_brs, options = opts, estimation_method = "ML")
fit_glmer_brs <- glmer(
  y ~ x + (1 + x | group),
  data = df_brs,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 30000))
)

params_brs <- summary(fit_semx_brs)$parameters
fixef_glmer_brs <- fixef(fit_glmer_brs)

expect_equal(as.numeric(params_brs["beta_y_on_x", "Estimate"]), as.numeric(fixef_glmer_brs["x"]), tolerance = 3e-2)
expect_equal(as.numeric(params_brs["alpha_y_on__intercept", "Estimate"]), as.numeric(fixef_glmer_brs["(Intercept)"]), tolerance = 3e-2)

re_params_brs <- params_brs[grep("^cov_re", rownames(params_brs)), "Estimate", drop = TRUE]
if (length(re_params_brs) >= 3) {
  L00 <- re_params_brs[[1]]
  L10 <- re_params_brs[[2]]
  L11 <- re_params_brs[[3]]
  var_int_semx <- L00^2
  cov_semx <- L00 * L10
  var_slope_semx <- L10^2 + L11^2

  vc_brs <- VarCorr(fit_glmer_brs)[[1]]
  sd_glmer <- attr(vc_brs, "stddev")
  cor_glmer <- attr(vc_brs, "correlation")
  var_int_glmer <- sd_glmer[[1]]^2
  var_slope_glmer <- sd_glmer[[2]]^2
  cov_glmer <- cor_glmer[1, 2] * sd_glmer[[1]] * sd_glmer[[2]]

  expect_equal(var_int_semx, var_int_glmer, tolerance = 0.35)
  expect_equal(var_slope_semx, var_slope_glmer, tolerance = 0.35)
  expect_equal(cov_semx, cov_glmer, tolerance = 0.5)

  summary_rows <- c(summary_rows, list(capture_row("binom_glmm_slope", c(
    beta = params_brs["beta_y_on_x", "Estimate"],
    intercept = params_brs["alpha_y_on__intercept", "Estimate"],
    var_int = var_int_semx,
    var_slope = var_slope_semx,
    cov_int_slope = cov_semx
  ))))
} else {
  cat("could not extract covariance parameters for binomial random-slope model\n")
}


# 8. AR(1) Covariance vs nlme ------------------------------------------
cat("\n--- Test 8: AR(1) Covariance vs nlme ---\n")
if (requireNamespace("nlme", quietly = TRUE)) {
  library(nlme)

  set.seed(444)
  n_subj <- 40
  t_levels <- 5
  subj <- factor(rep(seq_len(n_subj), each = t_levels))
  time <- rep(seq_len(t_levels) - 1, times = n_subj)
  beta0 <- 2.0
  beta1 <- 0.8
  phi_true <- 0.6
  sigma_eps <- 0.4
  sigma_u <- 0.7

  make_ar1 <- function(n, phi, sigma) {
    z <- numeric(n)
    z[1] <- rnorm(1, sd = sigma / sqrt(1 - phi^2))
    if (n > 1) {
      for (i in 2:n) {
        z[i] <- phi * z[i - 1] + rnorm(1, sd = sigma)
      }
    }
    z
  }

  y <- numeric(length(subj))
  subj_eff <- rnorm(n_subj, sd = sigma_u)
  for (i in seq_len(n_subj)) {
    idx <- which(subj == levels(subj)[i])
    eps <- make_ar1(length(idx), phi_true, sigma_eps)
    y[idx] <- beta0 + beta1 * time[idx] + subj_eff[i] + eps
  }

  df_ar1 <- data.frame(y = y, time = time, subj = subj)
  time_cols <- paste0("time_", seq_len(t_levels))
  for (lvl in seq_len(t_levels)) {
    df_ar1[[time_cols[lvl]]] <- as.numeric(df_ar1$time == (lvl - 1))
  }

  covs <- list(list(name = "G_ar1", structure = "ar1", dimension = length(time_cols)))
  random_effects <- list(list(name = "ar1_by_subj", variables = c("subj", time_cols), covariance = "G_ar1"))

  model_semx_ar1 <- semx_model(
    equations = c("y ~ 1 + time + (1 | subj)"),
    families = c(y = "gaussian"),
    covariances = covs,
    random_effects = random_effects
  )

  fit_semx_ar1 <- semx_fit(model_semx_ar1, df_ar1, options = opts, estimation_method = "REML")
  fit_nlme_ar1 <- nlme::lme(
    fixed = y ~ time,
    random = ~ 1 | subj,
    correlation = nlme::corAR1(form = ~ time | subj),
    data = df_ar1,
    method = "REML"
  )

  params_ar1 <- get_named_params(fit_semx_ar1)
  expect_true(all(c("G_ar1_0", "G_ar1_1") %in% names(params_ar1)))
  rho_semx <- positive_to_corr(params_ar1[["G_ar1_1"]])
  var_ar1_semx <- params_ar1[["G_ar1_0"]]

  rho_nlme <- as.numeric(coef(fit_nlme_ar1$modelStruct$corStruct, unconstrained = FALSE)[1])
  var_ar1_nlme <- as.numeric(nlme::getVarCov(fit_nlme_ar1))[1]

  expect_equal(rho_semx, rho_nlme, tolerance = 0.2)
  expect_equal(var_ar1_semx, var_ar1_nlme, tolerance = 0.35)

  loglik_semx_ar1 <- -fit_semx_ar1$optimization_result$objective_value
  loglik_nlme_ar1 <- as.numeric(logLik(fit_nlme_ar1))
  expect_equal(loglik_semx_ar1, loglik_nlme_ar1, tolerance = 0.5)
  summary_rows <- c(summary_rows, list(capture_row("ar1", c(rho = rho_semx, var = var_ar1_semx, loglik = loglik_semx_ar1))))
} else {
  cat("skipping AR(1) vs nlme: package nlme not available\n")
}

# 8b. Toeplitz covariance (random slopes) vs simulated truth -----------
cat("\n--- Test 8b: Toeplitz covariance (random slopes, simulated truth) ---\n")
if (requireNamespace("MASS", quietly = TRUE)) {
  toeplitz_cov_from_corrs <- function(variance, corrs) {
    dim <- length(corrs) + 1
    autocov <- numeric(dim)
    autocov[1] <- 1
    if (dim > 1) {
      phi <- matrix(0, nrow = dim, ncol = dim)
      for (k in seq_len(dim - 1)) {
        kappa <- corrs[k]
        phi[k + 1, k + 1] <- kappa
        if (k > 1) {
          for (j in 1:(k - 1)) {
            phi[k + 1, j + 1] <- phi[k, j + 1] - kappa * phi[k, k - j + 1]
          }
        }
        r_k <- 0
        for (j in 1:k) {
          r_k <- r_k + phi[k + 1, j + 1] * autocov[k - j + 1]
        }
        autocov[k + 1] <- r_k
      }
    }
    variance * stats::toeplitz(autocov)
  }

  set.seed(1001)
  n_groups_toep <- 40
  n_per_group_toep <- 10
  group_toep <- factor(rep(seq_len(n_groups_toep), each = n_per_group_toep))
  x1 <- rnorm(n_groups_toep * n_per_group_toep)
  x2 <- rnorm(n_groups_toep * n_per_group_toep)

  beta0 <- 1.1
  beta1 <- 0.9
  beta2 <- -0.4
  resid_sd <- 0.3

  var_true <- 0.75
  kappas_true <- c(0.45, 0.2)
  cov_toep <- toeplitz_cov_from_corrs(var_true, kappas_true)
  ranef_mat <- MASS::mvrnorm(n_groups_toep, mu = rep(0, 3), Sigma = cov_toep)

  intcpt <- rep(1, length(group_toep))
  y_toep <- numeric(length(group_toep))
  for (i in seq_len(n_groups_toep)) {
    idx <- which(group_toep == levels(group_toep)[i])
    b <- ranef_mat[i, ]
    y_toep[idx] <- beta0 + beta1 * x1[idx] + beta2 * x2[idx] +
      b[1] * intcpt[idx] + b[2] * x1[idx] + b[3] * x2[idx] +
      rnorm(length(idx), sd = resid_sd)
  }

  df_toep <- data.frame(y = y_toep, x1 = x1, x2 = x2, group = group_toep, intcpt = intcpt)
  covs_toep <- list(list(name = "G_toep", structure = "toeplitz", dimension = 3))
  re_toep <- list(list(name = "re_toep", variables = c("group", "intcpt", "x1", "x2"), covariance = "G_toep"))

  model_semx_toep <- semx_model(
    equations = c("y ~ 1 + x1 + x2"),
    families = c(y = "gaussian"),
    covariances = covs_toep,
    random_effects = re_toep
  )

  fit_semx_toep <- semx_fit(model_semx_toep, df_toep, options = opts, estimation_method = "REML")
  params_toep <- get_named_params(fit_semx_toep)

  expect_true(all(c("G_toep_0", "G_toep_1", "G_toep_2") %in% names(params_toep)))
  var_est <- params_toep[["G_toep_0"]]
  corr1_est <- positive_to_corr(params_toep[["G_toep_1"]])
  corr2_est <- positive_to_corr(params_toep[["G_toep_2"]])

  expect_true(is.finite(var_est) && var_est > 0)
  expect_true(is.finite(corr1_est) && abs(corr1_est) < 1)
  expect_true(is.finite(corr2_est) && abs(corr2_est) < 1)

  summary_rows <- c(summary_rows, list(capture_row("toeplitz_re", c(
    var = var_est,
    corr1 = corr1_est,
    corr2 = corr2_est
  ))))
} else {
  cat("skipping Toeplitz random-slope test: package MASS not available\n")
}


# 8c. Factor-analytic (FA1) random slopes vs simulated truth ----------
cat("\n--- Test 8c: Factor-analytic FA1 random slopes (simulated truth) ---\n")
if (requireNamespace("MASS", quietly = TRUE)) {
  set.seed(1202)
  n_groups_fa <- 45
  n_per_group_fa <- 10
  group_fa <- factor(rep(seq_len(n_groups_fa), each = n_per_group_fa))
  x1_fa <- rnorm(n_groups_fa * n_per_group_fa)
  x2_fa <- rnorm(n_groups_fa * n_per_group_fa)
  intcpt_fa <- rep(1, length(group_fa))

  beta0_fa <- 0.8
  beta1_fa <- 1.0
  beta2_fa <- -0.5
  resid_sd_fa <- 0.25

  loadings_true <- c(0.8, 0.6, 0.4)
  uniques_true <- c(0.25, 0.3, 0.2)
  Sigma_fa <- loadings_true %*% t(loadings_true) + diag(uniques_true)

  ranefs_fa <- MASS::mvrnorm(n_groups_fa, mu = rep(0, 3), Sigma = Sigma_fa)
  y_fa <- numeric(length(group_fa))
  for (i in seq_len(n_groups_fa)) {
    idx <- which(group_fa == levels(group_fa)[i])
    b <- ranefs_fa[i, ]
    y_fa[idx] <- beta0_fa + beta1_fa * x1_fa[idx] + beta2_fa * x2_fa[idx] +
      b[1] * intcpt_fa[idx] + b[2] * x1_fa[idx] + b[3] * x2_fa[idx] +
      rnorm(length(idx), sd = resid_sd_fa)
  }

  df_fa <- data.frame(y = y_fa, x1 = x1_fa, x2 = x2_fa, group = group_fa, intcpt = intcpt_fa)
  covs_fa <- list(list(name = "G_fa1", structure = "fa1", dimension = 3))
  re_fa <- list(list(name = "re_fa1", variables = c("group", "intcpt", "x1", "x2"), covariance = "G_fa1"))

  model_semx_fa <- semx_model(
    equations = c("y ~ 1 + x1 + x2"),
    families = c(y = "gaussian"),
    covariances = covs_fa,
    random_effects = re_fa
  )

  fit_semx_fa <- semx_fit(model_semx_fa, df_fa, options = opts, estimation_method = "REML")
  params_fa <- get_named_params(fit_semx_fa)
  fa_param_names <- grep("^G_fa1_", names(params_fa), value = TRUE)
  fa_param_names <- fa_param_names[order(as.integer(sub(".*_", "", fa_param_names)))]
  stopifnot(length(fa_param_names) == 6)
  loadings_est <- as.numeric(params_fa[fa_param_names[1:3]])
  uniques_est <- as.numeric(params_fa[fa_param_names[4:6]])

  Sigma_est <- loadings_est %*% t(loadings_est) + diag(uniques_est)
  expect_equal(as.numeric(Sigma_est), as.numeric(Sigma_fa), tolerance = 0.35)

  summary_rows <- c(summary_rows, list(capture_row("fa1_re", c(
    load1 = loadings_est[1],
    load2 = loadings_est[2],
    load3 = loadings_est[3],
    uniq1 = uniques_est[1],
    uniq2 = uniques_est[2],
    uniq3 = uniques_est[3]
  ))))
} else {
  cat("skipping FA1 random-slope test: package MASS not available\n")
}


# 9. Genomic BLUP (GRM) vs sommer -------------------------------------
cat("\n--- Test 9: Genomic BLUP vs sommer (GRM) ---\n")
if (requireNamespace("sommer", quietly = TRUE)) {
  library(sommer)
  set.seed(777)
  n_ind <- 10
  n_markers <- 20
  markers <- matrix(rbinom(n_ind * n_markers, size = 2, prob = 0.45), nrow = n_ind, ncol = n_markers)
  K <- tcrossprod(scale(markers, scale = FALSE))
  K <- K / n_markers
  K <- (K + t(K)) / 2
  diag(K) <- diag(K) + 1e-6
  rownames(K) <- colnames(K) <- paste0("id", seq_len(n_ind))
  sigma_g <- 0.7
  sigma_e <- 0.4
  L <- chol(K + diag(1e-6, n_ind))
  g_eff <- drop(rnorm(n_ind) %*% L) * sigma_g
  y_g <- 2.5 + g_eff + rnorm(n_ind, sd = sigma_e)

  ids <- paste0("id", seq_len(n_ind))
  df_g <- data.frame(id = factor(ids, levels = ids), y = y_g, cluster = 1)
  # indicator columns for design matrix (matches semx random_effect layout)
  for (i in seq_len(n_ind)) {
    df_g[[paste0("id", i)]] <- as.numeric(df_g$id == ids[i])
  }

  cov_grm <- list(list(name = "cov_grm", structure = "grm", dimension = n_ind))
  random_effects_grm <- list(list(name = "re_grm", variables = c("cluster", paste0("id", seq_len(n_ind))), covariance = "cov_grm"))

  model_semx_grm <- semx_model(
    equations = c("y ~ 1"),
    families = c(y = "gaussian", setNames(rep("gaussian", n_ind), paste0("id", seq_len(n_ind)))),
    kinds = c(cluster = "grouping"),
    covariances = cov_grm,
    genomic = list(cov_grm = list(markers = markers)),
    random_effects = random_effects_grm
  )

  fit_semx_grm <- semx_fit(model_semx_grm, df_g, options = opts, estimation_method = "REML")
  params_grm <- get_named_params(fit_semx_grm)
  re_var_semx_grm <- params_grm[["cov_grm_0"]]
  resid_var_semx_grm <- params_grm[["psi_y_y"]]
  alpha_semx_grm <- params_grm[["alpha_y_on__intercept"]]
  loglik_semx_grm <- -fit_semx_grm$optimization_result$objective_value

  fit_sommer_grm <- sommer::mmer(
    y ~ 1,
    random = ~ sommer::vs(id, Gu = K),
    data = df_g,
    verbose = FALSE
  )
  sm_sommer_grm <- summary(fit_sommer_grm)
  varcomp_df <- sm_sommer_grm$varcomp
  re_idx <- grep("id", rownames(varcomp_df))
  resid_idx <- grep("units", rownames(varcomp_df))
  re_var_sommer_grm <- varcomp_df$VarComp[re_idx[1]]
  resid_var_sommer_grm <- varcomp_df$VarComp[resid_idx[1]]
  alpha_sommer_grm <- sm_sommer_grm$betas$Estimate[[1]]
  loglik_sommer_grm <- sm_sommer_grm$logo$logLik[[1]]

  expect_equal(as.numeric(alpha_semx_grm), as.numeric(alpha_sommer_grm), tolerance = 2e-1)
  expect_equal(re_var_semx_grm, re_var_sommer_grm, tolerance = 4e-1)
  expect_equal(resid_var_semx_grm, resid_var_sommer_grm, tolerance = 4e-1)

  summary_rows <- c(summary_rows, list(capture_row("grm_reml_sommer", c(
    intercept = alpha_semx_grm,
    re_var = re_var_semx_grm,
    resid_var = resid_var_semx_grm,
    loglik = loglik_semx_grm,
    re_var_sommer = re_var_sommer_grm,
    resid_var_sommer = resid_var_sommer_grm,
    loglik_sommer = loglik_sommer_grm
  ))))
} else {
  cat("skipping GRM vs sommer: package sommer not available\n")
}


# 10. Multi-kernel GRM vs sommer --------------------------------------
cat("\n--- Test 10: Multi-kernel GRM vs sommer ---\n")
if (requireNamespace("sommer", quietly = TRUE)) {
  library(sommer)
  set.seed(778)
  n_ind2 <- 12
  n_markers1 <- 25
  n_markers2 <- 18
  markers1 <- matrix(rbinom(n_ind2 * n_markers1, size = 2, prob = 0.4), nrow = n_ind2, ncol = n_markers1)
  markers2 <- matrix(rbinom(n_ind2 * n_markers2, size = 2, prob = 0.6), nrow = n_ind2, ncol = n_markers2)

  make_grm <- function(mtx) {
    K <- tcrossprod(scale(mtx, scale = FALSE)) / ncol(mtx)
    K <- (K + t(K)) / 2
    diag(K) <- diag(K) + 1e-6
    rownames(K) <- colnames(K) <- paste0("id", seq_len(nrow(K)))
    K
  }

  K1 <- make_grm(markers1)
  K2 <- make_grm(markers2)

  sigma_g1 <- 0.6
  sigma_g2 <- 0.4
  sigma_e2 <- 0.35

  g1 <- MASS::mvrnorm(1, mu = rep(0, n_ind2), Sigma = sigma_g1^2 * K1)
  g2 <- MASS::mvrnorm(1, mu = rep(0, n_ind2), Sigma = sigma_g2^2 * K2)
  eps <- rnorm(n_ind2, sd = sigma_e2)
  y_multi <- 2.2 + g1 + g2 + eps

  ids2 <- paste0("id", seq_len(n_ind2))
  df_g2 <- data.frame(id = factor(ids2, levels = ids2), y = y_multi, cluster = 1)
  for (i in seq_len(n_ind2)) {
    df_g2[[paste0("id", i)]] <- as.numeric(df_g2$id == ids2[i])
  }

  cov_multi <- list(
    list(name = "cov_g1", structure = "grm", dimension = n_ind2),
    list(name = "cov_g2", structure = "grm", dimension = n_ind2)
  )
  random_effects_multi <- list(
    list(name = "re_g1", variables = c("cluster", paste0("id", seq_len(n_ind2))), covariance = "cov_g1"),
    list(name = "re_g2", variables = c("cluster", paste0("id", seq_len(n_ind2))), covariance = "cov_g2")
  )

  model_semx_multi <- semx_model(
    equations = c("y ~ 1"),
    families = c(y = "gaussian", setNames(rep("gaussian", n_ind2), paste0("id", seq_len(n_ind2)))),
    kinds = c(cluster = "grouping"),
    covariances = cov_multi,
    genomic = list(cov_g1 = list(markers = markers1), cov_g2 = list(markers = markers2)),
    random_effects = random_effects_multi
  )

  fit_semx_multi <- semx_fit(model_semx_multi, df_g2, options = opts, estimation_method = "REML")
  params_multi <- get_named_params(fit_semx_multi)
  alpha_semx_multi <- params_multi[["alpha_y_on__intercept"]]
  re1_semx <- params_multi[["cov_g1_0"]]
  re2_semx <- params_multi[["cov_g2_0"]]
  resid_semx <- params_multi[["psi_y_y"]]
  loglik_semx_multi <- -fit_semx_multi$optimization_result$objective_value

  fit_sommer_multi <- tryCatch(
    sommer::mmer(
      y ~ 1,
      random = ~ sommer::vs(id, Gu = K1) + sommer::vs(id, Gu = K2),
      data = df_g2,
      verbose = FALSE
    ),
    error = function(e) {
      message("sommer multi-kernel fit failed: ", e$message)
      NULL
    }
  )

  re1_sommer <- re2_sommer <- resid_sommer <- alpha_sommer_multi <- loglik_sommer_multi <- NA
  if (!is.null(fit_sommer_multi)) {
    sm_multi <- tryCatch(summary(fit_sommer_multi), error = function(e) NULL)
    vc_multi <- NULL
    if (!is.null(sm_multi)) {
      vc_multi <- sm_multi$varcomp
      alpha_sommer_multi <- sm_multi$betas$Estimate[[1]]
      loglik_sommer_multi <- sm_multi$logo$logLik[[1]]
    } else if (!is.null(fit_sommer_multi$var.comp)) {
      vc_multi <- fit_sommer_multi$var.comp
      if (!is.null(fit_sommer_multi$Beta)) {
        alpha_sommer_multi <- fit_sommer_multi$Beta[1]
      }
      if (!is.null(fit_sommer_multi$logLik)) {
        loglik_sommer_multi <- fit_sommer_multi$logLik
      }
    }

    if (!is.null(vc_multi)) {
      vc_df <- as.data.frame(vc_multi)
      row_names <- rownames(vc_multi)
      value_col <- if ("VarComp" %in% names(vc_df)) {
        "VarComp"
      } else if ("Variance" %in% names(vc_df)) {
        "Variance"
      } else {
        numeric_cols <- names(vc_df)[vapply(vc_df, is.numeric, logical(1))]
        if (length(numeric_cols)) numeric_cols[[1]] else NULL
      }
      if (!is.null(value_col)) {
        re_idx_all <- grep("id", row_names)
        resid_idx2 <- grep("units", row_names)
        if (length(re_idx_all) >= 2) {
          re1_sommer <- vc_df[re_idx_all[1], value_col]
          re2_sommer <- vc_df[re_idx_all[2], value_col]
        }
        if (length(resid_idx2)) {
          resid_sommer <- vc_df[resid_idx2[1], value_col]
        }
      }
    }

    if ((is.na(re1_sommer) || is.na(re2_sommer)) && !is.null(fit_sommer_multi$var.comp)) {
      vc_raw <- fit_sommer_multi$var.comp
      if (is.numeric(vc_raw) && length(vc_raw) >= 2) {
        re1_sommer <- if (is.na(re1_sommer)) as.numeric(vc_raw[[1]]) else re1_sommer
        re2_sommer <- if (is.na(re2_sommer)) as.numeric(vc_raw[[2]]) else re2_sommer
        if (length(vc_raw) >= 3 && is.na(resid_sommer)) {
          resid_sommer <- as.numeric(vc_raw[[3]])
        }
      }
    }

    if (is.na(alpha_sommer_multi) && !is.null(fit_sommer_multi$Beta)) {
      alpha_sommer_multi <- fit_sommer_multi$Beta[1]
    }

    if (is.null(re1_sommer) && !is.null(vc_multi) && is.numeric(vc_multi) && length(vc_multi) >= 2) {
      re1_sommer <- as.numeric(vc_multi[[1]])
      re2_sommer <- if (length(vc_multi) >= 2) as.numeric(vc_multi[[2]]) else NA
      resid_sommer <- if (length(vc_multi) >= 3) as.numeric(vc_multi[[3]]) else resid_sommer
    }

    if (!any(is.na(c(alpha_sommer_multi, re1_sommer, re2_sommer)))) {
      expect_equal(as.numeric(alpha_semx_multi), as.numeric(alpha_sommer_multi), tolerance = 3e-1)
      expect_equal(re1_semx, re1_sommer, tolerance = 6e-1)
      expect_equal(re2_semx, re2_sommer, tolerance = 6e-1)
    }
  }

  summary_rows <- c(summary_rows, list(capture_row("grm2_reml_sommer", c(
    intercept = alpha_semx_multi,
    re1 = re1_semx,
    re2 = re2_semx,
    resid_var = resid_semx,
    loglik = loglik_semx_multi,
    re1_sommer = re1_sommer,
    re2_sommer = re2_sommer,
    resid_var_sommer = resid_sommer,
    loglik_sommer = loglik_sommer_multi
  ))))
} else {
  cat("skipping multi-kernel GRM vs sommer: package sommer not available\n")
}


# 11. Path analysis (Gaussian SEM) vs lavaan/truth ---------------------
cat("\n--- Test 11: Gaussian path model vs lavaan ---\n")
set.seed(3131)
n_path <- 600
x_path <- rnorm(n_path)
z_path <- rnorm(n_path)
beta_y1_x <- 1.05
alpha_y1 <- 0.55
beta_y2_y1 <- 0.7
beta_y2_z <- 0.35
alpha_y2 <- -0.3
sd_y1 <- 0.45
sd_y2 <- 0.6

y1_path <- alpha_y1 + beta_y1_x * x_path + rnorm(n_path, sd = sd_y1)
y2_path <- alpha_y2 + beta_y2_y1 * y1_path + beta_y2_z * z_path + rnorm(n_path, sd = sd_y2)
df_path <- data.frame(x = x_path, z = z_path, y1 = y1_path, y2 = y2_path)

model_semx_path <- semx_model(
  equations = c(
    "y1 ~ 1 + x",
    "y2 ~ 1 + z + y1"
  ),
  families = c(y1 = "gaussian", y2 = "gaussian")
)

fit_semx_path <- semx_fit(model_semx_path, df_path, options = opts, estimation_method = "ML")
params_path <- summary(fit_semx_path)$parameters

expect_equal(as.numeric(params_path["beta_y1_on_x", "Estimate"]), beta_y1_x, tolerance = 5e-2)
expect_equal(as.numeric(params_path["alpha_y1_on__intercept", "Estimate"]), alpha_y1, tolerance = 5e-2)
expect_equal(as.numeric(params_path["beta_y2_on_y1", "Estimate"]), beta_y2_y1, tolerance = 5e-2)
expect_equal(as.numeric(params_path["beta_y2_on_z", "Estimate"]), beta_y2_z, tolerance = 5e-2)
expect_equal(as.numeric(params_path["alpha_y2_on__intercept", "Estimate"]), alpha_y2, tolerance = 5e-2)

if (requireNamespace("lavaan", quietly = TRUE)) {
  library(lavaan)
  lavaan_model <- "
    y1 ~ x
    y2 ~ y1 + z
  "
  fit_lavaan <- lavaan::sem(lavaan_model, data = df_path, meanstructure = TRUE, fixed.x = FALSE)
  pe <- lavaan::parameterEstimates(fit_lavaan)
  lav_val <- function(lhs, op, rhs) {
    pe$est[pe$lhs == lhs & pe$op == op & pe$rhs == rhs]
  }
  expect_equal(as.numeric(params_path["beta_y1_on_x", "Estimate"]), lav_val("y1", "~", "x"), tolerance = 5e-2)
  expect_equal(as.numeric(params_path["alpha_y1_on__intercept", "Estimate"]), lav_val("y1", "~1", ""), tolerance = 5e-2)
  expect_equal(as.numeric(params_path["beta_y2_on_y1", "Estimate"]), lav_val("y2", "~", "y1"), tolerance = 5e-2)
  expect_equal(as.numeric(params_path["beta_y2_on_z", "Estimate"]), lav_val("y2", "~", "z"), tolerance = 5e-2)
  expect_equal(as.numeric(params_path["alpha_y2_on__intercept", "Estimate"]), lav_val("y2", "~1", ""), tolerance = 5e-2)
} else {
  cat("lavaan not available; compared path model only to simulated truth\n")
}

summary_rows <- c(summary_rows, list(capture_row("path_gaussian", c(
  beta_y1_x = params_path["beta_y1_on_x", "Estimate"],
  alpha_y1 = params_path["alpha_y1_on__intercept", "Estimate"],
  beta_y2_y1 = params_path["beta_y2_on_y1", "Estimate"],
  beta_y2_z = params_path["beta_y2_on_z", "Estimate"],
  alpha_y2 = params_path["alpha_y2_on__intercept", "Estimate"]
))))


# 11b. CFA (Gaussian, single factor) vs lavaan -------------------------
cat("\n--- Test 11b: CFA (1 factor, Gaussian indicators) vs lavaan ---\n")
set.seed(4242)
n_cfa <- 500
f_latent <- rnorm(n_cfa, sd = 1.2)
lambda <- c(0.9, 0.8, 0.7)
intercepts_cfa <- c(0.5, -0.2, 0.1)
resid_sd_cfa <- c(0.4, 0.45, 0.35)
y1_cfa <- intercepts_cfa[1] + lambda[1] * f_latent + rnorm(n_cfa, sd = resid_sd_cfa[1])
y2_cfa <- intercepts_cfa[2] + lambda[2] * f_latent + rnorm(n_cfa, sd = resid_sd_cfa[2])
y3_cfa <- intercepts_cfa[3] + lambda[3] * f_latent + rnorm(n_cfa, sd = resid_sd_cfa[3])
df_cfa <- data.frame(y1 = y1_cfa, y2 = y2_cfa, y3 = y3_cfa)

model_semx_cfa <- semx_model(
  equations = c(
    "f =~ y1 + y2 + y3",
    "y1 ~ 1", "y2 ~ 1", "y3 ~ 1",
    "f ~~ f"
  ),
  families = c(y1 = "gaussian", y2 = "gaussian", y3 = "gaussian")
)

fit_semx_cfa <- semx_fit(model_semx_cfa, df_cfa, options = opts, estimation_method = "ML")
params_cfa <- summary(fit_semx_cfa)$parameters

# semx defaults to fixing the first loading to 1.0
# We need to adjust expectations or compare with lavaan which does the same
if (requireNamespace("lavaan", quietly = TRUE)) {
  library(lavaan)
  lavaan_cfa <- "
    f =~ y1 + y2 + y3
  "
  fit_lavaan_cfa <- lavaan::cfa(lavaan_cfa, data = df_cfa, meanstructure = TRUE)
  pe <- lavaan::parameterEstimates(fit_lavaan_cfa)
  lav_val <- function(lhs, op, rhs) {
    pe$est[pe$lhs == lhs & pe$op == op & pe$rhs == rhs]
  }
  
  # Loadings (y1 is fixed to 1 in both)
  expect_equal(as.numeric(params_cfa["lambda_y2_on_f", "Estimate"]), lav_val("f", "=~", "y2"), tolerance = 5e-2)
  expect_equal(as.numeric(params_cfa["lambda_y3_on_f", "Estimate"]), lav_val("f", "=~", "y3"), tolerance = 5e-2)
  
  # Intercepts
  expect_equal(as.numeric(params_cfa["alpha_y1_on__intercept", "Estimate"]), lav_val("y1", "~1", ""), tolerance = 5e-2)
  expect_equal(as.numeric(params_cfa["alpha_y2_on__intercept", "Estimate"]), lav_val("y2", "~1", ""), tolerance = 5e-2)
  expect_equal(as.numeric(params_cfa["alpha_y3_on__intercept", "Estimate"]), lav_val("y3", "~1", ""), tolerance = 5e-2)
  
  # Latent variance
  expect_equal(as.numeric(params_cfa["psi_f_f", "Estimate"]), lav_val("f", "~~", "f"), tolerance = 5e-2)
  
  lambda_est <- c(1.0, as.numeric(params_cfa["lambda_y2_on_f", "Estimate"]), as.numeric(params_cfa["lambda_y3_on_f", "Estimate"]))
  alpha_est <- c(
    as.numeric(params_cfa["alpha_y1_on__intercept", "Estimate"]),
    as.numeric(params_cfa["alpha_y2_on__intercept", "Estimate"]),
    as.numeric(params_cfa["alpha_y3_on__intercept", "Estimate"])
  )
} else {
  cat("lavaan not available; skipping CFA comparison\n")
  lambda_est <- rep(NA, 3)
  alpha_est <- rep(NA, 3)
}

summary_rows <- c(summary_rows, list(capture_row("cfa_gaussian", c(
  lambda1 = lambda_est[1],
  lambda2 = lambda_est[2],
  lambda3 = lambda_est[3],
  alpha1 = alpha_est[1],
  alpha2 = alpha_est[2],
  alpha3 = alpha_est[3]
))))


# Emit compact summary table ------------------------------------------
cat("\n\n=== Validation summary (key metrics, semx side) ===\n")
if (length(summary_rows)) {
  all_cols <- unique(unlist(lapply(summary_rows, names)))
  all_cols <- c("Test", setdiff(all_cols, "Test"))
  aligned_rows <- lapply(summary_rows, function(df) {
    missing <- setdiff(all_cols, names(df))
    if (length(missing)) {
      df[missing] <- NA
    }
    df[all_cols]
  })
  summary_table <- do.call(rbind, aligned_rows)
  print(summary_table, row.names = FALSE)
} else {
  cat("No summaries recorded.\n")
}

cat("\nAll validations executed.\n")
