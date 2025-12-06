library(semx)
library(lme4)
library(MASS) # For glm.nb
library(testthat)

set.seed(123)
n <- 500
x <- rnorm(n)
g <- sample(1:10, n, replace = TRUE)
df <- data.frame(x = x, g = factor(g))

# --- 1. Binomial (Logistic) ---
cat("\n--- Binomial GLM ---\n")
eta_bin <- 0.5 + 0.8 * x
prob_bin <- 1 / (1 + exp(-eta_bin))
df$y_bin <- rbinom(n, 1, prob_bin)

# GLM
fit_glm_bin <- glm(y_bin ~ x, data = df, family = binomial)
ll_glm_bin <- as.numeric(logLik(fit_glm_bin))
coef_glm_bin <- coef(fit_glm_bin)

# SEMX
model_bin <- semx_model(
  equations = "y_bin ~ 1 + x",
  families = c(y_bin = "binomial")
)
fit_semx_bin <- semx_fit(model_bin, df)
ll_semx_bin <- -fit_semx_bin$optimization_result$objective_value
coef_semx_bin <- fit_semx_bin$parameters

cat(sprintf("LogLik: GLM = %.4f, SEMX = %.4f\n", ll_glm_bin, ll_semx_bin))
print(coef_glm_bin)
print(coef_semx_bin)

expect_equal(ll_semx_bin, ll_glm_bin, tolerance = 1e-3)
# Note: Parameter names might differ slightly (e.g. alpha_y_bin_on__intercept vs (Intercept))

# --- 2. Poisson (Log) ---
cat("\n--- Poisson GLM ---\n")
eta_pois <- 0.2 + 0.5 * x
lambda_pois <- exp(eta_pois)
df$y_pois <- rpois(n, lambda_pois)

# GLM
fit_glm_pois <- glm(y_pois ~ x, data = df, family = poisson)
ll_glm_pois <- as.numeric(logLik(fit_glm_pois))
coef_glm_pois <- coef(fit_glm_pois)

# SEMX
model_pois <- semx_model(
  equations = "y_pois ~ 1 + x",
  families = c(y_pois = "poisson")
)
fit_semx_pois <- semx_fit(model_pois, df)
ll_semx_pois <- -fit_semx_pois$optimization_result$objective_value
coef_semx_pois <- fit_semx_pois$parameters

cat(sprintf("LogLik: GLM = %.4f, SEMX = %.4f\n", ll_glm_pois, ll_semx_pois))
print(coef_glm_pois)
print(coef_semx_pois)

expect_equal(ll_semx_pois, ll_glm_pois, tolerance = 1e-3)

# --- 3. Negative Binomial ---
cat("\n--- Negative Binomial GLM ---\n")
# theta = 2.0
eta_nb <- 0.5 + 0.5 * x
mu_nb <- exp(eta_nb)
theta_true <- 2.0
df$y_nb <- rnegbin(n, mu = mu_nb, theta = theta_true)

# GLM.NB
fit_glm_nb <- glm.nb(y_nb ~ x, data = df)
ll_glm_nb <- as.numeric(logLik(fit_glm_nb))
coef_glm_nb <- coef(fit_glm_nb)
theta_glm_nb <- fit_glm_nb$theta

# SEMX
# Note: semx parameterizes dispersion for NB. Need to check if it's theta or 1/theta.
# Usually dispersion = 1/theta (overdispersion parameter).
model_nb <- semx_model(
  equations = "y_nb ~ 1 + x",
  families = c(y_nb = "negative_binomial")
)
fit_semx_nb <- semx_fit(model_nb, df)
ll_semx_nb <- -fit_semx_nb$optimization_result$objective_value
coef_semx_nb <- fit_semx_nb$parameters

cat(sprintf("LogLik: GLM.NB = %.4f, SEMX = %.4f\n", ll_glm_nb, ll_semx_nb))
print(coef_glm_nb)
cat("Theta GLM:", theta_glm_nb, "\n")
print(coef_semx_nb)

expect_equal(ll_semx_nb, ll_glm_nb, tolerance = 1e-3)

# --- 4. Binomial GLMM (Random Intercept) ---
cat("\n--- Binomial GLMM ---\n")
# Random intercept per group
re_sd <- 0.5
re_vals <- rnorm(10, 0, re_sd)
df$eta_bin_re <- 0.5 + 0.8 * x + re_vals[df$g]
df$prob_bin_re <- 1 / (1 + exp(-df$eta_bin_re))
df$y_bin_re <- rbinom(n, 1, df$prob_bin_re)

# glmer
fit_glmer_bin <- glmer(y_bin_re ~ x + (1 | g), data = df, family = binomial)
ll_glmer_bin <- as.numeric(logLik(fit_glmer_bin))

# SEMX
covs <- list(list(name = "G_bin", structure = "diagonal", dimension = 1))
res <- list(list(name = "re_bin", variables = "g", covariance = "G_bin"))
model_bin_re <- semx_model(
  equations = "y_bin_re ~ 1 + x",
  families = c(y_bin_re = "binomial"),
  covariances = covs,
  random_effects = res
)
fit_semx_bin_re <- semx_fit(model_bin_re, df)
ll_semx_bin_re <- -fit_semx_bin_re$optimization_result$objective_value

cat(sprintf("LogLik: glmer = %.4f, SEMX = %.4f\n", ll_glmer_bin, ll_semx_bin_re))
expect_equal(ll_semx_bin_re, ll_glmer_bin, tolerance = 1e-2)

