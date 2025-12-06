library(semx)
library(lme4)
library(nlme)
library(testthat)

# Helper to capture results
results <- list()

# 1. Fixed-only Model (Gaussian)
# y ~ x
# REML should adjust for p (number of fixed effects) in variance estimation.
cat("\n--- Test 1: Fixed-only Model (Gaussian) ---\n")
set.seed(123)
n <- 100
x <- rnorm(n)
beta0 <- 1
beta1 <- 2
sigma <- 1.5
y <- beta0 + beta1 * x + rnorm(n, sd = sigma)
df_fixed <- data.frame(y = y, x = x)

# ML
fit_lm_ml <- gls(y ~ x, data = df_fixed, method = "ML")
model_semx_fixed <- semx_model(equations = "y ~ 1 + x", families = c(y = "gaussian"))
fit_semx_fixed_ml <- semx_fit(model_semx_fixed, df_fixed, estimation_method = "ML")

# REML
fit_lm_reml <- gls(y ~ x, data = df_fixed, method = "REML")
fit_semx_fixed_reml <- semx_fit(model_semx_fixed, df_fixed, estimation_method = "REML")

# Compare ML
cat("ML Comparison:\n")
print(summary(fit_semx_fixed_ml)$parameters)
print(coef(fit_lm_ml))
expect_equal(as.numeric(summary(fit_semx_fixed_ml)$parameters["beta_y_on_x", "Estimate"]), as.numeric(coef(fit_lm_ml)["x"]), tolerance = 1e-4)
expect_equal(-fit_semx_fixed_ml$optimization_result$objective_value, as.numeric(logLik(fit_lm_ml)), tolerance = 1e-4)

# Compare REML
cat("REML Comparison:\n")
print(summary(fit_semx_fixed_reml)$parameters)
print(coef(fit_lm_reml))
expect_equal(as.numeric(summary(fit_semx_fixed_reml)$parameters["beta_y_on_x", "Estimate"]), as.numeric(coef(fit_lm_reml)["x"]), tolerance = 1e-4)
expect_equal(-fit_semx_fixed_reml$optimization_result$objective_value, as.numeric(logLik(fit_lm_reml)), tolerance = 1e-4)

# Check Variance Estimates (Sigma^2)
# ML: RSS / n
# REML: RSS / (n - p)
sigma2_ml_semx <- summary(fit_semx_fixed_ml)$parameters["psi_y_y", "Estimate"]
sigma2_reml_semx <- summary(fit_semx_fixed_reml)$parameters["psi_y_y", "Estimate"]

sigma2_ml_gls <- (fit_lm_ml$sigma)^2 * ((n-1)/n) # gls reports REML-like sigma by default? No, check docs.
# Actually gls sigma is sqrt(RSS/(N-p)) for REML and sqrt(RSS/N) for ML?
# Let's check what gls returns.
sigma_ml_gls <- fit_lm_ml$sigma
sigma_reml_gls <- fit_lm_reml$sigma

cat("Sigma ML: SEMX =", sqrt(sigma2_ml_semx), " GLS =", sigma_ml_gls, "\n")
cat("Sigma REML: SEMX =", sqrt(sigma2_reml_semx), " GLS =", sigma_reml_gls, "\n")

# 2. Random-only Model (Gaussian)
# y ~ 1 + (1 | g)
cat("\n--- Test 2: Random-only Model (Gaussian) ---\n")
set.seed(456)
n_groups <- 20
n_per_group <- 5
g <- factor(rep(1:n_groups, each = n_per_group))
u <- rnorm(n_groups, sd = 2.0)
y_ro <- 10 + u[g] + rnorm(length(g), sd = 1.0)
df_ro <- data.frame(y = y_ro, g = g)

model_semx_ro <- semx_model(equations = "y ~ 1 + (1 | g)", families = c(y = "gaussian"))

# ML
fit_lmer_ro_ml <- lmer(y ~ 1 + (1 | g), data = df_ro, REML = FALSE)
fit_semx_ro_ml <- semx_fit(model_semx_ro, df_ro, estimation_method = "ML")

# REML
fit_lmer_ro_reml <- lmer(y ~ 1 + (1 | g), data = df_ro, REML = TRUE)
fit_semx_ro_reml <- semx_fit(model_semx_ro, df_ro, estimation_method = "REML")

# Compare ML
cat("ML Comparison (Random-Only):\n")
expect_equal(-fit_semx_ro_ml$optimization_result$objective_value, as.numeric(logLik(fit_lmer_ro_ml)), tolerance = 1e-3)

# Compare REML
cat("REML Comparison (Random-Only):\n")
expect_equal(-fit_semx_ro_reml$optimization_result$objective_value, as.numeric(logLik(fit_lmer_ro_reml)), tolerance = 1e-3)


# 3. Mixed Model (Gaussian)
# y ~ x + (1 | g)
cat("\n--- Test 3: Mixed Model (Gaussian) ---\n")
set.seed(789)
x_mix <- rnorm(length(g))
y_mix <- 10 + 2 * x_mix + u[g] + rnorm(length(g), sd = 1.0)
df_mix <- data.frame(y = y_mix, x = x_mix, g = g)

model_semx_mix <- semx_model(equations = "y ~ 1 + x + (1 | g)", families = c(y = "gaussian"))

# ML
fit_lmer_mix_ml <- lmer(y ~ x + (1 | g), data = df_mix, REML = FALSE)
fit_semx_mix_ml <- semx_fit(model_semx_mix, df_mix, estimation_method = "ML")

# REML
fit_lmer_mix_reml <- lmer(y ~ x + (1 | g), data = df_mix, REML = TRUE)
fit_semx_mix_reml <- semx_fit(model_semx_mix, df_mix, estimation_method = "REML")

# Compare ML
cat("ML Comparison (Mixed):\n")
expect_equal(-fit_semx_mix_ml$optimization_result$objective_value, as.numeric(logLik(fit_lmer_mix_ml)), tolerance = 1e-3)

# Compare REML
cat("REML Comparison (Mixed):\n")
expect_equal(-fit_semx_mix_reml$optimization_result$objective_value, as.numeric(logLik(fit_lmer_mix_reml)), tolerance = 1e-3)

cat("\nAll REML/ML Parity Tests Passed!\n")
