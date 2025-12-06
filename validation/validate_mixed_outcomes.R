library(semx)
library(lme4)
library(testthat)

# Helper to capture results
results <- list()

cat("\n--- Test 1: Joint Independent GLMMs (Gaussian + Binomial + Poisson) ---\n")
set.seed(2025)
n_groups <- 50
n_per_group <- 20
N <- n_groups * n_per_group
g <- factor(rep(1:n_groups, each = n_per_group))
x <- rnorm(N)

# 1. Gaussian Outcome
u1 <- rnorm(n_groups, sd = 1.0)
y1 <- 1 + 0.5 * x + u1[g] + rnorm(N, sd = 0.8)

# 2. Binomial Outcome
u2 <- rnorm(n_groups, sd = 0.8)
eta2 <- -0.5 + 0.8 * x + u2[g]
prob2 <- plogis(eta2)
y2 <- rbinom(N, 1, prob2)

# 3. Poisson Outcome
u3 <- rnorm(n_groups, sd = 0.5)
eta3 <- 1.5 + 0.3 * x + u3[g]
lambda3 <- exp(eta3)
y3 <- rpois(N, lambda3)

df_joint <- data.frame(y1 = y1, y2 = y2, y3 = y3, x = x, g = g)

# --- Fit Independent Models (lme4) ---
cat("Fitting lme4 models...\n")
fit_lmer_y1 <- lmer(y1 ~ x + (1 | g), data = df_joint, REML = FALSE)
fit_glmer_y2 <- glmer(y2 ~ x + (1 | g), data = df_joint, family = binomial)
fit_glmer_y3 <- glmer(y3 ~ x + (1 | g), data = df_joint, family = poisson)

ll_lme4_total <- as.numeric(logLik(fit_lmer_y1)) + as.numeric(logLik(fit_glmer_y2)) + as.numeric(logLik(fit_glmer_y3))
cat(sprintf("Total lme4 LogLik: %.4f\n", ll_lme4_total))

# --- Fit Independent Models (semx) ---
cat("Fitting semx individual models...\n")

# Gaussian
model_semx_y1 <- semx_model(equations = "y1 ~ 1 + x + (1 | g)", families = c(y1 = "gaussian"))
fit_semx_y1 <- semx_fit(model_semx_y1, df_joint, estimation_method = "ML")
cat(sprintf("SEMX y1 LogLik: %.4f\n", -fit_semx_y1$optimization_result$objective_value))

# Binomial
model_semx_y2 <- semx_model(equations = "y2 ~ 1 + x + (1 | g)", families = c(y2 = "binomial"))
fit_semx_y2 <- semx_fit(model_semx_y2, df_joint, estimation_method = "ML")
cat(sprintf("SEMX y2 LogLik: %.4f\n", -fit_semx_y2$optimization_result$objective_value))

# Poisson
model_semx_y3 <- semx_model(equations = "y3 ~ 1 + x + (1 | g)", families = c(y3 = "poisson"))
fit_semx_y3 <- semx_fit(model_semx_y3, df_joint, estimation_method = "ML")
cat(sprintf("SEMX y3 LogLik: %.4f\n", -fit_semx_y3$optimization_result$objective_value))

ll_semx_total_indep <- -fit_semx_y1$optimization_result$objective_value - fit_semx_y2$optimization_result$objective_value - fit_semx_y3$optimization_result$objective_value
cat(sprintf("Total SEMX Independent LogLik: %.4f\n", ll_semx_total_indep))

# --- Fit Joint Model (semx) ---
cat("Fitting semx joint model...\n")
# We specify 3 equations. By default, semx might correlate random effects on the same grouping factor.
# Let's see what happens.
model_semx_joint <- semx_model(
  equations = c(
    "y1 ~ 1 + x + (1 | g)",
    "y2 ~ 1 + x + (1 | g)",
    "y3 ~ 1 + x + (1 | g)"
  ),
  families = c(y1 = "gaussian", y2 = "binomial", y3 = "poisson")
)

fit_semx_joint <- semx_fit(model_semx_joint, df_joint, estimation_method = "ML")
ll_semx_joint <- -fit_semx_joint$optimization_result$objective_value
cat(sprintf("SEMX Joint LogLik: %.4f\n", ll_semx_joint))

# Compare LogLik
# Note: If semx fits a full covariance matrix for the random effects (3x3), it has more parameters than the 3 independent models (which assume zeros off-diagonal).
# So SEMX LL should be >= lme4 LL.
expect_true(ll_semx_joint >= ll_lme4_total - 1.0) 

# Compare Parameters
params <- summary(fit_semx_joint)$parameters
print(params)

# Gaussian Parameters
cat("\nChecking Gaussian Parameters...\n")
expect_equal(as.numeric(params["beta_y1_on_x", "Estimate"]), as.numeric(fixef(fit_lmer_y1)["x"]), tolerance = 0.05)
expect_equal(as.numeric(params["alpha_y1_on__intercept", "Estimate"]), as.numeric(fixef(fit_lmer_y1)["(Intercept)"]), tolerance = 0.05)

# Binomial Parameters
cat("\nChecking Binomial Parameters...\n")
expect_equal(as.numeric(params["beta_y2_on_x", "Estimate"]), as.numeric(fixef(fit_glmer_y2)["x"]), tolerance = 0.05)
expect_equal(as.numeric(params["alpha_y2_on__intercept", "Estimate"]), as.numeric(fixef(fit_glmer_y2)["(Intercept)"]), tolerance = 0.05)

# Poisson Parameters
cat("\nChecking Poisson Parameters...\n")
expect_equal(as.numeric(params["beta_y3_on_x", "Estimate"]), as.numeric(fixef(fit_glmer_y3)["x"]), tolerance = 0.05)
expect_equal(as.numeric(params["alpha_y3_on__intercept", "Estimate"]), as.numeric(fixef(fit_glmer_y3)["(Intercept)"]), tolerance = 0.05)

cat("\nJoint Independent GLMM Validation Passed!\n")
