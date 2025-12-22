library(semx)
library(lme4)
library(testthat)

# Stabilized LBFGS options to reduce line-search warnings and improve Hessians
lbfgs_opts <- list(
  max_iterations = 1200,
  tolerance = 1e-5,
  learning_rate = 0.05,
  max_linesearch = 60,
  linesearch_type = "wolfe",
  past = 10,
  delta = 1e-5,
  m = 10,
  force_laplace = TRUE
)

# Helper to capture results
results <- list()

cat("\n--- Test: Mixed Path Analysis (Mediation) with Random Effects ---\n")
cat("Model: X -> M (Gaussian) -> Y (Binomial)\n")
cat("       M ~ X + (1|g)\n")
cat("       Y ~ M + X + (1|g)\n")

set.seed(2025)
n_groups <- 50
n_per_group <- 20
N <- n_groups * n_per_group
g <- factor(rep(1:n_groups, each = n_per_group))
x <- rnorm(N)

# 1. Mediator (Gaussian)
# M = 0.5*X + u_M[g] + e_M
u_M <- rnorm(n_groups, sd = 0.8)
M <- 0.5 * x + u_M[g] + rnorm(N, sd = 1.0)

# 2. Outcome (Binomial)
# Y ~ Binomial(logit(eta))
# eta = -1 + 0.8*M + 0.3*X + u_Y[g]
u_Y <- rnorm(n_groups, sd = 0.6)
eta_Y <- -1.0 + 0.8 * M + 0.3 * x + u_Y[g]
prob_Y <- plogis(eta_Y)
Y <- rbinom(N, 1, prob_Y)

df_med <- data.frame(x = x, M = M, Y = Y, g = g)

# --- Fit Stepwise Models (lme4) ---
cat("Fitting lme4 models...\n")
fit_lmer_M <- lmer(M ~ x + (1 | g), data = df_med, REML = FALSE)
fit_glmer_Y <- glmer(Y ~ M + x + (1 | g), data = df_med, family = binomial)

ll_lme4_M <- as.numeric(logLik(fit_lmer_M))
ll_lme4_Y <- as.numeric(logLik(fit_glmer_Y))
ll_lme4_total <- ll_lme4_M + ll_lme4_Y

cat(sprintf("lme4 M LogLik: %.4f\n", ll_lme4_M))
cat(sprintf("lme4 Y LogLik: %.4f\n", ll_lme4_Y))
cat(sprintf("Total lme4 LogLik: %.4f\n", ll_lme4_total))

# --- Fit Joint Model (semx) ---
cat("Fitting semx joint model...\n")

model_semx <- semx_model(
  equations = c(
    "M ~ 1 + x + (1 | g)",
    "Y ~ 1 + M + x + (1 | g)"
  ),
  families = c(M = "gaussian", Y = "binomial")
)

# Note: We expect semx to handle the dependency Y ~ M correctly.
fit_semx <- semx_fit(model_semx, df_med, estimation_method = "ML", options = lbfgs_opts)
ll_semx <- -fit_semx$optimization_result$objective_value

cat(sprintf("SEMX Joint LogLik: %.4f\n", ll_semx))

# Compare LogLik
# Should be very close as the joint likelihood factorizes
expect_equal(ll_semx, ll_lme4_total, tolerance = 0.1)

# Compare Parameters
params <- summary(fit_semx)$parameters
print(params)

# M Parameters
cat("\nChecking M Parameters...\n")
expect_equal(as.numeric(params["beta_M_on_x", "Estimate"]), as.numeric(fixef(fit_lmer_M)["x"]), tolerance = 0.05)
expect_equal(as.numeric(params["alpha_M_on__intercept", "Estimate"]), as.numeric(fixef(fit_lmer_M)["(Intercept)"]), tolerance = 0.05)

# Y Parameters
cat("\nChecking Y Parameters...\n")
expect_equal(as.numeric(params["beta_Y_on_x", "Estimate"]), as.numeric(fixef(fit_glmer_Y)["x"]), tolerance = 0.05)
expect_equal(as.numeric(params["beta_Y_on_M", "Estimate"]), as.numeric(fixef(fit_glmer_Y)["M"]), tolerance = 0.05)
expect_equal(as.numeric(params["alpha_Y_on__intercept", "Estimate"]), as.numeric(fixef(fit_glmer_Y)["(Intercept)"]), tolerance = 0.05)

cat("\nMixed Path Analysis Validation Passed!\n")
