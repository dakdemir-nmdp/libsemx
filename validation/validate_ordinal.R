
library(semx)
library(MASS)
library(testthat)

# ------------------------------------------------------------------------------
# Test 1: Simple Ordinal Regression (Proportional Odds)
# ------------------------------------------------------------------------------
cat("\n--- Test 1: Ordinal Regression (Proportional Odds) ---\n")
set.seed(123)
N <- 500
x <- rnorm(N)
# Latent variable y* = 0.5*x + e, e ~ Logistic(0, 1) for Logit link (or Normal for Probit)
# libsemx Ordinal uses Probit link by default? Or Logit?
# Usually SEM uses Probit for ordinal (polychoric).
# Let's check documentation or assume Probit first.
# If Probit: y* = 0.5*x + e, e ~ N(0, 1)
# Thresholds: tau1 = -0.5, tau2 = 0.5
# y = 1 if y* < -0.5
# y = 2 if -0.5 <= y* < 0.5
# y = 3 if y* >= 0.5

beta <- 0.5
y_star <- beta * x + rnorm(N) # Probit link (Normal error)
tau1 <- -0.5
tau2 <- 0.5

y <- cut(y_star, breaks = c(-Inf, tau1, tau2, Inf), labels = FALSE)
# y is 1, 2, 3
# semx expects ordered factor or numeric?
# Usually ordered factor.

data <- data.frame(y = ordered(y), x = x)

# Fit with MASS::polr (uses Logistic by default, can set method="probit")
fit_polr <- polr(y ~ x, data = data, method = "probit")
print(summary(fit_polr))

# Fit with semx
# Note: semx ordinal support might need specific syntax for thresholds.
# Or it handles it automatically for "ordinal" family.
model <- semx_model(
  equations = c("y ~ x"),
  families = c(y = "ordinal")
)

print("Model variables:")
print(names(model$variables))
print(model$variables$y)

print("Model IR variables:")
print(model$ir$variables)

fit <- semx_fit(model, data)

print(summary(fit))

# Compare coefficients
# polr parameterization:
# P(Y <= j) = Phi(tau_j - eta) = Phi(tau_j - beta*x)
# semx parameterization might be:
# y* = beta*x + e
# y = j if tau_{j-1} < y* < tau_j
# This implies P(Y <= j) = P(y* < tau_j) = P(beta*x + e < tau_j) = P(e < tau_j - beta*x) = Phi(tau_j - beta*x)
# So signs should match.

est_beta <- fit$optimization_result$parameters[["beta_y_on_x"]]
ref_beta <- fit_polr$coefficients["x"]

cat(sprintf("Beta: Est=%.4f, Ref=%.4f\n", est_beta, ref_beta))
expect_equal(as.numeric(est_beta), as.numeric(ref_beta), tolerance = 0.1)

# Compare thresholds
# semx thresholds might be named 'tau_y_1', 'tau_y_2' etc.
# polr thresholds are '1|2', '2|3'
est_params <- fit$optimization_result$parameters
print(names(est_params))

# Check if thresholds are recovered
