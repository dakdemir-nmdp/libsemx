
# validation/generate_survival_baselines.R
options(echo=TRUE)

library(survival)

# Load data
data(ovarian)
# ovarian has columns: futime, fustat, age, resid.ds, rx, ecog.ps

# Save to CSV for C++ (if not already there, but we have it)
# write.csv(ovarian, "data/ovarian_survival.csv", row.names = FALSE)

# Fit Weibull model
# survreg uses log(T) = mu + sigma * W
# libsemx uses T ~ Weibull(k, lambda) where lambda = exp(eta)
# eta = mu (linear predictor)
# k = 1/sigma (shape)

# Model: futime ~ age + rx
fit_weibull <- survreg(Surv(futime, fustat) ~ age + rx, data = ovarian, dist = "weibull")

summary(fit_weibull)

cat("\n--- WEIBULL COMPARISON VALUES ---\n")
cat("LogLik:", fit_weibull$loglik[2], "\n")
cat("Scale (sigma):", fit_weibull$scale, "\n")
cat("Shape (k = 1/sigma):", 1/fit_weibull$scale, "\n")
cat("Coefficients:\n")
print(coef(fit_weibull))

# Fit Exponential model (Weibull with scale=1 fixed, so shape=1)
# survreg dist="exponential" fixes scale=1
fit_exp <- survreg(Surv(futime, fustat) ~ age + rx, data = ovarian, dist = "exponential")

summary(fit_exp)

cat("\n--- EXPONENTIAL COMPARISON VALUES ---\n")
cat("LogLik:", fit_exp$loglik[2], "\n")
cat("Scale (sigma):", fit_exp$scale, "\n") # Should be 1
cat("Coefficients:\n")
print(coef(fit_exp))
