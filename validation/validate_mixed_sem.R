# Validation script for Mixed Outcome SEM (Gaussian + Ordinal)
# This script tests if we can estimate a factor model with both continuous and ordinal indicators.

library(semx)
library(testthat)

cat("\n--- Test: Mixed Outcome SEM (Gaussian + Ordinal) ---\n")

# 1. Generate Data
set.seed(123)
N <- 2000
# True model:
# F ~ N(0, 1)
# y1 = 1.0*F + e1, e1 ~ N(0, 1)
# y2* = 0.8*F + e2, e2 ~ N(0, 1)
# y3 = 0.9*F + e3, e3 ~ N(0, 1)
# y2 = cut(y2*) at -0.5, 0.5

F_true <- rnorm(N, 0, 1)
y1 <- 1.0 * F_true + rnorm(N, 0, 1)
y3 <- 0.9 * F_true + rnorm(N, 0, 1)
y2_star <- 0.8 * F_true + rnorm(N, 0, 1)

# Cut y2 into 3 categories
y2 <- cut(y2_star, breaks = c(-Inf, -0.5, 0.5, Inf), labels = FALSE)
y2 <- as.ordered(y2)

data <- data.frame(y1 = y1, y2 = y2, y3 = y3)

# 2. Define Model
# We need 3 indicators for identification of a 1-factor model
# Free the first loading (NA*y1) because we fix Factor variance to 1
# Fix intercept of ordinal variable y2 to 0 for identification (since we estimate thresholds)
model_syntax <- "
  F =~ NA*y1 + y2 + y3
  y1 ~~ y1
  y3 ~~ y3
  y2 ~~ 1*y2  # Fix residual variance for ordinal
  F ~~ 1*F    # Fix factor variance for scale
  y2 ~ 0*1    # Fix intercept to 0
"

# 3. Fit Model
cat("Fitting model...\n")

# Define families explicitly
families <- c(
  y1 = "gaussian",
  y2 = "ordinal",
  y3 = "gaussian"
)

# Parse model
# Split syntax into lines and remove comments
equations <- strsplit(model_syntax, "\n")[[1]]
equations <- sub("#.*", "", equations) # Remove inline comments
equations <- trimws(equations)
equations <- equations[nzchar(equations)]

model <- semx_model(equations, families = families)
fit <- semx_fit(model, data)

# Inspect Results
print(fit)
cat("Full Parameter Table:\n")
print(fit$parameters)
summary(fit)

# Check estimates
summ <- summary(fit)
params <- summ$parameters
params$name <- rownames(params)
print(params)

# Extract specific parameters
lambda_y1 <- params$Estimate[params$name == "lambda_y1_on_F"]
lambda_y2 <- params$Estimate[params$name == "lambda_y2_on_F"]
lambda_y3 <- params$Estimate[params$name == "lambda_y3_on_F"]
tau1 <- params$Estimate[params$name == "y2_threshold_1"]
tau2 <- params$Estimate[params$name == "y2_threshold_2"]

cat("Lambda y1:", lambda_y1, "(Expected ~1.0)\n")
cat("Lambda y2:", lambda_y2, "(Expected ~0.8)\n")
cat("Lambda y3:", lambda_y3, "(Expected ~0.9)\n")
cat("Threshold 1:", tau1, "(Expected -0.5)\n")
cat("Threshold 2:", tau2, "(Expected 0.5)\n")

# Check if SEs are finite
if (any(is.na(params$Std.Error) | is.nan(params$Std.Error))) {
  warning("Some Standard Errors are NaN or NA!")
} else {
  cat("Standard Errors are finite.\n")
}
