# Validation of Mixed CFA (Continuous + Ordinal) against lavaan

library(semx)
library(lavaan)
library(testthat)

# 1. Simulate Data
# Model: f1 =~ y1 + y2 + u1 + u2
# y1, y2: Continuous (Gaussian)
# u1, u2: Ordinal (3 levels)
cat("--- Simulating Mixed Data ---\n")
set.seed(1234)
N <- 100
f1 <- rnorm(N)

# Loadings
lambda_y <- 1.0
lambda_u <- 1.0

# Continuous
y1 <- lambda_y * f1 + rnorm(N, 0, 1)
y2 <- lambda_y * f1 + rnorm(N, 0, 1)

# Ordinal Underlying
u1_star <- lambda_u * f1 + rnorm(N, 0, 1)
u2_star <- lambda_u * f1 + rnorm(N, 0, 1)

# Thresholds: -0.5, 0.5
cut_points <- c(-0.5, 0.5)
cut_ordinal <- function(x, cuts) {
  as.numeric(cut(x, c(-Inf, cuts, Inf))) - 1
}

u1 <- cut_ordinal(u1_star, cut_points)
u2 <- cut_ordinal(u2_star, cut_points)

df <- data.frame(
  y1 = y1,
  y2 = y2,
  u1 = ordered(u1),
  u2 = ordered(u2)
)

cat("Data summary:\n")
print(summary(df))

# 2. Fit with lavaan
cat("\n--- Fitting with lavaan ---\n")
lav_model <- '
  f1 =~ y1 + y2 + u1 + u2
'
# std.lv=TRUE fixes factor variance to 1.0
# ordered specifies ordinal variables
fit_lav <- cfa(lav_model, data=df, ordered=c("u1", "u2"), std.lv=TRUE)
print(summary(fit_lav))

pe <- parameterEstimates(fit_lav)
lav_loadings <- pe[pe$op == "=~", c("lhs", "rhs", "est")]
lav_thresholds <- pe[pe$op == "|", c("lhs", "rhs", "est")]

print("Lavaan Loadings:")
print(lav_loadings)

# 3. Fit with semx
cat("\n--- Fitting with semx ---\n")
semx_mod <- semx_model(
  equations = c(
    "f1 =~ NA*y1 + y2 + u1 + u2",
    "f1 ~~ 1*f1"
  ),
  families = c(
    y1="gaussian", 
    y2="gaussian", 
    u1="ordinal", 
    u2="ordinal"
  )
)

# Initialize parameters to help convergence
if (is.null(semx_mod$parameters)) semx_mod$parameters <- list()
semx_mod$parameters[["lambda_y1_on_f1"]] <- 1.0
semx_mod$parameters[["lambda_y2_on_f1"]] <- 1.0
semx_mod$parameters[["lambda_u1_on_f1"]] <- 1.0
semx_mod$parameters[["lambda_u2_on_f1"]] <- 1.0

opts <- new(OptimizationOptions)
opts$max_iterations <- 5000
opts$tolerance <- 1e-5

fit_semx <- semx_fit(semx_mod, df, options = opts)
print(summary(fit_semx))

# 4. Compare
cat("\n--- Comparison ---\n")

semx_params <- fit_semx$optimization_result$parameters
get_semx_param <- function(name) {
  if (name %in% names(semx_params)) return(semx_params[[name]])
  return(NA)
}

# Compare Loadings
for (var in c("y1", "y2", "u1", "u2")) {
  lav_est <- lav_loadings[lav_loadings$rhs == var, "est"]
  semx_name <- paste0("lambda_", var, "_on_f1")
  semx_est <- get_semx_param(semx_name)
  
  cat(sprintf("Loading %s: Lavaan=%.3f, Semx=%.3f, Diff=%.3f\n", 
              var, lav_est, semx_est, abs(lav_est - semx_est)))
  
  # Relaxed tolerance for mixed models with small N
  expect_true(abs(lav_est - semx_est) < 0.5)
}

# Compare Thresholds
for (var in c("u1", "u2")) {
  # Threshold 1
  lav_t1 <- lav_thresholds[lav_thresholds$lhs == var & lav_thresholds$rhs == "t1", "est"]
  semx_t1 <- get_semx_param(paste0(var, "_threshold_1"))
  
  cat(sprintf("Threshold %s|t1: Lavaan=%.3f, Semx=%.3f\n", var, lav_t1, semx_t1))
  expect_true(abs(lav_t1 - semx_t1) < 0.5)
  
  # Threshold 2
  lav_t2 <- lav_thresholds[lav_thresholds$lhs == var & lav_thresholds$rhs == "t2", "est"]
  semx_t2 <- get_semx_param(paste0(var, "_threshold_2"))
  
  cat(sprintf("Threshold %s|t2: Lavaan=%.3f, Semx=%.3f\n", var, lav_t2, semx_t2))
  expect_true(abs(lav_t2 - semx_t2) < 0.5)
}

cat("\nValidation Complete!\n")
