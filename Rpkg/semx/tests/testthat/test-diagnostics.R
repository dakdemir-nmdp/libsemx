test_that("semx diagnostics work", {
  library(semx)
  set.seed(42)
  n <- 100
  x <- rnorm(n)
  y <- 0.5 * x + rnorm(n, 0, 0.8)
  data <- data.frame(x = x, y = y)
  
  model <- semx_model(
    equations = c(
      "y ~~ y",
      "x ~~ x"
    ),
    families = c(y = "gaussian", x = "gaussian")
  )
  
  fit <- semx_fit(model, data)
  
  # Standardized Solution
  std <- semx_standardized_solution(fit, model, data)
  expect_true("edges" %in% names(std))
  expect_true("std_all" %in% names(std$edges))
  
  # Diagnostics
  diag <- semx_diagnostics(fit, model, data)
  expect_true("srmr" %in% names(diag))
  # Model is misspecified (missing y ~ x), so SRMR should be high
  expect_true(diag$srmr > 0.01) 
  
  # Modification Indices
  mi <- semx_modification_indices(fit, model, data)
  expect_true("mi" %in% names(mi))
  expect_true(nrow(mi) > 0)
  # Should suggest y ~ x or y ~~ x
  expect_true(any(mi$source == "x" & mi$target == "y") || any(mi$source == "y" & mi$target == "x"))
})
