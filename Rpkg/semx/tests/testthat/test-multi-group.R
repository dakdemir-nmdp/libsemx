library(semx)
library(testthat)

test_that("fit_multi_group works with pooled mean", {
  # Group 1
  builder1 <- new(ModelIRBuilder)
  builder1$add_variable("y", 0, "gaussian", "", "")
  builder1$add_variable("one", 3, "gaussian", "", "") # Intercept variable (Exogenous)
  builder1$register_parameter("mu", 0.0)
  # builder1$register_parameter("sigma2", 1.0) # Let add_edge register it as Positive
  builder1$add_edge(1, "one", "y", "mu") # Intercept (Regression)
  builder1$add_edge(2, "y", "y", "sigma2") # Variance
  model1 <- builder1$build()

  set.seed(123)
  data1 <- list(y = rnorm(100, mean = 5, sd = 1), one = rep(1, 100))

  # Group 2
  builder2 <- new(ModelIRBuilder)
  builder2$add_variable("y", 0, "gaussian", "", "")
  builder2$add_variable("one", 3, "gaussian", "", "") # Intercept variable (Exogenous)
  builder2$register_parameter("mu", 0.0) # Same parameter ID "mu"
  # builder2$register_parameter("sigma2", 1.0) # Same parameter ID "sigma2"
  builder2$add_edge(1, "one", "y", "mu")
  builder2$add_edge(2, "y", "y", "sigma2")
  model2 <- builder2$build()

  data2 <- list(y = rnorm(100, mean = 5, sd = 1), one = rep(1, 100))

  driver <- new(LikelihoodDriver)
  options <- new(OptimizationOptions)
  options$max_iterations <- 100
  options$tolerance <- 1e-5

  # Fit multi-group
  result <- driver$fit_multi_group(
    list(model1, model2),
    list(data1, data2),
    options,
    "lbfgs",
    NULL, # fixed_covariance_data_list
    NULL, # status_list
    NULL, # extra_param_mappings_list
    0 # GaussianML
  )

  expect_true(result$optimization_result$converged)
  
  names <- result$parameter_names
  values <- result$optimization_result$parameters
  
  mu_idx <- which(names == "mu")
  sigma2_idx <- which(names == "sigma2")
  
  expect_true(length(mu_idx) > 0)
  expect_true(length(sigma2_idx) > 0)
  
  expect_equal(values[mu_idx], 5.0, tolerance = 0.2)
  expect_equal(values[sigma2_idx], 1.0, tolerance = 0.2)
})

test_that("fit_multi_group works with group-specific means", {
  # Group 1
  builder1 <- new(ModelIRBuilder)
  builder1$add_variable("y", 0, "gaussian", "", "")
  builder1$add_variable("one", 3, "gaussian", "", "")
  builder1$register_parameter("mu1", 0.0)
  # builder1$register_parameter("sigma2", 1.0)
  builder1$add_edge(1, "one", "y", "mu1") # Intercept (Regression)
  builder1$add_edge(2, "y", "y", "sigma2") # Variance
  model1 <- builder1$build()

  set.seed(123)
  data1 <- list(y = rnorm(100, mean = 2, sd = 1), one = rep(1, 100))

  # Group 2
  builder2 <- new(ModelIRBuilder)
  builder2$add_variable("y", 0, "gaussian", "", "")
  builder2$add_variable("one", 3, "gaussian", "", "")
  builder2$register_parameter("mu2", 0.0) # Different parameter ID "mu2"
  # builder2$register_parameter("sigma2", 1.0)
  builder2$add_edge(1, "one", "y", "mu2")
  builder2$add_edge(2, "y", "y", "sigma2")
  model2 <- builder2$build()

  data2 <- list(y = rnorm(100, mean = 8, sd = 1), one = rep(1, 100))

  driver <- new(LikelihoodDriver)
  options <- new(OptimizationOptions)
  options$max_iterations <- 100
  options$tolerance <- 1e-5

  # Fit multi-group
  result <- driver$fit_multi_group(
    list(model1, model2),
    list(data1, data2),
    options,
    "lbfgs",
    NULL, # fixed_covariance_data_list
    NULL, # status_list
    NULL, # extra_param_mappings_list
    0 # GaussianML
  )

  expect_true(result$optimization_result$converged)
  
  names <- result$parameter_names
  values <- result$optimization_result$parameters
  
  mu1_idx <- which(names == "mu1")
  mu2_idx <- which(names == "mu2")
  sigma2_idx <- which(names == "sigma2")
  
  expect_true(length(mu1_idx) > 0)
  expect_true(length(mu2_idx) > 0)
  expect_true(length(sigma2_idx) > 0)
  
  expect_equal(values[mu1_idx], 2.0, tolerance = 0.2)
  expect_equal(values[mu2_idx], 8.0, tolerance = 0.2)
  expect_equal(values[sigma2_idx], 1.0, tolerance = 0.2)
})

test_that("semx_fit supports multi-group analysis", {
  set.seed(123)
  # Group 1: y = 1*x + 0 + e
  # Group 2: y = 2*x + 0 + e
  
  n <- 100
  x1 <- rnorm(n)
  y1 <- 1.0 * x1 + rnorm(n)
  g1 <- rep("A", n)
  
  x2 <- rnorm(n)
  y2 <- 2.0 * x2 + rnorm(n)
  g2 <- rep("B", n)
  
  df <- data.frame(
    x = c(x1, x2),
    y = c(y1, y2),
    group = c(g1, g2)
  )
  
  model <- semx_model(
    equations = "y ~ x",
    families = c(y = "gaussian")
  )
  
  # Fit without constraints
  fit <- semx_fit(model, df, group = "group")
  
  expect_true(fit$optimization_result$converged)
  
  # Check parameters
  params <- setNames(fit$fit_result$optimization_result$parameters, fit$fit_result$parameter_names)
  
  # Parameter IDs: beta_y_on_x_A, beta_y_on_x_B
  expect_true("beta_y_on_x_A" %in% names(params))
  expect_true("beta_y_on_x_B" %in% names(params))
  
  expect_equal(params[["beta_y_on_x_A"]], 1.0, tolerance = 0.2)
  expect_equal(params[["beta_y_on_x_B"]], 2.0, tolerance = 0.2)
  
  # Fit WITH constraints
  fit_eq <- semx_fit(model, df, group = "group", group.equal = "regressions")
  
  expect_true(fit_eq$optimization_result$converged)
  params_eq <- setNames(fit_eq$fit_result$optimization_result$parameters, fit_eq$fit_result$parameter_names)
  
  expect_true("beta_y_on_x" %in% names(params_eq))
  expect_false("beta_y_on_x_A" %in% names(params_eq))
  
  expect_equal(params_eq[["beta_y_on_x"]], 1.5, tolerance = 0.2)
})

test_that("semx_fit supports multi-group intercepts", {
  set.seed(123)
  n <- 100
  y1 <- rnorm(n, mean = 0)
  y2 <- rnorm(n, mean = 5)
  df <- data.frame(
    y = c(y1, y2),
    group = rep(c("A", "B"), each = n)
  )
  
  model <- semx_model(
    equations = "y ~ 1",
    families = c(y = "gaussian")
  )
  
  # Free intercepts
  fit <- semx_fit(model, df, group = "group")
  params <- setNames(fit$fit_result$optimization_result$parameters, fit$fit_result$parameter_names)
  
  # Intercept ID: alpha_y_on__intercept
  expect_true("alpha_y_on__intercept_A" %in% names(params))
  expect_equal(params[["alpha_y_on__intercept_A"]], 0.0, tolerance = 0.2)
  expect_equal(params[["alpha_y_on__intercept_B"]], 5.0, tolerance = 0.2)
  
  # Constrained intercepts AND variances to ensure pooled mean is recovered
  fit_eq <- semx_fit(model, df, group = "group", group.equal = c("intercepts", "covariances"))
  params_eq <- setNames(fit_eq$fit_result$optimization_result$parameters, fit_eq$fit_result$parameter_names)
  
  expect_true("alpha_y_on__intercept" %in% names(params_eq))
  expect_true("psi_y_y" %in% names(params_eq))
  expect_equal(params_eq[["alpha_y_on__intercept"]], 2.5, tolerance = 0.2)
})
