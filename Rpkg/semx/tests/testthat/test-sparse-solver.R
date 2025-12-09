test_that("Sparse solver works for large number of groups", {
  set.seed(123)
  n_groups <- 10
  n_per_group <- 5
  
  # Simulate random intercept model
  # y_ij = mu + u_i + e_ij
  mu <- 2.0
  sigma_u <- 1.5
  sigma_e <- 1.0
  
  groups <- rep(1:n_groups, each = n_per_group)
  u <- rnorm(n_groups, 0, sigma_u)
  y <- mu + u[groups] + rnorm(n_groups * n_per_group, 0, sigma_e)
  
  data <- data.frame(y = y, group = factor(groups))
  
  # Fit model
  model <- semx_model("y ~ 1 + (1 | group)", families = c(y = "gaussian"))
  fit <- semx_fit(model, data, options = list(force_laplace = TRUE))
  
  # Check convergence
  expect_true(fit$optimization_result$converged)
  
  # Check parameters
  coefs <- semx:::coef.semx_fit(fit)
  print(coefs)
  
  # Intercept (alpha_y_on__intercept)
  intercept_param <- coefs["alpha_y_on__intercept"]
  if (is.na(intercept_param)) {
      intercept_param <- coefs["y_on__intercept"]
  }
  expect_equal(as.numeric(intercept_param), mu, tolerance = 0.1)
  
  # Random effect variance
  vc <- semx_variance_components(fit)
  print(vc)
  # Column names are Group, Name1, Name2, Variance, Std.Dev, Corr
  est_u <- vc$Variance[vc$Group == "group" & vc$Name1 == "(Intercept)"]
  expect_equal(est_u, sigma_u^2, tolerance = 0.5)
  
  # Residual variance (psi_y_y)
  resid_param <- coefs["psi_y_y"]
  expect_equal(as.numeric(resid_param), sigma_e^2, tolerance = 0.5)
})
