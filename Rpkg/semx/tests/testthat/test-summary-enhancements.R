test_that("summary includes variance components and covariance weights", {
  # Create a mock fit object
  fit <- list(
    optimization_result = list(
      parameters = c(
        "cov_re_1_0" = 4.0, # Variance
        "cov_re_1_1" = 2.0, # Covariance
        "cov_re_1_2" = 2.0, # Covariance (duplicate for mock)
        "cov_re_1_3" = 9.0, # Variance
        "mk_cov_0" = 10.0, # Sigma^2
        "mk_cov_1" = 1.0,  # Theta 1
        "mk_cov_2" = 2.0   # Theta 2
      ),
      objective_value = 100.0,
      gradient_norm = 0.001,
      iterations = 10,
      converged = TRUE
    ),
    standard_errors = c(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
    parameter_names = c("cov_re_1_0", "cov_re_1_1", "cov_re_1_2", "cov_re_1_3", "mk_cov_0", "mk_cov_1", "mk_cov_2"),
    covariance_matrices = list(
      "cov_re_1" = c(4.0, 2.0, 2.0, 9.0) # 2x2 matrix: [4, 2; 2, 9]
    ),
    aic = 200, bic = 210, chi_square = 5, df = 2, p_value = 0.08,
    cfi = 0.95, tli = 0.94, rmsea = 0.05, srmr = 0.04,
    model = list(
      covariances = list(
        list(name = "cov_re_1", structure = "unstructured", dimension = 2),
        list(name = "mk_cov", structure = "multi_kernel_simplex", dimension = 100)
      ),
      random_effects = list(
        list(name = "re_1", variables = c("group", "x1", "x2"), covariance = "cov_re_1")
      ),
      ir = list(parameter_ids = function() c("cov_re_1_0", "cov_re_1_1", "cov_re_1_2", "cov_re_1_3", "mk_cov_0", "mk_cov_1", "mk_cov_2", "beta"))
    ),
    data = list(y = rnorm(100))
  )
  class(fit) <- "semx_fit"
  
  # Mock predict
  mock_predict <- function(object, newdata = NULL, ...) {
      list(y = rep(0, 100))
  }
  # We can't easily mock S3 method dispatch in testthat without defining it in global env or using mockery
  # But since we are testing plot.semx_fit which calls predict(), we rely on predict.semx_fit working.
  # predict.semx_fit relies on model structure.
  # Let's just mock the predict call by assigning it to the object if possible? No.
  
  # Let's make the model structure sufficient for predict.semx_fit
  fit$model$edges <- list(
      list(kind = 1L, source = "x", target = "y", parameter_id = "beta") # Regression
  )
  fit$optimization_result$parameters["beta"] <- 0.5
  fit$standard_errors <- c(fit$standard_errors, 0.1)
  fit$parameter_names <- c(fit$parameter_names, "beta")
  fit$data$x <- rnorm(100)
  
  # Test variance components
  vc <- semx_variance_components(fit)
  expect_equal(nrow(vc), 3) # Var(x1), Cov(x1, x2), Var(x2) - order depends on loop
  
  # Row 1: x1 variance
  expect_equal(vc$Variance[1], 4.0)
  expect_equal(vc$Std.Dev[1], 2.0)
  
  # Row 2: x1-x2 covariance
  expect_equal(vc$Variance[2], 2.0) # Covariance
  expect_equal(vc$Corr[2], 2.0 / (2.0 * 3.0)) # 2 / 6 = 0.3333
  
  # Row 3: x2 variance
  expect_equal(vc$Variance[3], 9.0)
  expect_equal(vc$Std.Dev[3], 3.0)
  
  # Test covariance weights
  cw <- semx_covariance_weights(fit, "mk_cov")
  expect_equal(cw$sigma_sq, 10.0)
  # Weights: exp(1-2)/(exp(1-2)+exp(2-2)) = exp(-1)/(exp(-1)+1) = 0.3678 / 1.3678 = 0.2689
  #          exp(2-2)/(exp(1-2)+exp(2-2)) = 1 / 1.3678 = 0.7310
  expect_equal(length(cw$weights), 2)
  expect_equal(sum(cw$weights), 1.0)
  
  # Test summary
  summ <- summary(fit)
  expect_true(!is.null(summ$variance_components))
  expect_true(!is.null(summ$covariance_weights$mk_cov))
  
  # Test plot (just that it runs)
  pdf(NULL)
  expect_error(plot(fit, type = "residuals"), NA)
  expect_error(plot(fit, type = "qq"), NA)
  dev.off()
})
