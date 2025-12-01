test_that("ModelIRBuilder works", {
  builder <- new(ModelIRBuilder)
  builder$add_variable("y", 0, "gaussian") # 0 = Observed
  
  model <- builder$build()
  expect_true(!is.null(model))
})

test_that("LikelihoodDriver works", {
  builder <- new(ModelIRBuilder)
  builder$add_variable("y", 0, "gaussian")
  model <- builder$build()
  
  driver <- new(LikelihoodDriver)
  
  data <- list(y = c(1.0, 2.0, 3.0))
  linear_predictors <- list(y = c(1.0, 2.0, 3.0))
  dispersions <- list(y = c(1.0, 1.0, 1.0))
  
  loglik <- driver$evaluate_model_loglik(model, data, linear_predictors, dispersions)
  
  expected <- 3 * (-0.5 * log(2 * pi))
  expect_equal(loglik, expected, tolerance = 1e-6)
})
