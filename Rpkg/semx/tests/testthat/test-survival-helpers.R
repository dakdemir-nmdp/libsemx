library(testthat)
library(semx)

test_that("semx_predict_survival works for exponential", {
  model <- list(
    ir = list(parameter_ids = function() c("beta_y_on_x", "psi_y_y")),
    variables = list(y = list(family = "exponential")),
    edges = list(
      list(kind = 1, source = "x", target = "y", parameter_id = "beta_y_on_x")
    ),
    uses_intercept_column = FALSE
  )
  
  fit <- structure(
    list(
      optimization_result = list(parameters = c(beta_y_on_x = 0.5, psi_y_y = 1.0)),
      model = model
    ),
    class = "semx_fit"
  )
  
  newdata <- data.frame(x = c(0, 1))
  
  res <- semx_predict_survival(fit, newdata, times = c(1), outcome = "y")
  
  expect_equal(res[1, 1], exp(-1), tolerance = 1e-4)
  expect_equal(res[2, 1], exp(-exp(-0.5)), tolerance = 1e-4)
})
