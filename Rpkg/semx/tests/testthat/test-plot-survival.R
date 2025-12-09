test_that("semx_plot_survival works", {
  library(semx)
  source("../../R/survival.R")
  source("../../R/plot_survival.R")
  
  # Mock fit object
  fit <- list(
    model = list(
      variables = list(
        time = list(name = "time", kind = 0, family = "weibull")
      ),
      edges = list(
        list(kind = 1, source = "x", target = "time", parameter_id = "time~x"),
        list(kind = 1, source = "_intercept", target = "time", parameter_id = "time~_intercept")
      ),
      uses_intercept_column = TRUE,
      ir = list(parameter_ids = function() c("time~x", "time~_intercept", "psi_time_time"))
    ),
    data = data.frame(time = c(10, 20, 30), x = c(1, 2, 3)),
    optimization_result = list(
      parameters = list(
        "time~x" = 0.5,
        "time~_intercept" = 2.0,
        "psi_time_time" = 1.5
      )
    ),
    parameter_names = c("time~x", "time~_intercept", "psi_time_time")
  )
  class(fit) <- "semx_fit"
  
  # We need ggplot2
  skip_if_not_installed("ggplot2")
  
  p <- semx_plot_survival(fit)
  expect_s3_class(p, "ggplot")
  
  # Test with newdata
  newdata <- data.frame(x = c(4))
  p2 <- semx_plot_survival(fit, newdata = newdata)
  expect_s3_class(p2, "ggplot")
  
  # Test dispatch (requires package reinstall to work)
  # p3 <- plot(fit, type = "survival")
  # expect_s3_class(p3, "ggplot")
})
