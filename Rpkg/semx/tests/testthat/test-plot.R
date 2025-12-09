context("plot")
library(semx)

test_that("plot.semx_fit supports path diagrams", {
  skip_if_not_installed("DiagrammeR")
  
  # Simple model
  set.seed(123)
  n <- 100
  x <- rnorm(n)
  y <- 0.5 * x + rnorm(n)
  data <- data.frame(x = x, y = y)
  
  model <- semx_model(
    equations = c("y ~ x"),
    families = c(y = "gaussian"),
    data = data
  )
  
  fit <- semx_fit(model, data)
  
  # Should return a DiagrammeR object (which is an htmlwidget)
  p <- plot(fit, type = "path")
  
  expect_s3_class(p, "htmlwidget")
  expect_true(inherits(p, "grViz"))
})
