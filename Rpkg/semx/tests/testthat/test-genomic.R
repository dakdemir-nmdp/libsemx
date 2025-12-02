library(testthat)
library(semx)

test_that("grm_vanraden (C++) builds expected kernel", {
  markers <- matrix(c(0, 1, 2, 1), nrow = 2, byrow = TRUE)
  grm <- grm_vanraden(markers)
  expect_equal(grm, c(1, -1, -1, 1), tolerance = 1e-6)
})

test_that("semx_model handles genomic markers", {
  markers <- matrix(c(0, 1, 2, 1), nrow = 2, byrow = TRUE)
  model <- semx_model(
    equations = c("y ~ id1 + id2"),
    families = c(y = "gaussian", id1 = "gaussian", id2 = "gaussian"),
    kinds = c(group = "grouping"),
    covariances = list(list(name = "cov_u", structure = "grm", dimension = 2L)),
    genomic = list(cov_u = list(markers = markers)),
    random_effects = list(list(name = "re_u", variables = c("group", "id1", "id2"), covariance = "cov_u"))
  )

  driver <- new(LikelihoodDriver)
  data <- list(
    y = c(1.0, 2.0),
    group = c(1.0, 1.0),
    id1 = c(1.0, 0.0),
    id2 = c(0.0, 1.0)
  )
  linear_predictors <- list(y = c(0.0, 0.0))
  dispersions <- list(y = c(1.0, 1.0))
  covariance_parameters <- list(cov_u = c(1.5))

  loglik <- driver$evaluate_model_loglik_full(
    model$ir,
    data,
    linear_predictors,
    dispersions,
    covariance_parameters,
    list(),
    list(),
    model$fixed_covariance_data,
    0L
  )
  expected <- -0.5 * (log(4.0) + 4.625 + 2.0 * log(2 * pi))
  expect_equal(loglik, expected, tolerance = 1e-6)
})
