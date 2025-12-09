library(testthat)
library(semx)

test_that("grm_vanraden (C++) builds expected kernel", {
  markers <- matrix(c(0, 1, 2, 1), nrow = 2, byrow = TRUE)
  grm <- grm_vanraden(markers)
  expected <- matrix(c(1, -1, -1, 1), nrow = 2)
  expect_equal(grm, expected, tolerance = 1e-6)
})

test_that("semx_model handles genomic markers", {
  markers <- matrix(c(0, 1, 2, 1), nrow = 2, byrow = TRUE)
  model <- semx_model(
    equations = c("y ~ re_u"),
    families = c(y = "gaussian"),
    kinds = c(group = "grouping", re_u = "latent"),
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

test_that("semx_gs fits a GBLUP model", {
  set.seed(123)
  n <- 50
  p <- 20
  M <- matrix(sample(0:2, n * p, replace = TRUE), n, p)
  g <- rnorm(p)
  y <- M %*% g + rnorm(n)
  df <- data.frame(y = y, id = as.factor(1:n))
  
  # Fit model
  fit <- semx_gs(y ~ 1, data = df, geno_id = "id", markers = M)
  
  expect_s3_class(fit, "semx_fit")
  expect_true(fit$optimization_result$converged)
  
  # Check variance components
  vc <- semx_variance_components(fit)
  
  # In semx GBLUP, the group is currently "_genomic_group_" and the term is the ID variable
  expect_true("_genomic_group_" %in% vc$Group)
  expect_true("id" %in% vc$Name1)
  
  # Check values (approximate)
  # expect_true("(Intercept)" %in% vc$Name1)
})
