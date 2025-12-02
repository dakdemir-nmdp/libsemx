test_that("mixed model intercept only", {
  # y ~ x + (1 | cluster)
  model <- semx_model(
    equations = c("y ~ x + (1 | cluster)"),
    families = c(y = "gaussian", x = "gaussian"),
    kinds = c(cluster = "grouping")
  )
  
  ir <- model$ir
  
  # Check random effects
  expect_equal(length(ir$random_effects), 1)
  re <- ir$random_effects[[1]]
  expect_equal(re$variables, "cluster")
  expect_true(!is.null(re$covariance_id))
  
  # Check covariance structure
  cov_idx <- which(sapply(ir$covariances, function(c) c$id == re$covariance_id))
  expect_true(length(cov_idx) == 1)
  cov <- ir$covariances[[cov_idx]]
  expect_equal(cov$structure, "unstructured")
  expect_equal(cov$dimension, 1)
})

test_that("mixed model random slope", {
  # y ~ x + (x | cluster)
  model <- semx_model(
    equations = c("y ~ x + (x | cluster)"),
    families = c(y = "gaussian", x = "gaussian"),
    kinds = c(cluster = "grouping")
  )
  
  ir <- model$ir
  re <- ir$random_effects[[1]]
  
  # Expect cluster, _intercept, x
  expect_equal(re$variables, c("cluster", "_intercept", "x"))
  
  cov_idx <- which(sapply(ir$covariances, function(c) c$id == re$covariance_id))
  cov <- ir$covariances[[cov_idx]]
  expect_equal(cov$dimension, 2)
})

test_that("mixed model explicit intercept and slope", {
  # y ~ x + (1 + x | cluster)
  model <- semx_model(
    equations = c("y ~ x + (1 + x | cluster)"),
    families = c(y = "gaussian", x = "gaussian"),
    kinds = c(cluster = "grouping")
  )
  
  ir <- model$ir
  re <- ir$random_effects[[1]]
  expect_equal(re$variables, c("cluster", "_intercept", "x"))
  
  cov_idx <- which(sapply(ir$covariances, function(c) c$id == re$covariance_id))
  cov <- ir$covariances[[cov_idx]]
  expect_equal(cov$dimension, 2)
})

test_that("mixed model multiple groups", {
  # y ~ x + (1 | school) + (1 | class)
  model <- semx_model(
    equations = c("y ~ x + (1 | school) + (1 | class)"),
    families = c(y = "gaussian", x = "gaussian"),
    kinds = c(school = "grouping", class = "grouping")
  )
  
  ir <- model$ir
  expect_equal(length(ir$random_effects), 2)
  
  re_vars <- lapply(ir$random_effects, function(r) r$variables)
  expect_true(any(sapply(re_vars, function(v) "school" %in% v)))
  expect_true(any(sapply(re_vars, function(v) "class" %in% v)))
})

test_that("mixed model no intercept", {
  # y ~ x + (0 + x | cluster)
  model <- semx_model(
    equations = c("y ~ x + (0 + x | cluster)"),
    families = c(y = "gaussian", x = "gaussian"),
    kinds = c(cluster = "grouping")
  )
  
  ir <- model$ir
  re <- ir$random_effects[[1]]
  expect_equal(re$variables, c("cluster", "x"))
  
  cov_idx <- which(sapply(ir$covariances, function(c) c$id == re$covariance_id))
  cov <- ir$covariances[[cov_idx]]
  expect_equal(cov$dimension, 1)
})
