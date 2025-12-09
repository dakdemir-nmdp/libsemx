test_that("semx_gxe works with diagonal structure", {
  df <- expand.grid(G = paste0("G", 1:5), E = paste0("E", 1:3))
  df$y <- rnorm(nrow(df))
  
  # We need to ensure semx is loaded
  # devtools::test() does this.
  
  fit <- semx_gxe("y ~ 1", df, "G", "E", gxe_structure = "diagonal")
  expect_s3_class(fit, "semx_fit")
  
  # Check if interaction random effect exists
  # We can inspect fit$model$random_effects
  # But fit object structure might be different.
  # fit$random_effects is a list of matrices (BLUPs)
})

test_that("semx_gxe works with Kronecker structure", {
  df <- expand.grid(G = paste0("G", 1:5), E = paste0("E", 1:3))
  df$y <- rnorm(nrow(df))
  K <- diag(5)
  
  fit <- semx_gxe("y ~ 1", df, "G", "E", genomic = K, gxe_structure = "kronecker")
  expect_s3_class(fit, "semx_fit")
})
