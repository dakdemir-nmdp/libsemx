test_that("semx_gs works", {
  n <- 50
  K <- matrix(rnorm(n*n), n, n)
  K <- crossprod(K)
  rownames(K) <- colnames(K) <- paste0("ID", 1:n)
  
  df <- data.frame(id = paste0("ID", 1:n), y = rnorm(n))
  
  fit <- semx_gs("y ~ 1", df, "id", kernel = K)
  expect_s3_class(fit, "semx_fit")
  
  # Check if we can get variance components
  vc <- semx_variance_components(fit)
  expect_true(nrow(vc) > 0)
  expect_true(any(vc$Group == "_genomic_group_"))
})
