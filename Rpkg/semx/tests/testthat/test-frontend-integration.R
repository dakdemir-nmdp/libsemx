test_that("semx_model round-trips Laplace GLMM", {
  model <- semx_model(
    equations = c("y ~ 1 + cluster"),
    families = c(y = "binomial"),
    kinds = c(cluster = "grouping"),
    covariances = list(
      list(name = "G_diag", structure = "diagonal", dimension = 1)
    ),
    random_effects = list(
      list(name = "u_cluster", variables = c("cluster"), covariance = "G_diag")
    )
  )
  
  ir <- model$ir
  expect_equal(length(ir$covariances), 1)
  expect_equal(ir$covariances[[1]]$id, "G_diag")
  expect_equal(ir$covariances[[1]]$structure, "diagonal")
  
  expect_equal(length(ir$random_effects), 1)
  expect_equal(ir$random_effects[[1]]$id, "u_cluster")
  expect_equal(ir$random_effects[[1]]$variables, c("cluster"))
  expect_equal(ir$random_effects[[1]]$covariance_id, "G_diag")
})

test_that("semx_model round-trips Survival", {
  model <- semx_model(
    equations = c("y ~ x"),
    families = c(y = "weibull", x = "gaussian")
  )
  
  ir <- model$ir
  vars <- ir$variables
  
  # Helper to find variable by name
  find_var <- function(name) {
    for (v in vars) {
      if (v$name == name) return(v)
    }
    NULL
  }
  
  y_var <- find_var("y")
  expect_false(is.null(y_var))
  expect_equal(y_var$family, "weibull")
  
  x_var <- find_var("x")
  expect_false(is.null(x_var))
  # VariableKind.Observed is 0
  expect_equal(x_var$kind, 0) 
})

test_that("semx_model round-trips Kronecker", {
  model <- semx_model(
    equations = c("y ~ 1 + env + geno"),
    families = c(y = "gaussian"),
    kinds = c(env = "grouping", geno = "grouping"),
    covariances = list(
      list(name = "G_kron", structure = "kronecker", dimension = 1)
    ),
    random_effects = list(
      list(name = "u_gxe", variables = c("env", "geno"), covariance = "G_kron")
    )
  )
  
  ir <- model$ir
  expect_equal(length(ir$covariances), 1)
  expect_equal(ir$covariances[[1]]$id, "G_kron")
  expect_equal(ir$covariances[[1]]$structure, "kronecker")
  
  expect_equal(length(ir$random_effects), 1)
  expect_equal(ir$random_effects[[1]]$id, "u_gxe")
  expect_equal(ir$random_effects[[1]]$variables, c("env", "geno"))
})
