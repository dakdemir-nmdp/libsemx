library(semx)

as_row_major <- function(mat) {
  as.numeric(t(mat))
}

test_that("grm_vanraden builds expected kernel", {
  markers <- matrix(c(0, 1, 2, 1), nrow = 2, byrow = TRUE)
  grm <- grm_vanraden(markers)
  expect_equal(length(grm), 4L)
  expect_equal(grm, c(1, -1, -1, 1), tolerance = 1e-6)
})

build_nb_model <- function() {
  builder <- new(ModelIRBuilder)
  builder$add_variable("y", 0, "negative_binomial")
  builder$add_variable("x", 0, "gaussian")
  builder$add_variable("cluster", 2, "")
  builder$add_edge(1, "x", "y", "beta")
  builder$add_covariance("G_nb", "diagonal", 1)
  builder$add_random_effect("u_cluster", c("cluster"), "G_nb")
  builder$build()
}

nb_data <- function() {
  list(
    y = c(0, 1, 1, 2, 3, 2, 1, 4),
    x = c(-1.4, -0.9, -0.3, 0.2, 0.7, 1.1, 1.5, 1.9),
    cluster = c(1, 1, 2, 2, 3, 3, 4, 4)
  )
}

nb_dispersions <- function(n) {
  list(y = rep(1.0, n))
}

build_ordinal_model <- function() {
  builder <- new(ModelIRBuilder)
  builder$add_variable("y", 0, "ordinal")
  builder$add_variable("x", 0, "gaussian")
  builder$add_variable("cluster", 2, "")
  builder$add_edge(1, "x", "y", "beta")
  builder$add_covariance("G_ord", "diagonal", 1)
  builder$add_random_effect("u_cluster", c("cluster"), "G_ord")
  builder$build()
}

ordinal_data <- function() {
  list(
    y = c(0, 1, 1, 2, 1, 2, 2, 0),
    x = c(-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0),
    cluster = c(1, 1, 2, 2, 3, 3, 4, 4)
  )
}

ordinal_thresholds <- function() {
  c(-0.4, 0.6)
}

build_kronecker_model <- function() {
  builder <- new(ModelIRBuilder)
  builder$add_variable("y", 0, "binomial")
  builder$add_variable("x", 0, "gaussian")
  builder$add_variable("cluster", 2, "")
  builder$add_variable("t1e1", 0, "gaussian")
  builder$add_variable("t1e2", 0, "gaussian")
  builder$add_variable("t2e1", 0, "gaussian")
  builder$add_variable("t2e2", 0, "gaussian")
  builder$add_edge(1, "x", "y", "beta")
  builder$add_covariance("G_kron", "multi_kernel", 4)
  builder$add_covariance("G_diag", "diagonal", 1)
  builder$add_random_effect("u_kron", c("cluster", "t1e1", "t1e2", "t2e1", "t2e2"), "G_kron")
  builder$add_random_effect("u_diag", c("cluster"), "G_diag")
  builder$build()
}

kronecker_data <- function() {
  list(
    y = c(0, 1, 0, 1, 0, 1, 0, 1),
    x = c(-1.6, -1.1, -0.6, -0.1, 0.4, 0.9, 1.4, 1.9),
    cluster = c(1, 2, 1, 2, 1, 2, 1, 2),
    t1e1 = c(1, 0, 0, 0, 1, 0, 0, 0),
    t1e2 = c(0, 1, 0, 0, 0, 1, 0, 0),
    t2e1 = c(0, 0, 1, 0, 0, 0, 1, 0),
    t2e2 = c(0, 0, 0, 1, 0, 0, 0, 1)
  )
}

kronecker_fixed_cov <- function() {
  trait_cov <- matrix(c(1.0, 0.35, 0.35, 1.0), nrow = 2, byrow = TRUE)
  env_cov <- matrix(c(1.0, 0.2, 0.2, 1.0), nrow = 2, byrow = TRUE)
  identity2 <- diag(2)
  list(
    G_kron = list(
      as_row_major(kronecker(trait_cov, identity2)),
      as_row_major(kronecker(identity2, env_cov))
    )
  )
}

LOG_SQRT_TWO_PI <- 0.5 * log(2 * pi)

lognormal_loglik <- function(time, eta, sigma, status) {
  z <- (log(time) - eta) / sigma
  if (status > 0) {
    return(-0.5 * z * z - log(time) - log(sigma) - LOG_SQRT_TWO_PI)
  }
  return(pnorm(-z, log.p = TRUE))
}

loglogistic_loglik <- function(time, eta, gamma, status) {
  diff <- log(time) - eta
  u <- exp(gamma * diff)
  if (status > 0) {
    return(log(gamma) + gamma * diff - log(time) - 2.0 * log1p(u))
  }
  return(-log1p(u))
}

test_that("ModelIRBuilder enforces graph invariants", {
  expect_error({
    builder <- new(ModelIRBuilder)
    builder$build()
  }, "must contain at least one variable")

  builder <- new(ModelIRBuilder)
  expect_error(builder$add_variable("y", 0L, ""), "requires outcome family")

  builder$add_variable("y", 0L, "gaussian")
  expect_error(builder$add_variable("y", 0L, "gaussian"), "duplicate variable name")

  builder$add_variable("eta", 1L, "")
  expect_error(builder$add_edge(1L, "eta", "missing", "beta"), "edge target not registered")
  expect_error(builder$add_edge(1L, "missing", "y", "beta"), "edge source not registered")
  expect_error(builder$add_edge(1L, "eta", "y", ""), "parameter id must be non-empty")

  expect_error(builder$add_covariance("", "diagonal", 1L), "covariance id must be non-empty")
  expect_error(builder$add_covariance("G", "", 1L), "structure identifier")
  expect_error(builder$add_covariance("G", "diagonal", 0L), "dimension must be positive")
  builder$add_covariance("G", "diagonal", 1L)

  expect_error(builder$add_random_effect("u", character(), "G"), "at least one variable")
  expect_error(builder$add_random_effect("u", c("missing"), "G"), "unknown variable")
  expect_error(builder$add_random_effect("u", c("eta", "eta"), "G"), "multiple times")
  expect_error(builder$add_random_effect("u", c("eta"), "missing"), "unknown covariance id")
})

test_that("ModelIR surfaces parameter identifiers", {
  builder <- new(ModelIRBuilder)
  builder$add_variable("y", 0L, "gaussian")
  builder$add_variable("x1", 1L, "")
  builder$add_variable("x2", 1L, "")
  builder$add_edge(1L, "x1", "y", "beta_x1")
  builder$add_edge(1L, "x2", "y", "beta_x2")
  builder$add_edge(1L, "x2", "y", "beta_x2")

  model <- builder$build()
  expect_equal(model$parameter_ids(), c("beta_x1", "beta_x2"))
})

test_that("semx_model compiles lavaan-style formulas", {
  mod <- semx_model(
    equations = c("eta =~ y1 + y2", "y1 ~ eta + x1", "y1 ~~ y2"),
    families = c(y1 = "gaussian", y2 = "gaussian", x1 = "gaussian")
  )

  expect_s3_class(mod, "semx_model")
  expect_true(inherits(mod$ir, c("ModelIR", "Rcpp_ModelIR")))
  expect_setequal(names(mod$variables), c("eta", "y1", "y2", "x1"))
  expect_equal(mod$variables$eta$kind, 1L) # latent
  expect_equal(mod$variables$y1$family, "gaussian")
  kinds <- vapply(mod$edges, function(e) e$kind, integer(1L))
  expect_equal(sum(kinds == 0L), 2L) # loadings
  expect_equal(sum(kinds == 1L), 2L) # regressions
  expect_equal(sum(kinds == 2L), 1L) # covariance

  expect_error(
    semx_model("y ~ x", families = c(x = "gaussian")),
    "requires a family"
  )
})

test_that("Gaussian gradient alignment follows parameter registry", {
  builder <- new(ModelIRBuilder)
  builder$add_variable("y", 0L, "gaussian")
  builder$add_variable("intercept", 0L, "gaussian")
  builder$add_variable("x1", 0L, "gaussian")
  builder$add_variable("x2", 0L, "gaussian")
  builder$add_variable("cluster", 2L, "")
  builder$add_edge(1L, "intercept", "y", "beta_intercept")
  builder$add_edge(1L, "x1", "y", "beta_x1")
  builder$add_edge(1L, "x2", "y", "beta_x2")
  builder$add_covariance("G_cluster", "diagonal", 1L)
  builder$add_random_effect("u_cluster", c("cluster"), "G_cluster")

  model <- builder$build()
  driver <- new(LikelihoodDriver)

  data <- list(
    y = c(-0.4, 0.5, 0.2, 1.2, 1.8, 2.4),
    intercept = rep(1.0, 6L),
    x1 = c(-1.5, -0.5, 0.0, 0.5, 1.0, 1.5),
    x2 = c(0.3, -0.2, 0.4, -0.1, 0.7, -0.4),
    cluster = c(1, 1, 2, 2, 3, 3)
  )

  opts <- new(OptimizationOptions)
  opts$max_iterations <- 400L
  opts$tolerance <- 1e-6

  fit <- driver$fit(model, data, opts, "lbfgs")
  expect_true(fit$optimization_result$converged)

  param_ids <- model$parameter_ids()
  expect_equal(param_ids, c("beta_intercept", "beta_x1", "beta_x2"))
  betas <- stats::setNames(fit$optimization_result$parameters[seq_along(param_ids)], param_ids)
  sigma <- fit$optimization_result$parameters[length(param_ids) + 1L]

  build_linear_predictors <- function(beta_vals) {
    list(
      y = beta_vals["beta_intercept"] * data$intercept +
        beta_vals["beta_x1"] * data$x1 +
        beta_vals["beta_x2"] * data$x2
    )
  }
  dispersions <- list(y = rep(1.0, length(data$y)))

  loglik <- function(beta_vals, sigma_val) {
    driver$evaluate_model_loglik_full(
      model,
      data,
      build_linear_predictors(beta_vals),
      dispersions,
      list(G_cluster = sigma_val),
      list(),
      list(),
      NULL,
      0L
    )
  }

  gradients <- driver$evaluate_model_gradient(
    model,
    data,
    build_linear_predictors(betas),
    dispersions,
    list(G_cluster = sigma),
    list(),
    list(),
    NULL,
    0L
  )

  beta_step <- 5e-4
  for (param in param_ids) {
    beta_plus <- betas
    beta_minus <- betas
    beta_plus[param] <- beta_plus[param] + beta_step
    beta_minus[param] <- beta_minus[param] - beta_step
    fd <- (loglik(beta_plus, sigma) - loglik(beta_minus, sigma)) / (2 * beta_step)
    expect_equal(gradients[[param]], fd, tolerance = 1e-3)
  }

  theta_step <- 1e-3
  sigma_plus <- sigma * exp(theta_step)
  sigma_minus <- sigma * exp(-theta_step)
  fd_sigma <- (loglik(betas, sigma_plus) - loglik(betas, sigma_minus)) / (sigma_plus - sigma_minus)
  expect_equal(gradients$G_cluster_0, fd_sigma, tolerance = 5e-3)
})

test_that("ModelIRBuilder works", {
  builder <- new(ModelIRBuilder)
  builder$add_variable("y", 0, "gaussian") # 0 = Observed
  
  model <- builder$build()
  expect_true(!is.null(model))
})

test_that("LikelihoodDriver evaluates lognormal survival outcomes via R bindings", {
  builder <- new(ModelIRBuilder)
  builder$add_variable("y", 0L, "lognormal")
  model <- builder$build()

  data <- list(y = c(1.4, 2.2, 0.95))
  linear_predictors <- list(y = c(0.3, -0.2, 0.1))
  dispersions <- list(y = c(0.9, 1.0, 0.85))
  status <- list(y = c(1.0, 0.0, 1.0))

  driver <- new(LikelihoodDriver)
  loglik <- driver$evaluate_model_loglik_full(
    model,
    data,
    linear_predictors,
    dispersions,
    list(),
    status,
    list(),
    NULL,
    0L
  )

  expected <- sum(mapply(
    lognormal_loglik,
    data$y,
    linear_predictors$y,
    dispersions$y,
    status$y
  ))

  expect_equal(loglik, expected, tolerance = 1e-10)
})

test_that("LikelihoodDriver evaluates loglogistic survival outcomes via R bindings", {
  builder <- new(ModelIRBuilder)
  builder$add_variable("y", 0L, "loglogistic")
  model <- builder$build()

  data <- list(y = c(1.2, 2.5, 3.0))
  linear_predictors <- list(y = c(0.1, 0.2, -0.4))
  dispersions <- list(y = c(1.1, 0.9, 1.3))
  status <- list(y = c(0.0, 1.0, 1.0))

  driver <- new(LikelihoodDriver)
  loglik <- driver$evaluate_model_loglik_full(
    model,
    data,
    linear_predictors,
    dispersions,
    list(),
    status,
    list(),
    NULL,
    0L
  )

  expected <- sum(mapply(
    loglogistic_loglik,
    data$y,
    linear_predictors$y,
    dispersions$y,
    status$y
  ))

  expect_equal(loglik, expected, tolerance = 1e-10)
})

test_that("LikelihoodDriver accumulates CIF-style competing risks via R bindings", {
  builder <- new(ModelIRBuilder)
  builder$add_variable("cause_lognormal", 0L, "lognormal")
  builder$add_variable("cause_loglogistic", 0L, "loglogistic")
  model <- builder$build()

  times <- c(1.3, 2.0, 3.6)
  data <- list(
    cause_lognormal = times,
    cause_loglogistic = times
  )
  linear_predictors <- list(
    cause_lognormal = c(0.2, -0.15, 0.25),
    cause_loglogistic = c(-0.3, 0.35, 0.05)
  )
  dispersions <- list(
    cause_lognormal = c(0.8, 1.05, 0.9),
    cause_loglogistic = c(1.2, 1.0, 1.1)
  )
  status <- list(
    cause_lognormal = c(1.0, 0.0, 0.0),
    cause_loglogistic = c(0.0, 1.0, 0.0)
  )

  driver <- new(LikelihoodDriver)
  loglik <- driver$evaluate_model_loglik_full(
    model,
    data,
    linear_predictors,
    dispersions,
    list(),
    status,
    list(),
    NULL,
    0L
  )

  expected_lognormal <- sum(mapply(
    lognormal_loglik,
    data$cause_lognormal,
    linear_predictors$cause_lognormal,
    dispersions$cause_lognormal,
    status$cause_lognormal
  ))
  expected_loglogistic <- sum(mapply(
    loglogistic_loglik,
    data$cause_loglogistic,
    linear_predictors$cause_loglogistic,
    dispersions$cause_loglogistic,
    status$cause_loglogistic
  ))

  expect_equal(loglik, expected_lognormal + expected_loglogistic, tolerance = 1e-10)
})

test_that("ModelIRBuilder serializes new covariance structures", {
  cases <- list(
    list(structure = "compound_symmetry", dim = 2L),
    list(structure = "cs", dim = 3L),
    list(structure = "ar1", dim = 2L),
    list(structure = "toeplitz", dim = 3L),
    list(structure = "fa1", dim = 3L)
  )

  for (case in cases) {
    builder <- new(ModelIRBuilder)
    builder$add_variable("y", 0L, "gaussian")
    builder$add_variable("cluster", 2L, "")
    if (case$dim > 1L) {
      for (idx in seq_len(case$dim)) {
        builder$add_variable(paste0("z", idx), 1L, "")
      }
    }
    builder$add_edge(1L, "cluster", "y", "beta")
    builder$add_covariance("G_new", case$structure, case$dim)
    design_vars <- "cluster"
    if (case$dim > 1L) {
      design_vars <- c(design_vars, paste0("z", seq_len(case$dim)))
    }
    builder$add_random_effect("u_new", design_vars, "G_new")
    model <- builder$build()
    expect_true(length(model$parameter_ids()) >= 1L)
  }
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

test_that("Laplace gradients are exposed via bindings", {
  builder <- new(ModelIRBuilder)
  builder$add_variable("y", 0, "binomial")
  builder$add_variable("x", 1, "")
  builder$add_variable("cluster", 2, "")
  builder$add_edge(1, "x", "y", "beta")
  builder$add_covariance("G", "diagonal", 1)
  builder$add_random_effect("u", c("cluster"), "G")

  model <- builder$build()
  driver <- new(LikelihoodDriver)

  y <- c(0, 1, 0, 1, 0, 1)
  x <- c(-1.0, -0.5, 0.2, 0.7, 1.0, 1.5)
  cluster <- c(1, 1, 2, 2, 3, 3)

  beta <- 0.8
  sigma <- 0.6

  data <- list(y = y, x = x, cluster = cluster)
  linear_predictors <- list(y = beta * x)
  dispersions <- list(y = rep(1.0, length(y)))

  gradients <- driver$evaluate_model_gradient(
    model,
    data,
    linear_predictors,
    dispersions,
    list(G = sigma),
    list(),
    list(),
    NULL,
    0L
  )

  loglik <- function(beta_val, sigma_val) {
    lp <- list(y = beta_val * x)
    driver$evaluate_model_loglik_full(
      model,
      data,
      lp,
      dispersions,
      list(G = sigma_val),
      list(),
      list(),
      NULL,
      0L
    )
  }

  eps <- 1e-5
  fd_beta <- (loglik(beta + eps, sigma) - loglik(beta - eps, sigma)) / (2 * eps)
  fd_sigma <- (loglik(beta, sigma + eps) - loglik(beta, sigma - eps)) / (2 * eps)

  expect_equal(gradients$beta, fd_beta, tolerance = 1e-4)
  expect_equal(gradients$G_0, fd_sigma, tolerance = 1e-4)
})

test_that("Negative binomial Laplace gradients match finite differences", {
  model <- build_nb_model()
  driver <- new(LikelihoodDriver)

  data <- nb_data()
  opts <- new(OptimizationOptions)
  opts$max_iterations <- 400
  opts$tolerance <- 5e-4
  opts$learning_rate <- 0.1

  fit <- driver$fit(model, data, opts, "lbfgs")
  expect_true(fit$optimization_result$converged)
  expect_length(fit$optimization_result$parameters, 2)
  beta <- fit$optimization_result$parameters[[1]]
  sigma <- fit$optimization_result$parameters[[2]]

  dispersions <- nb_dispersions(length(data$y))
  linear_predictors <- list(y = beta * data$x)

  gradients <- driver$evaluate_model_gradient(
    model,
    data,
    linear_predictors,
    dispersions,
    list(G_nb = sigma),
    list(),
    list(),
    NULL,
    0L
  )

  loglik <- function(beta_val, sigma_val) {
    lp <- list(y = beta_val * data$x)
    driver$evaluate_model_loglik_full(
      model,
      data,
      lp,
      dispersions,
      list(G_nb = sigma_val),
      list(),
      list(),
      NULL,
      0L
    )
  }

  step_beta <- 5e-4
  step_sigma <- max(1e-5, 0.1 * sigma)

  fd_beta <- (loglik(beta + step_beta, sigma) - loglik(beta - step_beta, sigma)) / (2 * step_beta)
  fd_sigma <- (loglik(beta, sigma + step_sigma) - loglik(beta, sigma - step_sigma)) / (2 * step_sigma)

  expect_equal(gradients$beta, fd_beta, tolerance = 2e-3)
  expect_equal(gradients$G_nb_0, fd_sigma, tolerance = 2e-3)
})

test_that("Ordinal Laplace gradients match finite differences", {
  model <- build_ordinal_model()
  driver <- new(LikelihoodDriver)

  data <- ordinal_data()
  thresholds <- ordinal_thresholds()

  beta <- 0.7
  sigma <- 0.8

  linear_predictors <- list(y = beta * data$x)
  dispersions <- list(y = rep(1.0, length(data$y)))
  extra <- list(y = thresholds)

  gradients <- driver$evaluate_model_gradient(
    model,
    data,
    linear_predictors,
    dispersions,
    list(G_ord = sigma),
    list(),
    extra,
    NULL,
    0L
  )

  loglik <- function(beta_val, sigma_val) {
    lp <- list(y = beta_val * data$x)
    driver$evaluate_model_loglik_full(
      model,
      data,
      lp,
      dispersions,
      list(G_ord = sigma_val),
      list(),
      extra,
      NULL,
      0L
    )
  }

  step_beta <- 5e-4
  step_sigma <- max(1e-5, 0.1 * sigma)

  fd_beta <- (loglik(beta + step_beta, sigma) - loglik(beta - step_beta, sigma)) / (2 * step_beta)
  fd_sigma <- (loglik(beta, sigma + step_sigma) - loglik(beta, sigma - step_sigma)) / (2 * step_sigma)

  expect_equal(gradients$beta, fd_beta, tolerance = 2e-3)
  expect_equal(gradients$G_ord_0, fd_sigma, tolerance = 2e-3)
})

test_that("Laplace fit converges through bindings", {
  builder <- new(ModelIRBuilder)
  builder$add_variable("y", 0, "binomial")
  builder$add_variable("x", 0, "gaussian")
  builder$add_variable("cluster", 2, "")
  builder$add_edge(1, "x", "y", "beta")
  builder$add_covariance("G", "diagonal", 1)
  builder$add_random_effect("u", c("cluster"), "G")

  model <- builder$build()
  driver <- new(LikelihoodDriver)

  data <- list(
    y = c(0, 1, 0, 1, 0, 1),
    x = c(-1.0, -0.5, 0.2, 0.7, 1.0, 1.5),
    cluster = c(1, 1, 2, 2, 3, 3)
  )

  opts <- new(OptimizationOptions)
  opts$max_iterations <- 200
  opts$tolerance <- 1e-4

  fit <- driver$fit(model, data, opts, "lbfgs")
  expect_true(fit$optimization_result$converged)
  expect_length(fit$optimization_result$parameters, 2)
  beta <- fit$optimization_result$parameters[[1]]
  sigma <- fit$optimization_result$parameters[[2]]
  expect_gt(sigma, 0)

  linear_predictors <- list(y = beta * data$x)
  dispersions <- list(y = rep(1.0, length(data$y)))
  gradients <- driver$evaluate_model_gradient(
    model,
    data,
    linear_predictors,
    dispersions,
    list(G = sigma),
    list(),
    list(),
    NULL,
    0L
  )

  expect_lt(abs(gradients$beta), 1e-3)
  expect_lt(sigma, 1e-3)
  expect_true(gradients$G_0 <= 0)
})

test_that("Laplace multi-effect fit converges through bindings", {
  builder <- new(ModelIRBuilder)
  builder$add_variable("y", 0, "binomial")
  builder$add_variable("x", 0, "gaussian")
  builder$add_variable("cluster", 2, "")
  builder$add_variable("batch", 2, "")
  builder$add_edge(1, "x", "y", "beta")
  builder$add_covariance("G_cluster", "diagonal", 1)
  builder$add_covariance("G_batch", "diagonal", 1)
  builder$add_random_effect("u_cluster", c("cluster"), "G_cluster")
  builder$add_random_effect("u_batch", c("batch"), "G_batch")

  model <- builder$build()
  driver <- new(LikelihoodDriver)

  data <- list(
    y = c(0, 1, 0, 1, 0, 1, 0, 1),
    x = c(-1.5, -0.8, -0.2, 0.4, 0.9, 1.2, 1.5, 1.8),
    cluster = c(1, 1, 2, 2, 3, 3, 4, 4),
    batch = c(1, 1, 1, 1, 2, 2, 2, 2)
  )

  opts <- new(OptimizationOptions)
  opts$max_iterations <- 250
  opts$tolerance <- 5e-4

  fit <- driver$fit(model, data, opts, "lbfgs")
  expect_true(fit$optimization_result$converged)
  expect_length(fit$optimization_result$parameters, 3)
  beta <- fit$optimization_result$parameters[[1]]
  sigma_cluster <- fit$optimization_result$parameters[[2]]
  sigma_batch <- fit$optimization_result$parameters[[3]]
  expect_gt(sigma_cluster, 0)
  expect_gt(sigma_batch, 0)

  linear_predictors <- list(y = beta * data$x)
  dispersions <- list(y = rep(1.0, length(data$y)))
  gradients <- driver$evaluate_model_gradient(
    model,
    data,
    linear_predictors,
    dispersions,
    list(G_cluster = sigma_cluster, G_batch = sigma_batch),
    list(),
    list(),
    NULL,
    0L
  )

  expect_lt(abs(gradients$beta), 1e-3)
  expect_true(gradients$G_cluster_0 <= 0)
  expect_true(gradients$G_batch_0 <= 0)
})

test_that("Laplace mixed covariance fit converges through bindings", {
  builder <- new(ModelIRBuilder)
  builder$add_variable("y", 0, "binomial")
  builder$add_variable("x", 0, "gaussian")
  builder$add_variable("cluster", 2, "")
  builder$add_variable("batch", 2, "")
  builder$add_edge(1, "x", "y", "beta")
  builder$add_covariance("G_cluster", "diagonal", 1)
  builder$add_covariance("G_batch_fixed", "scaled_fixed", 1)
  builder$add_random_effect("u_cluster", c("cluster"), "G_cluster")
  builder$add_random_effect("u_batch", c("batch"), "G_batch_fixed")

  model <- builder$build()
  driver <- new(LikelihoodDriver)

  data <- list(
    y = c(0, 1, 0, 1, 0, 1, 0, 1),
    x = c(-1.5, -0.8, -0.2, 0.4, 0.9, 1.2, 1.5, 1.8),
    cluster = c(1, 1, 2, 2, 3, 3, 4, 4),
    batch = c(1, 1, 1, 1, 2, 2, 2, 2)
  )

  opts <- new(OptimizationOptions)
  opts$max_iterations <- 250
  opts$tolerance <- 5e-4

  fixed_cov <- list(G_batch_fixed = list(matrix(1.0, nrow = 1, ncol = 1)))

  fit <- driver$fit_with_fixed(model, data, opts, "lbfgs", fixed_cov)
  expect_true(fit$optimization_result$converged)
  expect_length(fit$optimization_result$parameters, 3)
  beta <- fit$optimization_result$parameters[[1]]
  sigma_cluster <- fit$optimization_result$parameters[[2]]
  sigma_batch_scale <- fit$optimization_result$parameters[[3]]
  expect_gt(sigma_cluster, 0)
  expect_gt(sigma_batch_scale, 0)

  linear_predictors <- list(y = beta * data$x)
  dispersions <- list(y = rep(1.0, length(data$y)))
  gradients <- driver$evaluate_model_gradient(
    model,
    data,
    linear_predictors,
    dispersions,
    list(G_cluster = sigma_cluster, G_batch_fixed = sigma_batch_scale),
    list(),
    list(),
    fixed_cov,
    0L
  )

  expect_lt(abs(gradients$beta), 1e-3)
  expect_true(gradients$G_cluster_0 <= 0)
  expect_true(gradients$G_batch_fixed_0 <= 0)
})

test_that("Laplace random-slope fit converges through bindings", {
  builder <- new(ModelIRBuilder)
  builder$add_variable("y", 0, "binomial")
  builder$add_variable("x", 0, "gaussian")
  builder$add_variable("cluster", 2, "")
  builder$add_variable("intercept_col", 0, "gaussian")
  builder$add_variable("z", 0, "gaussian")
  builder$add_edge(1, "x", "y", "beta")
  builder$add_covariance("G_cluster2", "unstructured", 2)
  builder$add_random_effect("u_cluster2", c("cluster", "intercept_col", "z"), "G_cluster2")

  model <- builder$build()
  driver <- new(LikelihoodDriver)

  data <- list(
    y = c(0, 1, 0, 1, 0, 1, 0, 1),
    x = c(-1.4, -0.9, -0.3, 0.2, 0.7, 1.1, 1.5, 1.9),
    cluster = c(1, 1, 2, 2, 3, 3, 4, 4),
    intercept_col = rep(1, 8),
    z = c(-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 0.9, 1.3)
  )

  opts <- new(OptimizationOptions)
  opts$max_iterations <- 300
  opts$tolerance <- 5e-4

  fit <- driver$fit(model, data, opts, "lbfgs")
  expect_true(fit$optimization_result$converged)
  expect_length(fit$optimization_result$parameters, 4)
  beta <- fit$optimization_result$parameters[[1]]
  sigma_intercept <- fit$optimization_result$parameters[[2]]
  cov_term <- fit$optimization_result$parameters[[3]]
  sigma_slope <- fit$optimization_result$parameters[[4]]
  expect_gt(sigma_intercept, 0)
  expect_gt(sigma_slope, 0)
  expect_lt(abs(cov_term), 1)

  linear_predictors <- list(y = beta * data$x)
  dispersions <- list(y = rep(1.0, length(data$y)))
  gradients <- driver$evaluate_model_gradient(
    model,
    data,
    linear_predictors,
    dispersions,
    list(G_cluster2 = c(sigma_intercept, cov_term, sigma_slope)),
    list(),
    list(),
    NULL,
    0L
  )

  expect_lt(abs(gradients$beta), 1e-3)
  expect_true(gradients$G_cluster2_0 <= 0)
  expect_lt(abs(gradients$G_cluster2_1), 5e-3)
  expect_true(gradients$G_cluster2_2 <= 0)
})

test_that("Laplace Kronecker + diagonal fit converges through bindings", {
  model <- build_kronecker_model()
  driver <- new(LikelihoodDriver)

  data <- kronecker_data()
  fixed_cov <- kronecker_fixed_cov()

  opts <- new(OptimizationOptions)
  opts$max_iterations <- 600
  opts$tolerance <- 5e-4
  opts$learning_rate <- 0.2

  fit <- driver$fit_with_fixed(model, data, opts, "lbfgs", fixed_cov)
  expect_true(fit$optimization_result$converged)
  expect_length(fit$optimization_result$parameters, 5)
  beta <- fit$optimization_result$parameters[[1]]
  sigma_kron <- fit$optimization_result$parameters[[2]]
  weight_trait <- fit$optimization_result$parameters[[3]]
  weight_env <- fit$optimization_result$parameters[[4]]
  sigma_diag <- fit$optimization_result$parameters[[5]]
  expect_gt(sigma_kron, 0)
  expect_gt(sigma_diag, 0)

  linear_predictors <- list(y = beta * data$x)
  dispersions <- list(y = rep(1.0, length(data$y)))
  gradients <- driver$evaluate_model_gradient(
    model,
    data,
    linear_predictors,
    dispersions,
    list(G_kron = c(sigma_kron, weight_trait, weight_env), G_diag = sigma_diag),
    list(),
    list(),
    fixed_cov,
    0L
  )

  expect_lt(abs(gradients$beta), 2e-3)
  expect_true(gradients$G_kron_0 <= 0)
  expect_lt(abs(gradients$G_kron_1), 5e-3)
  expect_lt(abs(gradients$G_kron_2), 5e-3)
  expect_true(gradients$G_diag_0 <= 0)
})

test_that("Kronecker Laplace gradients match finite differences", {
  model <- build_kronecker_model()
  driver <- new(LikelihoodDriver)

  data <- kronecker_data()
  fixed_cov <- kronecker_fixed_cov()

  opts <- new(OptimizationOptions)
  opts$max_iterations <- 600
  opts$tolerance <- 5e-4
  opts$learning_rate <- 0.2

  fit <- driver$fit_with_fixed(model, data, opts, "lbfgs", fixed_cov)
  expect_true(fit$optimization_result$converged)
  expect_length(fit$optimization_result$parameters, 5)
  beta <- fit$optimization_result$parameters[[1]]
  sigma_kron <- fit$optimization_result$parameters[[2]]
  weight_trait <- fit$optimization_result$parameters[[3]]
  weight_env <- fit$optimization_result$parameters[[4]]
  sigma_diag <- fit$optimization_result$parameters[[5]]

  dispersions <- list(y = rep(1.0, length(data$y)))
  linear_predictors <- list(y = beta * data$x)
  gradients <- driver$evaluate_model_gradient(
    model,
    data,
    linear_predictors,
    dispersions,
    list(G_kron = c(sigma_kron, weight_trait, weight_env), G_diag = sigma_diag),
    list(),
    list(),
    fixed_cov,
    0L
  )

  loglik <- function(beta_val, sigma_kron_val, weight_trait_val, weight_env_val, sigma_diag_val) {
    lp <- list(y = beta_val * data$x)
    driver$evaluate_model_loglik_full(
      model,
      data,
      lp,
      dispersions,
      list(
        G_kron = c(sigma_kron_val, weight_trait_val, weight_env_val),
        G_diag = sigma_diag_val
      ),
      list(),
      list(),
      fixed_cov,
      0L
    )
  }

  step_fixed <- 5e-4
  step_sigma <- 1e-4

  fd_beta <- (loglik(beta + step_fixed, sigma_kron, weight_trait, weight_env, sigma_diag) -
    loglik(beta - step_fixed, sigma_kron, weight_trait, weight_env, sigma_diag)) / (2 * step_fixed)
  fd_trait <- (loglik(beta, sigma_kron, weight_trait + step_fixed, weight_env, sigma_diag) -
    loglik(beta, sigma_kron, weight_trait - step_fixed, weight_env, sigma_diag)) / (2 * step_fixed)
  fd_env <- (loglik(beta, sigma_kron, weight_trait, weight_env + step_fixed, sigma_diag) -
    loglik(beta, sigma_kron, weight_trait, weight_env - step_fixed, sigma_diag)) / (2 * step_fixed)
  fd_sigma <- (loglik(beta, sigma_kron + step_sigma, weight_trait, weight_env, sigma_diag) -
    loglik(beta, sigma_kron - step_sigma, weight_trait, weight_env, sigma_diag)) / (2 * step_sigma)
  fd_diag <- (loglik(beta, sigma_kron, weight_trait, weight_env, sigma_diag + step_sigma) -
    loglik(beta, sigma_kron, weight_trait, weight_env, sigma_diag - step_sigma)) / (2 * step_sigma)

  expect_equal(gradients$beta, fd_beta, tolerance = 2e-3)
  expect_equal(gradients$G_kron_1, fd_trait, tolerance = 3e-3)
  expect_equal(gradients$G_kron_2, fd_env, tolerance = 3e-3)
  expect_equal(gradients$G_kron_0, fd_sigma, tolerance = 3e-3)
  expect_equal(gradients$G_diag_0, fd_diag, tolerance = 3e-3)
})

test_that("semx_model and LikelihoodDriver handle genomic markers", {
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

test_that("genomic Kronecker kernels flow through semx_model", {
  markers <- matrix(c(0, 1, 2, 1), nrow = 2, byrow = TRUE)
  trait_cov <- matrix(c(1.0, 0.35, 0.35, 1.0), nrow = 2, byrow = TRUE)
  env_cov <- matrix(c(1.0, 0.2, 0.2, 1.0), nrow = 2, byrow = TRUE)

  grm <- matrix(grm_vanraden(markers), nrow = 2, byrow = TRUE)
  kron <- grm_kronecker(grm, env_cov)

  model <- semx_model(
    equations = c("y ~ t1e1 + t1e2 + t2e1 + t2e2"),
    families = c(
      y = "gaussian", t1e1 = "gaussian", t1e2 = "gaussian",
      t2e1 = "gaussian", t2e2 = "gaussian"
    ),
    kinds = c(group = "grouping"),
    covariances = list(list(name = "cov_gxe", structure = "grm", dimension = 4L)),
    genomic = list(cov_gxe = list(markers = matrix(kron, nrow = 4, byrow = TRUE), precomputed = TRUE)),
    random_effects = list(
      list(name = "re_gxe", variables = c("group", "t1e1", "t1e2", "t2e1", "t2e2"), covariance = "cov_gxe")
    )
  )

  driver <- new(LikelihoodDriver)
  data <- list(
    y = c(0.5, 1.2, -0.3, 0.9),
    group = c(1.0, 1.0, 1.0, 1.0),
    t1e1 = c(1, 0, 0, 0),
    t1e2 = c(0, 1, 0, 0),
    t2e1 = c(0, 0, 1, 0),
    t2e2 = c(0, 0, 0, 1)
  )
  linear_predictors <- list(y = rep(0.0, 4))
  dispersions <- list(y = rep(1.0, 4))
  covariance_parameters <- list(cov_gxe = c(1.2))

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
  expect_true(is.finite(loglik))
})
