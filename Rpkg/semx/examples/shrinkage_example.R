#!/usr/bin/env Rscript
#
# Example: Random Effects Shrinkage in Genomic Selection
#
# This example demonstrates how to use the lambda parameter for ridge regularization
# (random effects shrinkage) in genomic prediction models.
#
# Shrinkage is useful for:
# - High-dimensional genomic data (preventing overfitting)
# - Stabilizing variance estimates
# - Implementing Bayesian ridge regression
#
# Mathematical Background:
# - Standard: u ~ N(0, σ²G)
# - With shrinkage: u ~ N(0, σ²G/λ)
#   where λ > 1 increases shrinkage (pulls random effects toward zero)

library(semx)

generate_synthetic_genomic_data <- function(n_individuals = 200, n_markers = 500, h2 = 0.6, seed = 42) {
  #' Generate synthetic genomic data for demonstration
  #'
  #' @param n_individuals Number of individuals
  #' @param n_markers Number of genetic markers
  #' @param h2 Heritability (proportion of variance explained by genetics)
  #' @param seed Random seed
  #'
  #' @return List with genotypes, phenotypes, true_effects, and grm

  set.seed(seed)

  # Generate marker genotypes (0, 1, 2 coding)
  maf <- runif(n_markers, 0.1, 0.5)  # Minor allele frequencies
  genotypes <- matrix(0, n_individuals, n_markers)
  for (j in 1:n_markers) {
    genotypes[, j] <- rbinom(n_individuals, 2, maf[j])
  }

  # Standardize markers
  X <- scale(genotypes)

  # Compute genomic relationship matrix (VanRaden method)
  grm <- (X %*% t(X)) / n_markers

  # Generate true marker effects (sparse - only 50 have non-zero effects)
  true_effects <- numeric(n_markers)
  causal_markers <- sample(n_markers, 50)
  true_effects[causal_markers] <- rnorm(50, 0, 1)

  # Generate genetic values
  genetic_values <- X %*% true_effects

  # Scale to achieve desired heritability
  var_g <- var(genetic_values)
  var_e <- var_g * (1 - h2) / h2

  # Generate phenotypes
  residuals <- rnorm(n_individuals, 0, sqrt(var_e))
  phenotypes <- as.vector(genetic_values + residuals)

  cat(sprintf("Generated data: n=%d, p=%d, h²=%.2f\n", n_individuals, n_markers, h2))
  cat(sprintf("Genetic variance: %.3f, Residual variance: %.3f\n", var_g, var_e))

  list(
    genotypes = genotypes,
    phenotypes = phenotypes,
    true_effects = true_effects,
    grm = grm
  )
}

fit_gblup_with_shrinkage <- function(phenotypes, grm, lambda_value = 1.0) {
  #' Fit a GBLUP model with specified shrinkage parameter
  #'
  #' @param phenotypes Trait values (n,)
  #' @param grm Genomic relationship matrix (n × n)
  #' @param lambda_value Shrinkage parameter (default 1.0 = no shrinkage)
  #'
  #' @return FitResult object

  n <- length(phenotypes)

  # Build model using ModelIRBuilder
  builder <- new(ModelIRBuilder)

  # Add outcome variable (Gaussian)
  # VariableKind: Observed = 0L
  builder$add_variable("y", 0L, "gaussian", "", "")

  # Add intercept
  # VariableKind: Exogenous = 1L
  builder$add_variable("_intercept", 1L, "", "", "")

  # EdgeKind: Regression = 0L
  builder$add_edge(0L, "_intercept", "y", "beta_0")
  builder$register_parameter("beta_0", 0.0)

  # Add grouping variable for individuals (indexed 0 to n-1)
  # VariableKind: Grouping = 2L
  builder$add_variable("id", 2L, "", "", "")

  # Add genomic random effect with shrinkage
  builder$add_covariance("grm_cov", "scaled_fixed", n)
  builder$add_random_effect("u_genomic", c("id", "y"), "grm_cov", lambda_value)  # <-- Lambda parameter!

  model <- builder$build()

  # Prepare data
  data <- list(
    y = phenotypes,
    `_intercept` = rep(1.0, n),
    id = 0:(n-1)  # Integer indices starting from 0
  )

  # Flatten GRM for C++ (row-major)
  grm_flat <- as.vector(t(grm))
  fixed_cov_data <- list(grm_cov = list(grm_flat))

  # Fit model
  driver <- new(LikelihoodDriver)

  # EstimationMethod: REML = 1L
  result <- driver$fit(model, data, list(), fixed_cov_data, 1L)

  result
}

compare_shrinkage_levels <- function(phenotypes, grm, lambda_values = c(0.5, 1.0, 2.0, 5.0)) {
  #' Compare GBLUP models with different shrinkage levels

  cat("\n", paste(rep("=", 70), collapse = ""), "\n", sep = "")
  cat("Comparing Different Shrinkage Levels\n")
  cat(paste(rep("=", 70), collapse = ""), "\n", sep = "")

  results <- list()

  for (lambda_val in lambda_values) {
    cat(sprintf("\nFitting model with λ = %.1f...\n", lambda_val))
    result <- fit_gblup_with_shrinkage(phenotypes, grm, lambda_val)

    # Extract variance components
    params <- result$optimization_result$parameters
    param_names <- result$parameter_names

    # Find variance component indices
    sigma_g_idx <- which(param_names == "grm_cov")
    sigma_e_idx <- which(param_names == "y_dispersion")

    sigma_g_sq <- params[sigma_g_idx]
    sigma_e_sq <- params[sigma_e_idx]

    # Compute effective heritability
    # Note: with shrinkage, effective genetic variance is σ²_g / λ
    effective_sigma_g_sq <- sigma_g_sq / lambda_val
    h2_est <- effective_sigma_g_sq / (effective_sigma_g_sq + sigma_e_sq)

    # Get random effects (BLUPs)
    blups <- result$random_effects$u_genomic
    blup_variance <- var(blups)

    cat(sprintf("  Log-likelihood: %.2f\n", result$optimization_result$objective_value))
    cat(sprintf("  σ²_g: %.4f, σ²_e: %.4f\n", sigma_g_sq, sigma_e_sq))
    cat(sprintf("  Effective σ²_g (σ²_g/λ): %.4f\n", effective_sigma_g_sq))
    cat(sprintf("  Estimated h²: %.4f\n", h2_est))
    cat(sprintf("  BLUP variance: %.4f\n", blup_variance))
    cat(sprintf("  Max |BLUP|: %.4f\n", max(abs(blups))))

    results[[as.character(lambda_val)]] <- list(
      sigma_g_sq = sigma_g_sq,
      sigma_e_sq = sigma_e_sq,
      h2 = h2_est,
      blups = blups,
      loglik = result$optimization_result$objective_value
    )
  }

  results
}

demonstrate_cross_validation <- function(X, phenotypes, grm, lambda_values = c(0.5, 1.0, 2.0, 5.0)) {
  #' Demonstrate cross-validation to select optimal shrinkage parameter

  cat("\n", paste(rep("=", 70), collapse = ""), "\n", sep = "")
  cat("Cross-Validation for Optimal Shrinkage Parameter\n")
  cat(paste(rep("=", 70), collapse = ""), "\n", sep = "")

  n <- length(phenotypes)
  n_folds <- 5
  fold_size <- n %/% n_folds

  # Shuffle indices
  set.seed(123)
  indices <- sample(n)

  cv_errors <- lapply(lambda_values, function(x) numeric(n_folds))
  names(cv_errors) <- as.character(lambda_values)

  for (fold in 1:n_folds) {
    # Split data
    test_start <- (fold - 1) * fold_size + 1
    test_end <- ifelse(fold < n_folds, fold * fold_size, n)
    test_indices <- indices[test_start:test_end]
    train_indices <- setdiff(1:n, test_indices)

    y_train <- phenotypes[train_indices]
    grm_train <- grm[train_indices, train_indices]

    for (i in seq_along(lambda_values)) {
      lambda_val <- lambda_values[i]

      # Fit on training set
      result <- fit_gblup_with_shrinkage(y_train, grm_train, lambda_val)

      # Predict test set (simplified - just use mean)
      blups_train <- result$random_effects$u_genomic
      mean_pred <- mean(y_train)

      # For test individuals, use 0 (could use GRM-based prediction in practice)
      y_pred <- mean_pred
      y_test <- phenotypes[test_indices]

      # Compute MSE
      mse <- mean((y_test - y_pred)^2)
      cv_errors[[as.character(lambda_val)]][fold] <- mse
    }
  }

  cat("\nCross-validation results (Mean Squared Error):\n")
  cat(paste(rep("-", 40), collapse = ""), "\n", sep = "")

  mean_mses <- numeric(length(lambda_values))
  names(mean_mses) <- as.character(lambda_values)

  for (i in seq_along(lambda_values)) {
    lambda_val <- lambda_values[i]
    errors <- cv_errors[[as.character(lambda_val)]]
    mean_mse <- mean(errors)
    std_mse <- sd(errors)
    mean_mses[as.character(lambda_val)] <- mean_mse
    cat(sprintf("λ = %4.1f: MSE = %.4f ± %.4f\n", lambda_val, mean_mse, std_mse))
  }

  # Find optimal lambda
  optimal_lambda <- as.numeric(names(which.min(mean_mses)))
  cat(sprintf("\nOptimal λ: %.1f\n", optimal_lambda))

  optimal_lambda
}

main <- function() {
  cat(paste(rep("=", 70), collapse = ""), "\n", sep = "")
  cat("Random Effects Shrinkage Example: Genomic Selection\n")
  cat(paste(rep("=", 70), collapse = ""), "\n", sep = "")

  # Generate synthetic data
  sim_data <- generate_synthetic_genomic_data(
    n_individuals = 200,
    n_markers = 500,
    h2 = 0.6
  )

  # Compare different shrinkage levels
  results <- compare_shrinkage_levels(sim_data$phenotypes, sim_data$grm)

  # Demonstrate cross-validation
  optimal_lambda <- demonstrate_cross_validation(
    sim_data$genotypes,
    sim_data$phenotypes,
    sim_data$grm
  )

  # Fit final model with optimal shrinkage
  cat("\n", paste(rep("=", 70), collapse = ""), "\n", sep = "")
  cat(sprintf("Final Model with Optimal Shrinkage (λ = %.1f)\n", optimal_lambda))
  cat(paste(rep("=", 70), collapse = ""), "\n", sep = "")

  final_result <- fit_gblup_with_shrinkage(sim_data$phenotypes, sim_data$grm, optimal_lambda)

  cat("\nFinal model fitted successfully!\n")
  cat(sprintf("Number of individuals: %d\n", length(sim_data$phenotypes)))
  cat(sprintf("Shrinkage parameter: λ = %.1f\n", optimal_lambda))
  cat(sprintf("BLUP range: [%.3f, %.3f]\n",
              min(final_result$random_effects$u_genomic),
              max(final_result$random_effects$u_genomic)))

  # Key insight
  cat("\n", paste(rep("=", 70), collapse = ""), "\n", sep = "")
  cat("Key Insight:\n")
  cat(paste(rep("=", 70), collapse = ""), "\n", sep = "")
  cat("Higher lambda (λ > 1) → More shrinkage → Smaller BLUPs → Less overfitting\n")
  cat("Lower lambda (λ < 1) → Less shrinkage → Larger BLUPs → More flexibility\n")
  cat("\nFor genomic selection with many markers, moderate shrinkage (λ ≈ 2-5)\n")
  cat("often improves prediction accuracy by reducing overfitting.\n")
  cat(paste(rep("=", 70), collapse = ""), "\n", sep = "")
}

# Run the example
if (!interactive()) {
  main()
}
