#!/usr/bin/env python3
"""
Example: Random Effects Shrinkage in Genomic Selection

This example demonstrates how to use the lambda parameter for ridge regularization
(random effects shrinkage) in genomic prediction models.

Shrinkage is useful for:
- High-dimensional genomic data (preventing overfitting)
- Stabilizing variance estimates
- Implementing Bayesian ridge regression

Mathematical Background:
- Standard: u ~ N(0, σ²G)
- With shrinkage: u ~ N(0, σ²G/λ)
  where λ > 1 increases shrinkage (pulls random effects toward zero)
"""

import numpy as np
import sys
import os

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), '../../build')
sys.path.insert(0, build_dir)

try:
    import _libsemx as semx
except ImportError:
    print(f"Error: Could not import _libsemx from {build_dir}")
    print("Make sure you have built the project with: cmake --build build")
    sys.exit(1)

def generate_synthetic_genomic_data(n_individuals=200, n_markers=500, h2=0.6, seed=42):
    """
    Generate synthetic genomic data for demonstration.

    Args:
        n_individuals: Number of individuals
        n_markers: Number of genetic markers
        h2: Heritability (proportion of variance explained by genetics)
        seed: Random seed

    Returns:
        genotypes: Marker matrix (n × p)
        phenotypes: Trait values (n,)
        true_effects: True marker effects
        grm: Genomic relationship matrix
    """
    np.random.seed(seed)

    # Generate marker genotypes (0, 1, 2 coding)
    maf = np.random.uniform(0.1, 0.5, n_markers)  # Minor allele frequencies
    genotypes = np.random.binomial(2, maf, size=(n_individuals, n_markers))

    # Standardize markers
    X = (genotypes - genotypes.mean(axis=0)) / (genotypes.std(axis=0) + 1e-6)

    # Compute genomic relationship matrix (VanRaden method)
    grm = (X @ X.T) / n_markers

    # Generate true marker effects (sparse - only 50 have non-zero effects)
    true_effects = np.zeros(n_markers)
    causal_markers = np.random.choice(n_markers, size=50, replace=False)
    true_effects[causal_markers] = np.random.normal(0, 1, size=50)

    # Generate genetic values
    genetic_values = X @ true_effects

    # Scale to achieve desired heritability
    var_g = np.var(genetic_values)
    var_e = var_g * (1 - h2) / h2

    # Generate phenotypes
    residuals = np.random.normal(0, np.sqrt(var_e), n_individuals)
    phenotypes = genetic_values + residuals

    print(f"Generated data: n={n_individuals}, p={n_markers}, h²={h2:.2f}")
    print(f"Genetic variance: {var_g:.3f}, Residual variance: {var_e:.3f}")

    return X, phenotypes, true_effects, grm


def fit_gblup_with_shrinkage(phenotypes, grm, lambda_value=1.0):
    """
    Fit a GBLUP model with specified shrinkage parameter.

    Args:
        phenotypes: Trait values (n,)
        grm: Genomic relationship matrix (n × n)
        lambda_value: Shrinkage parameter (default 1.0 = no shrinkage)

    Returns:
        FitResult object
    """
    n = len(phenotypes)

    # Build model using ModelIRBuilder
    builder = semx.ModelIRBuilder()

    # Add outcome variable (Gaussian)
    builder.add_variable("y", semx.VariableKind.Observed, "gaussian")

    # Add intercept
    builder.add_variable("_intercept", semx.VariableKind.Exogenous)
    builder.add_edge(semx.EdgeKind.Regression, "_intercept", "y", "beta_0")
    builder.register_parameter("beta_0", 0.0)

    # Add genomic random effect with shrinkage
    builder.add_covariance("grm_cov", "genomic", n)
    builder.add_random_effect("u_genomic", ["y"], "grm_cov", lambda_value)  # <-- Lambda parameter!

    model = builder.build()

    # Prepare data
    data = {
        "y": phenotypes.tolist(),
        "_intercept": [1.0] * n
    }

    # Flatten GRM for C++ (row-major)
    grm_flat = grm.flatten().tolist()
    fixed_cov_data = {"grm_cov": [grm_flat]}

    # Fit model
    driver = semx.LikelihoodDriver()
    result = driver.fit(model, data, {}, fixed_cov_data, semx.EstimationMethod.REML)

    return result


def compare_shrinkage_levels(phenotypes, grm, lambda_values=[0.5, 1.0, 2.0, 5.0]):
    """
    Compare GBLUP models with different shrinkage levels.
    """
    print("\n" + "="*70)
    print("Comparing Different Shrinkage Levels")
    print("="*70)

    results = {}

    for lambda_val in lambda_values:
        print(f"\nFitting model with λ = {lambda_val}...")
        result = fit_gblup_with_shrinkage(phenotypes, grm, lambda_val)

        # Extract variance components
        params = result.optimization_result.parameters
        param_names = result.parameter_names

        # Find variance component indices
        sigma_g_idx = param_names.index("grm_cov")
        sigma_e_idx = param_names.index("y_dispersion")

        sigma_g_sq = params[sigma_g_idx]
        sigma_e_sq = params[sigma_e_idx]

        # Compute effective heritability
        # Note: with shrinkage, effective genetic variance is σ²_g / λ
        effective_sigma_g_sq = sigma_g_sq / lambda_val
        h2_est = effective_sigma_g_sq / (effective_sigma_g_sq + sigma_e_sq)

        # Get random effects (BLUPs)
        blups = result.random_effects["u_genomic"]
        blup_variance = np.var(blups)

        print(f"  Log-likelihood: {result.optimization_result.objective_value:.2f}")
        print(f"  σ²_g: {sigma_g_sq:.4f}, σ²_e: {sigma_e_sq:.4f}")
        print(f"  Effective σ²_g (σ²_g/λ): {effective_sigma_g_sq:.4f}")
        print(f"  Estimated h²: {h2_est:.4f}")
        print(f"  BLUP variance: {blup_variance:.4f}")
        print(f"  Max |BLUP|: {np.max(np.abs(blups)):.4f}")

        results[lambda_val] = {
            'sigma_g_sq': sigma_g_sq,
            'sigma_e_sq': sigma_e_sq,
            'h2': h2_est,
            'blups': blups,
            'loglik': result.optimization_result.objective_value
        }

    return results


def demonstrate_cross_validation(X, phenotypes, grm, lambda_values=[0.5, 1.0, 2.0, 5.0]):
    """
    Demonstrate cross-validation to select optimal shrinkage parameter.
    """
    print("\n" + "="*70)
    print("Cross-Validation for Optimal Shrinkage Parameter")
    print("="*70)

    n = len(phenotypes)
    n_folds = 5
    fold_size = n // n_folds

    # Shuffle indices
    np.random.seed(123)
    indices = np.random.permutation(n)

    cv_errors = {lambda_val: [] for lambda_val in lambda_values}

    for fold in range(n_folds):
        # Split data
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_folds - 1 else n
        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])

        y_train = phenotypes[train_indices]
        grm_train = grm[np.ix_(train_indices, train_indices)]

        for lambda_val in lambda_values:
            # Fit on training set
            result = fit_gblup_with_shrinkage(y_train, grm_train, lambda_val)

            # Predict test set (simplified - just use mean + random effect)
            blups_train = np.array(result.random_effects["u_genomic"])
            mean_pred = np.mean(y_train)

            # For test individuals, use 0 (could use GRM-based prediction in practice)
            y_pred = mean_pred
            y_test = phenotypes[test_indices]

            # Compute MSE
            mse = np.mean((y_test - y_pred) ** 2)
            cv_errors[lambda_val].append(mse)

    print("\nCross-validation results (Mean Squared Error):")
    print("-" * 40)
    for lambda_val in lambda_values:
        mean_mse = np.mean(cv_errors[lambda_val])
        std_mse = np.std(cv_errors[lambda_val])
        print(f"λ = {lambda_val:4.1f}: MSE = {mean_mse:.4f} ± {std_mse:.4f}")

    # Find optimal lambda
    mean_mses = {lam: np.mean(cv_errors[lam]) for lam in lambda_values}
    optimal_lambda = min(mean_mses, key=mean_mses.get)
    print(f"\nOptimal λ: {optimal_lambda}")

    return optimal_lambda


def main():
    print("="*70)
    print("Random Effects Shrinkage Example: Genomic Selection")
    print("="*70)

    # Generate synthetic data
    X, phenotypes, true_effects, grm = generate_synthetic_genomic_data(
        n_individuals=200,
        n_markers=500,
        h2=0.6
    )

    # Compare different shrinkage levels
    results = compare_shrinkage_levels(phenotypes, grm)

    # Demonstrate cross-validation
    optimal_lambda = demonstrate_cross_validation(X, phenotypes, grm)

    # Fit final model with optimal shrinkage
    print("\n" + "="*70)
    print(f"Final Model with Optimal Shrinkage (λ = {optimal_lambda})")
    print("="*70)

    final_result = fit_gblup_with_shrinkage(phenotypes, grm, optimal_lambda)

    print(f"\nFinal model fitted successfully!")
    print(f"Number of individuals: {len(phenotypes)}")
    print(f"Shrinkage parameter: λ = {optimal_lambda}")
    print(f"BLUP range: [{np.min(final_result.random_effects['u_genomic']):.3f}, "
          f"{np.max(final_result.random_effects['u_genomic']):.3f}]")

    # Key insight
    print("\n" + "="*70)
    print("Key Insight:")
    print("="*70)
    print("Higher lambda (λ > 1) → More shrinkage → Smaller BLUPs → Less overfitting")
    print("Lower lambda (λ < 1) → Less shrinkage → Larger BLUPs → More flexibility")
    print("\nFor genomic selection with many markers, moderate shrinkage (λ ≈ 2-5)")
    print("often improves prediction accuracy by reducing overfitting.")
    print("="*70)


if __name__ == "__main__":
    main()
