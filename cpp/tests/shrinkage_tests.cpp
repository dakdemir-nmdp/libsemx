#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "libsemx/model_ir.hpp"
#include "libsemx/likelihood_driver.hpp"
#include <cmath>

using namespace libsemx;
using Catch::Matchers::WithinAbs;

TEST_CASE("Random effects shrinkage - basic functionality", "[shrinkage]") {
    SECTION("Lambda parameter is stored in RandomEffectSpec") {
        ModelIRBuilder builder;
        builder.add_variable("y", VariableKind::Observed, "gaussian");
        builder.add_variable("group", VariableKind::Grouping);
        builder.add_covariance("cov1", "unstructured", 2);

        // Test default lambda (1.0 = no shrinkage)
        builder.add_random_effect("re1", {"group", "y"}, "cov1");
        auto model = builder.build();
        REQUIRE(model.random_effects.size() == 1);
        REQUIRE(model.random_effects[0].lambda == 1.0);

        // Test custom lambda
        ModelIRBuilder builder2;
        builder2.add_variable("y", VariableKind::Observed, "gaussian");
        builder2.add_variable("group", VariableKind::Grouping);
        builder2.add_covariance("cov1", "unstructured", 2);
        builder2.add_random_effect("re1", {"group", "y"}, "cov1", 2.5);
        auto model2 = builder2.build();
        REQUIRE(model2.random_effects[0].lambda == 2.5);
    }

    SECTION("Lambda must be positive") {
        ModelGraph graph;
        graph.add_variable("y", VariableKind::Observed, "gaussian");
        graph.add_variable("group", VariableKind::Grouping);
        graph.add_covariance("cov1", "unstructured", 2);

        REQUIRE_THROWS_AS(
            graph.add_random_effect("re1", {"group", "y"}, "cov1", 0.0),
            std::invalid_argument
        );

        REQUIRE_THROWS_AS(
            graph.add_random_effect("re1", {"group", "y"}, "cov1", -1.0),
            std::invalid_argument
        );
    }
}

TEST_CASE("Shrinkage affects likelihood correctly - Laplace mode", "[shrinkage]") {
    // Create a simple GLMM with shrinkage
    // y ~ Binomial(p), logit(p) = intercept + u[group]
    // u[group] ~ N(0, sigma^2 / lambda)

    ModelIRBuilder builder;
    builder.add_variable("y", VariableKind::Observed, "binomial");
    builder.add_variable("_intercept", VariableKind::Exogenous);
    builder.add_variable("group", VariableKind::Grouping);

    builder.add_edge(EdgeKind::Regression, "_intercept", "y", "beta0");
    builder.register_parameter("beta0", 0.0);

    builder.add_covariance("group_cov", "unstructured", 1);

    double lambda1 = 1.0;  // No shrinkage
    double lambda2 = 2.0;  // 2x shrinkage

    builder.add_random_effect("re_group", {"group", "y"}, "group_cov", lambda1);

    auto model1 = builder.build();

    // Create data: 4 groups with 5 observations each
    std::unordered_map<std::string, std::vector<double>> data;
    data["y"] = {0,1,1,0,1,  1,1,0,1,0,  0,0,1,1,1,  1,0,1,0,1};  // 20 observations
    data["_intercept"] = std::vector<double>(20, 1.0);
    data["group"] = {1,1,1,1,1,  2,2,2,2,2,  3,3,3,3,3,  4,4,4,4,4};

    // Evaluate likelihood with lambda=1.0 (no shrinkage)
    LikelihoodDriver driver;

    std::unordered_map<std::string, std::vector<double>> linear_preds1;
    linear_preds1["y"] = std::vector<double>(20, 0.0);  // Start at 0

    std::unordered_map<std::string, std::vector<double>> dispersions1;
    dispersions1["y"] = std::vector<double>(20, 1.0);

    std::unordered_map<std::string, std::vector<double>> cov_params1;
    cov_params1["group_cov"] = {0.5};  // Variance

    double loglik1 = driver.evaluate_model_loglik(
        model1, data, linear_preds1, dispersions1, cov_params1);

    // Now test with lambda=2.0 (more shrinkage)
    ModelIRBuilder builder2;
    builder2.add_variable("y", VariableKind::Observed, "binomial");
    builder2.add_variable("_intercept", VariableKind::Exogenous);
    builder2.add_variable("group", VariableKind::Grouping);
    builder2.add_edge(EdgeKind::Regression, "_intercept", "y", "beta0");
    builder2.register_parameter("beta0", 0.0);
    builder2.add_covariance("group_cov", "unstructured", 1);
    builder2.add_random_effect("re_group", {"group", "y"}, "group_cov", lambda2);

    auto model2 = builder2.build();

    double loglik2 = driver.evaluate_model_loglik(
        model2, data, linear_preds1, dispersions1, cov_params1);

    // With shrinkage, the effective prior variance is reduced (sigma^2/lambda),
    // which should affect the log-likelihood
    // More shrinkage (lambda=2) should give different likelihood than no shrinkage (lambda=1)
    REQUIRE(loglik1 != loglik2);

    // The log-prior contribution changes with lambda:
    // log p(u) = -0.5 * [q*log(2π) + log|G/lambda| + lambda * u^T G^{-1} u]
    //          = -0.5 * [q*log(2π) + log|G| - q*log(lambda) + lambda * u^T G^{-1} u]
    // So higher lambda should generally give different (usually lower) likelihood
    // unless random effects are very close to zero
}

TEST_CASE("Shrinkage affects BLUP estimation", "[shrinkage]") {
    // Create a simple Gaussian model with random intercepts
    // y = intercept + u[group] + e
    // u[group] ~ N(0, sigma^2_u / lambda)

    ModelIRBuilder builder;
    builder.add_variable("y", VariableKind::Observed, "gaussian");
    builder.add_variable("_intercept", VariableKind::Exogenous);
    builder.add_variable("group", VariableKind::Grouping);

    builder.add_edge(EdgeKind::Regression, "_intercept", "y", "beta0");
    builder.register_parameter("beta0", 5.0);

    builder.add_covariance("group_cov", "unstructured", 1);
    builder.add_random_effect("re_group", {"group", "y"}, "group_cov", 2.0);  // lambda = 2.0

    auto model = builder.build();

    // Simple data: 3 groups, 2 obs each
    std::unordered_map<std::string, std::vector<double>> data;
    data["y"] = {5.5, 5.3,  6.2, 6.1,  4.8, 4.9};  // Group means: ~5.4, ~6.15, ~4.85
    data["_intercept"] = std::vector<double>(6, 1.0);
    data["group"] = {1, 1,  2, 2,  3, 3};

    std::unordered_map<std::string, std::vector<double>> linear_preds;
    linear_preds["y"] = std::vector<double>(6, 5.0);  // Fixed effect prediction

    std::unordered_map<std::string, std::vector<double>> dispersions;
    dispersions["y"] = std::vector<double>(6, 0.1);  // Small residual variance

    std::unordered_map<std::string, std::vector<double>> cov_params;
    cov_params["group_cov"] = {0.5};  // Random effect variance

    LikelihoodDriver driver;
    auto random_effects = driver.compute_random_effects(
        model, data, linear_preds, dispersions, cov_params);

    REQUIRE(random_effects.count("re_group") > 0);
    const auto& blups = random_effects.at("re_group");

    // With shrinkage (lambda=2), BLUPs should be shrunk more towards zero
    // compared to no shrinkage (lambda=1)
    // The shrinkage formula: BLUP = (1 / (1 + lambda*sigma_e^2/sigma_u^2)) * group_mean_deviation

    // All BLUPs should be non-zero but relatively small due to shrinkage
    REQUIRE(blups.size() == 3);
    for (double blup : blups) {
        REQUIRE(std::abs(blup) < 1.0);  // Heavily shrunk
    }
}

TEST_CASE("Shrinkage works with spectral decomposition", "[shrinkage][spectral]") {
    // Test that shrinkage parameter is correctly applied in spectral mode
    // This requires a Gaussian outcome with a genomic/scaled_fixed kernel

    ModelIRBuilder builder;
    builder.add_variable("y", VariableKind::Observed, "gaussian");
    builder.add_variable("_intercept", VariableKind::Exogenous);

    builder.add_edge(EdgeKind::Regression, "_intercept", "y", "beta0");
    builder.register_parameter("beta0", 0.0);

    builder.add_covariance("kernel", "scaled_fixed", 5);
    builder.add_random_effect("re_genetic", {"y"}, "kernel", 1.5);  // lambda = 1.5

    auto model = builder.build();

    // Create simple data
    std::unordered_map<std::string, std::vector<double>> data;
    data["y"] = {1.2, 0.8, 1.5, 0.9, 1.1};
    data["_intercept"] = {1, 1, 1, 1, 1};

    // Create a simple 5x5 kernel matrix (identity for simplicity)
    std::vector<std::vector<double>> kernel_data;
    std::vector<double> kernel_flat(25, 0.0);
    for (int i = 0; i < 5; ++i) {
        kernel_flat[i * 5 + i] = 1.0;
    }
    kernel_data.push_back(kernel_flat);

    std::unordered_map<std::string, std::vector<std::vector<double>>> fixed_cov_data;
    fixed_cov_data["kernel"] = kernel_data;

    std::unordered_map<std::string, std::vector<double>> linear_preds;
    linear_preds["y"] = {1.0, 1.0, 1.0, 1.0, 1.0};

    std::unordered_map<std::string, std::vector<double>> dispersions;
    dispersions["y"] = {1.0, 1.0, 1.0, 1.0, 1.0};

    std::unordered_map<std::string, std::vector<double>> cov_params;
    cov_params["kernel"] = {0.5};  // Genetic variance
    cov_params["y_dispersion"] = {0.2};  // Residual variance

    LikelihoodDriver driver;
    double loglik = driver.evaluate_model_loglik(
        model, data, linear_preds, dispersions, cov_params,
        {}, {}, fixed_cov_data, EstimationMethod::ML);

    // Should not throw and should return a finite value
    REQUIRE(std::isfinite(loglik));

    // Test gradient as well
    auto gradient = driver.evaluate_model_gradient(
        model, data, linear_preds, dispersions, cov_params,
        {}, {}, fixed_cov_data, EstimationMethod::ML);

    // Should have gradients for both variance components
    REQUIRE(gradient.count("kernel") > 0);
    REQUIRE(std::isfinite(gradient.at("kernel")));
}

TEST_CASE("Shrinkage parameter validation in spectral mode", "[shrinkage][spectral]") {
    // Verify that lambda is extracted and used correctly in spectral decomposition

    ModelIRBuilder builder;
    builder.add_variable("y", VariableKind::Observed, "gaussian");
    builder.add_variable("_intercept", VariableKind::Exogenous);
    builder.add_edge(EdgeKind::Regression, "_intercept", "y", "beta0");
    builder.register_parameter("beta0", 0.0);
    builder.add_covariance("kernel", "scaled_fixed", 3);

    double lambda_val = 3.0;
    builder.add_random_effect("re_genetic", {"y"}, "kernel", lambda_val);

    auto model = builder.build();

    // Verify lambda is stored correctly
    REQUIRE(model.random_effects.size() == 1);
    REQUIRE(model.random_effects[0].lambda == lambda_val);

    // Create simple data and kernel
    std::unordered_map<std::string, std::vector<double>> data;
    data["y"] = {1.0, 2.0, 1.5};
    data["_intercept"] = {1, 1, 1};

    std::vector<std::vector<double>> kernel_data;
    std::vector<double> kernel_flat = {
        1.0, 0.5, 0.3,
        0.5, 1.0, 0.4,
        0.3, 0.4, 1.0
    };
    kernel_data.push_back(kernel_flat);

    std::unordered_map<std::string, std::vector<std::vector<double>>> fixed_cov_data;
    fixed_cov_data["kernel"] = kernel_data;

    std::unordered_map<std::string, std::vector<double>> linear_preds;
    linear_preds["y"] = {1.5, 1.5, 1.5};

    std::unordered_map<std::string, std::vector<double>> dispersions;
    dispersions["y"] = {1.0, 1.0, 1.0};

    std::unordered_map<std::string, std::vector<double>> cov_params;
    cov_params["kernel"] = {1.0};  // Genetic variance
    cov_params["y_dispersion"] = {0.5};  // Residual variance

    LikelihoodDriver driver;

    // Evaluate with lambda = 3.0
    double loglik_shrunk = driver.evaluate_model_loglik(
        model, data, linear_preds, dispersions, cov_params,
        {}, {}, fixed_cov_data, EstimationMethod::ML);

    // Compare with lambda = 1.0 (no shrinkage)
    ModelIRBuilder builder_no_shrink;
    builder_no_shrink.add_variable("y", VariableKind::Observed, "gaussian");
    builder_no_shrink.add_variable("_intercept", VariableKind::Exogenous);
    builder_no_shrink.add_edge(EdgeKind::Regression, "_intercept", "y", "beta0");
    builder_no_shrink.register_parameter("beta0", 0.0);
    builder_no_shrink.add_covariance("kernel", "scaled_fixed", 3);
    builder_no_shrink.add_random_effect("re_genetic", {"y"}, "kernel", 1.0);

    auto model_no_shrink = builder_no_shrink.build();

    double loglik_no_shrink = driver.evaluate_model_loglik(
        model_no_shrink, data, linear_preds, dispersions, cov_params,
        {}, {}, fixed_cov_data, EstimationMethod::ML);

    // Likelihoods should differ when lambda differs
    REQUIRE(loglik_shrunk != loglik_no_shrink);
    REQUIRE(std::isfinite(loglik_shrunk));
    REQUIRE(std::isfinite(loglik_no_shrink));
}
