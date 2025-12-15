#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "libsemx/likelihood_driver.hpp"
#include "libsemx/model_ir.hpp"

#include <cmath>
#include <vector>
#include <unordered_map>

TEST_CASE("LikelihoodDriver evaluates REML for Gaussian Random Intercept", "[reml][mixed]") {
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("cluster", libsemx::VariableKind::Grouping);
    // Use Exogenous to avoid LikelihoodDriver treating it as an outcome, but provide data for it.
    builder.add_variable("const", libsemx::VariableKind::Exogenous);
    builder.add_variable("u_cluster", libsemx::VariableKind::Latent);

    // Random intercept: u ~ N(0, tau^2)
    builder.add_covariance("tau_sq", "diagonal", 1);
    builder.add_random_effect("u_cluster", {"cluster"}, "tau_sq");

    // Fixed effect: y ~ const
    builder.add_edge(libsemx::EdgeKind::Regression, "const", "y", "beta_0");
    // Connect random effect to outcome
    builder.add_edge(libsemx::EdgeKind::Regression, "u_cluster", "y", "1");

    auto model = builder.build();

    // Data: 2 clusters, 2 obs each
    std::vector<double> y = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> cluster = {1.0, 1.0, 2.0, 2.0};
    std::vector<double> constant = {1.0, 1.0, 1.0, 1.0};
    
    // Fixed effects: beta=0 => preds=0
    std::vector<double> preds = {0.0, 0.0, 0.0, 0.0};
    
    // Residual variance: sigma^2 = 1
    std::vector<double> disps = {1.0, 1.0, 1.0, 1.0};

    // Random effect variance: tau^2 = 1
    std::vector<double> tau_sq_params = {1.0};

    std::unordered_map<std::string, std::vector<double>> data = {
        {"y", y},
        {"cluster", cluster},
        {"const", constant}
    };
    std::unordered_map<std::string, std::vector<double>> linear_predictors = {
        {"y", preds}
    };
    std::unordered_map<std::string, std::vector<double>> dispersions = {
        {"y", disps}
    };
    std::unordered_map<std::string, std::vector<double>> cov_params = {
        {"tau_sq", tau_sq_params}
    };

    libsemx::LikelihoodDriver driver;
    
    // 1. Calculate ML
    double ml_loglik = driver.evaluate_model_loglik(model, data, linear_predictors, dispersions, cov_params, {}, {}, {}, libsemx::EstimationMethod::ML);

    // 2. Calculate REML
    double reml_loglik = driver.evaluate_model_loglik(model, data, linear_predictors, dispersions, cov_params, {}, {}, {}, libsemx::EstimationMethod::REML);

    // Expected Difference
    // V_i = [[2, 1], [1, 2]]
    // V_i^-1 = 1/3 * [[2, -1], [-1, 2]]
    // X_i = [1, 1]^T
    // X_i^T V_i^-1 X_i = 1/3 * [1, 1] * [1, 1]^T = 2/3.
    // Wait, earlier I calculated 2/3 per group?
    // 1^T V^-1 1 = sum(V^-1).
    // sum([[2, -1], [-1, 2]]) = 2-1-1+2 = 2.
    // So 1/3 * 2 = 2/3.
    // Total X' V^-1 X = 2/3 + 2/3 = 4/3.
    
    double log_det_XtVinvX = std::log(4.0/3.0);
    double log_2pi = std::log(2.0 * 3.14159265358979323846);
    double p = 1.0;
    
    double expected_diff = -0.5 * log_det_XtVinvX + 0.5 * p * log_2pi;
    
    REQUIRE_THAT(reml_loglik - ml_loglik, Catch::Matchers::WithinRel(expected_diff, 1e-5));
}
