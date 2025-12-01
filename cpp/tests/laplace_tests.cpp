#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "libsemx/likelihood_driver.hpp"
#include "libsemx/model_ir.hpp"

#include <cmath>
#include <vector>
#include <unordered_map>

TEST_CASE("LikelihoodDriver evaluates Laplace for Binomial GLMM", "[laplace][mixed]") {
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "binomial");
    builder.add_variable("cluster", libsemx::VariableKind::Grouping);
    
    // Random intercept: u ~ N(0, tau^2)
    builder.add_covariance("tau_sq", "diagonal", 1);
    builder.add_random_effect("u_cluster", {"cluster"}, "tau_sq");

    auto model = builder.build();

    // Data: 1 cluster, 2 obs
    // y = [1, 0]
    std::vector<double> y = {1.0, 0.0};
    std::vector<double> cluster = {1.0, 1.0};
    
    // Fixed effects: mu=0 => preds=0
    std::vector<double> preds = {0.0, 0.0};
    
    // Dispersion: 1 (ignored for binomial usually, but passed)
    std::vector<double> disps = {1.0, 1.0};

    // Random effect variance: tau^2 = 1
    std::vector<double> tau_sq_params = {1.0};

    std::unordered_map<std::string, std::vector<double>> data = {
        {"y", y},
        {"cluster", cluster}
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
    double loglik = driver.evaluate_model_loglik(model, data, linear_predictors, dispersions, cov_params);

    // Expected: -2 log(2) - 0.5 log(1.5)
    double expected = -2.0 * std::log(2.0) - 0.5 * std::log(1.5);
    
    REQUIRE_THAT(loglik, Catch::Matchers::WithinRel(expected, 1e-4));
}
