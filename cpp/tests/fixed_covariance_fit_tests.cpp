#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "libsemx/likelihood_driver.hpp"
#include "libsemx/model_ir.hpp"

#include <vector>
#include <unordered_map>
#include <cmath>

TEST_CASE("LikelihoodDriver fits model with fixed covariance data", "[optimization][mixed][fixed_cov]") {
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("cluster", libsemx::VariableKind::Grouping);
    
    // Scaled fixed covariance: G = sigma_g^2 * K
    // We provide K via fixed_covariance_data.
    // We estimate sigma_g^2.
    builder.add_covariance("K", "scaled_fixed", 2); // 2x2 matrix
    builder.add_random_effect("u", {"cluster"}, "K");

    auto model = builder.build();

    // Data: 1 cluster, 2 obs
    // Cluster 1: y = [1, 2]
    std::vector<double> y = {1.0, 2.0};
    std::vector<double> cluster = {1.0, 1.0};
    
    std::unordered_map<std::string, std::vector<double>> data = {
        {"y", y},
        {"cluster", cluster}
    };
    
    // Fixed covariance matrix K (2x2 identity for simplicity)
    // Flattened: [1, 0, 0, 1]
    std::vector<double> K = {1.0, 0.0, 0.0, 1.0};
    std::unordered_map<std::string, std::vector<std::vector<double>>> fixed_cov_data = {
        {"K", {K}}
    };
    
    libsemx::LikelihoodDriver driver;
    libsemx::OptimizationOptions options;
    options.max_iterations = 1; // Just check initialization and first step
    
    // If fixed_covariance_data is not passed correctly, this will throw "Missing fixed covariance data for: K"
    REQUIRE_NOTHROW(driver.fit(model, data, options, "lbfgs", fixed_cov_data));
}
