#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <unordered_map>
#include <string>

#include "libsemx/scaled_fixed_covariance.hpp"
#include "libsemx/likelihood_driver.hpp"
#include "libsemx/model_ir.hpp"

TEST_CASE("ScaledFixedCovariance materializes scaled matrix", "[covariance][grm]") {
    // 2x2 Identity matrix as fixed structure
    std::vector<double> fixed_matrix = {1.0, 0.0, 0.0, 1.0};
    libsemx::ScaledFixedCovariance cov(fixed_matrix, 2);

    REQUIRE(cov.dimension() == 2);
    REQUIRE(cov.parameter_count() == 1); // Only the scaling factor (variance)

    SECTION("Scale by 1.0") {
        std::vector<double> params = {1.0};
        auto mat = cov.materialize(params);
        REQUIRE(mat[0] == Catch::Approx(1.0));
        REQUIRE(mat[1] == Catch::Approx(0.0));
        REQUIRE(mat[2] == Catch::Approx(0.0));
        REQUIRE(mat[3] == Catch::Approx(1.0));
    }

    SECTION("Scale by 2.0") {
        std::vector<double> params = {2.0};
        auto mat = cov.materialize(params);
        REQUIRE(mat[0] == Catch::Approx(2.0));
        REQUIRE(mat[1] == Catch::Approx(0.0));
        REQUIRE(mat[2] == Catch::Approx(0.0));
        REQUIRE(mat[3] == Catch::Approx(2.0));
    }
    
    SECTION("Invalid parameters") {
        REQUIRE_THROWS_AS(cov.materialize({}), std::invalid_argument);
        REQUIRE_THROWS_AS(cov.materialize({1.0, 2.0}), std::invalid_argument);
        REQUIRE_THROWS_AS(cov.materialize({-1.0}), std::invalid_argument); // Variance must be non-negative
    }
}

TEST_CASE("LikelihoodDriver handles scaled_fixed covariance", "[likelihood_driver][grm]") {
    libsemx::LikelihoodDriver driver;
    libsemx::ModelIRBuilder builder;

    // Define a model with 2 observations in 1 group.
    // y = u + e
    // u ~ N(0, sigma_u^2 * K)
    // To achieve Z=I, we use 2 dummy variables as random slopes.
    // We must declare them as variables so ModelIRBuilder accepts them.
    // We add them BEFORE "y" so that LikelihoodDriver picks "y" as the outcome (it picks the last Observed variable).
    // We must provide a family, even if it's dummy, because ModelIRBuilder enforces it for Observed variables.
    builder.add_variable("id1", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("id2", libsemx::VariableKind::Observed, "gaussian");
    
    builder.add_variable("y", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("group", libsemx::VariableKind::Grouping);
    
    // Define covariance structure
    // Dimension 2 (for 2 observations)
    builder.add_covariance("cov_u", "scaled_fixed", 2);

    // Define random effect
    // Variables: "id1", "id2" (dummy variables)
    // Also need "group" to identify the group.
    // The first variable in the list that is of kind Grouping is the grouping variable.
    // So we need to add "group" to the list.
    builder.add_random_effect("re_u", {"group", "id1", "id2"}, "cov_u");

    auto model = builder.build();

    // Data
    std::vector<double> y = {1.0, 2.0};
    std::vector<double> group = {1.0, 1.0};
    std::vector<double> id1 = {1.0, 0.0};
    std::vector<double> id2 = {0.0, 1.0};
    
    std::unordered_map<std::string, std::vector<double>> data;
    data["y"] = y;
    data["group"] = group;
    data["id1"] = id1;
    data["id2"] = id2;

    std::unordered_map<std::string, std::vector<double>> linear_predictors;
    linear_predictors["y"] = {0.0, 0.0}; // Mean 0

    std::unordered_map<std::string, std::vector<double>> dispersions;
    dispersions["y"] = {1.0, 1.0}; // Residual variance 1.0

    std::unordered_map<std::string, std::vector<double>> covariance_parameters;
    covariance_parameters["cov_u"] = {2.0}; // Random effect variance = 2.0

    std::unordered_map<std::string, std::vector<double>> status;
    std::unordered_map<std::string, std::vector<double>> extra_params;

    // Fixed covariance data (K matrix)
    // 2x2 Identity
    std::unordered_map<std::string, std::vector<std::vector<double>>> fixed_covariance_data;
    fixed_covariance_data["cov_u"] = {{1.0, 0.0, 0.0, 1.0}};

    // Evaluate
    // V = sigma_u^2 * K + sigma_e^2 * I
    //   = 2.0 * I + 1.0 * I = 3.0 * I
    // LogLik = sum( -0.5 * (log(2*pi*3.0) + y_i^2 / 3.0) )
    // y = {1, 2}
    // LL = -0.5 * (log(6pi) + 1/3) - 0.5 * (log(6pi) + 4/3)
    //    = -log(6pi) - 0.5 * (5/3)
    
    double expected_loglik = -std::log(2 * 3.141592653589793 * 3.0) - 0.5 * (5.0 / 3.0);

    double loglik = driver.evaluate_model_loglik(
        model,
        data,
        linear_predictors,
        dispersions,
        covariance_parameters,
        status,
        extra_params,
        fixed_covariance_data,
        libsemx::EstimationMethod::ML
    );

    REQUIRE(loglik == Catch::Approx(expected_loglik));
}

TEST_CASE("LikelihoodDriver throws on missing fixed covariance data", "[likelihood_driver][grm]") {
    libsemx::LikelihoodDriver driver;
    libsemx::ModelIRBuilder builder;

    builder.add_variable("y", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("group", libsemx::VariableKind::Grouping);

    builder.add_covariance("cov_u", "scaled_fixed", 1);
    builder.add_random_effect("re_u", {"group"}, "cov_u"); // Random intercept

    auto model = builder.build();

    std::unordered_map<std::string, std::vector<double>> data;
    data["y"] = {1.0};
    data["group"] = {1.0};
    
    std::unordered_map<std::string, std::vector<double>> linear_predictors;
    linear_predictors["y"] = {0.0};
    
    std::unordered_map<std::string, std::vector<double>> dispersions;
    dispersions["y"] = {1.0};
    
    std::unordered_map<std::string, std::vector<double>> covariance_parameters;
    covariance_parameters["cov_u"] = {1.0};
    
    std::unordered_map<std::string, std::vector<double>> status;
    std::unordered_map<std::string, std::vector<double>> extra_params;
    
    // Empty fixed covariance data
    std::unordered_map<std::string, std::vector<std::vector<double>>> fixed_covariance_data;

    REQUIRE_THROWS_AS(driver.evaluate_model_loglik(
        model,
        data,
        linear_predictors,
        dispersions,
        covariance_parameters,
        status,
        extra_params,
        fixed_covariance_data,
        libsemx::EstimationMethod::ML
    ), std::runtime_error);
}
