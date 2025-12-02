#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <string>

#include "libsemx/scaled_fixed_covariance.hpp"
#include "libsemx/likelihood_driver.hpp"
#include "libsemx/model_ir.hpp"
#include "libsemx/genomic_kernel.hpp"

TEST_CASE("GenomicRelationshipMatrix computes VanRaden GRM", "[genomic][grm]") {
    // Two individuals, two markers (row-major)
    std::vector<double> markers = {
        0.0, 1.0,
        2.0, 1.0
    };

    auto grm = libsemx::GenomicRelationshipMatrix::vanraden(markers, 2, 2);

    REQUIRE(grm.size() == 4);
    // Expected GRM = [[1, -1], [-1, 1]] after centering/normalization
    REQUIRE(grm[0] == Catch::Approx(1.0));
    REQUIRE(grm[1] == Catch::Approx(-1.0));
    REQUIRE(grm[2] == Catch::Approx(-1.0));
    REQUIRE(grm[3] == Catch::Approx(1.0));
}

TEST_CASE("GenomicRelationshipMatrix builds GxE Kronecker kernels", "[genomic][grm][kronecker]") {
    std::vector<double> g = {
        1.0, -0.5,
       -0.5, 1.0
    };
    std::vector<double> e = {
        2.0, 0.5,
        0.5, 1.0
    };

    auto kron = libsemx::GenomicRelationshipMatrix::kronecker(g, 2, e, 2);
    REQUIRE(kron.size() == 16);

    // Spot-check a few entries
    // Block (0,0) should be 1.0 * E
    REQUIRE(kron[0] == Catch::Approx(2.0));
    REQUIRE(kron[1] == Catch::Approx(0.5));
    REQUIRE(kron[4] == Catch::Approx(0.5));
    REQUIRE(kron[5] == Catch::Approx(1.0));

    // Block (1,0) should be -0.5 * E
    REQUIRE(kron[8] == Catch::Approx(-1.0));
    REQUIRE(kron[9] == Catch::Approx(-0.25));
    REQUIRE(kron[12] == Catch::Approx(-0.25));
    REQUIRE(kron[13] == Catch::Approx(-0.5));
}

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

TEST_CASE("LikelihoodDriver consumes GRM covariance", "[likelihood_driver][genomic]") {
    libsemx::LikelihoodDriver driver;
    libsemx::ModelIRBuilder builder;

    // Dummy variables to achieve Z = I
    builder.add_variable("id1", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("id2", libsemx::VariableKind::Observed, "gaussian");

    builder.add_variable("y", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("group", libsemx::VariableKind::Grouping);

    builder.add_covariance("cov_u", "grm", 2);
    builder.add_random_effect("re_u", {"group", "id1", "id2"}, "cov_u");

    auto model = builder.build();

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
    linear_predictors["y"] = {0.0, 0.0};

    std::unordered_map<std::string, std::vector<double>> dispersions;
    dispersions["y"] = {1.0, 1.0};

    std::unordered_map<std::string, std::vector<double>> covariance_parameters;
    covariance_parameters["cov_u"] = {1.5}; // variance scale for GRM

    std::unordered_map<std::string, std::vector<double>> status;
    std::unordered_map<std::string, std::vector<double>> extra_params;

    // Build GRM from markers
    std::vector<double> markers = {
        0.0, 1.0,
        2.0, 1.0
    };
    auto grm = libsemx::GenomicRelationshipMatrix::vanraden(markers, 2, 2);

    std::unordered_map<std::string, std::vector<std::vector<double>>> fixed_covariance_data;
    fixed_covariance_data["cov_u"] = {grm};

    // Expected V = 1.5 * GRM + I
    // GRM = [[1, -1], [-1, 1]]
    // V = [[2.5, -1.5], [-1.5, 2.5]]
    // loglik = -0.5 * (log|V| + y' V^{-1} y + n log(2*pi))
    // |V| = 4, V^{-1} = [[0.625, 0.375], [0.375, 0.625]], y' V^{-1} y = 4.625
    const double two_pi = 2.0 * std::acos(-1.0);
    double expected_loglik = -0.5 * (std::log(4.0) + 4.625 + 2.0 * std::log(two_pi));

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
