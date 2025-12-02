#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>
#include <unordered_map>
#include <string>
#include <cmath>

#include "libsemx/multi_kernel_covariance.hpp"
#include "libsemx/likelihood_driver.hpp"
#include "libsemx/model_ir.hpp"

TEST_CASE("MultiKernelCovariance materializes weighted sum", "[covariance][mkl]") {
    // 2x2 Identity matrix
    std::vector<double> K1 = {1.0, 0.0, 0.0, 1.0};
    // 2x2 Ones matrix
    std::vector<double> K2 = {1.0, 1.0, 1.0, 1.0};
    
    libsemx::MultiKernelCovariance cov({K1, K2}, 2);

    REQUIRE(cov.dimension() == 2);
    REQUIRE(cov.parameter_count() == 3); // sigma^2, w1, w2

    SECTION("Equal weights") {
        // sigma^2 = 1.0, w1 = 0.5, w2 = 0.5
        // Matrix = 1.0 * (0.5 * K1 + 0.5 * K2)
        // K1 = [[1,0],[0,1]], K2 = [[1,1],[1,1]]
        // Sum = [[1, 0.5], [0.5, 1]]
        std::vector<double> params = {1.0, 0.5, 0.5};
        auto mat = cov.materialize(params);
        REQUIRE(mat[0] == Catch::Approx(1.0));
        REQUIRE(mat[1] == Catch::Approx(0.5));
        REQUIRE(mat[2] == Catch::Approx(0.5));
        REQUIRE(mat[3] == Catch::Approx(1.0));
    }

    SECTION("Zero variance") {
        std::vector<double> params = {0.0, 0.5, 0.5};
        auto mat = cov.materialize(params);
        for (double v : mat) {
            REQUIRE(v == Catch::Approx(0.0));
        }
    }
    
    SECTION("Invalid parameters") {
        REQUIRE_THROWS_AS(cov.materialize({1.0, 0.5}), std::invalid_argument); // Too few
        REQUIRE_THROWS_AS(cov.materialize({1.0, 0.5, 0.5, 0.1}), std::invalid_argument); // Too many
        REQUIRE_THROWS_AS(cov.materialize({-1.0, 0.5, 0.5}), std::invalid_argument); // Negative variance
    }
}

TEST_CASE("LikelihoodDriver handles multi_kernel covariance", "[likelihood_driver][mkl]") {
    libsemx::LikelihoodDriver driver;
    libsemx::ModelIRBuilder builder;

    builder.add_variable("group", libsemx::VariableKind::Grouping);
    
    // Define covariance structure
    builder.add_covariance("cov_u", "multi_kernel", 2);

    // Define random effect
    // Dummy variables for random slopes to match dimension 2
    builder.add_variable("id1", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("id2", libsemx::VariableKind::Observed, "gaussian");
    
    // Outcome variable LAST
    builder.add_variable("y", libsemx::VariableKind::Observed, "gaussian");
    
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
    linear_predictors["y"] = {0.0, 0.0};

    std::unordered_map<std::string, std::vector<double>> dispersions;
    dispersions["y"] = {1.0, 1.0};

    std::unordered_map<std::string, std::vector<double>> covariance_parameters;
    // sigma^2 = 2.0, w1 = 0.5, w2 = 0.5
    // K1 = I, K2 = I (for simplicity)
    // V_u = 2.0 * (0.5*I + 0.5*I) = 2.0 * I
    // Total V = 2.0*I + 1.0*I = 3.0*I
    covariance_parameters["cov_u"] = {2.0, 0.5, 0.5};

    std::unordered_map<std::string, std::vector<double>> status;
    std::unordered_map<std::string, std::vector<double>> extra_params;

    // Fixed covariance data (2 kernels)
    std::unordered_map<std::string, std::vector<std::vector<double>>> fixed_covariance_data;
    std::vector<double> I = {1.0, 0.0, 0.0, 1.0};
    fixed_covariance_data["cov_u"] = {I, I};

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

TEST_CASE("MultiKernelCovariance: Simplex weights", "[covariance][mkl]") {
    std::vector<double> K1 = {1.0, 0.0, 0.0, 1.0};
    std::vector<double> K2 = {1.0, 1.0, 1.0, 1.0};
    
    libsemx::MultiKernelCovariance cov({K1, K2}, 2, true); // simplex_weights = true

    REQUIRE(cov.parameter_count() == 3);

    // Params: sigma=2.0, theta1=0.0, theta2=0.0
    // Weights: exp(0)/(exp(0)+exp(0)) = 0.5 each.
    std::vector<double> params = {2.0, 0.0, 0.0};
    auto mat = cov.materialize(params);
    
    // Matrix = 2.0 * (0.5*K1 + 0.5*K2) = K1 + K2 = [[2, 1], [1, 2]]
    REQUIRE_THAT(mat[0], Catch::Matchers::WithinRel(2.0));
    REQUIRE_THAT(mat[1], Catch::Matchers::WithinRel(1.0));
    REQUIRE_THAT(mat[2], Catch::Matchers::WithinRel(1.0));
    REQUIRE_THAT(mat[3], Catch::Matchers::WithinRel(2.0));
    
    // Gradients
    auto analytic = cov.parameter_gradients(params);
    double eps = 1e-6;
    for (size_t i = 0; i < params.size(); ++i) {
        std::vector<double> p_plus = params;
        std::vector<double> p_minus = params;
        p_plus[i] += eps;
        p_minus[i] -= eps;
        
        auto m_plus = cov.materialize(p_plus);
        auto m_minus = cov.materialize(p_minus);
        
        for (size_t j = 0; j < m_plus.size(); ++j) {
            double num_grad = (m_plus[j] - m_minus[j]) / (2 * eps);
            REQUIRE_THAT(analytic[i][j], Catch::Matchers::WithinRel(num_grad, 1e-4));
        }
    }
}
