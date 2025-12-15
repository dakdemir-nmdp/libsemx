#include "libsemx/covariance_structure.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>

using namespace libsemx;

// ============================================================================
// ARMA(1,1) Covariance Tests
// ============================================================================

TEST_CASE("ARMA11Covariance: Basic construction", "[covariance][arma11]") {
    ARMA11Covariance cov(5);
    REQUIRE(cov.dimension() == 5);
    REQUIRE(cov.parameter_count() == 3);  // variance, rho, lambda
}

TEST_CASE("ARMA11Covariance: Single dimension", "[covariance][arma11]") {
    ARMA11Covariance cov(1);
    REQUIRE(cov.dimension() == 1);
    REQUIRE(cov.parameter_count() == 1);  // Only variance

    std::vector<double> params = {2.0};
    auto matrix = cov.materialize(params);
    REQUIRE(matrix.size() == 1);
    REQUIRE_THAT(matrix[0], Catch::Matchers::WithinAbs(2.0, 1e-10));
}

TEST_CASE("ARMA11Covariance: Materialize matrix", "[covariance][arma11]") {
    ARMA11Covariance cov(5);

    // Parameters: variance=1.0, rho=0.0 (tanh(0)=0), lambda=0.0 (tanh(0)=0)
    std::vector<double> params = {1.0, 0.0, 0.0};
    auto matrix = cov.materialize(params);

    // With rho=0 and lambda=0, only diagonal should be non-zero
    REQUIRE(matrix.size() == 25);
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            if (i == j) {
                REQUIRE_THAT(matrix[i * 5 + j], Catch::Matchers::WithinAbs(1.0, 1e-10));
            } else {
                REQUIRE_THAT(matrix[i * 5 + j], Catch::Matchers::WithinAbs(0.0, 1e-10));
            }
        }
    }
}

TEST_CASE("ARMA11Covariance: Non-zero autocorrelation", "[covariance][arma11]") {
    ARMA11Covariance cov(4);

    // Parameters: variance=2.0, rho=atanh(0.5), lambda=atanh(0.8)
    const double rho_raw = std::atanh(0.5);  // Transform to get rho=0.5
    const double lambda_raw = std::atanh(0.8);  // Transform to get lambda=0.8
    std::vector<double> params = {2.0, rho_raw, lambda_raw};
    auto matrix = cov.materialize(params);

    // Check diagonal
    for (size_t i = 0; i < 4; ++i) {
        REQUIRE_THAT(matrix[i * 4 + i], Catch::Matchers::WithinAbs(2.0, 1e-10));
    }

    // Check lag 1: variance * lambda = 2.0 * 0.8 = 1.6
    REQUIRE_THAT(matrix[0 * 4 + 1], Catch::Matchers::WithinAbs(1.6, 1e-10));
    REQUIRE_THAT(matrix[1 * 4 + 0], Catch::Matchers::WithinAbs(1.6, 1e-10));

    // Check lag 2: variance * lambda * rho^1 = 2.0 * 0.8 * 0.5 = 0.8
    REQUIRE_THAT(matrix[0 * 4 + 2], Catch::Matchers::WithinAbs(0.8, 1e-10));
    REQUIRE_THAT(matrix[2 * 4 + 0], Catch::Matchers::WithinAbs(0.8, 1e-10));

    // Check lag 3: variance * lambda * rho^2 = 2.0 * 0.8 * 0.25 = 0.4
    REQUIRE_THAT(matrix[0 * 4 + 3], Catch::Matchers::WithinAbs(0.4, 1e-10));
    REQUIRE_THAT(matrix[3 * 4 + 0], Catch::Matchers::WithinAbs(0.4, 1e-10));
}

TEST_CASE("ARMA11Covariance: Symmetry", "[covariance][arma11]") {
    ARMA11Covariance cov(6);
    std::vector<double> params = {1.5, 0.3, -0.2};
    auto matrix = cov.materialize(params);

    for (size_t i = 0; i < 6; ++i) {
        for (size_t j = 0; j < 6; ++j) {
            REQUIRE_THAT(matrix[i * 6 + j], Catch::Matchers::WithinAbs(matrix[j * 6 + i], 1e-10));
        }
    }
}

TEST_CASE("ARMA11Covariance: Parameter validation", "[covariance][arma11]") {
    ARMA11Covariance cov(4);

    // Negative variance should throw
    std::vector<double> bad_params = {-1.0, 0.0, 0.0};
    REQUIRE_THROWS_AS(cov.materialize(bad_params), std::invalid_argument);

    // Zero variance should throw
    bad_params = {0.0, 0.0, 0.0};
    REQUIRE_THROWS_AS(cov.materialize(bad_params), std::invalid_argument);
}

TEST_CASE("ARMA11Covariance: Gradient computation", "[covariance][arma11]") {
    ARMA11Covariance cov(4);
    std::vector<double> params = {1.0, 0.2, -0.1};

    auto grads = cov.parameter_gradients(params);
    REQUIRE(grads.size() == 3);
    REQUIRE(grads[0].size() == 16);
    REQUIRE(grads[1].size() == 16);
    REQUIRE(grads[2].size() == 16);

    // Verify gradients via finite differences
    const double eps = 1e-7;
    for (size_t p = 0; p < 3; ++p) {
        auto params_plus = params;
        auto params_minus = params;
        params_plus[p] += eps;
        params_minus[p] -= eps;

        auto matrix_plus = cov.materialize(params_plus);
        auto matrix_minus = cov.materialize(params_minus);

        for (size_t i = 0; i < 16; ++i) {
            double fd_grad = (matrix_plus[i] - matrix_minus[i]) / (2.0 * eps);
            REQUIRE_THAT(grads[p][i], Catch::Matchers::WithinAbs(fd_grad, 1e-5));
        }
    }
}

// ============================================================================
// RBF Kernel Tests
// ============================================================================

TEST_CASE("RBFKernel: Basic construction", "[covariance][rbf]") {
    // 1D coordinates: 0, 1, 2, 3
    std::vector<double> coords = {0.0, 1.0, 2.0, 3.0};
    RBFKernel kernel(coords, 4);

    REQUIRE(kernel.dimension() == 4);
    REQUIRE(kernel.parameter_count() == 2);  // variance, lengthscale
}

TEST_CASE("RBFKernel: 2D coordinates", "[covariance][rbf]") {
    // 2D coordinates: (0,0), (1,0), (0,1), (1,1)
    std::vector<double> coords = {
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0
    };
    RBFKernel kernel(coords, 4);

    REQUIRE(kernel.dimension() == 4);
    REQUIRE(kernel.parameter_count() == 2);
}

TEST_CASE("RBFKernel: Materialize matrix", "[covariance][rbf]") {
    // 1D coordinates: 0, 1, 2
    std::vector<double> coords = {0.0, 1.0, 2.0};
    RBFKernel kernel(coords, 3);

    // Parameters: variance=1.0, log(inv_lengthscale)=0 => inv_lengthscale=1
    std::vector<double> params = {1.0, 0.0};
    auto matrix = kernel.materialize(params);

    // K(x, x') = exp(-d²)
    // K(0, 0) = exp(0) = 1
    REQUIRE_THAT(matrix[0 * 3 + 0], Catch::Matchers::WithinAbs(1.0, 1e-10));

    // K(0, 1) = exp(-1²) = exp(-1) ≈ 0.3679
    REQUIRE_THAT(matrix[0 * 3 + 1], Catch::Matchers::WithinAbs(std::exp(-1.0), 1e-10));

    // K(0, 2) = exp(-2²) = exp(-4) ≈ 0.0183
    REQUIRE_THAT(matrix[0 * 3 + 2], Catch::Matchers::WithinAbs(std::exp(-4.0), 1e-10));

    // K(1, 2) = exp(-1²) = exp(-1)
    REQUIRE_THAT(matrix[1 * 3 + 2], Catch::Matchers::WithinAbs(std::exp(-1.0), 1e-10));
}

TEST_CASE("RBFKernel: Variance scaling", "[covariance][rbf]") {
    std::vector<double> coords = {0.0, 1.0};
    RBFKernel kernel(coords, 2);

    // variance=2.0, inv_lengthscale=1
    std::vector<double> params = {2.0, 0.0};
    auto matrix = kernel.materialize(params);

    // All values should be scaled by variance
    REQUIRE_THAT(matrix[0 * 2 + 0], Catch::Matchers::WithinAbs(2.0, 1e-10));
    REQUIRE_THAT(matrix[0 * 2 + 1], Catch::Matchers::WithinAbs(2.0 * std::exp(-1.0), 1e-10));
}

TEST_CASE("RBFKernel: Symmetry", "[covariance][rbf]") {
    std::vector<double> coords = {0.0, 0.5, 1.0, 1.5};
    RBFKernel kernel(coords, 4);
    std::vector<double> params = {1.5, 0.3};
    auto matrix = kernel.materialize(params);

    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            REQUIRE_THAT(matrix[i * 4 + j], Catch::Matchers::WithinAbs(matrix[j * 4 + i], 1e-10));
        }
    }
}

TEST_CASE("RBFKernel: Parameter validation", "[covariance][rbf]") {
    std::vector<double> coords = {0.0, 1.0};
    RBFKernel kernel(coords, 2);

    // Negative variance should throw
    std::vector<double> bad_params = {-1.0, 0.0};
    REQUIRE_THROWS_AS(kernel.materialize(bad_params), std::invalid_argument);
}

TEST_CASE("RBFKernel: Gradient computation", "[covariance][rbf]") {
    std::vector<double> coords = {0.0, 1.0, 2.0};
    RBFKernel kernel(coords, 3);
    std::vector<double> params = {1.0, 0.0};

    auto grads = kernel.parameter_gradients(params);
    REQUIRE(grads.size() == 2);

    // Verify gradients via finite differences
    const double eps = 1e-7;
    for (size_t p = 0; p < 2; ++p) {
        auto params_plus = params;
        auto params_minus = params;
        params_plus[p] += eps;
        params_minus[p] -= eps;

        auto matrix_plus = kernel.materialize(params_plus);
        auto matrix_minus = kernel.materialize(params_minus);

        for (size_t i = 0; i < 9; ++i) {
            double fd_grad = (matrix_plus[i] - matrix_minus[i]) / (2.0 * eps);
            REQUIRE_THAT(grads[p][i], Catch::Matchers::WithinAbs(fd_grad, 1e-5));
        }
    }
}

// ============================================================================
// Exponential Kernel Tests
// ============================================================================

TEST_CASE("ExponentialKernel: Basic construction", "[covariance][exponential]") {
    std::vector<double> coords = {0.0, 1.0, 2.0, 3.0};
    ExponentialKernel kernel(coords, 4);

    REQUIRE(kernel.dimension() == 4);
    REQUIRE(kernel.parameter_count() == 2);
}

TEST_CASE("ExponentialKernel: Materialize matrix", "[covariance][exponential]") {
    // 1D coordinates: 0, 1, 2
    std::vector<double> coords = {0.0, 1.0, 2.0};
    ExponentialKernel kernel(coords, 3);

    // Parameters: variance=1.0, log(inv_lengthscale)=0 => inv_lengthscale=1
    std::vector<double> params = {1.0, 0.0};
    auto matrix = kernel.materialize(params);

    // K(x, x') = exp(-d)
    // K(0, 0) = exp(0) = 1
    REQUIRE_THAT(matrix[0 * 3 + 0], Catch::Matchers::WithinAbs(1.0, 1e-10));

    // K(0, 1) = exp(-1) ≈ 0.3679
    REQUIRE_THAT(matrix[0 * 3 + 1], Catch::Matchers::WithinAbs(std::exp(-1.0), 1e-10));

    // K(0, 2) = exp(-2) ≈ 0.1353
    REQUIRE_THAT(matrix[0 * 3 + 2], Catch::Matchers::WithinAbs(std::exp(-2.0), 1e-10));

    // K(1, 2) = exp(-1)
    REQUIRE_THAT(matrix[1 * 3 + 2], Catch::Matchers::WithinAbs(std::exp(-1.0), 1e-10));
}

TEST_CASE("ExponentialKernel: Variance scaling", "[covariance][exponential]") {
    std::vector<double> coords = {0.0, 1.0};
    ExponentialKernel kernel(coords, 2);

    // variance=2.5, inv_lengthscale=1
    std::vector<double> params = {2.5, 0.0};
    auto matrix = kernel.materialize(params);

    REQUIRE_THAT(matrix[0 * 2 + 0], Catch::Matchers::WithinAbs(2.5, 1e-10));
    REQUIRE_THAT(matrix[0 * 2 + 1], Catch::Matchers::WithinAbs(2.5 * std::exp(-1.0), 1e-10));
}

TEST_CASE("ExponentialKernel: Symmetry", "[covariance][exponential]") {
    std::vector<double> coords = {0.0, 0.5, 1.0, 1.5};
    ExponentialKernel kernel(coords, 4);
    std::vector<double> params = {1.5, 0.3};
    auto matrix = kernel.materialize(params);

    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            REQUIRE_THAT(matrix[i * 4 + j], Catch::Matchers::WithinAbs(matrix[j * 4 + i], 1e-10));
        }
    }
}

TEST_CASE("ExponentialKernel: Gradient computation", "[covariance][exponential]") {
    std::vector<double> coords = {0.0, 1.0, 2.0};
    ExponentialKernel kernel(coords, 3);
    std::vector<double> params = {1.0, 0.0};

    auto grads = kernel.parameter_gradients(params);
    REQUIRE(grads.size() == 2);

    // Verify gradients via finite differences
    const double eps = 1e-7;
    for (size_t p = 0; p < 2; ++p) {
        auto params_plus = params;
        auto params_minus = params;
        params_plus[p] += eps;
        params_minus[p] -= eps;

        auto matrix_plus = kernel.materialize(params_plus);
        auto matrix_minus = kernel.materialize(params_minus);

        for (size_t i = 0; i < 9; ++i) {
            double fd_grad = (matrix_plus[i] - matrix_minus[i]) / (2.0 * eps);
            REQUIRE_THAT(grads[p][i], Catch::Matchers::WithinAbs(fd_grad, 1e-5));
        }
    }
}

TEST_CASE("ExponentialKernel: 2D spatial coordinates", "[covariance][exponential]") {
    // 2D grid: (0,0), (1,0), (0,1)
    std::vector<double> coords = {
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0
    };
    ExponentialKernel kernel(coords, 3);
    std::vector<double> params = {1.0, 0.0};  // inv_lengthscale = 1
    auto matrix = kernel.materialize(params);

    // Distance from (0,0) to (1,0) is 1
    REQUIRE_THAT(matrix[0 * 3 + 1], Catch::Matchers::WithinAbs(std::exp(-1.0), 1e-10));

    // Distance from (0,0) to (0,1) is 1
    REQUIRE_THAT(matrix[0 * 3 + 2], Catch::Matchers::WithinAbs(std::exp(-1.0), 1e-10));

    // Distance from (1,0) to (0,1) is sqrt(2)
    REQUIRE_THAT(matrix[1 * 3 + 2], Catch::Matchers::WithinAbs(std::exp(-std::sqrt(2.0)), 1e-10));
}

// ============================================================================
// Comparison: RBF vs Exponential
// ============================================================================

TEST_CASE("RBF vs Exponential: Decay rates", "[covariance][rbf][exponential]") {
    std::vector<double> coords = {0.0, 1.0, 2.0, 3.0};
    RBFKernel rbf(coords, 4);
    ExponentialKernel exp_kernel(coords, 4);

    std::vector<double> params = {1.0, 0.0};  // Same parameters
    auto rbf_matrix = rbf.materialize(params);
    auto exp_matrix = exp_kernel.materialize(params);

    // RBF decays faster than exponential at larger distances
    // At distance 1: RBF = exp(-1) = 0.368, Exp = exp(-1) = 0.368
    REQUIRE_THAT(rbf_matrix[0 * 4 + 1], Catch::Matchers::WithinAbs(exp_matrix[0 * 4 + 1], 1e-10));

    // At distance 2: RBF = exp(-4) = 0.018, Exp = exp(-2) = 0.135
    // RBF should be smaller
    REQUIRE(rbf_matrix[0 * 4 + 2] < exp_matrix[0 * 4 + 2]);

    // At distance 3: RBF = exp(-9) = 0.0001, Exp = exp(-3) = 0.050
    // RBF should be much smaller
    REQUIRE(rbf_matrix[0 * 4 + 3] < exp_matrix[0 * 4 + 3]);
}
