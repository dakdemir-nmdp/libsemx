#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "libsemx/covariance_structure.hpp"

namespace {

std::vector<double> finite_difference_gradient(const libsemx::CovarianceStructure& cov,
                                               const std::vector<double>& params,
                                               std::size_t param_index) {
    const double base = params[param_index];
    const double step = std::max(1e-6, std::abs(base) * 1e-6);
    auto plus = params;
    auto minus = params;
    plus[param_index] = base + step;
    minus[param_index] = base - step;
    const auto matrix_plus = cov.materialize(plus);
    const auto matrix_minus = cov.materialize(minus);
    std::vector<double> grad(matrix_plus.size());
    for (std::size_t i = 0; i < grad.size(); ++i) {
        grad[i] = (matrix_plus[i] - matrix_minus[i]) / (2.0 * step);
    }
    return grad;
}

void expect_gradient_match(const libsemx::CovarianceStructure& cov,
                           const std::vector<double>& params,
                           double tolerance = 1e-5) {
    const auto grads = cov.parameter_gradients(params);
    for (std::size_t p = 0; p < grads.size(); ++p) {
        const auto numeric = finite_difference_gradient(cov, params, p);
        for (std::size_t idx = 0; idx < numeric.size(); ++idx) {
            REQUIRE(grads[p][idx] == Catch::Approx(numeric[idx]).margin(tolerance));
        }
    }
}

}  // namespace

TEST_CASE("UnstructuredCovariance materializes symmetric matrix", "[covariance]") {
    libsemx::UnstructuredCovariance cov(3);
    REQUIRE(cov.dimension() == 3);
    REQUIRE(cov.parameter_count() == 6);

    const std::vector<double> params{1.0, 0.2, 2.0, 0.3, 0.4, 3.0};
    const auto matrix = cov.materialize(params);

    REQUIRE(matrix.size() == 9);
    REQUIRE(matrix[0] == Catch::Approx(1.0));
    REQUIRE(matrix[1] == Catch::Approx(0.2));
    REQUIRE(matrix[2] == Catch::Approx(0.3));
    REQUIRE(matrix[3] == Catch::Approx(0.2));
    REQUIRE(matrix[4] == Catch::Approx(2.0));
    REQUIRE(matrix[5] == Catch::Approx(0.4));
    REQUIRE(matrix[6] == Catch::Approx(0.3));
    REQUIRE(matrix[7] == Catch::Approx(0.4));
    REQUIRE(matrix[8] == Catch::Approx(3.0));
}

TEST_CASE("UnstructuredCovariance rejects invalid parameters", "[covariance]") {
    libsemx::UnstructuredCovariance cov(2);
    REQUIRE_THROWS_AS(cov.materialize({1.0, 0.3, -0.1}), std::invalid_argument);
    REQUIRE_THROWS_AS(cov.materialize({1.0, 0.2}), std::invalid_argument);
}

TEST_CASE("DiagonalCovariance enforces positive diagonal", "[covariance]") {
    libsemx::DiagonalCovariance cov(2);
    const std::vector<double> params{1.5, 0.75};
    const auto matrix = cov.materialize(params);
    REQUIRE(matrix[0] == Catch::Approx(1.5));
    REQUIRE(matrix[1] == Catch::Approx(0.0));
    REQUIRE(matrix[2] == Catch::Approx(0.0));
    REQUIRE(matrix[3] == Catch::Approx(0.75));

    REQUIRE_THROWS_AS(cov.materialize({1.0, 0.0}), std::invalid_argument);
    REQUIRE_THROWS_AS(cov.materialize({1.0}), std::invalid_argument);
}

TEST_CASE("CompoundSymmetryCovariance materializes block structure", "[covariance]") {
    libsemx::CompoundSymmetryCovariance cov(3);
    const std::vector<double> params{0.6, 0.3};
    const auto matrix = cov.materialize(params);
    REQUIRE(matrix[0] == Catch::Approx(0.9));
    REQUIRE(matrix[1] == Catch::Approx(0.3));
    REQUIRE(matrix[3] == Catch::Approx(0.3));
    REQUIRE(matrix[4] == Catch::Approx(0.9));
    REQUIRE(matrix[8] == Catch::Approx(0.9));
    expect_gradient_match(cov, params);
}

TEST_CASE("AR1Covariance produces decaying correlations", "[covariance]") {
    libsemx::AR1Covariance cov(4);
    const std::vector<double> params{1.2, 3.0};
    const auto matrix = cov.materialize(params);
    REQUIRE(matrix[0] == Catch::Approx(1.2));
    REQUIRE(matrix[1] == Catch::Approx(0.6));
    REQUIRE(matrix[2] == Catch::Approx(0.3));
    REQUIRE(matrix[5] == Catch::Approx(1.2));
    expect_gradient_match(cov, params, 5e-5);
}

TEST_CASE("ToeplitzCovariance honors lag parameters", "[covariance]") {
    libsemx::ToeplitzCovariance cov(3);
    const std::vector<double> params{2.0, 3.0, 1.0};
    const auto matrix = cov.materialize(params);
    REQUIRE(matrix[0] == Catch::Approx(2.0));
    REQUIRE(matrix[1] == Catch::Approx(1.0));
    REQUIRE(matrix[2] == Catch::Approx(0.5));
    REQUIRE(matrix[3] == Catch::Approx(1.0));
    REQUIRE(matrix[4] == Catch::Approx(2.0));
    REQUIRE(matrix[5] == Catch::Approx(1.0));
    expect_gradient_match(cov, params, 2e-4);
}

TEST_CASE("FactorAnalyticCovariance builds low-rank structure", "[covariance]") {
    libsemx::FactorAnalyticCovariance cov(3, 1);
    const std::vector<double> params{0.8, 0.4, -0.2, 0.5, 0.6, 0.7};
    const auto matrix = cov.materialize(params);
    REQUIRE(matrix[0] == Catch::Approx(0.8 * 0.8 + 0.5));
    REQUIRE(matrix[1] == Catch::Approx(0.8 * 0.4));
    REQUIRE(matrix[4] == Catch::Approx(0.4 * 0.4 + 0.6));
    REQUIRE(matrix[8] == Catch::Approx(0.7 + 0.04));
    expect_gradient_match(cov, params, 5e-5);
}

TEST_CASE("DiagonalCovariance supports sparse materialization", "[covariance][sparse]") {
    libsemx::DiagonalCovariance cov(3);
    REQUIRE(cov.is_sparse());
    
    const std::vector<double> params{1.0, 2.0, 3.0};
    
    // Check dense materialization
    const auto dense = cov.materialize(params);
    REQUIRE(dense[0] == 1.0);
    REQUIRE(dense[4] == 2.0);
    REQUIRE(dense[8] == 3.0);
    REQUIRE(dense[1] == 0.0);
    
    // Check sparse materialization
    const auto sparse = cov.materialize_sparse(params);
    REQUIRE(sparse.rows() == 3);
    REQUIRE(sparse.cols() == 3);
    REQUIRE(sparse.nonZeros() == 3);
    
    REQUIRE(sparse.coeff(0, 0) == 1.0);
    REQUIRE(sparse.coeff(1, 1) == 2.0);
    REQUIRE(sparse.coeff(2, 2) == 3.0);
    REQUIRE(sparse.coeff(0, 1) == 0.0);
}

TEST_CASE("UnstructuredCovariance default sparse fallback works", "[covariance][sparse]") {
    libsemx::UnstructuredCovariance cov(2);
    REQUIRE_FALSE(cov.is_sparse());
    
    const std::vector<double> params{1.0, 0.5, 2.0};
    const auto sparse = cov.materialize_sparse(params);
    
    REQUIRE(sparse.rows() == 2);
    REQUIRE(sparse.cols() == 2);
    REQUIRE(sparse.coeff(0, 0) == 1.0);
    REQUIRE(sparse.coeff(1, 0) == 0.5);
    REQUIRE(sparse.coeff(0, 1) == 0.5);
    REQUIRE(sparse.coeff(1, 1) == 4.25);
}

