#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <vector>

#include "libsemx/covariance_structure.hpp"

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
