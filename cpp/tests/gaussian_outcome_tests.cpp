#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <stdexcept>

#include "libsemx/gaussian_outcome.hpp"

TEST_CASE("GaussianOutcome returns analytic derivatives", "[outcome][gaussian]") {
    libsemx::GaussianOutcome family;
    const double y = 1.5;
    const double eta = 1.0;
    const double sigma2 = 0.25;  // variance

    const auto eval = family.evaluate(y, eta, sigma2);

    const double residual = y - eta;
    const double inv_var = 1.0 / sigma2;
    const double expected_loglik = -0.5 * (std::log(6.28318530717958647692 * sigma2) + residual * residual * inv_var);
    const double expected_grad = residual * inv_var;
    const double expected_hess = -inv_var;

    REQUIRE(eval.log_likelihood == Catch::Approx(expected_loglik));
    REQUIRE(eval.first_derivative == Catch::Approx(expected_grad));
    REQUIRE(eval.second_derivative == Catch::Approx(expected_hess));
}

TEST_CASE("GaussianOutcome rejects non-positive dispersion", "[outcome][gaussian]") {
    libsemx::GaussianOutcome family;
    REQUIRE_THROWS_AS(family.evaluate(0.0, 0.0, 0.0), std::invalid_argument);
    REQUIRE_THROWS_AS(family.evaluate(0.0, 0.0, -1.0), std::invalid_argument);
}

TEST_CASE("GaussianOutcome default dispersion defined", "[outcome][gaussian]") {
    libsemx::GaussianOutcome family;
    REQUIRE(family.default_dispersion(10) == Catch::Approx(1.0));
    REQUIRE_THROWS_AS(family.default_dispersion(0), std::invalid_argument);
}
