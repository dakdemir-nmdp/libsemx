#include "libsemx/lognormal_outcome.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace {
const double kLogSqrtTwoPi = 0.5 * std::log(2.0 * std::numbers::pi_v<double>);
double pdf(double z) {
    const double inv = 1.0 / std::sqrt(2.0 * std::numbers::pi_v<double>);
    return inv * std::exp(-0.5 * z * z);
}
double surv(double z) {
    return 0.5 * std::erfc(z / std::numbers::sqrt2_v<double>);
}
}

TEST_CASE("LognormalOutcome evaluates analytic log-likelihood pieces", "[survival][lognormal_outcome]") {
    libsemx::LognormalOutcome outcome;

    const double t = 2.3;
    const double eta = 0.4;
    const double sigma = 0.8;
    const double log_t = std::log(t);
    const double z = (log_t - eta) / sigma;

    SECTION("Event") {
        const auto eval = outcome.evaluate(t, eta, sigma, 1.0);
        const double expected_ll = -0.5 * z * z - log_t - std::log(sigma) - kLogSqrtTwoPi;
        const double expected_grad = z / sigma;
        const double expected_hess = -1.0 / (sigma * sigma);
        REQUIRE_THAT(eval.log_likelihood, Catch::Matchers::WithinRel(expected_ll));
        REQUIRE_THAT(eval.first_derivative, Catch::Matchers::WithinRel(expected_grad));
        REQUIRE_THAT(eval.second_derivative, Catch::Matchers::WithinRel(expected_hess));
        REQUIRE(eval.third_derivative == Catch::Approx(0.0));
    }

    SECTION("Censored") {
        const auto eval = outcome.evaluate(t, eta, sigma, 0.0);
        const double s = surv(z);
        const double density = pdf(z);
        const double expected_ll = std::log(s);
        const double expected_grad = density / (sigma * s);
        const double ratio_prime = (density / (s * s)) * (-z * s + density);
        const double expected_hess = -ratio_prime / (sigma * sigma);
        const double ratio_second = (density / (s * s * s)) * (((z * z - 1.0) * s * s) - 3.0 * z * density * s + 2.0 * density * density);
        const double expected_third = ratio_second / (sigma * sigma * sigma);

        REQUIRE_THAT(eval.log_likelihood, Catch::Matchers::WithinRel(expected_ll));
        REQUIRE_THAT(eval.first_derivative, Catch::Matchers::WithinRel(expected_grad));
        REQUIRE_THAT(eval.second_derivative, Catch::Matchers::WithinRel(expected_hess));
        REQUIRE_THAT(eval.third_derivative, Catch::Matchers::WithinRel(expected_third));
    }

    SECTION("Validation and dispersion") {
        REQUIRE(outcome.has_dispersion());
        REQUIRE(outcome.default_dispersion(5) == Catch::Approx(1.0));
        REQUIRE_THROWS_AS(outcome.evaluate(0.0, eta, sigma, 1.0), std::invalid_argument);
        REQUIRE_THROWS_AS(outcome.evaluate(t, eta, 0.0, 1.0), std::invalid_argument);
    }
}
