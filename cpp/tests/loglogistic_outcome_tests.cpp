#include "libsemx/loglogistic_outcome.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <stdexcept>

TEST_CASE("LogLogisticOutcome survival contributions evaluate analytic expressions", "[survival][loglogistic_outcome]") {
    libsemx::LogLogisticOutcome outcome;

    const double t = 1.7;
    const double eta = 0.25;
    const double gamma = 1.3;
    const double log_t = std::log(t);
    const double diff = log_t - eta;
    const double u = std::exp(gamma * diff);
    const double one_plus_u = 1.0 + u;
    const double log1pu = std::log1p(u);

    SECTION("Event") {
        const auto eval = outcome.evaluate(t, eta, gamma, 1.0);
        const double expected_ll = std::log(gamma) + gamma * diff - log_t - 2.0 * log1pu;
        const double expected_grad = gamma * (u - 1.0) / one_plus_u;
        const double expected_hess = -2.0 * gamma * gamma * u / (one_plus_u * one_plus_u);
        const double expected_third = 2.0 * gamma * gamma * gamma * u * (1.0 - u) /
                                      (one_plus_u * one_plus_u * one_plus_u);

        REQUIRE_THAT(eval.log_likelihood, Catch::Matchers::WithinRel(expected_ll));
        REQUIRE_THAT(eval.first_derivative, Catch::Matchers::WithinRel(expected_grad));
        REQUIRE_THAT(eval.second_derivative, Catch::Matchers::WithinRel(expected_hess));
        REQUIRE_THAT(eval.third_derivative, Catch::Matchers::WithinRel(expected_third));
    }

    SECTION("Censored") {
        const auto eval = outcome.evaluate(t, eta, gamma, 0.0);
        const double expected_ll = -log1pu;
        const double expected_grad = gamma * u / one_plus_u;
        const double expected_hess = -gamma * gamma * u / (one_plus_u * one_plus_u);
        const double expected_third = gamma * gamma * gamma * u * (1.0 - u) /
                                      (one_plus_u * one_plus_u * one_plus_u);

        REQUIRE_THAT(eval.log_likelihood, Catch::Matchers::WithinRel(expected_ll));
        REQUIRE_THAT(eval.first_derivative, Catch::Matchers::WithinRel(expected_grad));
        REQUIRE_THAT(eval.second_derivative, Catch::Matchers::WithinRel(expected_hess));
        REQUIRE_THAT(eval.third_derivative, Catch::Matchers::WithinRel(expected_third));
    }

    SECTION("Validation and dispersion") {
        REQUIRE(outcome.has_dispersion());
        REQUIRE(outcome.default_dispersion(4) == 1.0);
        REQUIRE_THROWS_AS(outcome.evaluate(0.0, eta, gamma, 1.0), std::invalid_argument);
        REQUIRE_THROWS_AS(outcome.evaluate(t, eta, 0.0, 1.0), std::invalid_argument);
    }
}
