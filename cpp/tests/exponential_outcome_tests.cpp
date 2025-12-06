#include "libsemx/exponential_outcome.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

TEST_CASE("ExponentialOutcome survival contributions match the Weibull special case", "[survival][exponential_outcome]") {
    libsemx::ExponentialOutcome expo;

    const double t = 2.0;
    const double eta = 0.35;
    const double lambda = std::exp(eta);
    const double z = t / lambda;

    SECTION("Event contribution") {
        const auto eval = expo.evaluate(t, eta, 1.0, 1.0);
        const double expected_ll = -eta - z;
        const double expected_grad = z - 1.0;
        const double expected_hess = -z;
        const double expected_third = z;

        REQUIRE_THAT(eval.log_likelihood, Catch::Matchers::WithinRel(expected_ll));
        REQUIRE_THAT(eval.first_derivative, Catch::Matchers::WithinRel(expected_grad));
        REQUIRE_THAT(eval.second_derivative, Catch::Matchers::WithinRel(expected_hess));
        REQUIRE_THAT(eval.third_derivative, Catch::Matchers::WithinRel(expected_third));
    }

    SECTION("Censored contribution") {
        const auto eval = expo.evaluate(t, eta, 1.0, 0.0);
        const double expected_ll = -z;
        const double expected_grad = z;
        const double expected_hess = -z;
        const double expected_third = z;

        REQUIRE_THAT(eval.log_likelihood, Catch::Matchers::WithinRel(expected_ll));
        REQUIRE_THAT(eval.first_derivative, Catch::Matchers::WithinRel(expected_grad));
        REQUIRE_THAT(eval.second_derivative, Catch::Matchers::WithinRel(expected_hess));
        REQUIRE_THAT(eval.third_derivative, Catch::Matchers::WithinRel(expected_third));
    }

    SECTION("Dispersion contract") {
        REQUIRE_FALSE(expo.has_dispersion());
        REQUIRE(expo.default_dispersion(3) == 1.0);
    }

    SECTION("Input validation") {
        REQUIRE_THROWS_AS(expo.evaluate(0.0, eta, 1.0, 1.0), std::runtime_error);
        REQUIRE_THROWS_AS(expo.evaluate(-1.0, eta, 1.0, 1.0), std::runtime_error);
    }
}
