#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "libsemx/weibull_outcome.hpp"

#include <cmath>

TEST_CASE("WeibullOutcome evaluates log-likelihood", "[outcome][weibull]") {
    libsemx::WeibullOutcome weibull;

    SECTION("Event (uncensored)") {
        double t = 2.0;
        double eta = 0.5;
        double k = 1.5;
        double status = 1.0;

        auto result = weibull.evaluate(t, eta, k, status);

        // Manual calculation
        // z = (t * exp(-eta))^k = (2 * exp(-0.5))^1.5
        double z = std::pow(2.0 * std::exp(-0.5), 1.5);
        // LL = log(k) + (k-1)log(t) - k*eta - z
        double expected_ll = std::log(k) + (k - 1.0) * std::log(t) - k * eta - z;
        
        // Grad = k(z - 1)
        double expected_grad = k * (z - 1.0);
        
        // Hess = -k^2 * z
        double expected_hess = -k * k * z;

        REQUIRE_THAT(result.log_likelihood, Catch::Matchers::WithinRel(expected_ll));
        REQUIRE_THAT(result.first_derivative, Catch::Matchers::WithinRel(expected_grad));
        REQUIRE_THAT(result.second_derivative, Catch::Matchers::WithinRel(expected_hess));
    }

    SECTION("Censored") {
        double t = 2.0;
        double eta = 0.5;
        double k = 1.5;
        double status = 0.0;

        auto result = weibull.evaluate(t, eta, k, status);

        // Manual calculation
        // z = (t * exp(-eta))^k
        double z = std::pow(2.0 * std::exp(-0.5), 1.5);
        // LL = -z
        double expected_ll = -z;
        
        // Grad = k(z - 0) = kz
        double expected_grad = k * z;
        
        // Hess = -k^2 * z
        double expected_hess = -k * k * z;

        REQUIRE_THAT(result.log_likelihood, Catch::Matchers::WithinRel(expected_ll));
        REQUIRE_THAT(result.first_derivative, Catch::Matchers::WithinRel(expected_grad));
        REQUIRE_THAT(result.second_derivative, Catch::Matchers::WithinRel(expected_hess));
    }
}

TEST_CASE("WeibullOutcome rejects invalid inputs", "[outcome][weibull]") {
    libsemx::WeibullOutcome weibull;

    REQUIRE_THROWS_AS( weibull.evaluate(0.0, 0.0, 1.0, 1.0), std::runtime_error );
    REQUIRE_THROWS_AS( weibull.evaluate(-1.0, 0.0, 1.0, 1.0), std::runtime_error );
    REQUIRE_THROWS_AS( weibull.evaluate(1.0, 0.0, 0.0, 1.0), std::runtime_error );
    REQUIRE_THROWS_AS( weibull.evaluate(1.0, 0.0, -1.0, 1.0), std::runtime_error );
}

TEST_CASE("WeibullOutcome default dispersion", "[outcome][weibull]") {
    libsemx::WeibullOutcome weibull;
    REQUIRE(weibull.default_dispersion(10) == 1.0);
}
