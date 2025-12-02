#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "libsemx/poisson_outcome.hpp"

#include <cmath>
#include <vector>

TEST_CASE("PoissonOutcome evaluates correctly", "[outcome][poisson]") {
    libsemx::PoissonOutcome family;

    SECTION("Evaluates log-likelihood and derivatives") {
        double y = 2.0;
        double eta = 0.5; // lambda = exp(0.5) approx 1.6487
        double lambda = std::exp(eta);
        
        auto eval = family.evaluate(y, eta, 1.0);

        // log_lik = y * eta - lambda - lgamma(y+1)
        double expected_ll = y * eta - lambda - std::lgamma(y + 1.0);
        
        REQUIRE_THAT(eval.log_likelihood, Catch::Matchers::WithinRel(expected_ll, 1e-6));
        REQUIRE_THAT(eval.first_derivative, Catch::Matchers::WithinRel(y - lambda, 1e-6));
        REQUIRE_THAT(eval.second_derivative, Catch::Matchers::WithinRel(-lambda, 1e-6));
        REQUIRE_THAT(eval.third_derivative, Catch::Matchers::WithinRel(-lambda, 1e-6));
    }

    SECTION("Throws on negative observed value") {
        REQUIRE_THROWS_AS(family.evaluate(-1.0, 0.0, 1.0), std::invalid_argument);
    }
    
    SECTION("Handles zero observed value") {
        double y = 0.0;
        double eta = 0.0; // lambda = 1
        auto eval = family.evaluate(y, eta, 1.0);
        
        // log_lik = 0 - 1 - log(1) = -1
        REQUIRE_THAT(eval.log_likelihood, Catch::Matchers::WithinRel(-1.0, 1e-6));
        REQUIRE_THAT(eval.first_derivative, Catch::Matchers::WithinRel(-1.0, 1e-6));
    }
}
