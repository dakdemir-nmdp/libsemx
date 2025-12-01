#include "libsemx/binomial_outcome.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("BinomialOutcome evaluates log-likelihood and derivatives", "[binomial_outcome]") {
    libsemx::BinomialOutcome binomial;

    SECTION("Success case y=1") {
        const double observed = 1.0;
        const double linear_predictor = 0.0;  // logit(p) = 0 => p = 0.5
        const double dispersion = 1.0;  // ignored

        const auto eval = binomial.evaluate(observed, linear_predictor, dispersion);

        // p = 0.5, loglik = 1*log(0.5) + 0*log(0.5) = log(0.5)
        const double expected_loglik = std::log(0.5);
        REQUIRE_THAT(eval.log_likelihood, Catch::Matchers::WithinRel(expected_loglik));

        // first_derivative = y - p = 1 - 0.5 = 0.5
        REQUIRE_THAT(eval.first_derivative, Catch::Matchers::WithinRel(0.5));

        // second_derivative = -p*(1-p) = -0.5*0.5 = -0.25
        REQUIRE_THAT(eval.second_derivative, Catch::Matchers::WithinRel(-0.25));
    }

    SECTION("Success case y=0") {
        const double observed = 0.0;
        const double linear_predictor = 0.0;
        const double dispersion = 1.0;

        const auto eval = binomial.evaluate(observed, linear_predictor, dispersion);

        const double expected_loglik = std::log(0.5);
        REQUIRE_THAT(eval.log_likelihood, Catch::Matchers::WithinRel(expected_loglik));

        REQUIRE_THAT(eval.first_derivative, Catch::Matchers::WithinRel(-0.5));
        REQUIRE_THAT(eval.second_derivative, Catch::Matchers::WithinRel(-0.25));
    }

    SECTION("Extreme linear_predictor") {
        const double observed = 1.0;
        const double linear_predictor = 10.0;  // p ≈ 1
        const double dispersion = 1.0;

        const auto eval = binomial.evaluate(observed, linear_predictor, dispersion);

        // Should not be -inf
        REQUIRE(std::isfinite(eval.log_likelihood));
        REQUIRE(std::isfinite(eval.first_derivative));
        REQUIRE(std::isfinite(eval.second_derivative));
    }
}

TEST_CASE("BinomialOutcome rejects invalid observed values", "[binomial_outcome]") {
    libsemx::BinomialOutcome binomial;

    REQUIRE_THROWS_AS(binomial.evaluate(0.5, 0.0, 1.0), std::invalid_argument);
    REQUIRE_THROWS_AS(binomial.evaluate(-1.0, 0.0, 1.0), std::invalid_argument);
    REQUIRE_THROWS_AS(binomial.evaluate(2.0, 0.0, 1.0), std::invalid_argument);
}

TEST_CASE("BinomialOutcome default dispersion", "[binomial_outcome]") {
    libsemx::BinomialOutcome binomial;

    REQUIRE(binomial.default_dispersion(1) == 1.0);
    REQUIRE_FALSE(binomial.has_dispersion());
}