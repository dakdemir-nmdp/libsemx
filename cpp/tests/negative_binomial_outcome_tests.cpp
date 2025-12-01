#include "libsemx/negative_binomial_outcome.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("NegativeBinomialOutcome evaluates log-likelihood", "[negative_binomial_outcome]") {
    libsemx::NegativeBinomialOutcome nb;

    SECTION("Basic case") {
        const double observed = 2.0;
        const double linear_predictor = 0.0;  // log(mu) = 0 => mu = 1
        const double dispersion = 1.0;  // k = 1

        const auto eval = nb.evaluate(observed, linear_predictor, dispersion);

        REQUIRE(std::isfinite(eval.log_likelihood));
        REQUIRE(std::isfinite(eval.first_derivative));
        REQUIRE(std::isfinite(eval.second_derivative));
    }

    SECTION("Zero observed") {
        const double observed = 0.0;
        const double linear_predictor = 0.0;
        const double dispersion = 1.0;

        const auto eval = nb.evaluate(observed, linear_predictor, dispersion);

        REQUIRE(std::isfinite(eval.log_likelihood));
        REQUIRE(std::isfinite(eval.first_derivative));
        REQUIRE(std::isfinite(eval.second_derivative));
    }

    SECTION("Check derivatives consistency") {
        // For mu=1, k=1, y=2: score should be (2-1)*1 / (1 + 1^2/1) = 1*1/2 = 0.5
        const double observed = 2.0;
        const double linear_predictor = 0.0;
        const double dispersion = 1.0;

        const auto eval = nb.evaluate(observed, linear_predictor, dispersion);

        REQUIRE_THAT(eval.first_derivative, Catch::Matchers::WithinRel(0.5, 1e-10));
    }
}

TEST_CASE("NegativeBinomialOutcome rejects invalid inputs", "[negative_binomial_outcome]") {
    libsemx::NegativeBinomialOutcome nb;

    REQUIRE_THROWS_AS(nb.evaluate(-1.0, 0.0, 1.0), std::invalid_argument);
    REQUIRE_THROWS_AS(nb.evaluate(1.0, 0.0, 0.0), std::invalid_argument);
    REQUIRE_THROWS_AS(nb.evaluate(1.0, 0.0, -1.0), std::invalid_argument);
}

TEST_CASE("NegativeBinomialOutcome default dispersion", "[negative_binomial_outcome]") {
    libsemx::NegativeBinomialOutcome nb;

    REQUIRE(nb.default_dispersion(1) == 1.0);
    REQUIRE(nb.has_dispersion());
}