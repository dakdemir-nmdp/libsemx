#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "libsemx/ordinal_outcome.hpp"
#include <cmath>
#include <vector>

using namespace libsemx;

// Helper for standard normal CDF
double normal_cdf(double x) {
    return 0.5 * std::erfc(-x * 0.707106781186547524401);
}

TEST_CASE("OrdinalOutcome: Basic Evaluation", "[ordinal]") {
    OrdinalOutcome outcome;
    // 3 categories: 0, 1, 2.
    // 2 thresholds: tau_1 = -0.5, tau_2 = 0.5
    std::vector<double> thresholds = {-0.5, 0.5};
    
    SECTION("Category 0 (y=0)") {
        // P(y=0) = Phi(tau_1 - eta) - Phi(-inf - eta) = Phi(-0.5 - 0) = Phi(-0.5)
        double eta = 0.0;
        auto result = outcome.evaluate(0.0, eta, 1.0, 1.0, thresholds);
        
        double expected_prob = normal_cdf(-0.5);
        REQUIRE_THAT(result.log_likelihood, Catch::Matchers::WithinRel(std::log(expected_prob), 1e-6));
    }

    SECTION("Category 1 (y=1)") {
        // P(y=1) = Phi(tau_2 - eta) - Phi(tau_1 - eta) = Phi(0.5) - Phi(-0.5)
        double eta = 0.0;
        auto result = outcome.evaluate(1.0, eta, 1.0, 1.0, thresholds);
        
        double expected_prob = normal_cdf(0.5) - normal_cdf(-0.5);
        REQUIRE_THAT(result.log_likelihood, Catch::Matchers::WithinRel(std::log(expected_prob), 1e-6));
    }

    SECTION("Category 2 (y=2)") {
        // P(y=2) = Phi(inf - eta) - Phi(tau_2 - eta) = 1 - Phi(0.5)
        double eta = 0.0;
        auto result = outcome.evaluate(2.0, eta, 1.0, 1.0, thresholds);
        
        double expected_prob = 1.0 - normal_cdf(0.5);
        REQUIRE_THAT(result.log_likelihood, Catch::Matchers::WithinRel(std::log(expected_prob), 1e-6));
    }
}

TEST_CASE("OrdinalOutcome: Derivatives", "[ordinal]") {
    OrdinalOutcome outcome;
    std::vector<double> thresholds = {0.0}; // 2 categories: 0, 1. Cutpoint at 0.
    
    // Check derivatives numerically
    double eta = 0.5;
    double h = 1e-5;
    double y = 0.0; // Category 0: P = Phi(0 - 0.5) = Phi(-0.5)
    
    auto res = outcome.evaluate(y, eta, 1.0, 1.0, thresholds);
    auto res_plus = outcome.evaluate(y, eta + h, 1.0, 1.0, thresholds);
    auto res_minus = outcome.evaluate(y, eta - h, 1.0, 1.0, thresholds);
    
    double num_grad = (res_plus.log_likelihood - res_minus.log_likelihood) / (2 * h);
    double num_hess = (res_plus.log_likelihood - 2 * res.log_likelihood + res_minus.log_likelihood) / (h * h);
    
    REQUIRE_THAT(res.first_derivative, Catch::Matchers::WithinRel(num_grad, 1e-4));
    REQUIRE_THAT(res.second_derivative, Catch::Matchers::WithinRel(num_hess, 1e-4));
}

TEST_CASE("OrdinalOutcome: Error Handling", "[ordinal]") {
    OrdinalOutcome outcome;
    std::vector<double> thresholds = {0.0, 1.0};
    
    SECTION("Missing thresholds") {
        REQUIRE_THROWS_AS(outcome.evaluate(0.0, 0.0, 1.0, 1.0, {}), std::invalid_argument);
    }
    
    SECTION("Unsorted thresholds") {
        std::vector<double> bad_thresholds = {1.0, 0.0};
        REQUIRE_THROWS_AS(outcome.evaluate(0.0, 0.0, 1.0, 1.0, bad_thresholds), std::invalid_argument);
    }
    
    SECTION("Invalid observed value (negative)") {
        REQUIRE_THROWS_AS(outcome.evaluate(-1.0, 0.0, 1.0, 1.0, thresholds), std::invalid_argument);
    }
    
    SECTION("Invalid observed value (non-integer)") {
        REQUIRE_THROWS_AS(outcome.evaluate(0.5, 0.0, 1.0, 1.0, thresholds), std::invalid_argument);
    }
    
    SECTION("Observed value out of range") {
        // 2 thresholds => 3 categories (0, 1, 2). 3 is invalid.
        REQUIRE_THROWS_AS(outcome.evaluate(3.0, 0.0, 1.0, 1.0, thresholds), std::invalid_argument);
    }
}
