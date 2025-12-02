#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "libsemx/likelihood_driver.hpp"
#include "libsemx/model_ir.hpp"
#include "libsemx/weibull_outcome.hpp"

#include <cmath>
#include <vector>
#include <unordered_map>

TEST_CASE("LikelihoodDriver evaluates survival Competing Risks Model", "[survival][competing_risks]") {
    // Competing risks: Two causes of failure.
    // Modeled as two cause-specific hazard functions.
    // T_1 ~ Weibull(k1, lambda1), T_2 ~ Weibull(k2, lambda2)
    // Observed T = min(T_1, T_2)
    // Status = 1 if T_1 < T_2, 2 if T_2 < T_1, 0 if censored (both > C)
    
    // We model this as two outcomes:
    // y1: time = T, status = (Status == 1)
    // y2: time = T, status = (Status == 2)
    
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y1", libsemx::VariableKind::Observed, "weibull");
    builder.add_variable("y2", libsemx::VariableKind::Observed, "weibull");
    
    auto model = builder.build();

    // Data: 3 subjects
    // 1. Event 1 at t=2.0
    // 2. Event 2 at t=3.0
    // 3. Censored at t=4.0
    
    std::vector<double> time = {2.0, 3.0, 4.0};
    
    // Status for y1 (Cause 1)
    std::vector<double> status1 = {1.0, 0.0, 0.0};
    
    // Status for y2 (Cause 2)
    std::vector<double> status2 = {0.0, 1.0, 0.0};
    
    // Predictors (constant for simplicity)
    std::vector<double> preds1 = {0.5, 0.5, 0.5}; // eta1
    std::vector<double> preds2 = {-0.5, -0.5, -0.5}; // eta2
    
    // Shapes
    std::vector<double> shape1 = {1.5, 1.5, 1.5};
    std::vector<double> shape2 = {0.8, 0.8, 0.8};

    std::unordered_map<std::string, std::vector<double>> data = {
        {"y1", time},
        {"y2", time}
    };
    
    std::unordered_map<std::string, std::vector<double>> linear_predictors = {
        {"y1", preds1},
        {"y2", preds2}
    };
    
    std::unordered_map<std::string, std::vector<double>> dispersions = {
        {"y1", shape1},
        {"y2", shape2}
    };
    
    std::unordered_map<std::string, std::vector<double>> status = {
        {"y1", status1},
        {"y2", status2}
    };

    libsemx::LikelihoodDriver driver;
    double loglik = driver.evaluate_model_loglik(model, data, linear_predictors, dispersions, {}, status);

    // Expected: Sum of individual Weibull log-likelihoods with appropriate censoring
    libsemx::WeibullOutcome weibull;
    double expected = 0.0;
    
    // y1 contributions
    for(size_t i=0; i<3; ++i) {
        expected += weibull.evaluate(time[i], preds1[i], shape1[i], status1[i]).log_likelihood;
    }
    // y2 contributions
    for(size_t i=0; i<3; ++i) {
        expected += weibull.evaluate(time[i], preds2[i], shape2[i], status2[i]).log_likelihood;
    }

    REQUIRE_THAT(loglik, Catch::Matchers::WithinRel(expected, 1e-5));
}
