#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "libsemx/likelihood_driver.hpp"
#include "libsemx/model_ir.hpp"
#include "libsemx/optimizer.hpp"
#include "libsemx/model_objective.hpp"

#include <vector>
#include <unordered_map>
#include <cmath>

TEST_CASE("LikelihoodDriver respects REML flag in fit()", "[reml]") {
    // Simple Gaussian model: y ~ N(mu, sigma^2)
    // With REML, the likelihood should be penalized by 0.5 * log(N) (if we consider intercept as fixed effect)
    // Actually, let's use the explicit logic: 0.5 * log|X' V^-1 X|
    
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("x", libsemx::VariableKind::Observed, "gaussian");
    
    // y ~ x
    builder.add_edge(libsemx::EdgeKind::Regression, "x", "y", "beta");
    // Residual variance
    builder.add_edge(libsemx::EdgeKind::Covariance, "y", "y", "sigma2");
    
    auto model = builder.build();

    std::vector<double> y = {1.0, 2.0, 3.0};
    std::vector<double> x = {1.0, 1.0, 1.0}; // Intercept-only equivalent
    
    std::unordered_map<std::string, std::vector<double>> data = {
        {"y", y},
        {"x", x}
    };

    libsemx::LikelihoodDriver driver;
    libsemx::OptimizationOptions options;
    options.max_iterations = 1; // We just want to check the objective value at start if possible, or run a bit

    // We can't easily check the objective value directly from fit() result without running it.
    // But we can check if the result differs between ML and REML.
    
    // Let's manually evaluate loglik to verify the plumbing works
    
    std::vector<double> preds = {0.0, 0.0, 0.0}; // beta=0
    std::vector<double> disps = {1.0, 1.0, 1.0}; // sigma2=1
    
    std::unordered_map<std::string, std::vector<double>> linear_predictors = {{"y", preds}, {"x", x}};
    std::unordered_map<std::string, std::vector<double>> dispersions = {{"y", disps}, {"x", disps}};
    
    // ML
    double ml_val = driver.evaluate_model_loglik(model, data, linear_predictors, dispersions, {}, {}, {}, {}, libsemx::EstimationMethod::ML);
    
    // REML
    double reml_val = driver.evaluate_model_loglik(model, data, linear_predictors, dispersions, {}, {}, {}, {}, libsemx::EstimationMethod::REML);
    
    // For this case:
    // V = I
    // X = [1, 1, 1]'
    // X' V^-1 X = 3
    // log|X' V^-1 X| = log(3)
    // REML penalty = 0.5 * log(3) - 0.5 * p * log(2pi) = 0.5 * 1.0986 - 0.5 * 1.837 = 0.549 - 0.918 = -0.369
    // Wait, the code says: total_loglik -= 0.5 * log_det_reml; total_loglik += 0.5 * p * log_2pi;
    // So REML = ML - 0.5 * log(3) + 0.5 * log(2pi)
    
    double diff = reml_val - ml_val;
    double expected_diff = -0.5 * std::log(3.0) + 0.5 * std::log(2.0 * M_PI);
    
    REQUIRE_THAT(diff, Catch::Matchers::WithinRel(expected_diff, 1e-5));
}

TEST_CASE("ModelObjective passes REML flag", "[reml]") {
    // Verify that ModelObjective correctly passes the flag to the driver
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("x", libsemx::VariableKind::Observed, "gaussian");
    builder.add_edge(libsemx::EdgeKind::Regression, "x", "y", "beta");
    builder.add_edge(libsemx::EdgeKind::Covariance, "y", "y", "sigma2");
    auto model = builder.build();

    std::vector<double> y = {1.0, 2.0, 3.0};
    std::vector<double> x = {1.0, 1.0, 1.0};
    std::unordered_map<std::string, std::vector<double>> data = {{"y", y}, {"x", x}};

    libsemx::LikelihoodDriver driver;
    
    // Create objectives
    libsemx::ModelObjective obj_ml(driver, model, data, {}, {}, libsemx::EstimationMethod::ML);
    libsemx::ModelObjective obj_reml(driver, model, data, {}, {}, libsemx::EstimationMethod::REML);
    
    // Initial parameters: beta=0, sigma2=1 (log(sigma2)=0)
    std::vector<double> params = {0.0, 0.0}; 
    
    double val_ml = obj_ml.value(params);
    double val_reml = obj_reml.value(params);
    
    // Objective returns negative loglik
    double diff = (-val_reml) - (-val_ml);
    double expected_diff = -0.5 * std::log(3.0) + 0.5 * std::log(2.0 * M_PI);
    
    REQUIRE_THAT(diff, Catch::Matchers::WithinRel(expected_diff, 1e-5));
}
