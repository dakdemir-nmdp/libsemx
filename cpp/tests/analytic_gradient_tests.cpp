#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "libsemx/likelihood_driver.hpp"
#include "libsemx/model_ir.hpp"

#include <vector>
#include <unordered_map>
#include <cmath>

TEST_CASE("LikelihoodDriver evaluates analytic gradients for Gaussian model", "[gradient]") {
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("x", libsemx::VariableKind::Observed, "gaussian"); // x is predictor
    
    // y ~ N(beta * x, 1)
    builder.add_edge(libsemx::EdgeKind::Regression, "x", "y", "beta");
    
    auto model = builder.build();

    // Data
    std::vector<double> y = {1.0, 2.0, 3.0};
    std::vector<double> x = {1.0, 1.0, 2.0};
    
    // Parameters: beta = 0.5
    double beta = 0.5;
    
    // Linear predictors: eta = beta * x
    std::vector<double> preds = {0.5, 0.5, 1.0};
    
    // Dispersions: 1.0
    std::vector<double> disps = {1.0, 1.0, 1.0};

    std::unordered_map<std::string, std::vector<double>> data = {
        {"y", y},
        {"x", x}
    };
    std::unordered_map<std::string, std::vector<double>> linear_predictors = {
        {"y", preds},
        {"x", x} // x is observed, so its LP is itself (or irrelevant if not outcome)
    };
    std::unordered_map<std::string, std::vector<double>> dispersions = {
        {"y", disps},
        {"x", disps}
    };

    libsemx::LikelihoodDriver driver;
    auto gradients = driver.evaluate_model_gradient(model, data, linear_predictors, dispersions);

    // Expected gradient:
    // d(loglik)/d(beta) = sum( d(loglik)/d(eta_i) * d(eta_i)/d(beta) )
    // d(loglik)/d(eta_i) = (y_i - eta_i) / sigma^2 = (y_i - eta_i)
    // d(eta_i)/d(beta) = x_i
    // Grad = sum( (y_i - eta_i) * x_i )
    // i=0: (1.0 - 0.5) * 1.0 = 0.5
    // i=1: (2.0 - 0.5) * 1.0 = 1.5
    // i=2: (3.0 - 1.0) * 2.0 = 4.0
    // Total = 0.5 + 1.5 + 4.0 = 6.0

    REQUIRE(gradients.count("beta"));
    REQUIRE_THAT(gradients.at("beta"), Catch::Matchers::WithinRel(6.0, 1e-5));
}
