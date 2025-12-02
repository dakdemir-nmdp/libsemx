#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "libsemx/model_objective.hpp"
#include "libsemx/likelihood_driver.hpp"
#include "libsemx/model_ir.hpp"

#include <vector>
#include <unordered_map>
#include <iostream>

using namespace libsemx;

TEST_CASE("ModelObjective computes correct gradients for SEM", "[sem][gradient]") {
    // Define a simple CFA: F -> y1, y2
    ModelIRBuilder builder;
    builder.add_variable("y1", VariableKind::Observed, "gaussian");
    builder.add_variable("y2", VariableKind::Observed, "gaussian");
    builder.add_variable("F", VariableKind::Latent, "gaussian");
    
    // Loadings
    builder.add_edge(EdgeKind::Loading, "F", "y1", "lambda1");
    builder.add_edge(EdgeKind::Loading, "F", "y2", "lambda2");
    
    // Variances
    builder.add_edge(EdgeKind::Covariance, "F", "F", "psi_F");
    builder.add_edge(EdgeKind::Covariance, "y1", "y1", "theta1");
    builder.add_edge(EdgeKind::Covariance, "y2", "y2", "theta2");
    
    auto model = builder.build();
    
    // Data (N=10)
    std::vector<double> y1(10, 1.0);
    std::vector<double> y2(10, 2.0);
    
    std::unordered_map<std::string, std::vector<double>> data = {
        {"y1", y1},
        {"y2", y2}
    };
    
    LikelihoodDriver driver;
    ModelObjective objective(driver, model, data);
    
    // Initial parameters
    auto params = objective.initial_parameters();
    // Set some non-trivial values
    for(auto& p : params) p = 0.5;
    
    // Analytic Gradient
    auto analytic_grad = objective.gradient(params);
    
    // Finite Difference Gradient
    double epsilon = 1e-5;
    std::vector<double> fd_grad(params.size());
    double base_val = objective.value(params);
    
    for(size_t i=0; i<params.size(); ++i) {
        auto perturbed = params;
        perturbed[i] += epsilon;
        double val_plus = objective.value(perturbed);
        fd_grad[i] = (val_plus - base_val) / epsilon;
    }
    
    // Compare
    for(size_t i=0; i<params.size(); ++i) {
        INFO("Parameter " << i << " Name: " << objective.parameter_names()[i]);
        CHECK_THAT(analytic_grad[i], Catch::Matchers::WithinRel(fd_grad[i], 1e-3) || Catch::Matchers::WithinAbs(fd_grad[i], 1e-4));
    }
}
