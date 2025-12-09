#include "libsemx/likelihood_driver.hpp"
#include "libsemx/model_ir.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>
#include <string>
#include <unordered_map>

using namespace libsemx;

TEST_CASE("Multi-group analysis with Gaussian outcomes", "[multi_group]") {
    LikelihoodDriver driver;
    
    // Generate data for two groups
    // Group 1: N(1.0, 1.0), n=100
    // Group 2: N(2.0, 1.0), n=100
    
    std::vector<double> y1(100);
    std::vector<double> y2(100);
    std::vector<double> one(100, 1.0);
    
    for (int i = 0; i < 100; ++i) {
        y1[i] = 1.0 + (i % 2 == 0 ? 0.1 : -0.1); 
    }
    for (int i = 0; i < 100; ++i) {
        y2[i] = 2.0 + (i % 2 == 0 ? 0.1 : -0.1);
    }
    
    std::vector<std::unordered_map<std::string, std::vector<double>>> data(2);
    data[0]["y"] = y1;
    data[0]["one"] = one;
    data[1]["y"] = y2;
    data[1]["one"] = one;
    
    SECTION("Equality constraints across groups (Pooled Mean)") {
        // Both groups share parameter "mu" for the intercept
        
        std::vector<ModelIR> models(2);
        
        // Model 1
        ModelIRBuilder b1;
        b1.add_variable("y", VariableKind::Observed, "gaussian");
        b1.add_variable("one", VariableKind::Exogenous);
        b1.add_edge(EdgeKind::Regression, "one", "y", "mu"); // Shared name "mu"
        b1.add_edge(EdgeKind::Covariance, "y", "y", "var_y"); // Shared variance "var_y"
        models[0] = b1.build();
        
        // Model 2
        ModelIRBuilder b2;
        b2.add_variable("y", VariableKind::Observed, "gaussian");
        b2.add_variable("one", VariableKind::Exogenous);
        b2.add_edge(EdgeKind::Regression, "one", "y", "mu"); // Shared name "mu"
        b2.add_edge(EdgeKind::Covariance, "y", "y", "var_y"); // Shared variance "var_y"
        models[1] = b2.build();
        
        OptimizationOptions options;
        options.max_iterations = 100;
        
        FitResult result = driver.fit_multi_group(models, data, options);
        
        REQUIRE(result.optimization_result.converged);
        
        // Expected pooled mean = (1.0 * 100 + 2.0 * 100) / 200 = 1.5
        // Find "mu" in parameters
        double mu_est = 0.0;
        bool found = false;
        for (size_t i = 0; i < result.parameter_names.size(); ++i) {
            if (result.parameter_names[i] == "mu") {
                mu_est = result.optimization_result.parameters[i];
                found = true;
                break;
            }
        }
        REQUIRE(found);
        REQUIRE_THAT(mu_est, Catch::Matchers::WithinAbs(1.5, 1e-4));
    }
    
    SECTION("Free parameters across groups (Group-specific Means)") {
        // Group 1 uses "mu1", Group 2 uses "mu2"
        
        std::vector<ModelIR> models(2);
        
        // Model 1
        ModelIRBuilder b1;
        b1.add_variable("y", VariableKind::Observed, "gaussian");
        b1.add_variable("one", VariableKind::Exogenous);
        b1.add_edge(EdgeKind::Regression, "one", "y", "mu1"); 
        b1.add_edge(EdgeKind::Covariance, "y", "y", "var_y"); // Shared variance
        models[0] = b1.build();
        
        // Model 2
        ModelIRBuilder b2;
        b2.add_variable("y", VariableKind::Observed, "gaussian");
        b2.add_variable("one", VariableKind::Exogenous);
        b2.add_edge(EdgeKind::Regression, "one", "y", "mu2"); 
        b2.add_edge(EdgeKind::Covariance, "y", "y", "var_y"); // Shared variance
        models[1] = b2.build();
        
        OptimizationOptions options;
        options.max_iterations = 100;
        
        FitResult result = driver.fit_multi_group(models, data, options);
        
        REQUIRE(result.optimization_result.converged);
        
        double mu1_est = 0.0;
        double mu2_est = 0.0;
        
        for (size_t i = 0; i < result.parameter_names.size(); ++i) {
            if (result.parameter_names[i] == "mu1") {
                mu1_est = result.optimization_result.parameters[i];
            } else if (result.parameter_names[i] == "mu2") {
                mu2_est = result.optimization_result.parameters[i];
            }
        }
        
        REQUIRE_THAT(mu1_est, Catch::Matchers::WithinAbs(1.0, 1e-4));
        REQUIRE_THAT(mu2_est, Catch::Matchers::WithinAbs(2.0, 1e-4));
    }
}
