#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "libsemx/likelihood_driver.hpp"
#include "libsemx/model_ir.hpp"

#include <vector>
#include <unordered_map>
#include <cmath>

TEST_CASE("LikelihoodDriver optimizes covariance parameters", "[optimization][mixed]") {
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("cluster", libsemx::VariableKind::Grouping);
    
    // Random intercept: u ~ N(0, tau^2)
    // We want to estimate tau^2.
    // The parameter ID for the covariance structure is "tau_sq".
    // In ModelObjective, we need to map "tau_sq" to optimization parameters.
    // However, CovarianceSpec doesn't explicitly list parameter IDs for its internal parameters.
    // Usually, CovarianceStructure manages its own parameters.
    // But ModelObjective needs to know about them to pass them to the optimizer.
    // 
    // Wait, CovarianceSpec has an ID ("tau_sq").
    // But a covariance structure might have multiple parameters (e.g. unstructured).
    // The current ModelIR doesn't seem to expose individual parameter IDs for covariance structures.
    // 
    // Let's assume for "diagonal" structure, the ID "tau_sq" refers to the structure itself,
    // and maybe we need a convention for parameter names?
    // Or maybe we just use the covariance ID as a prefix?
    // 
    // For now, let's see if we can just use "tau_sq" as the parameter ID if it's 1D.
    builder.add_variable("u_cluster", libsemx::VariableKind::Latent);
    builder.add_covariance("tau_sq", "diagonal", 1);
    builder.add_random_effect("u_cluster", {"cluster"}, "tau_sq");
    builder.add_edge(libsemx::EdgeKind::Regression, "u_cluster", "y", "1");

    auto model = builder.build();

    // Data: 100 clusters, 5 obs each
    // True tau^2 = 4.0
    // True sigma^2 = 1.0
    // y_ij = u_i + e_ij
    std::vector<double> y;
    std::vector<double> cluster;
    std::vector<double> preds;
    std::vector<double> disps;
    
    // Simple synthetic data generation
    // We won't actually run optimization to convergence here because we don't have a full optimizer setup in this test,
    // but we want to check if the objective function respects the covariance parameter.
    
    // Let's just check if changing the covariance parameter changes the likelihood.
    // If ModelObjective ignores covariance parameters, the likelihood will be constant with respect to them.
    
    y = {1.0, 2.0, 3.0, 4.0};
    cluster = {1.0, 1.0, 2.0, 2.0};
    preds = {0.0, 0.0, 0.0, 0.0};
    disps = {1.0, 1.0, 1.0, 1.0};

    std::unordered_map<std::string, std::vector<double>> data = {
        {"y", y},
        {"cluster", cluster}
    };
    
    libsemx::LikelihoodDriver driver;
    libsemx::OptimizationOptions options;
    options.max_iterations = 1; // We just want to initialize and check gradients/values
    
    // We can't easily access the internal ModelObjective from here.
    // But we can use fit() and see if it fails or if we can infer something.
    // Actually, let's just try to fit and see if it throws or if it returns a result.
    // If it doesn't optimize covariance, the result will likely be the initial value.
    
    // However, we need to know what the parameter names are.
    // The current implementation of ModelObjective only looks at edge.parameter_id.
    // It ignores model.covariances.
    
    // So, we expect that if we run fit(), the covariance parameter won't be in the parameter list,
    // and thus won't be optimized.
    
    // Let's verify this by checking if we can even specify a covariance parameter.
    // In the current ModelIR, there is no place to attach a parameter ID to a covariance element.
    // The CovarianceSpec just has an ID for the whole structure.
    
    // This suggests we need to update ModelIR or ModelObjective to handle this.
    // For "diagonal" or "unstructured", the parameters are implicit.
    
    // Let's try to run fit and see what happens.
    // If ModelObjective doesn't include covariance parameters, the optimizer will only see fixed effects (none here).
    // So it might complain about no parameters to optimize, or just return immediately.
    
    auto result = driver.fit(model, data, options, "lbfgs");
    
    // If covariance parameters were included, we should see them in the result.
    // But result.optimization_result.parameters is a vector, and we don't know the mapping.
    // Wait, OptimizationResult doesn't seem to return parameter names?
    // Let's check LikelihoodDriver::fit return type.
}
