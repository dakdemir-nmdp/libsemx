#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "libsemx/likelihood_driver.hpp"
#include "libsemx/model_ir.hpp"
#include "libsemx/optimizer.hpp"

#include <cmath>
#include <vector>
#include <numeric>
#include <iostream>

using namespace libsemx;

TEST_CASE("Fit statistics for simple regression", "[fit][statistics]") {
    // y ~ x
    ModelIRBuilder builder;
    builder.add_variable("y", VariableKind::Observed, "gaussian");
    builder.add_variable("x", VariableKind::Observed, "gaussian");
    builder.add_edge(EdgeKind::Regression, "x", "y", "beta");
    
    ModelIR model = builder.build();
    
    // Generate data
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {2.1, 3.9, 6.1, 8.0, 10.1}; // Approx y = 2x
    
    std::unordered_map<std::string, std::vector<double>> data;
    data["x"] = x;
    data["y"] = y;
    
    LikelihoodDriver driver;
    OptimizationOptions options;
    options.max_iterations = 100;
    
    FitResult result = driver.fit(model, data, options);
    
    REQUIRE(result.optimization_result.converged);
    
    // Check SEs are not NaN and positive
    REQUIRE(result.standard_errors.size() > 0);
    for (double se : result.standard_errors) {
        REQUIRE(!std::isnan(se));
        REQUIRE(se > 0.0);
    }
    
    // Check AIC/BIC
    REQUIRE(!std::isnan(result.aic));
    REQUIRE(!std::isnan(result.bic));
    
    // Check Vcov is symmetric
    size_t n = result.standard_errors.size();
    REQUIRE(result.vcov.size() == n * n);
    for(size_t i=0; i<n; ++i) {
        for(size_t j=0; j<n; ++j) {
            REQUIRE(result.vcov[i*n + j] == Catch::Approx(result.vcov[j*n + i]));
        }
    }
}
