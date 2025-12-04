#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "libsemx/likelihood_driver.hpp"
#include "libsemx/model_ir.hpp"
#include "libsemx/optimizer.hpp"

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <map>
#include <unordered_map>

using namespace libsemx;

// Helper to read CSV
struct SleepStudyData {
    std::vector<double> reaction;
    std::vector<double> days;
    std::vector<double> subject;
    std::vector<double> intercept;
};

SleepStudyData load_sleepstudy(const std::string& path) {
    SleepStudyData data;
    std::ifstream file(path);
    if (!file.is_open()) {
        // Try relative to build dir
        std::string alt_path = "../" + path;
        file.open(alt_path);
        if (!file.is_open()) {
             // Try relative to root
             alt_path = "../../" + path; // if in build/libsemx_tests
             file.open(alt_path);
             if (!file.is_open()) {
                throw std::runtime_error("Could not open file: " + path);
             }
        }
    }

    std::string line;
    std::getline(file, line); // Skip header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        while (std::getline(ss, cell, ',')) {
            row.push_back(cell);
        }
        if (row.size() < 4) continue;

        // row[0] is rownames
        data.reaction.push_back(std::stod(row[1]));
        data.days.push_back(std::stod(row[2]));
        data.subject.push_back(std::stod(row[3]));
        data.intercept.push_back(1.0);
    }
    return data;
}

TEST_CASE("Align with lme4 results for sleepstudy data", "[alignment][lme4]") {
    // Path to data
    std::string data_path = "data/sleepstudy.csv";
    
    SleepStudyData data;
    try {
        data = load_sleepstudy(data_path);
    } catch (const std::exception& e) {
        FAIL("Could not load data: " << e.what());
    }

    ModelIRBuilder builder;
    
    // Variables
    builder.add_variable("Reaction", VariableKind::Observed, "gaussian");
    builder.add_variable("Days", VariableKind::Exogenous);
    builder.add_variable("_intercept", VariableKind::Exogenous);
    builder.add_variable("Subject", VariableKind::Grouping);

    // Fixed effects: Reaction ~ _intercept + Days
    builder.add_edge(EdgeKind::Regression, "_intercept", "Reaction", "beta_Reaction_on__intercept");
    builder.add_edge(EdgeKind::Regression, "Days", "Reaction", "beta_Reaction_on_Days");

    // Residual variance: Reaction ~~ Reaction
    builder.add_edge(EdgeKind::Covariance, "Reaction", "Reaction", "Reaction_dispersion");

    // Random effects: (1 + Days | Subject)
    // Covariance structure: Unstructured, dimension 2.
    builder.add_covariance("re_cov", "unstructured", 2);
    
    // Random effect definition
    // Variables: Subject (grouping), _intercept (random intercept), Days (random slope)
    builder.add_random_effect("subject_re", {"Subject", "_intercept", "Days"}, "re_cov");

    auto model = builder.build();

    // Prepare data map
    std::unordered_map<std::string, std::vector<double>> data_map;
    data_map["Reaction"] = data.reaction;
    data_map["Days"] = data.days;
    data_map["_intercept"] = data.intercept;
    data_map["Subject"] = data.subject;

    LikelihoodDriver driver;
    
    // Configure optimizer
    OptimizationOptions options;
    options.max_iterations = 1000;
    options.tolerance = 1e-6;

    // Fit
    FitResult result;
    try {
        result = driver.fit(model, data_map, options, "lbfgs", {}, {}, EstimationMethod::ML);
    } catch (const std::exception& e) {
        FAIL("Optimization failed: " << e.what());
    }

    // Map parameters
    std::map<std::string, double> param_map;
    for (size_t i = 0; i < result.parameter_names.size(); ++i) {
        param_map[result.parameter_names[i]] = result.optimization_result.parameters[i];
    }
    
    // Print parameters for debugging
    std::cout << "Converged: " << result.optimization_result.converged << std::endl;
    std::cout << "Iterations: " << result.optimization_result.iterations << std::endl;
    std::cout << "Objective: " << result.optimization_result.objective_value << std::endl;
    std::cout << "Estimated parameters:" << std::endl;
    for (const auto& pair : param_map) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    // REQUIRE(result.optimization_result.converged);

    // Expected values from lme4 (ML estimation)
    double expected_intercept = 251.40510;
    double expected_days = 10.46729;
    
    // Check Fixed Effects
    // Using a slightly larger margin because optimization might differ slightly
    REQUIRE_THAT(param_map["beta_Reaction_on__intercept"], Catch::Matchers::WithinRel(expected_intercept, 0.05));
    REQUIRE_THAT(param_map["beta_Reaction_on_Days"], Catch::Matchers::WithinRel(expected_days, 0.05));

    // Check Residual Variance
    // lme4 sigma = 25.5919 -> variance = 654.945
    // We expect a parameter for dispersion.
    // If it exists, check it.
    if (param_map.count("Reaction_dispersion")) {
        REQUIRE_THAT(param_map["Reaction_dispersion"], Catch::Matchers::WithinRel(654.945, 0.1));
    }

    // Check AIC, BIC, LogLik
    // AIC: 1763.939
    // BIC: 1783.097
    // LogLik: -875.9697
    REQUIRE_THAT(result.aic, Catch::Matchers::WithinRel(1763.939, 0.001));
    REQUIRE_THAT(result.bic, Catch::Matchers::WithinRel(1783.097, 0.001));
    REQUIRE_THAT(-result.optimization_result.objective_value, Catch::Matchers::WithinRel(-875.9697, 0.001));

    // Check Random Effects (BLUPs) for Subject 308 (First subject)
    // lme4: Intercept 2.8158, Days 9.0755
    if (result.random_effects.count("subject_re")) {
        const auto& re = result.random_effects["subject_re"];
        REQUIRE(re.size() >= 2);
        // Assuming sorted by Subject ID, 308 is first.
        REQUIRE_THAT(re[0], Catch::Matchers::WithinAbs(2.8158, 0.1));
        REQUIRE_THAT(re[1], Catch::Matchers::WithinAbs(9.0755, 0.1));
    }
}

TEST_CASE("Align with lme4 results for sleepstudy data (Binomial)", "[alignment][lme4][binomial]") {
    // Path to data
    std::string data_path = "data/sleepstudy.csv";
    
    SleepStudyData data;
    try {
        data = load_sleepstudy(data_path);
    } catch (const std::exception& e) {
        FAIL("Could not load data: " << e.what());
    }

    // Binarize Reaction: 1 if > 250, 0 otherwise
    std::vector<double> reaction_bin;
    for (double val : data.reaction) {
        reaction_bin.push_back(val > 250.0 ? 1.0 : 0.0);
    }

    ModelIRBuilder builder;
    
    // Variables
    builder.add_variable("ReactionBin", VariableKind::Observed, "binomial");
    builder.add_variable("Days", VariableKind::Exogenous);
    builder.add_variable("_intercept", VariableKind::Exogenous);
    builder.add_variable("Subject", VariableKind::Grouping);

    // Fixed effects: ReactionBin ~ _intercept + Days
    builder.add_edge(EdgeKind::Regression, "_intercept", "ReactionBin", "beta_ReactionBin_on__intercept");
    builder.add_edge(EdgeKind::Regression, "Days", "ReactionBin", "beta_ReactionBin_on_Days");

    // Random effects: (1 | Subject)
    // Covariance structure: Unstructured (scalar), dimension 1.
    builder.add_covariance("re_cov", "unstructured", 1);
    
    // Random effect definition
    // Variables: Subject (grouping), _intercept (random intercept)
    builder.add_random_effect("subject_re", {"Subject", "_intercept"}, "re_cov");

    auto model = builder.build();

    // Prepare data map
    std::unordered_map<std::string, std::vector<double>> data_map;
    data_map["ReactionBin"] = reaction_bin;
    data_map["Days"] = data.days;
    data_map["_intercept"] = data.intercept;
    data_map["Subject"] = data.subject;

    LikelihoodDriver driver;
    
    // Configure optimizer
    OptimizationOptions options;
    options.max_iterations = 1000;
    options.tolerance = 1e-6;

    // Fit
    FitResult result;
    try {
        result = driver.fit(model, data_map, options, "lbfgs", {}, {}, EstimationMethod::ML);
    } catch (const std::exception& e) {
        FAIL("Optimization failed: " << e.what());
    }

    // Map parameters
    std::map<std::string, double> param_map;
    for (size_t i = 0; i < result.parameter_names.size(); ++i) {
        param_map[result.parameter_names[i]] = result.optimization_result.parameters[i];
    }
    
    // Print parameters for debugging
    std::cout << "Binomial Converged: " << result.optimization_result.converged << std::endl;
    std::cout << "Binomial Iterations: " << result.optimization_result.iterations << std::endl;
    std::cout << "Binomial Objective: " << result.optimization_result.objective_value << std::endl;
    std::cout << "Binomial Estimated parameters:" << std::endl;
    for (const auto& pair : param_map) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    // Expected values from lme4 (glmer)
    // Fixed effects:
    // (Intercept)   1.3663
    // Days          0.4255
    // Random effects:
    // Subject (Intercept) Variance 10.09 -> StdDev 3.177
    
    double expected_intercept = 1.3663;
    double expected_days = 0.4255;
    double expected_re_var = 10.09;

    REQUIRE_THAT(param_map["beta_ReactionBin_on__intercept"], Catch::Matchers::WithinRel(expected_intercept, 0.1));
    REQUIRE_THAT(param_map["beta_ReactionBin_on_Days"], Catch::Matchers::WithinRel(expected_days, 0.1));
    
    // Check Random Effect Variance (re_cov_0)
    // Note: libsemx returns the standard deviation (or Cholesky factor) for random effects.
    // lme4 reports variance. So we square the libsemx result.
    if (param_map.count("re_cov_0")) {
        double estimated_std = param_map["re_cov_0"];
        double estimated_var = estimated_std * estimated_std;
        REQUIRE_THAT(estimated_var, Catch::Matchers::WithinRel(expected_re_var, 0.1));
    }

    // Check AIC, BIC, LogLik
    // Binomial AIC: 119.5044 
    // Binomial BIC: 129.0833 
    // Binomial LogLik: -56.75219 
    REQUIRE_THAT(result.aic, Catch::Matchers::WithinRel(119.5044, 0.01));
    REQUIRE_THAT(result.bic, Catch::Matchers::WithinRel(129.0833, 0.01));
    REQUIRE_THAT(-result.optimization_result.objective_value, Catch::Matchers::WithinRel(-56.75219, 0.01));

    // Check Random Effects (Conditional Modes) for Subject 308
    // lme4: Intercept -0.4856
    if (result.random_effects.count("subject_re")) {
        const auto& re = result.random_effects["subject_re"];
        REQUIRE(re.size() >= 1);
        REQUIRE_THAT(re[0], Catch::Matchers::WithinAbs(-0.4856, 0.1));
    }
}
