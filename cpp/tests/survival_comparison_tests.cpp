
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
#include <cmath>

using namespace libsemx;

// Helper to load Ovarian data
std::unordered_map<std::string, std::vector<double>> load_ovarian_data(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Cannot open " + path);

    std::unordered_map<std::string, std::vector<double>> data;
    std::vector<std::string> headers;
    
    std::string line, cell;
    std::getline(file, line); // Header
    std::stringstream ss(line);
    while (std::getline(ss, cell, ',')) {
        if (cell.size() >= 2 && cell.front() == '"' && cell.back() == '"') {
            cell = cell.substr(1, cell.size() - 2);
        }
        headers.push_back(cell);
        data[cell] = {};
    }

    // Add Intercept column
    data["Intercept"] = {};

    while (std::getline(file, line)) {
        std::stringstream ss2(line);
        int col_idx = 0;
        while (std::getline(ss2, cell, ',')) {
            if (col_idx < headers.size()) {
                if (cell.size() >= 2 && cell.front() == '"' && cell.back() == '"') {
                    cell = cell.substr(1, cell.size() - 2);
                }
                try {
                    double val = std::stod(cell);
                    data[headers[col_idx]].push_back(val);
                } catch (...) {
                    // Should not happen with cleaned data
                }
            }
            col_idx++;
        }
        data["Intercept"].push_back(1.0);
    }
    return data;
}

TEST_CASE("Compare Weibull Survival with survival::survreg (Ovarian)", "[comparison][survival][weibull]") {
    std::string data_path = "../data/ovarian_survival.csv";
    std::ifstream check(data_path);
    if (!check.good()) {
        data_path = "data/ovarian_survival.csv";
    }
    
    auto data = load_ovarian_data(data_path);
    size_t N = data["futime"].size();
    REQUIRE(N == 26);

    ModelIRBuilder builder;

    // Variables
    builder.add_variable("futime", VariableKind::Observed, "weibull");
    builder.add_variable("age", VariableKind::Exogenous);
    builder.add_variable("rx", VariableKind::Exogenous);
    builder.add_variable("Intercept", VariableKind::Exogenous);

    // Regressions: futime ~ Intercept + age + rx
    builder.add_edge(EdgeKind::Regression, "Intercept", "futime", "beta_int");
    builder.add_edge(EdgeKind::Regression, "age", "futime", "beta_age");
    builder.add_edge(EdgeKind::Regression, "rx", "futime", "beta_rx");

    // Dispersion (Shape parameter k)
    // libsemx models dispersion as a parameter. For Weibull, dispersion = k (shape).
    // survreg reports Scale = 1/k.
    // We need to estimate k.
    builder.add_edge(EdgeKind::Covariance, "futime", "futime", "shape_k");

    auto model = builder.build();

    // Prepare status map
    std::unordered_map<std::string, std::vector<double>> status;
    status["futime"] = data["fustat"];

    // Initial values
    std::unordered_map<std::string, std::vector<double>> init_status;
    init_status["beta_int"] = {10.0}; // From survreg ~10.46
    init_status["beta_age"] = {-0.1}; // From survreg ~-0.08
    init_status["beta_rx"] = {0.5};   // From survreg ~0.57
    init_status["shape_k"] = {1.8};   // From survreg ~1.81

    LikelihoodDriver driver;
    OptimizationOptions options;
    options.max_iterations = 1000;
    options.tolerance = 1e-6;

    auto result = driver.fit(model, data, options, "lbfgs", {}, status);

    REQUIRE(result.optimization_result.converged);

    // Comparison Values (from R)
    // LogLik: -88.76171
    // Shape (k): 1.816069
    // Intercept: 10.46261528
    // age: -0.07909537
    // rx: 0.56727556

    double r_loglik = -88.76171;
    double r_shape = 1.816069;
    double r_int = 10.46261528;
    double r_age = -0.07909537;
    double r_rx = 0.56727556;

    // libsemx returns NLL
    double libsemx_loglik = -result.optimization_result.objective_value;
    
    CHECK_THAT(libsemx_loglik, Catch::Matchers::WithinRel(r_loglik, 0.001));

    // Check parameters
    // We need to find the parameter values in the result
    auto find_param = [&](const std::string& name) {
        for (size_t i = 0; i < result.parameter_names.size(); ++i) {
            if (result.parameter_names[i] == name) {
                return result.optimization_result.parameters[i];
            }
        }
        throw std::runtime_error("Parameter not found: " + name);
    };

    CHECK_THAT(find_param("shape_k"), Catch::Matchers::WithinRel(r_shape, 0.01));
    CHECK_THAT(find_param("beta_int"), Catch::Matchers::WithinRel(r_int, 0.01));
    CHECK_THAT(find_param("beta_age"), Catch::Matchers::WithinRel(r_age, 0.01));
    CHECK_THAT(find_param("beta_rx"), Catch::Matchers::WithinRel(r_rx, 0.01));
}

TEST_CASE("Compare Exponential Survival with survival::survreg (Ovarian)", "[comparison][survival][exponential]") {
    std::string data_path = "../data/ovarian_survival.csv";
    std::ifstream check(data_path);
    if (!check.good()) {
        data_path = "data/ovarian_survival.csv";
    }
    
    auto data = load_ovarian_data(data_path);
    size_t N = data["futime"].size();
    REQUIRE(N == 26);

    ModelIRBuilder builder;

    // Variables
    builder.add_variable("futime", VariableKind::Observed, "exponential");
    builder.add_variable("age", VariableKind::Exogenous);
    builder.add_variable("rx", VariableKind::Exogenous);
    builder.add_variable("Intercept", VariableKind::Exogenous);

    // Regressions: futime ~ Intercept + age + rx
    builder.add_edge(EdgeKind::Regression, "Intercept", "futime", "beta_int");
    builder.add_edge(EdgeKind::Regression, "age", "futime", "beta_age");
    builder.add_edge(EdgeKind::Regression, "rx", "futime", "beta_rx");

    // Exponential has no dispersion parameter (shape=1 fixed)
    // So no covariance edge for futime

    auto model = builder.build();

    // Prepare status map
    std::unordered_map<std::string, std::vector<double>> status;
    status["futime"] = data["fustat"];

    // Initial values
    std::unordered_map<std::string, std::vector<double>> init_status;
    init_status["beta_int"] = {12.0}; 
    init_status["beta_age"] = {-0.1}; 
    init_status["beta_rx"] = {0.6};   

    LikelihoodDriver driver;
    OptimizationOptions options;
    options.max_iterations = 1000;
    options.tolerance = 1e-6;

    auto result = driver.fit(model, data, options, "lbfgs", {}, status);

    REQUIRE(result.optimization_result.converged);

    // Comparison Values (from R)
    // LogLik: -91.20899
    // Intercept: 12.1225260
    // age: -0.1050638
    // rx: 0.6610630

    double r_loglik = -91.20899;
    double r_int = 12.1225260;
    double r_age = -0.1050638;
    double r_rx = 0.6610630;

    // libsemx returns NLL
    double libsemx_loglik = -result.optimization_result.objective_value;
    
    CHECK_THAT(libsemx_loglik, Catch::Matchers::WithinRel(r_loglik, 0.001));

    // Check parameters
    auto find_param = [&](const std::string& name) {
        for (size_t i = 0; i < result.parameter_names.size(); ++i) {
            if (result.parameter_names[i] == name) {
                return result.optimization_result.parameters[i];
            }
        }
        throw std::runtime_error("Parameter not found: " + name);
    };

    CHECK_THAT(find_param("beta_int"), Catch::Matchers::WithinRel(r_int, 0.01));
    CHECK_THAT(find_param("beta_age"), Catch::Matchers::WithinRel(r_age, 0.01));
    CHECK_THAT(find_param("beta_rx"), Catch::Matchers::WithinRel(r_rx, 0.01));
}
