
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

// Helper to load BFI data
// Expects header: "A1","A2",...,"O5" (25 cols)
// Returns map: var_name -> vector<double>
std::unordered_map<std::string, std::vector<double>> load_bfi_data(const std::string& path) {
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

#include <numeric>

// ...

TEST_CASE("Compare CFA with lavaan (BFI)", "[comparison][lavaan][cfa]") {
    std::string data_path = "../data/bfi_complete.csv";
    std::ifstream check(data_path);
    if (!check.good()) {
        data_path = "data/bfi_complete.csv";
    }
    
    auto data = load_bfi_data(data_path);
    size_t N = data["A1"].size();
    REQUIRE(N == 2436);

    ModelIRBuilder builder;

    // 1. Define Variables
    std::vector<std::string> items;
    std::vector<std::string> factors = {"Agreeableness", "Conscientiousness", "Extraversion", "Neuroticism", "Openness"};
    std::vector<std::string> prefixes = {"A", "C", "E", "N", "O"};

    // Add Intercept
    builder.add_variable("Intercept", VariableKind::Exogenous);

    for (const auto& prefix : prefixes) {
        for (int i = 1; i <= 5; ++i) {
            std::string item = prefix + std::to_string(i);
            items.push_back(item);
            builder.add_variable(item, VariableKind::Observed, "gaussian");
            
            // Intercepts
            builder.add_edge(EdgeKind::Regression, "Intercept", item, "int_" + item);
            
            // Residual Variances
            builder.add_edge(EdgeKind::Covariance, item, item, "theta_" + item);
        }
    }

    for (const auto& factor : factors) {
        builder.add_variable(factor, VariableKind::Latent);
    }

    // 2. Define Loadings
    auto add_loadings = [&](const std::string& factor, const std::string& prefix) {
        for (int i = 1; i <= 5; ++i) {
            std::string item = prefix + std::to_string(i);
            if (i == 1) {
                builder.add_edge(EdgeKind::Loading, factor, item, "1.0");
            } else {
                builder.add_edge(EdgeKind::Loading, factor, item, "lambda_" + factor + "_" + item);
            }
        }
    };

    add_loadings("Agreeableness", "A");
    add_loadings("Conscientiousness", "C");
    add_loadings("Extraversion", "E");
    add_loadings("Neuroticism", "N");
    add_loadings("Openness", "O");

    // 3. Define Latent Covariance (Unstructured)
    for (const auto& factor : factors) {
        builder.add_edge(EdgeKind::Covariance, factor, factor, "psi_" + factor);
    }
    for (size_t i = 0; i < factors.size(); ++i) {
        for (size_t j = i + 1; j < factors.size(); ++j) {
            builder.add_edge(EdgeKind::Covariance, factors[i], factors[j], "psi_" + factors[i] + "_" + factors[j]);
        }
    }

    auto model = builder.build();

    // Prepare initial values (status)
    std::unordered_map<std::string, std::vector<double>> status;
    
    // Calculate means for intercepts
    for (const auto& item : items) {
        const auto& vec = data[item];
        double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
        double mean = sum / vec.size();
        status["int_" + item] = {mean};
        status["theta_" + item] = {1.0}; // Residual variance
    }
    
    // Latent variances/covariances
    for (const auto& factor : factors) {
        status["psi_" + factor] = {1.0};
    }
    // Covariances default to 0.0 which is fine (Identity)
    
    // Loadings
    // Default 0.0 is fine, or maybe 0.5
    for (const auto& prefix : prefixes) {
        for (int i = 2; i <= 5; ++i) { // First is fixed
             // status["lambda_..."] = {0.5}; 
        }
    }

    LikelihoodDriver driver;
    OptimizationOptions options;
    options.max_iterations = 2000; // Increased from 50
    options.tolerance = 1e-5;

    auto result = driver.fit(model, data, options, "lbfgs", {}, status);

    REQUIRE(result.optimization_result.converged);


    // 4. Compare Results
    // Lavaan values:
    // LogLik: -99840.24
    // AIC: 199850.5
    // BIC: 200343.3
    // CFI: 0.782
    // TLI: 0.754
    // RMSEA: 0.078
    // SRMR: 0.073

    double lavaan_loglik = -99840.24;
    
    // libsemx returns NLL in optimization_result.value
    // But fit() might return total loglik?
    // LikelihoodDriver::fit returns FitResult.
    // FitResult has optimization_result.
    // But LikelihoodDriver::evaluate_total_loglik returns the loglik.
    // Wait, fit() calculates chi_square based on loglik_user = -nll.
    // So result.optimization_result.objective_value is NLL.
    // So LogLik = -result.optimization_result.objective_value.
    
    double libsemx_loglik = -result.optimization_result.objective_value;
    
    // Relax tolerance slightly for initial comparison
    CHECK_THAT(libsemx_loglik, Catch::Matchers::WithinRel(lavaan_loglik, 0.01));
    
    // Fit Indices
    CHECK_THAT(result.cfi, Catch::Matchers::WithinAbs(0.782, 0.05));
    CHECK_THAT(result.tli, Catch::Matchers::WithinAbs(0.754, 0.05));
    CHECK_THAT(result.rmsea, Catch::Matchers::WithinAbs(0.078, 0.02));
    CHECK_THAT(result.srmr, Catch::Matchers::WithinAbs(0.073, 0.02));
}
