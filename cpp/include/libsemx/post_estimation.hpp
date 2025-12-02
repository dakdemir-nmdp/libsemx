#pragma once

#include "libsemx/model_types.hpp"
#include <vector>
#include <string>
#include <unordered_map>

namespace libsemx {

struct StandardizedEdgeResult {
    double std_lv;
    double std_all;
};

struct StandardizedSolution {
    // One result per edge in model.edges
    std::vector<StandardizedEdgeResult> edges;
    
    // TODO: Add standardized values for covariance blocks if needed
};

struct ModelDiagnostics {
    // Model-implied moments (flattened row-major)
    std::vector<double> implied_means;
    std::vector<double> implied_covariance; 

    // Residuals (Sample - Implied)
    std::vector<double> mean_residuals;
    std::vector<double> covariance_residuals; 
    
    // Correlation Residuals: (S_ij - Sigma_ij) / sqrt(S_ii * S_jj)
    std::vector<double> correlation_residuals; 
    
    // Standardized Root Mean Square Residual
    double srmr;
};

struct ModificationIndex {
    std::string source;
    std::string target;
    EdgeKind kind;
    double mi;      // Modification Index (approximate Chi-square change)
    double epc;     // Expected Parameter Change
    double gradient; // Gradient of log-likelihood
};

StandardizedSolution compute_standardized_estimates(
    const ModelIR& model,
    const std::vector<std::string>& parameter_names,
    const std::vector<double>& parameter_values,
    const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data = {}
);

ModelDiagnostics compute_model_diagnostics(
    const ModelIR& model,
    const std::vector<std::string>& parameter_names,
    const std::vector<double>& parameter_values,
    const std::vector<double>& sample_means,
    const std::vector<double>& sample_covariance, // Flattened row-major, matching model.variables order
    const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data = {}
);

std::vector<ModificationIndex> compute_modification_indices(
    const ModelIR& model,
    const std::vector<std::string>& parameter_names,
    const std::vector<double>& parameter_values,
    const std::vector<double>& sample_covariance,
    size_t sample_size,
    const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data = {}
);

} // namespace libsemx
