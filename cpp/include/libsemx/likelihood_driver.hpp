#pragma once

#include "libsemx/outcome_family.hpp"
#include "libsemx/model_ir.hpp"
#include "libsemx/optimizer.hpp"

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Sparse>

namespace libsemx {

enum class EstimationMethod {
    ML,
    REML
};

struct FitResult {
    OptimizationResult optimization_result;
    std::vector<double> standard_errors;
    std::vector<double> vcov; // Flattened n x n matrix
    std::vector<std::string> parameter_names;
    std::unordered_map<std::string, std::vector<double>> covariance_matrices;
    std::unordered_map<std::string, std::vector<double>> random_effects; // BLUPs / Conditional Modes
    double aic{0.0};
    double bic{0.0};
    double chi_square{std::numeric_limits<double>::quiet_NaN()};
    double df{std::numeric_limits<double>::quiet_NaN()};
    double p_value{std::numeric_limits<double>::quiet_NaN()};
    double cfi{std::numeric_limits<double>::quiet_NaN()};
    double tli{std::numeric_limits<double>::quiet_NaN()};
    double rmsea{std::numeric_limits<double>::quiet_NaN()};
    double srmr{std::numeric_limits<double>::quiet_NaN()};
};

struct SparseHessianAccumulator {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    bool initialized = false;
    Eigen::SparseMatrix<double> hessian;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    
    struct ObsContribution {
        int row_idx;
        int col_idx;
        double z_product;
    };
    
    struct OutcomeRecipe {
        std::vector<std::vector<ObsContribution>> obs_recipes;
    };
    
    std::vector<OutcomeRecipe> outcome_recipes;
    
    // prior_contributions -> list of {block_idx, row, col, value}
    // Actually, for sparse matrix, we can just rebuild it or update values if structure is fixed.
    // For now, let's keep it simple and rebuild triplets if needed, or use the efficient update if structure is static.
};

class LikelihoodDriver {
public:
    [[nodiscard]] double evaluate_total_loglik(const std::vector<double>& observed,
                                                const std::vector<double>& linear_predictors,
                                                const std::vector<double>& dispersions,
                                                const OutcomeFamily& family,
                                                const std::vector<double>& status = {},
                                                const std::vector<double>& extra_params = {}) const;

    [[nodiscard]] double evaluate_total_loglik_mixed(const std::vector<double>& observed,
                                                      const std::vector<double>& linear_predictors,
                                                      const std::vector<double>& dispersions,
                                                      const std::vector<const OutcomeFamily*>& families,
                                                      const std::vector<double>& status = {},
                                                      const std::vector<std::vector<double>>& extra_params = {}) const;

    [[nodiscard]] double evaluate_model_loglik(const ModelIR& model,
                                                const std::unordered_map<std::string, std::vector<double>>& data,
                                                const std::unordered_map<std::string, std::vector<double>>& linear_predictors,
                                                const std::unordered_map<std::string, std::vector<double>>& dispersions,
                                                const std::unordered_map<std::string, std::vector<double>>& covariance_parameters = {},
                                                const std::unordered_map<std::string, std::vector<double>>& status = {},
                                                const std::unordered_map<std::string, std::vector<double>>& extra_params = {},
                                                const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data = {},
                                                EstimationMethod method = EstimationMethod::ML,
                                                bool force_laplace = false) const;

    struct DataParamMapping {
        std::vector<std::string> pattern;
        std::size_t stride = 1;
    };

    [[nodiscard]] std::unordered_map<std::string, double> evaluate_model_gradient(const ModelIR& model,
                                                const std::unordered_map<std::string, std::vector<double>>& data,
                                                const std::unordered_map<std::string, std::vector<double>>& linear_predictors,
                                                const std::unordered_map<std::string, std::vector<double>>& dispersions,
                                                const std::unordered_map<std::string, std::vector<double>>& covariance_parameters = {},
                                                const std::unordered_map<std::string, std::vector<double>>& status = {},
                                                const std::unordered_map<std::string, std::vector<double>>& extra_params = {},
                                                const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data = {},
                                                EstimationMethod method = EstimationMethod::ML,
                                                const std::unordered_map<std::string, DataParamMapping>& data_param_mappings = {},
                                                const std::unordered_map<std::string, DataParamMapping>& dispersion_param_mappings = {},
                                                const std::unordered_map<std::string, std::vector<std::string>>& extra_param_mappings = {},
                                                bool force_laplace = false) const;

    [[nodiscard]] std::pair<double, std::unordered_map<std::string, double>> evaluate_model_loglik_and_gradient(
                                                const ModelIR& model,
                                                const std::unordered_map<std::string, std::vector<double>>& data,
                                                const std::unordered_map<std::string, std::vector<double>>& linear_predictors,
                                                const std::unordered_map<std::string, std::vector<double>>& dispersions,
                                                const std::unordered_map<std::string, std::vector<double>>& covariance_parameters = {},
                                                const std::unordered_map<std::string, std::vector<double>>& status = {},
                                                const std::unordered_map<std::string, std::vector<double>>& extra_params = {},
                                                const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data = {},
                                                EstimationMethod method = EstimationMethod::ML,
                                                const std::unordered_map<std::string, DataParamMapping>& data_param_mappings = {},
                                                const std::unordered_map<std::string, DataParamMapping>& dispersion_param_mappings = {},
                                                const std::unordered_map<std::string, std::vector<std::string>>& extra_param_mappings = {},
                                                bool force_laplace = false,
                                                SparseHessianAccumulator* sparse_accumulator = nullptr) const;

    [[nodiscard]] std::unordered_map<std::string, std::vector<double>> compute_random_effects(
                                                const ModelIR& model,
                                                const std::unordered_map<std::string, std::vector<double>>& data,
                                                const std::unordered_map<std::string, std::vector<double>>& linear_predictors,
                                                const std::unordered_map<std::string, std::vector<double>>& dispersions,
                                                const std::unordered_map<std::string, std::vector<double>>& covariance_parameters = {},
                                                const std::unordered_map<std::string, std::vector<double>>& status = {},
                                                const std::unordered_map<std::string, std::vector<double>>& extra_params = {},
                                                const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data = {}) const;

    [[nodiscard]] FitResult fit(const ModelIR& model,
                                const std::unordered_map<std::string, std::vector<double>>& data,
                                const OptimizationOptions& options,
                                const std::string& optimizer_name = "lbfgs",
                                const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data = {},
                                const std::unordered_map<std::string, std::vector<double>>& status = {},
                                EstimationMethod method = EstimationMethod::ML,
                                const std::unordered_map<std::string, std::vector<std::string>>& extra_param_mappings = {}) const;

    [[nodiscard]] FitResult fit_multi_group(const std::vector<ModelIR>& models,
                                            const std::vector<std::unordered_map<std::string, std::vector<double>>>& data,
                                            const OptimizationOptions& options,
                                            const std::string& optimizer_name = "lbfgs",
                                            const std::vector<std::unordered_map<std::string, std::vector<std::vector<double>>>>& fixed_covariance_data = {},
                                            const std::vector<std::unordered_map<std::string, std::vector<double>>>& status = {},
                                            EstimationMethod method = EstimationMethod::ML,
                                            const std::vector<std::unordered_map<std::string, std::vector<std::string>>>& extra_param_mappings = {}) const;

private:
    // Cached Laplace system to avoid rebuilding block/group structure on every evaluation.
    mutable std::shared_ptr<void> laplace_cache_;
};

}  // namespace libsemx
