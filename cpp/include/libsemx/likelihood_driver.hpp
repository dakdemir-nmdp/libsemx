#pragma once

#include "libsemx/outcome_family.hpp"
#include "libsemx/model_ir.hpp"
#include "libsemx/optimizer.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

namespace libsemx {

enum class EstimationMethod {
    ML,
    REML
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
                                                EstimationMethod method = EstimationMethod::ML) const;

    [[nodiscard]] std::unordered_map<std::string, double> evaluate_model_gradient(const ModelIR& model,
                                                const std::unordered_map<std::string, std::vector<double>>& data,
                                                const std::unordered_map<std::string, std::vector<double>>& linear_predictors,
                                                const std::unordered_map<std::string, std::vector<double>>& dispersions,
                                                const std::unordered_map<std::string, std::vector<double>>& covariance_parameters = {},
                                                const std::unordered_map<std::string, std::vector<double>>& status = {},
                                                const std::unordered_map<std::string, std::vector<double>>& extra_params = {},
                                                const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data = {},
                                                EstimationMethod method = EstimationMethod::ML) const;

    [[nodiscard]] OptimizationResult fit(const ModelIR& model,
                                         const std::unordered_map<std::string, std::vector<double>>& data,
                                         const OptimizationOptions& options,
                                         const std::string& optimizer_name = "lbfgs",
                                         const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data = {}) const;
};

}  // namespace libsemx