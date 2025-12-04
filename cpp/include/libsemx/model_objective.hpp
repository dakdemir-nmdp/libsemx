#pragma once

#include "libsemx/optimizer.hpp"
#include "libsemx/model_ir.hpp"
#include "libsemx/parameter_catalog.hpp"
#include "libsemx/likelihood_driver.hpp"

#include <vector>
#include <unordered_map>
#include <string>

namespace libsemx {

class ModelObjective : public ObjectiveFunction {
public:
    ModelObjective(const LikelihoodDriver& driver,
                   const ModelIR& model,
                   const std::unordered_map<std::string, std::vector<double>>& data,
                   const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data = {},
                   const std::unordered_map<std::string, std::vector<double>>& status = {},
                   EstimationMethod method = EstimationMethod::ML);

    [[nodiscard]] double value(const std::vector<double>& parameters) const override;

    [[nodiscard]] std::vector<double> gradient(const std::vector<double>& parameters) const override;

    [[nodiscard]] double value_and_gradient(const std::vector<double>& parameters,
                                            std::vector<double>& gradient) const override;

    [[nodiscard]] const std::vector<std::string>& parameter_names() const;

    [[nodiscard]] std::vector<double> initial_parameters() const;

    [[nodiscard]] std::vector<double> to_constrained(const std::vector<double>& unconstrained) const;

    [[nodiscard]] std::vector<double> constrained_derivatives(const std::vector<double>& unconstrained) const;

    // Converts optimizer parameters (which might be Cholesky factors) back to model parameters (Variances/Covariances)
    [[nodiscard]] std::vector<double> convert_to_model_parameters(const std::vector<double>& optimizer_parameters) const;

    [[nodiscard]] std::unordered_map<std::string, std::vector<double>> get_covariance_matrices(const std::vector<double>& constrained_parameters) const;

    // Helper to expose workspaces for diagnostics
    void build_prediction_workspaces(const std::vector<double>& constrained_parameters,
                                     std::unordered_map<std::string, std::vector<double>>& linear_predictors,
                                     std::unordered_map<std::string, std::vector<double>>& dispersions,
                                     std::unordered_map<std::string, std::vector<double>>& covariance_parameters) const;

private:
    void prepare_sem_structures();
    void update_sem_data(const std::vector<double>& constrained_parameters) const;

    const LikelihoodDriver& driver_;
    const ModelIR& model_;
    const std::unordered_map<std::string, std::vector<double>>& data_;
    const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data_;
    const std::unordered_map<std::string, std::vector<double>>& status_;
    EstimationMethod method_;
    ParameterCatalog catalog_;
    std::unordered_map<std::string, std::pair<size_t, size_t>> covariance_param_ranges_;

    // SEM support
    bool sem_mode_{false};
    ModelIR sem_model_;
    mutable std::unordered_map<std::string, std::vector<double>> sem_data_;
    std::vector<std::string> sem_outcomes_;
    std::vector<std::string> sem_latents_;
    
    struct LoadingInfo {
        size_t param_index; // Index in catalog
        size_t outcome_index; // Index in sem_outcomes_
        size_t latent_index;  // Index in sem_latents_
        double fixed_value{0.0}; // If param_index is npos
    };
    std::vector<LoadingInfo> loading_infos_;

    struct ResidualInfo {
        size_t param_index; // Index in catalog
        size_t outcome_index; // Index in sem_outcomes_
        double fixed_value{1.0}; // If param_index is npos
    };
    std::vector<ResidualInfo> residual_infos_;

    struct SemCovarianceMapping {
        std::string id;
        struct Element {
            size_t param_index; // Index in catalog, or npos
            double fixed_value; // Used if param_index is npos
        };
        std::vector<Element> elements;
    };
    std::vector<SemCovarianceMapping> sem_covariance_mappings_;
};

}  // namespace libsemx
