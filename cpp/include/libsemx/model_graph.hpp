#pragma once

#include "libsemx/model_types.hpp"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace libsemx {

class ModelGraph {
public:
    ModelGraph() = default;

    void add_variable(std::string name, VariableKind kind, std::string family = {}, std::string label = {}, std::string measurement_level = {});

    void add_edge(EdgeKind kind, std::string source, std::string target, std::string parameter_id);

    void add_covariance(std::string id, std::string structure, std::size_t dimension, std::vector<std::string> component_ids = {});

    void add_random_effect(std::string id, std::vector<std::string> variables, std::string covariance_id);

    void register_parameter(std::string id,
                            ParameterConstraint constraint = ParameterConstraint::Free,
                            double initial_value = 0.0);

    void set_parameter_initial_value(const std::string& id, double initial_value);

    [[nodiscard]] const std::vector<VariableSpec>& variables() const noexcept;

    [[nodiscard]] const std::vector<EdgeSpec>& edges() const noexcept;

    [[nodiscard]] const std::vector<CovarianceSpec>& covariances() const noexcept;

    [[nodiscard]] const std::vector<RandomEffectSpec>& random_effects() const noexcept;

    [[nodiscard]] const std::vector<ParameterSpec>& parameters() const noexcept;

    [[nodiscard]] ModelIR to_model_ir() const;

private:
    std::vector<VariableSpec> variables_;
    std::vector<EdgeSpec> edges_;
    std::vector<CovarianceSpec> covariances_;
    std::vector<RandomEffectSpec> random_effects_;
    std::unordered_map<std::string, VariableKind> variable_index_;
    std::unordered_map<std::string, std::size_t> covariance_index_;
    std::vector<ParameterSpec> parameters_;
    std::unordered_map<std::string, std::size_t> parameter_index_;
};

}  // namespace libsemx
