#pragma once

#include "libsemx/model_graph.hpp"
#include "libsemx/model_types.hpp"

#include <string>
#include <vector>

namespace libsemx {

class ModelIRBuilder {
public:
    ModelIRBuilder() = default;

    void add_variable(std::string name, VariableKind kind, std::string family = {}, std::string label = {}, std::string measurement_level = {});

    void add_edge(EdgeKind kind, std::string source, std::string target, std::string parameter_id);

    void add_covariance(std::string id, std::string structure, std::size_t dimension, std::vector<std::string> component_ids = {});

    void add_random_effect(std::string id, std::vector<std::string> variables, std::string covariance_id, double lambda = 1.0);

    void register_parameter(std::string id, double initial_value = 0.0);

    void set_parameter_initial_value(std::string id, double initial_value);

    [[nodiscard]] ModelIR build() const;

private:
    ModelGraph graph_;
};

}  // namespace libsemx
