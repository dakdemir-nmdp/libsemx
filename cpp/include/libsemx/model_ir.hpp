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

    void add_covariance(std::string id, std::string structure, std::size_t dimension);

    void add_random_effect(std::string id, std::vector<std::string> variables, std::string covariance_id);

    [[nodiscard]] ModelIR build() const;

private:
    ModelGraph graph_;
};

}  // namespace libsemx
