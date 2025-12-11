#include "libsemx/model_ir.hpp"

namespace libsemx {

void ModelIRBuilder::add_variable(std::string name, VariableKind kind, std::string family, std::string label, std::string measurement_level) {
    graph_.add_variable(std::move(name), kind, std::move(family), std::move(label), std::move(measurement_level));
}

void ModelIRBuilder::add_edge(EdgeKind kind, std::string source, std::string target, std::string parameter_id) {
    graph_.add_edge(kind, std::move(source), std::move(target), std::move(parameter_id));
}

void ModelIRBuilder::add_covariance(std::string id, std::string structure, std::size_t dimension, std::vector<std::string> component_ids) {
    graph_.add_covariance(std::move(id), std::move(structure), dimension, std::move(component_ids));
}

void ModelIRBuilder::add_random_effect(std::string id, std::vector<std::string> variables, std::string covariance_id) {
    graph_.add_random_effect(std::move(id), std::move(variables), std::move(covariance_id));
}

void ModelIRBuilder::register_parameter(std::string id, double initial_value) {
    graph_.register_parameter(std::move(id), ParameterConstraint::Free, initial_value);
}

void ModelIRBuilder::set_parameter_initial_value(std::string id, double initial_value) {
    graph_.set_parameter_initial_value(id, initial_value);
}

ModelIR ModelIRBuilder::build() const {
    return graph_.to_model_ir();
}

}  // namespace libsemx
