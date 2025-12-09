#include "libsemx/model_graph.hpp"

#include <cstdlib>
#include <stdexcept>

namespace libsemx {

namespace {
[[nodiscard]] bool requires_family(VariableKind kind) {
    return kind == VariableKind::Observed;
}

[[nodiscard]] bool is_numeric_literal(const std::string& value) {
    if (value.empty()) {
        return false;
    }
    char* end = nullptr;
    std::strtod(value.c_str(), &end);
    return end != nullptr && end != value.c_str() && *end == '\0';
}

constexpr double kDefaultCoefficientInit = 0.0;
}  // namespace

void ModelGraph::add_variable(std::string name, VariableKind kind, std::string family, std::string label, std::string measurement_level) {
    if (name.empty()) {
        throw std::invalid_argument("variable name must be non-empty");
    }
    if (variable_index_.contains(name)) {
        throw std::invalid_argument("duplicate variable name: " + name);
    }
    if (requires_family(kind)) {
        if (family.empty()) {
            throw std::invalid_argument("observed variable requires outcome family identifier");
        }
    } else {
        family.clear();
    }

    variable_index_.emplace(name, kind);
    variables_.push_back(VariableSpec{std::move(name), kind, std::move(family), std::move(label), std::move(measurement_level)});
}

void ModelGraph::add_edge(EdgeKind kind, std::string source, std::string target, std::string parameter_id) {
    if (source.empty() || target.empty()) {
        throw std::invalid_argument("edge endpoints must be non-empty");
    }
    if (parameter_id.empty()) {
        throw std::invalid_argument("edge parameter id must be non-empty");
    }
    if (!variable_index_.contains(source)) {
        throw std::invalid_argument("edge source not registered: " + source);
    }
    if (!variable_index_.contains(target)) {
        throw std::invalid_argument("edge target not registered: " + target);
    }
    if (!parameter_id.empty() && !is_numeric_literal(parameter_id)) {
        double init = kDefaultCoefficientInit;
        ParameterConstraint constraint = ParameterConstraint::Free;
        
        if (kind == EdgeKind::Covariance && source == target) {
            init = 0.5;
            constraint = ParameterConstraint::Positive;
        }
        
        register_parameter(parameter_id, constraint, init);
    }
    edges_.push_back(EdgeSpec{kind, std::move(source), std::move(target), std::move(parameter_id)});
}

void ModelGraph::add_covariance(std::string id, std::string structure, std::size_t dimension, std::vector<std::string> component_ids) {
    if (id.empty()) {
        throw std::invalid_argument("covariance id must be non-empty");
    }
    if (structure.empty()) {
        throw std::invalid_argument("covariance structure identifier must be non-empty");
    }
    if (dimension == 0) {
        throw std::invalid_argument("covariance dimension must be positive");
    }
    if (covariance_index_.contains(id)) {
        throw std::invalid_argument("duplicate covariance id: " + id);
    }

    std::vector<CovarianceSpec> components;
    components.reserve(component_ids.size());
    for (const auto& comp_id : component_ids) {
        auto it = covariance_index_.find(comp_id);
        if (it == covariance_index_.end()) {
            throw std::invalid_argument("covariance component not found: " + comp_id);
        }
        components.push_back(covariances_[it->second]);
    }

    covariance_index_.emplace(id, covariances_.size());
    covariances_.push_back(CovarianceSpec{std::move(id), std::move(structure), dimension, std::move(components)});
}

void ModelGraph::add_random_effect(std::string id, std::vector<std::string> variables, std::string covariance_id) {
    if (id.empty()) {
        throw std::invalid_argument("random effect id must be non-empty");
    }
    if (variables.empty()) {
        throw std::invalid_argument("random effect must reference at least one variable");
    }
    if (covariance_id.empty()) {
        throw std::invalid_argument("random effect covariance id must be non-empty");
    }
    std::unordered_set<std::string> seen;
    seen.reserve(variables.size());
    for (const auto& var_name : variables) {
        if (!variable_index_.contains(var_name)) {
            throw std::invalid_argument("random effect references unknown variable: " + var_name);
        }
        if (!seen.insert(var_name).second) {
            throw std::invalid_argument("random effect references variable multiple times: " + var_name);
        }
    }
    if (!covariance_index_.contains(covariance_id)) {
        throw std::invalid_argument("random effect references unknown covariance id: " + covariance_id);
    }
    random_effects_.push_back(RandomEffectSpec{std::move(id), std::move(variables), std::move(covariance_id)});
}

void ModelGraph::register_parameter(std::string id,
                                    ParameterConstraint constraint,
                                    double initial_value) {
    if (id.empty()) {
        throw std::invalid_argument("parameter id must be non-empty");
    }
    auto it = parameter_index_.find(id);
    if (it != parameter_index_.end()) {
        const auto& existing = parameters_[it->second];
        if (existing.constraint != constraint) {
            throw std::invalid_argument("parameter registered with conflicting constraint: " + id);
        }
        return;
    }
    ParameterSpec spec{std::move(id), constraint, initial_value};
    parameter_index_.emplace(spec.id, parameters_.size());
    parameters_.push_back(std::move(spec));
}

const std::vector<VariableSpec>& ModelGraph::variables() const noexcept {
    return variables_;
}

const std::vector<EdgeSpec>& ModelGraph::edges() const noexcept {
    return edges_;
}

const std::vector<CovarianceSpec>& ModelGraph::covariances() const noexcept {
    return covariances_;
}

const std::vector<RandomEffectSpec>& ModelGraph::random_effects() const noexcept {
    return random_effects_;
}

const std::vector<ParameterSpec>& ModelGraph::parameters() const noexcept {
    return parameters_;
}

ModelIR ModelGraph::to_model_ir() const {
    if (variables_.empty()) {
        throw std::invalid_argument("model graph must contain at least one variable");
    }
    ModelIR ir;
    ir.variables = variables_;
    ir.edges = edges_;
    ir.covariances = covariances_;
    ir.random_effects = random_effects_;
    ir.parameters = parameters_;
    return ir;
}

}  // namespace libsemx
