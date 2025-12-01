#include "libsemx/model_ir.hpp"

#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace libsemx {

namespace {
[[nodiscard]] bool requires_family(VariableKind kind) {
    return kind == VariableKind::Observed;
}
}  // namespace

void ModelIRBuilder::add_variable(std::string name, VariableKind kind, std::string family) {
    if (name.empty()) {
        throw std::invalid_argument("variable name must be non-empty");
    }
    if (variable_index_.contains(name)) {
        throw std::invalid_argument("duplicate variable name: " + name);
    }
    if (requires_family(kind) && family.empty()) {
        throw std::invalid_argument("observed variable requires outcome family identifier");
    }
    if (!requires_family(kind)) {
        family.clear();
    }
    variable_index_.emplace(name, kind);
    variables_.push_back(VariableSpec{std::move(name), kind, std::move(family)});
}

void ModelIRBuilder::add_edge(EdgeKind kind, std::string source, std::string target, std::string parameter_id) {
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
    edges_.push_back(EdgeSpec{kind, std::move(source), std::move(target), std::move(parameter_id)});
}

void ModelIRBuilder::add_covariance(std::string id, std::string structure, std::size_t dimension) {
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
    covariance_index_.emplace(id, covariances_.size());
    covariances_.push_back(CovarianceSpec{std::move(id), std::move(structure), dimension});
}

void ModelIRBuilder::add_random_effect(std::string id, std::vector<std::string> variables, std::string covariance_id) {
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

ModelIR ModelIRBuilder::build() const {
    if (variables_.empty()) {
        throw std::invalid_argument("model IR must contain at least one variable");
    }
    ModelIR ir;
    ir.variables = variables_;
    ir.edges = edges_;
    ir.covariances = covariances_;
    ir.random_effects = random_effects_;
    return ir;
}

}  // namespace libsemx
