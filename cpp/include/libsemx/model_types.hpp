#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace libsemx {

enum class VariableKind {
    Observed,
    Latent,
    Grouping
};

enum class EdgeKind {
    Loading,
    Regression,
    Covariance
};

enum class ParameterConstraint {
    Free,
    Positive
};

struct VariableSpec {
    std::string name;
    VariableKind kind;
    std::string family;  // outcome family identifier, empty for latent/grouping
    std::string label;
    std::string measurement_level;
};

struct EdgeSpec {
    EdgeKind kind;
    std::string source;
    std::string target;
    std::string parameter_id;
};

struct CovarianceSpec {
    std::string id;
    std::string structure;
    std::size_t dimension;
};

struct RandomEffectSpec {
    std::string id;
    std::vector<std::string> variables;
    std::string covariance_id;
};

struct ParameterSpec {
    std::string id;
    ParameterConstraint constraint{ParameterConstraint::Free};
    double initial_value{0.0};
};

struct ModelIR {
    std::vector<VariableSpec> variables;
    std::vector<EdgeSpec> edges;
    std::vector<CovarianceSpec> covariances;
    std::vector<RandomEffectSpec> random_effects;
    std::vector<ParameterSpec> parameters;
};

}  // namespace libsemx
