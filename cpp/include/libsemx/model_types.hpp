#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace libsemx {

enum class VariableKind {
    Observed,
    Latent,
    Grouping,
    Exogenous
};

enum class EdgeKind {
    Loading,
    Regression,
    Covariance
};

enum class ParameterConstraint {
    Free,
    Positive,
    Fixed
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
    std::vector<CovarianceSpec> components;
};

struct RandomEffectSpec {
    std::string id;
    std::vector<std::string> variables;
    std::string covariance_id;
    double lambda{1.0};  // Shrinkage/ridge parameter (1.0 = no shrinkage, >1 = more shrinkage)
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
