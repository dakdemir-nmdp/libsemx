#pragma once

#include <optional>
#include <string>
#include <unordered_map>
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

struct VariableSpec {
    std::string name;
    VariableKind kind;
    std::string family;  // outcome family identifier, empty for latent/grouping
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

struct ModelIR {
    std::vector<VariableSpec> variables;
    std::vector<EdgeSpec> edges;
    std::vector<CovarianceSpec> covariances;
    std::vector<RandomEffectSpec> random_effects;
};

class ModelIRBuilder {
public:
    ModelIRBuilder() = default;

    void add_variable(std::string name, VariableKind kind, std::string family = {});

    void add_edge(EdgeKind kind, std::string source, std::string target, std::string parameter_id);

    void add_covariance(std::string id, std::string structure, std::size_t dimension);

    void add_random_effect(std::string id, std::vector<std::string> variables, std::string covariance_id);

    [[nodiscard]] ModelIR build() const;

private:
    std::vector<VariableSpec> variables_;
    std::vector<EdgeSpec> edges_;
    std::vector<CovarianceSpec> covariances_;
    std::vector<RandomEffectSpec> random_effects_;
    std::unordered_map<std::string, VariableKind> variable_index_;
    std::unordered_map<std::string, std::size_t> covariance_index_;
};

}  // namespace libsemx
