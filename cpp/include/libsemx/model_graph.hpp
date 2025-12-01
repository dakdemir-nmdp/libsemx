#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace libsemx {

enum class VariableType {
    Observed,
    Latent
};

struct Variable {
    std::string name;
    VariableType type;
};

class ModelGraph {
public:
    ModelGraph() = default;

    void add_variable(std::string name, VariableType type);

    [[nodiscard]] bool contains(const std::string& name) const;

    [[nodiscard]] const Variable& get(const std::string& name) const;

    [[nodiscard]] std::vector<std::string> variable_names() const;

private:
    std::unordered_map<std::string, Variable> variables_;
};

}  // namespace libsemx
