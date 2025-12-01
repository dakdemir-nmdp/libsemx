#include "libsemx/model_graph.hpp"

#include <stdexcept>
#include <utility>

namespace libsemx {

void ModelGraph::add_variable(std::string name, VariableType type) {
    if (name.empty()) {
        throw std::invalid_argument("variable name must be non-empty");
    }
    Variable variable{name, type};
    if (!variables_.try_emplace(variable.name, std::move(variable)).second) {
        throw std::invalid_argument("variable already exists: " + name);
    }
}

bool ModelGraph::contains(const std::string& name) const {
    return variables_.find(name) != variables_.end();
}

const Variable& ModelGraph::get(const std::string& name) const {
    auto it = variables_.find(name);
    if (it == variables_.end()) {
        throw std::out_of_range("variable not found: " + name);
    }
    return it->second;
}

std::vector<std::string> ModelGraph::variable_names() const {
    std::vector<std::string> names;
    names.reserve(variables_.size());
    for (const auto& entry : variables_) {
        names.push_back(entry.first);
    }
    return names;
}

}  // namespace libsemx
