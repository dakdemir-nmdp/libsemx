#include "libsemx/parameter_catalog.hpp"

#include "libsemx/parameter_transform.hpp"

#include <stdexcept>

namespace libsemx {

std::size_t ParameterCatalog::register_parameter(const std::string& name,
                                                 double initial_value,
                                                 std::shared_ptr<const ParameterTransform> transform) {
    if (name.empty()) {
        throw std::invalid_argument("parameter name must be non-empty");
    }
    if (!transform) {
        transform = make_identity_transform();
    }

    auto it = index_.find(name);
    if (it != index_.end()) {
        return it->second;
    }

    parameters_.emplace_back(name, initial_value, std::move(transform));
    names_.push_back(name);
    std::size_t idx = parameters_.size() - 1;
    index_.emplace(name, idx);
    return idx;
}

std::size_t ParameterCatalog::size() const noexcept {
    return parameters_.size();
}

bool ParameterCatalog::contains(const std::string& name) const noexcept {
    return index_.contains(name);
}

std::size_t ParameterCatalog::find_index(const std::string& name) const noexcept {
    auto it = index_.find(name);
    if (it == index_.end()) {
        return npos;
    }
    return it->second;
}

const std::vector<std::string>& ParameterCatalog::names() const noexcept {
    return names_;
}

std::vector<double> ParameterCatalog::initial_unconstrained() const {
    std::vector<double> values(parameters_.size());
    for (std::size_t i = 0; i < parameters_.size(); ++i) {
        values[i] = parameters_[i].unconstrained_value();
    }
    return values;
}

std::vector<double> ParameterCatalog::constrain(const std::vector<double>& unconstrained) const {
    if (unconstrained.size() != parameters_.size()) {
        throw std::invalid_argument("unconstrained vector size mismatch");
    }
    std::vector<double> constrained(parameters_.size());
    for (std::size_t i = 0; i < parameters_.size(); ++i) {
        constrained[i] = parameters_[i].transform()->to_constrained(unconstrained[i]);
    }
    return constrained;
}

std::vector<double> ParameterCatalog::constrained_derivatives(const std::vector<double>& unconstrained) const {
    if (unconstrained.size() != parameters_.size()) {
        throw std::invalid_argument("unconstrained vector size mismatch");
    }
    std::vector<double> derivatives(parameters_.size());
    for (std::size_t i = 0; i < parameters_.size(); ++i) {
        derivatives[i] = parameters_[i].transform()->constrained_derivative(unconstrained[i]);
    }
    return derivatives;
}

}  // namespace libsemx
