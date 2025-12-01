#pragma once

#include "libsemx/parameter.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace libsemx {

class ParameterCatalog {
public:
    static constexpr std::size_t npos = std::numeric_limits<std::size_t>::max();

    std::size_t register_parameter(const std::string& name,
                                   double initial_value,
                                   std::shared_ptr<const ParameterTransform> transform);

    [[nodiscard]] std::size_t size() const noexcept;

    [[nodiscard]] bool contains(const std::string& name) const noexcept;

    [[nodiscard]] std::size_t find_index(const std::string& name) const noexcept;

    [[nodiscard]] const std::vector<std::string>& names() const noexcept;

    [[nodiscard]] std::vector<double> initial_unconstrained() const;

    [[nodiscard]] std::vector<double> constrain(const std::vector<double>& unconstrained) const;

    [[nodiscard]] std::vector<double> constrained_derivatives(const std::vector<double>& unconstrained) const;

private:
    std::vector<Parameter> parameters_;
    std::vector<std::string> names_;
    std::unordered_map<std::string, std::size_t> index_;
};

}  // namespace libsemx
