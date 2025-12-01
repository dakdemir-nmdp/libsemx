#pragma once

#include <memory>
#include <string>

namespace libsemx {

class ParameterTransform;

class Parameter {
public:
    Parameter(std::string name, double value);

    Parameter(std::string name, double value, std::shared_ptr<const ParameterTransform> transform);

    [[nodiscard]] const std::string& name() const noexcept;

    [[nodiscard]] double value() const noexcept;

    void set_value(double value);

    [[nodiscard]] double unconstrained_value() const noexcept;

    void set_unconstrained_value(double value);

    void set_transform(std::shared_ptr<const ParameterTransform> transform);

    [[nodiscard]] std::shared_ptr<const ParameterTransform> transform() const noexcept;

    void set_fixed(bool fixed) noexcept;

    [[nodiscard]] bool is_fixed() const noexcept;

private:
    std::string name_;
    double unconstrained_value_;
    std::shared_ptr<const ParameterTransform> transform_;
    bool fixed_{false};
};

}  // namespace libsemx
