#include "libsemx/parameter.hpp"

#include "libsemx/parameter_transform.hpp"

#include <stdexcept>
#include <utility>

namespace libsemx {

Parameter::Parameter(std::string name, double value)
    : Parameter(std::move(name), value, make_identity_transform()) {}

Parameter::Parameter(std::string name, double value, std::shared_ptr<const ParameterTransform> transform)
    : name_(std::move(name)), transform_(std::move(transform)) {
    if (name_.empty()) {
        throw std::invalid_argument("parameter name must be non-empty");
    }
    if (!transform_) {
        throw std::invalid_argument("parameter transform must be non-null");
    }
    if (!transform_->is_valid_constrained(value)) {
        throw std::out_of_range("initial value violates transform constraints");
    }
    unconstrained_value_ = transform_->to_unconstrained(value);
}

const std::string& Parameter::name() const noexcept {
    return name_;
}

double Parameter::value() const noexcept {
    return transform_->to_constrained(unconstrained_value_);
}

void Parameter::set_value(double value) {
    if (!transform_->is_valid_constrained(value)) {
        throw std::out_of_range("value violates transform constraints");
    }
    unconstrained_value_ = transform_->to_unconstrained(value);
}

double Parameter::unconstrained_value() const noexcept {
    return unconstrained_value_;
}

void Parameter::set_unconstrained_value(double value) {
    unconstrained_value_ = value;
}

void Parameter::set_transform(std::shared_ptr<const ParameterTransform> transform) {
    if (!transform) {
        throw std::invalid_argument("parameter transform must be non-null");
    }
    const double current_constrained = value();
    if (!transform->is_valid_constrained(current_constrained)) {
        throw std::invalid_argument("new transform cannot represent current parameter value");
    }
    transform_ = std::move(transform);
    unconstrained_value_ = transform_->to_unconstrained(current_constrained);
}

std::shared_ptr<const ParameterTransform> Parameter::transform() const noexcept {
    return transform_;
}

void Parameter::set_fixed(bool fixed) noexcept {
    fixed_ = fixed;
}

bool Parameter::is_fixed() const noexcept {
    return fixed_;
}

}  // namespace libsemx
