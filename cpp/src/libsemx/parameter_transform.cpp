#include "libsemx/parameter_transform.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>

namespace libsemx {

double IdentityTransform::to_constrained(double unconstrained) const {
    return unconstrained;
}

double IdentityTransform::to_unconstrained(double constrained) const {
    return constrained;
}

bool IdentityTransform::is_valid_constrained(double /*constrained*/) const noexcept {
    return true;
}

double LogTransform::to_constrained(double unconstrained) const {
    return std::exp(unconstrained);
}

double LogTransform::to_unconstrained(double constrained) const {
    if (constrained <= 0.0) {
        throw std::domain_error("log transform input must be positive");
    }
    return std::log(constrained);
}

bool LogTransform::is_valid_constrained(double constrained) const noexcept {
    return constrained > 0.0;
}

LogisticTransform::LogisticTransform(double lower_bound, double upper_bound)
    : lower_(lower_bound), upper_(upper_bound) {
    if (!(upper_ > lower_)) {
        throw std::invalid_argument("logistic transform requires upper > lower");
    }
}

double LogisticTransform::to_constrained(double unconstrained) const {
    const double span = upper_ - lower_;
    const double logistic = 1.0 / (1.0 + std::exp(-unconstrained));
    return lower_ + span * logistic;
}

double LogisticTransform::to_unconstrained(double constrained) const {
    if (!is_valid_constrained(constrained)) {
        throw std::domain_error("logistic transform input outside open interval");
    }
    const double span = upper_ - lower_;
    const double scaled = (constrained - lower_) / span;
    return std::log(scaled / (1.0 - scaled));
}

bool LogisticTransform::is_valid_constrained(double constrained) const noexcept {
    return (constrained > lower_) && (constrained < upper_);
}

double LogisticTransform::lower() const noexcept {
    return lower_;
}

double LogisticTransform::upper() const noexcept {
    return upper_;
}

std::shared_ptr<const ParameterTransform> make_identity_transform() {
    static const std::shared_ptr<const ParameterTransform> kIdentity = std::make_shared<IdentityTransform>();
    return kIdentity;
}

std::shared_ptr<const ParameterTransform> make_log_transform() {
    static const std::shared_ptr<const ParameterTransform> kLog = std::make_shared<LogTransform>();
    return kLog;
}

std::shared_ptr<const ParameterTransform> make_logistic_transform(double lower_bound, double upper_bound) {
    return std::make_shared<LogisticTransform>(lower_bound, upper_bound);
}

}  // namespace libsemx
