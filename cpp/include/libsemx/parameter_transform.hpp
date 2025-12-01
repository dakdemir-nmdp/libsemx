#pragma once

#include <memory>

namespace libsemx {

class ParameterTransform {
public:
    virtual ~ParameterTransform() = default;

    [[nodiscard]] virtual double to_constrained(double unconstrained) const = 0;

    [[nodiscard]] virtual double to_unconstrained(double constrained) const = 0;

    [[nodiscard]] virtual bool is_valid_constrained(double constrained) const noexcept = 0;

    [[nodiscard]] virtual double constrained_derivative(double unconstrained) const noexcept = 0;
};

class IdentityTransform final : public ParameterTransform {
public:
    [[nodiscard]] double to_constrained(double unconstrained) const override;

    [[nodiscard]] double to_unconstrained(double constrained) const override;

    [[nodiscard]] bool is_valid_constrained(double constrained) const noexcept override;

    [[nodiscard]] double constrained_derivative(double unconstrained) const noexcept override;
};

class LogTransform final : public ParameterTransform {
public:
    [[nodiscard]] double to_constrained(double unconstrained) const override;

    [[nodiscard]] double to_unconstrained(double constrained) const override;

    [[nodiscard]] bool is_valid_constrained(double constrained) const noexcept override;

    [[nodiscard]] double constrained_derivative(double unconstrained) const noexcept override;
};

class LogisticTransform final : public ParameterTransform {
public:
    LogisticTransform(double lower_bound, double upper_bound);

    [[nodiscard]] double to_constrained(double unconstrained) const override;

    [[nodiscard]] double to_unconstrained(double constrained) const override;

    [[nodiscard]] bool is_valid_constrained(double constrained) const noexcept override;

    [[nodiscard]] double lower() const noexcept;

    [[nodiscard]] double upper() const noexcept;

    [[nodiscard]] double constrained_derivative(double unconstrained) const noexcept override;

private:
    double lower_;
    double upper_;
};

std::shared_ptr<const ParameterTransform> make_identity_transform();

std::shared_ptr<const ParameterTransform> make_log_transform();

std::shared_ptr<const ParameterTransform> make_logistic_transform(double lower_bound, double upper_bound);

}  // namespace libsemx
