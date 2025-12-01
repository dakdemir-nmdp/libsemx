#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace libsemx {

struct OptimizationOptions {
    std::size_t max_iterations{100};
    double tolerance{1e-6};
    double learning_rate{0.1};
};

struct OptimizationResult {
    std::vector<double> parameters;
    double objective_value{0.0};
    double gradient_norm{0.0};
    std::size_t iterations{0};
    bool converged{false};
};

class ObjectiveFunction {
public:
    virtual ~ObjectiveFunction() = default;

    [[nodiscard]] virtual double value(const std::vector<double>& parameters) const = 0;

    [[nodiscard]] virtual std::vector<double> gradient(const std::vector<double>& parameters) const = 0;
};

class Optimizer {
public:
    virtual ~Optimizer() = default;

    [[nodiscard]] virtual std::string name() const = 0;

    [[nodiscard]] virtual OptimizationResult optimize(const ObjectiveFunction& function,
                                                       std::vector<double> initial_parameters,
                                                       const OptimizationOptions& options) const = 0;
};

class GradientDescentOptimizer final : public Optimizer {
public:
    [[nodiscard]] std::string name() const override;

    [[nodiscard]] OptimizationResult optimize(const ObjectiveFunction& function,
                                               std::vector<double> initial_parameters,
                                               const OptimizationOptions& options) const override;
};

class LBFGSOptimizer final : public Optimizer {
public:
    [[nodiscard]] std::string name() const override;

    [[nodiscard]] OptimizationResult optimize(const ObjectiveFunction& function,
                                               std::vector<double> initial_parameters,
                                               const OptimizationOptions& options) const override;
};

[[nodiscard]] std::unique_ptr<Optimizer> make_gradient_descent_optimizer();

[[nodiscard]] std::unique_ptr<Optimizer> make_lbfgs_optimizer();

}  // namespace libsemx
