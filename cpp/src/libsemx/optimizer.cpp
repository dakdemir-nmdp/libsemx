#include "libsemx/optimizer.hpp"

#include <Eigen/Core>
#include <LBFGS.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace libsemx {

namespace {
[[nodiscard]] double norm2(const std::vector<double>& values) {
    double sum_sq = 0.0;
    for (double v : values) {
        sum_sq += v * v;
    }
    return std::sqrt(sum_sq);
}

[[nodiscard]] bool valid_options(const OptimizationOptions& options) {
    return options.max_iterations > 0 && options.tolerance > 0.0 && options.learning_rate > 0.0;
}
}  // namespace

std::string GradientDescentOptimizer::name() const {
    return "gradient_descent";
}

OptimizationResult GradientDescentOptimizer::optimize(const ObjectiveFunction& function,
                                                      std::vector<double> parameters,
                                                      const OptimizationOptions& options) const {
    if (!valid_options(options)) {
        throw std::invalid_argument("invalid optimization options");
    }
    OptimizationResult result;
    result.parameters = std::move(parameters);

    const double decrease_factor = 0.5;
    const double increase_factor = 1.05;
    const double min_step = 1e-8;
    double step_size = options.learning_rate;
    std::vector<double> candidate(result.parameters.size());
    std::vector<double> candidate_gradient(result.parameters.size());

    for (std::size_t iter = 0; iter < options.max_iterations; ++iter) {
        const auto gradient = function.gradient(result.parameters);
        const double grad_norm = norm2(gradient);
        const double objective = function.value(result.parameters);

        result.iterations = iter + 1;
        result.gradient_norm = grad_norm;
        result.objective_value = objective;

        if (grad_norm <= options.tolerance) {
            result.converged = true;
            break;
        }

        bool accepted = false;
        double step = step_size;
        double candidate_objective = objective;
        double candidate_grad_norm = grad_norm;
        for (int backtrack = 0; backtrack < 12; ++backtrack) {
            for (std::size_t i = 0; i < result.parameters.size(); ++i) {
                candidate[i] = result.parameters[i] - step * gradient[i];
            }
            candidate_objective = function.value(candidate);
            candidate_gradient = function.gradient(candidate);
            candidate_grad_norm = norm2(candidate_gradient);
            if (candidate_objective <= objective || candidate_grad_norm < grad_norm) {
                accepted = true;
                break;
            }
            step *= decrease_factor;
            if (step < min_step) {
                break;
            }
        }

        if (!accepted) {
            step_size = std::max(step_size * decrease_factor, min_step);
            continue;  // try again with a smaller global step
        }

        result.parameters = candidate;
        result.objective_value = candidate_objective;
        result.gradient_norm = candidate_grad_norm;
        step_size = std::min(step * increase_factor, options.learning_rate * 4.0);
    }

    if (!result.converged) {
        result.objective_value = function.value(result.parameters);
        result.gradient_norm = norm2(function.gradient(result.parameters));
    }

    return result;
}

std::unique_ptr<Optimizer> make_gradient_descent_optimizer() {
    return std::make_unique<GradientDescentOptimizer>();
}

class LBFGSFunctor {
public:
    LBFGSFunctor(const ObjectiveFunction& function) : function_(function) {}

    double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        std::vector<double> params(x.data(), x.data() + x.size());
        double value = function_.value(params);
        std::vector<double> g = function_.gradient(params);
        
        if (g.size() != static_cast<std::size_t>(grad.size())) {
             throw std::runtime_error("Gradient dimension mismatch");
        }
        for(std::size_t i=0; i<g.size(); ++i) grad[i] = g[i];
        
        return value;
    }

private:
    const ObjectiveFunction& function_;
};

std::string LBFGSOptimizer::name() const {
    return "lbfgs";
}

OptimizationResult LBFGSOptimizer::optimize(const ObjectiveFunction& function,
                                            std::vector<double> initial_parameters,
                                            const OptimizationOptions& options) const {
    if (!valid_options(options)) {
        throw std::invalid_argument("invalid optimization options");
    }

    LBFGSpp::LBFGSParam<double> param;
    param.epsilon = options.tolerance;
    param.max_iterations = options.max_iterations;
    
    LBFGSpp::LBFGSSolver<double> solver(param);
    LBFGSFunctor functor(function);
    
    Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(initial_parameters.data(), initial_parameters.size());
    double fx = 0.0;
    
    int niter = 0;
    try {
        niter = solver.minimize(functor, x, fx);
    } catch (...) {
        GradientDescentOptimizer fallback;
        std::vector<double> current(x.data(), x.data() + x.size());
        return fallback.optimize(function, current, options);
    }

    OptimizationResult result;
    result.parameters.assign(x.data(), x.data() + x.size());
    result.objective_value = fx;
    result.iterations = niter;

    std::vector<double> final_grad = function.gradient(result.parameters);
    result.gradient_norm = norm2(final_grad);
    result.converged = (result.gradient_norm <= options.tolerance);

    if (!result.converged) {
        GradientDescentOptimizer fallback;
        return fallback.optimize(function, result.parameters, options);
    }

    return result;
}

std::unique_ptr<Optimizer> make_lbfgs_optimizer() {
    return std::make_unique<LBFGSOptimizer>();
}

}  // namespace libsemx
