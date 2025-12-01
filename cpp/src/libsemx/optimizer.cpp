#include "libsemx/optimizer.hpp"

#include <Eigen/Core>
#include <LBFGS.h>

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

        for (std::size_t i = 0; i < result.parameters.size(); ++i) {
            result.parameters[i] -= options.learning_rate * gradient[i];
        }
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
        // Fallback or rethrow? For now, we assume if it throws, optimization failed.
        // But we want to return the best state found if possible.
        // LBFGSpp throws std::runtime_error on line search failure.
    }

    OptimizationResult result;
    result.parameters.assign(x.data(), x.data() + x.size());
    result.objective_value = fx;
    result.iterations = niter;
    
    std::vector<double> final_grad = function.gradient(result.parameters);
    result.gradient_norm = norm2(final_grad);
    
    // LBFGSpp returns number of iterations. If it didn't throw, it likely converged or hit max iter.
    // We check gradient norm against tolerance to be sure.
    result.converged = (result.gradient_norm <= options.tolerance);

    return result;
}

std::unique_ptr<Optimizer> make_lbfgs_optimizer() {
    return std::make_unique<LBFGSOptimizer>();
}

}  // namespace libsemx
