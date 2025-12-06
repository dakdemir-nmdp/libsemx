#include "libsemx/gaussian_outcome.hpp"

#include <cmath>
#include <stdexcept>

namespace libsemx {

namespace {
constexpr double kTwoPi = 6.28318530717958647692;
}

OutcomeEvaluation GaussianOutcome::evaluate(double observed,
                                            double linear_predictor,
                                            double dispersion,
                                            double /*status*/,
                                            const std::vector<double>& /*extra_params*/) const {
    if (!(dispersion > 0.0)) {
        throw std::invalid_argument("gaussian dispersion must be positive: " + std::to_string(dispersion));
    }
    const double variance = dispersion;
    const double residual = observed - linear_predictor;
    const double inv_variance = 1.0 / variance;
    OutcomeEvaluation eval;
    eval.log_likelihood = -0.5 * (std::log(kTwoPi * variance) + residual * residual * inv_variance);
    eval.first_derivative = residual * inv_variance;
    eval.second_derivative = -inv_variance;
    eval.third_derivative = 0.0;
    
    // d(ll)/d(sigma^2) = -0.5/sigma^2 + 0.5 * resid^2 / (sigma^2)^2
    //                  = 0.5 * (resid^2/sigma^2 - 1) / sigma^2
    eval.d_dispersion = 0.5 * (residual * residual * inv_variance - 1.0) * inv_variance;
    
    return eval;
}

double GaussianOutcome::default_dispersion(std::size_t n) const {
    if (n == 0) {
        throw std::invalid_argument("gaussian default dispersion requires at least one observation");
    }
    return 1.0;
}

}  // namespace libsemx
