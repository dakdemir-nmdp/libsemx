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
        throw std::invalid_argument("gaussian dispersion must be positive");
    }
    const double variance = dispersion;
    const double residual = observed - linear_predictor;
    const double inv_variance = 1.0 / variance;
    OutcomeEvaluation eval;
    eval.log_likelihood = -0.5 * (std::log(kTwoPi * variance) + residual * residual * inv_variance);
    eval.first_derivative = residual * inv_variance;
    eval.second_derivative = -inv_variance;
    eval.third_derivative = 0.0;
    return eval;
}

double GaussianOutcome::default_dispersion(std::size_t n) const {
    if (n == 0) {
        throw std::invalid_argument("gaussian default dispersion requires at least one observation");
    }
    return 1.0;
}

}  // namespace libsemx
