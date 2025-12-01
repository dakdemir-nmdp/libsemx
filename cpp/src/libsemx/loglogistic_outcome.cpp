#include "libsemx/loglogistic_outcome.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace libsemx {

OutcomeEvaluation LogLogisticOutcome::evaluate(double observed,
                                               double linear_predictor,
                                               double dispersion,
                                               double status,
                                               const std::vector<double>& /*extra_params*/) const {
    if (observed <= 0.0) {
        throw std::invalid_argument("Log-logistic observed time must be positive");
    }
    if (dispersion <= 0.0) {
        throw std::invalid_argument("Log-logistic dispersion (shape) must be positive");
    }

    const double gamma = dispersion;
    const double log_t = std::log(observed);
    const double diff = log_t - linear_predictor;
    const double u = std::exp(gamma * diff);
    const double one_plus_u = 1.0 + u;

    const double delta = status;
    const double censor = 1.0 - delta;

    const double loglik_event = std::log(gamma) + gamma * diff - log_t - 2.0 * std::log1p(u);
    const double loglik_censored = -std::log1p(u);
    const double loglik = delta * loglik_event + censor * loglik_censored;

    const double inv_one_plus_u = 1.0 / one_plus_u;
    const double inv_one_plus_u_sq = inv_one_plus_u * inv_one_plus_u;
    const double inv_one_plus_u_cu = inv_one_plus_u_sq * inv_one_plus_u;

    const double grad_event = gamma * (u - 1.0) * inv_one_plus_u;
    const double grad_censored = gamma * u * inv_one_plus_u;
    const double grad = delta * grad_event + censor * grad_censored;

    const double base = gamma * gamma * u * inv_one_plus_u_sq;
    const double hess = -delta * (2.0 * base) - censor * base;

    const double cubic_base = gamma * gamma * gamma * u * (1.0 - u) * inv_one_plus_u_cu;
    const double third = delta * (2.0 * cubic_base) + censor * cubic_base;

    return {loglik, grad, hess, third};
}

}  // namespace libsemx
