#include "libsemx/weibull_outcome.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace libsemx {

OutcomeEvaluation WeibullOutcome::evaluate(double observed,
                                            double linear_predictor,
                                            double dispersion,
                                            double status,
                                            const std::vector<double>& /*extra_params*/) const {
    if (observed <= 0.0) {
        throw std::runtime_error("Weibull observed time must be positive");
    }
    if (dispersion <= 0.0) {
        throw std::runtime_error("Weibull dispersion (shape) must be positive");
    }
    if (status != 0.0 && status != 1.0) {
        // Allow weighted status? For now, strict 0/1 or maybe just warn?
        // Standard survival is 0/1. Competing risks might use other values if we use weights?
        // Let's assume status is the event indicator.
        // If status is a weight, the formula might be different.
        // For now, let's assume status is delta (0 or 1).
        // But let's not throw if it's e.g. 1.0.
    }

    const double k = dispersion;
    const double eta = linear_predictor;
    const double t = observed;
    const double delta = status;

    // z = (t * exp(-eta))^k = (t / lambda)^k
    // log(z) = k * (log(t) - eta)
    const double log_t = std::log(t);
    const double log_z = k * (log_t - eta);
    const double z = std::exp(log_z);

    // LL = delta * (log(k) + log(z) - log(t)) - z
    //    = delta * (log(k) + k*log(t) - k*eta - log(t)) - z
    //    = delta * (log(k) + (k-1)*log(t) - k*eta) - z
    
    double log_lik = -z;
    if (delta > 0.0) {
        log_lik += delta * (std::log(k) + (k - 1.0) * log_t - k * eta);
    }

    // dLL/deta = k(z - delta)
    const double grad = k * (z - delta);

    // d2LL/deta2 = -k^2 * z
    const double hess = -k * k * z;
    const double third = k * k * k * z;

    // dLL/dk
    // z = exp(k * (log_t - eta))
    // dz/dk = z * (log_t - eta)
    // LL = -z + delta * (log(k) + (k-1)log_t - k*eta)
    // dLL/dk = -z(log_t - eta) + delta * (1/k + log_t - eta)
    //        = (delta - z) * (log_t - eta) + delta/k
    
    double d_dispersion = (delta - z) * (log_t - eta);
    if (delta > 0.0) {
        d_dispersion += delta / k;
    }

    return {log_lik, grad, hess, third, d_dispersion, 0.0};
}

double WeibullOutcome::default_dispersion(std::size_t /*n*/) const {
    return 1.0; // Exponential distribution by default (k=1)
}

}  // namespace libsemx
