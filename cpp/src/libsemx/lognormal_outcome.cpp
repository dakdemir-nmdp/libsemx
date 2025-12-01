#include "libsemx/lognormal_outcome.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace libsemx {
namespace {

const double kLogSqrtTwoPi = 0.5 * std::log(2.0 * std::numbers::pi_v<double>);
const double kInvSqrtTwoPi = 1.0 / std::sqrt(2.0 * std::numbers::pi_v<double>);

double normal_pdf(double z) {
    return kInvSqrtTwoPi * std::exp(-0.5 * z * z);
}

double normal_survival(double z) {
    return 0.5 * std::erfc(z / std::numbers::sqrt2_v<double>);
}

}  // namespace

OutcomeEvaluation LognormalOutcome::evaluate(double observed,
                                             double linear_predictor,
                                             double dispersion,
                                             double status,
                                             const std::vector<double>& /*extra_params*/) const {
    if (observed <= 0.0) {
        throw std::invalid_argument("Lognormal observed time must be positive");
    }
    if (dispersion <= 0.0) {
        throw std::invalid_argument("Lognormal dispersion (sigma) must be positive");
    }

    const double sigma = dispersion;
    const double inv_sigma = 1.0 / sigma;
    const double inv_sigma_sq = inv_sigma * inv_sigma;

    const double log_t = std::log(observed);
    const double z = (log_t - linear_predictor) * inv_sigma;
    const double pdf = normal_pdf(z);
    const double survival = std::max(normal_survival(z), std::numeric_limits<double>::min());

    const double delta = status;
    const double censor = 1.0 - delta;

    const double loglik_event = -0.5 * z * z - log_t - std::log(sigma) - kLogSqrtTwoPi;
    const double loglik_censored = std::log(survival);
    double loglik = delta * loglik_event + censor * loglik_censored;

    double grad = delta * (z * inv_sigma);
    double hess = -delta * inv_sigma_sq;
    double third = 0.0;

    if (censor != 0.0) {
        const double hazard_ratio = pdf / survival;
        grad += censor * (hazard_ratio * inv_sigma);

        const double first_ratio = (pdf / (survival * survival)) * (-z * survival + pdf);
        hess -= censor * (first_ratio * inv_sigma_sq);

        const double second_ratio = (pdf / (survival * survival * survival)) *
            (((z * z - 1.0) * survival * survival) - 3.0 * z * pdf * survival + 2.0 * pdf * pdf);
        third += censor * (second_ratio * inv_sigma_sq * inv_sigma);
    }

    return {loglik, grad, hess, third};
}

}  // namespace libsemx
