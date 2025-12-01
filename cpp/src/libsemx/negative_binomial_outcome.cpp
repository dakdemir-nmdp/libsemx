#include "libsemx/negative_binomial_outcome.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace libsemx {

OutcomeEvaluation NegativeBinomialOutcome::evaluate(double observed,
                                                     double linear_predictor,
                                                     double dispersion,
                                                     double /*status*/,
                                                     const std::vector<double>& /*extra_params*/) const {
    if (observed < 0 || !std::isfinite(observed)) {
        throw std::invalid_argument("negative binomial observed value must be non-negative finite");
    }
    if (dispersion <= 0) {
        throw std::invalid_argument("negative binomial dispersion (size) must be positive");
    }

    const double mu = std::exp(linear_predictor);
    const double k = dispersion;
    const double y = observed;

    // Log-likelihood for negative binomial
    // loglik = lgamma(y + k) - lgamma(k) - lgamma(y+1) + k*log(k) - (k+y)*log(k+mu) + y*log(mu)
    double loglik = std::lgamma(y + k) - std::lgamma(k) - std::lgamma(y + 1.0);
    loglik += k * std::log(k) - (k + y) * std::log(k + mu) + y * std::log(mu);

    // GLM derivatives for NB
    const double variance = mu + mu * mu / k;  // V = mu + mu^2/k
    const double score = (y - mu) * mu / variance;  // d loglik / d eta

    // Hessian: d² loglik / d eta²
    // = mu/V * (y - mu - mu * dV/d mu) + (y - mu) * d/d eta (mu/V)
    // dV/d mu = 1 + 2*mu/k
    // d/d eta (mu/V) = mu * (-1/V^2) * dV/d eta, where dV/d eta = dV/d mu * mu
    const double dV_dmu = 1.0 + 2.0 * mu / k;
    const double dVdeta = dV_dmu * mu;
    const double d_mudivV_deta = mu * (-1.0 / (variance * variance)) * dVdeta;
    const double hessian = mu / variance * (y - mu - mu * dV_dmu) + (y - mu) * d_mudivV_deta;

    OutcomeEvaluation eval;
    eval.log_likelihood = loglik;
    eval.first_derivative = score;
    eval.second_derivative = hessian;

    return eval;
}

double NegativeBinomialOutcome::default_dispersion(std::size_t n) const {
    if (n == 0) {
        throw std::invalid_argument("negative binomial default dispersion requires at least one observation");
    }
    return 1.0;  // Default size parameter
}

}  // namespace libsemx