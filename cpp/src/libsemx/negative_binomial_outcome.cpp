#include "libsemx/negative_binomial_outcome.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace libsemx {

namespace {
double digamma(double x) {
    double result = 0, xx, xx2, xx4;
    for ( ; x < 7; ++x)
        result -= 1.0/x;
    x -= 1.0/2.0;
    xx = 1.0/x;
    xx2 = xx*xx;
    xx4 = xx2*xx2;
    result += std::log(x) + (1.0/24.0)*xx2 - (7.0/960.0)*xx4 + (31.0/8064.0)*xx4*xx2 - (127.0/30720.0)*xx4*xx4;
    return result;
}
}

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

    // Derivatives wrt eta (where mu = exp(eta))
    // First derivative (score): k(y - mu) / (k + mu)
    const double denom = k + mu;
    const double score = k * (y - mu) / denom;

    // Second derivative (Hessian): -k * mu * (k + y) / (k + mu)^2
    const double hessian = -k * mu * (k + y) / (denom * denom);

    // Third derivative: -k * mu * (k + y) * (k - mu) / (k + mu)^3
    const double third = -k * mu * (k + y) * (k - mu) / (denom * denom * denom);

    OutcomeEvaluation eval;
    eval.log_likelihood = loglik;
    eval.first_derivative = score;
    eval.second_derivative = hessian;
    eval.third_derivative = third;
    
    // d(ll)/dk = psi(y+k) - psi(k) + log(k) + 1 - log(k+mu) - (k+y)/(k+mu)
    eval.d_dispersion = digamma(y + k) - digamma(k) + std::log(k) + 1.0 - std::log(k + mu) - (k + y) / denom;

    return eval;
}

double NegativeBinomialOutcome::default_dispersion(std::size_t n) const {
    if (n == 0) {
        throw std::invalid_argument("negative binomial default dispersion requires at least one observation");
    }
    return 1.0;  // Default size parameter
}

}  // namespace libsemx