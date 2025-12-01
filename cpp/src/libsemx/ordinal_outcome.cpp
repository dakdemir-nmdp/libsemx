#include "libsemx/ordinal_outcome.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>

namespace libsemx {

namespace {
// Standard Normal CDF
double phi(double x) {
    return 0.5 * std::erfc(-x * 0.707106781186547524401); // 1/sqrt(2)
}

// Standard Normal PDF
double dphi(double x) {
    constexpr double inv_sqrt_2pi = 0.39894228040143267794;
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}
}

OutcomeEvaluation OrdinalOutcome::evaluate(double observed,
                                            double linear_predictor,
                                            double /*dispersion*/,
                                            double /*status*/,
                                            const std::vector<double>& extra_params) const {
    // extra_params contains thresholds: tau_1, tau_2, ..., tau_{K-1}
    // Categories are 0, 1, ..., K
    // tau_0 = -inf, tau_K = +inf
    
    if (extra_params.empty()) {
        throw std::invalid_argument("Ordinal outcome requires thresholds in extra_params");
    }

    // Validate thresholds are sorted
    for (size_t i = 0; i < extra_params.size() - 1; ++i) {
        if (extra_params[i] >= extra_params[i+1]) {
            throw std::invalid_argument("Ordinal thresholds must be strictly increasing");
        }
    }

    int k = static_cast<int>(observed);
    if (static_cast<double>(k) != observed || k < 0) {
        throw std::invalid_argument("Ordinal observed value must be a non-negative integer");
    }
    
    // Number of thresholds is K-1. Max category is K.
    // If we have M thresholds, we have M+1 categories: 0, ..., M.
    if (k > static_cast<int>(extra_params.size())) {
        throw std::invalid_argument("Observed category exceeds number of thresholds + 1");
    }

    double eta = linear_predictor;
    
    double lower_tau = (k == 0) ? -std::numeric_limits<double>::infinity() : extra_params[k-1];
    double upper_tau = (k == static_cast<int>(extra_params.size())) ? std::numeric_limits<double>::infinity() : extra_params[k];

    double prob_upper = (upper_tau == std::numeric_limits<double>::infinity()) ? 1.0 : phi(upper_tau - eta);
    double prob_lower = (lower_tau == -std::numeric_limits<double>::infinity()) ? 0.0 : phi(lower_tau - eta);
    
    double prob = prob_upper - prob_lower;
    
    // Numerical stability
    if (prob < 1e-15) prob = 1e-15;

    double log_lik = std::log(prob);

    // Derivatives w.r.t eta
    // P = Phi(u - eta) - Phi(l - eta)
    // dP/deta = -phi(u - eta) - (-phi(l - eta)) = phi(l - eta) - phi(u - eta)
    // dLL/deta = (1/P) * dP/deta
    
    double dens_upper = (upper_tau == std::numeric_limits<double>::infinity()) ? 0.0 : dphi(upper_tau - eta);
    double dens_lower = (lower_tau == -std::numeric_limits<double>::infinity()) ? 0.0 : dphi(lower_tau - eta);
    
    double dP_deta = dens_lower - dens_upper;
    double grad = dP_deta / prob;

    // d2LL/deta2 = (P * d2P/deta2 - (dP/deta)^2) / P^2
    // d2P/deta2 = d/deta (phi(l-eta) - phi(u-eta))
    // d/dx phi(x) = -x * phi(x)
    // d/deta phi(c - eta) = -phi'(c-eta) = -(-(c-eta)*phi(c-eta)) = (c-eta)*phi(c-eta)
    
    double d2P_deta2 = 0.0;
    if (lower_tau != -std::numeric_limits<double>::infinity()) {
        d2P_deta2 += (lower_tau - eta) * dens_lower;
    }
    if (upper_tau != std::numeric_limits<double>::infinity()) {
        d2P_deta2 -= (upper_tau - eta) * dens_upper;
    }
    
    double hess = (prob * d2P_deta2 - dP_deta * dP_deta) / (prob * prob);
    double d3P_deta3 = 0.0;
    if (lower_tau != -std::numeric_limits<double>::infinity()) {
        double x = lower_tau - eta;
        d3P_deta3 += (x * x - 1.0) * dens_lower;
    }
    if (upper_tau != std::numeric_limits<double>::infinity()) {
        double x = upper_tau - eta;
        d3P_deta3 -= (x * x - 1.0) * dens_upper;
    }
    const double prob_sq = prob * prob;
    const double prob_cu = prob_sq * prob;
    double third = (prob_sq * d3P_deta3 - 3.0 * prob * d2P_deta2 * dP_deta + 2.0 * dP_deta * dP_deta * dP_deta) / prob_cu;
    return {log_lik, grad, hess, third};
}

double OrdinalOutcome::default_dispersion(std::size_t /*n*/) const {
    return 1.0; 
}

}  // namespace libsemx
