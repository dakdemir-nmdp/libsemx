#include "libsemx/binomial_outcome.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace libsemx {

namespace {
constexpr double kEps = 1e-15;  // For numerical stability
}

OutcomeEvaluation BinomialOutcome::evaluate(double observed,
                                             double linear_predictor,
                                             double /*dispersion*/,
                                             double /*status*/,
                                             const std::vector<double>& /*extra_params*/) const {
    if (!(observed == 0.0 || observed == 1.0)) {
        throw std::invalid_argument("binomial observed value must be 0 or 1");
    }

    // p = 1 / (1 + exp(-linear_predictor))
    // Use stable computation to avoid overflow
    double p;
    if (linear_predictor > 0) {
        const double exp_neg_eta = std::exp(-linear_predictor);
        p = 1.0 / (1.0 + exp_neg_eta);
    } else {
        const double exp_eta = std::exp(linear_predictor);
        p = exp_eta / (1.0 + exp_eta);
    }

    // Clamp p to avoid log(0)
    if (p < kEps) p = kEps;
    if (p > 1.0 - kEps) p = 1.0 - kEps;

    const double log_p = std::log(p);
    const double log_1mp = std::log(1.0 - p);

    OutcomeEvaluation eval;
    eval.log_likelihood = observed * log_p + (1.0 - observed) * log_1mp;
    eval.first_derivative = observed - p;
    eval.second_derivative = -p * (1.0 - p);

    return eval;
}

double BinomialOutcome::default_dispersion(std::size_t /*n*/) const {
    return 1.0;  // Not used
}

}  // namespace libsemx