#include "libsemx/poisson_outcome.hpp"

#include <cmath>
#include <stdexcept>

namespace libsemx {

OutcomeEvaluation PoissonOutcome::evaluate(double observed,
                                            double linear_predictor,
                                            double /*dispersion*/,
                                            double /*status*/,
                                            const std::vector<double>& /*extra_params*/) const {
    if (observed < 0.0) {
        throw std::invalid_argument("Poisson observed value must be non-negative");
    }

    // lambda = exp(eta)
    const double lambda = std::exp(linear_predictor);

    // log_lik = y * eta - lambda - log(y!)
    // log(y!) = lgamma(y + 1)
    const double log_factorial = std::lgamma(observed + 1.0);
    
    OutcomeEvaluation eval;
    eval.log_likelihood = observed * linear_predictor - lambda - log_factorial;
    eval.first_derivative = observed - lambda;
    eval.second_derivative = -lambda;
    eval.third_derivative = -lambda;

    return eval;
}

double PoissonOutcome::default_dispersion(std::size_t /*n*/) const {
    return 1.0;
}

}  // namespace libsemx
