#include "libsemx/exponential_outcome.hpp"

#include "libsemx/weibull_outcome.hpp"

namespace libsemx {

OutcomeEvaluation ExponentialOutcome::evaluate(double observed,
                                               double linear_predictor,
                                               double dispersion,
                                               double status,
                                               const std::vector<double>& extra_params) const {
    (void)dispersion;
    static const WeibullOutcome kWeibull;
    return kWeibull.evaluate(observed, linear_predictor, 1.0, status, extra_params);
}

}  // namespace libsemx
