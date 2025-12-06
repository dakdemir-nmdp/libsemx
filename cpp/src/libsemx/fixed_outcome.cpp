#include "libsemx/fixed_outcome.hpp"

namespace libsemx {

OutcomeEvaluation FixedOutcome::evaluate(double /*observed*/,
                                          double /*linear_predictor*/,
                                          double /*dispersion*/,
                                          double /*status*/,
                                          const std::vector<double>& /*extra_params*/) const {
    return {0.0, 0.0, 0.0, 0.0, 0.0, {}};
}

}  // namespace libsemx
