#pragma once

#include "libsemx/outcome_family.hpp"

namespace libsemx {

class LognormalOutcome final : public OutcomeFamily {
public:
    [[nodiscard]] OutcomeEvaluation evaluate(double observed,
                                              double linear_predictor,
                                              double dispersion,
                                              double status = 1.0,
                                              const std::vector<double>& extra_params = {}) const override;
};

}  // namespace libsemx
