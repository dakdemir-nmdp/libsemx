#pragma once

#include "libsemx/outcome_family.hpp"

namespace libsemx {

class GaussianOutcome final : public OutcomeFamily {
public:
    [[nodiscard]] OutcomeEvaluation evaluate(double observed,
                                              double linear_predictor,
                                              double dispersion,
                                              double status = 1.0,
                                              const std::vector<double>& extra_params = {}) const override;

    [[nodiscard]] double default_dispersion(std::size_t n) const override;
};

}  // namespace libsemx
