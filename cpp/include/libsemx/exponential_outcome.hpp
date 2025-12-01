#pragma once

#include "libsemx/outcome_family.hpp"

namespace libsemx {

class ExponentialOutcome final : public OutcomeFamily {
public:
    [[nodiscard]] OutcomeEvaluation evaluate(double observed,
                                              double linear_predictor,
                                              double dispersion,
                                              double status = 1.0,
                                              const std::vector<double>& extra_params = {}) const override;

    [[nodiscard]] bool has_dispersion() const noexcept override { return false; }

    [[nodiscard]] double default_dispersion(std::size_t /*n*/) const override { return 1.0; }
};

}  // namespace libsemx
