#pragma once

#include "libsemx/outcome_family.hpp"
#include <vector>
#include <memory>
#include <string>

namespace libsemx {

class MixedOutcome : public OutcomeFamily {
public:
    explicit MixedOutcome(const std::string& config);

    [[nodiscard]] OutcomeEvaluation evaluate(double observed,
                                              double linear_predictor,
                                              double dispersion,
                                              double status = 1.0,
                                              const std::vector<double>& extra_params = {}) const override;

    [[nodiscard]] bool has_dispersion() const noexcept override;
    [[nodiscard]] double default_dispersion(std::size_t n) const override;

private:
    struct SubFamily {
        std::unique_ptr<OutcomeFamily> family;
        std::size_t extra_param_count;
        std::size_t extra_param_offset;
    };

    std::vector<SubFamily> families_;
};

}  // namespace libsemx
