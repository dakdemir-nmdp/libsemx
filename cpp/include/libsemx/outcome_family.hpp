#pragma once

#include <cstddef>
#include <vector>

namespace libsemx {

struct OutcomeEvaluation {
    double log_likelihood;
    double first_derivative;
    double second_derivative;
    double third_derivative;
};

class OutcomeFamily {
public:
    virtual ~OutcomeFamily() = default;

    [[nodiscard]] virtual OutcomeEvaluation evaluate(double observed,
                                                      double linear_predictor,
                                                      double dispersion,
                                                      double status = 1.0,
                                                      const std::vector<double>& extra_params = {}) const = 0;

    [[nodiscard]] virtual bool has_dispersion() const noexcept { return true; }

    [[nodiscard]] virtual double default_dispersion(std::size_t /*n*/) const { return 1.0; }
};

}  // namespace libsemx
