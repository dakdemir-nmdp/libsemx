#pragma once

#include <cstddef>
#include <vector>

namespace libsemx {

struct OutcomeEvaluation {
    double log_likelihood{0.0};
    double first_derivative{0.0};
    double second_derivative{0.0};
    double third_derivative{0.0};
    double d_dispersion{0.0};
    std::vector<double> d_extra_params{};
    std::vector<double> d_hessian_d_extra_params{};
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
