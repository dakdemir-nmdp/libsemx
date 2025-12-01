#pragma once

#include "libsemx/covariance_structure.hpp"
#include <memory>
#include <vector>

namespace libsemx {

class KroneckerCovariance final : public CovarianceStructure {
public:
    KroneckerCovariance(std::vector<std::unique_ptr<CovarianceStructure>> components, bool learn_scale = false);

protected:
    void fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const override;

    [[nodiscard]] std::vector<std::vector<double>> parameter_gradients(const std::vector<double>& parameters) const override;

private:
    std::vector<std::unique_ptr<CovarianceStructure>> components_;
    bool learn_scale_;
};

}  // namespace libsemx
