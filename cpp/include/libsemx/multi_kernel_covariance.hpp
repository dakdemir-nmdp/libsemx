#pragma once

#include "libsemx/covariance_structure.hpp"
#include <vector>

namespace libsemx {

class MultiKernelCovariance final : public CovarianceStructure {
public:
    MultiKernelCovariance(std::vector<std::vector<double>> kernels, std::size_t dimension, bool simplex_weights = false);

    [[nodiscard]] std::vector<std::vector<double>> parameter_gradients(const std::vector<double>& parameters) const override;

protected:
    void fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const override;

private:
    std::vector<std::vector<double>> kernels_;
    bool simplex_weights_;
};

}  // namespace libsemx
