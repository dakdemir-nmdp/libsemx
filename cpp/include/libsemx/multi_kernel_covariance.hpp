#pragma once

#include "libsemx/covariance_structure.hpp"
#include <vector>

namespace libsemx {

class MultiKernelCovariance final : public CovarianceStructure {
public:
    MultiKernelCovariance(std::vector<std::vector<double>> kernels, std::size_t dimension);

protected:
    void fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const override;

private:
    std::vector<std::vector<double>> kernels_;
};

}  // namespace libsemx
