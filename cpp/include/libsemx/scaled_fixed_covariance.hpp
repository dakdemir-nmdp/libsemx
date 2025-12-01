#pragma once

#include "libsemx/covariance_structure.hpp"
#include <vector>

namespace libsemx {

class ScaledFixedCovariance final : public CovarianceStructure {
public:
    ScaledFixedCovariance(std::vector<double> fixed_matrix, std::size_t dimension);

protected:
    void fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const override;

private:
    std::vector<double> fixed_matrix_;
};

}  // namespace libsemx
