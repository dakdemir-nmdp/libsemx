#include "libsemx/scaled_fixed_covariance.hpp"
#include <stdexcept>

namespace libsemx {

ScaledFixedCovariance::ScaledFixedCovariance(std::vector<double> fixed_matrix, std::size_t dimension)
    : CovarianceStructure(dimension, 1), fixed_matrix_(std::move(fixed_matrix)) {
    if (fixed_matrix_.size() != dimension * dimension) {
        throw std::invalid_argument("Fixed matrix size does not match dimension squared");
    }
}

void ScaledFixedCovariance::fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const {
    double scale = parameters[0];
    if (scale < 0) {
        throw std::invalid_argument("Scale parameter must be non-negative");
    }
    
    for (std::size_t i = 0; i < fixed_matrix_.size(); ++i) {
        matrix[i] = fixed_matrix_[i] * scale;
    }
}

}  // namespace libsemx
