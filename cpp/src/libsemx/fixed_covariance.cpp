#include "libsemx/fixed_covariance.hpp"
#include <stdexcept>

namespace libsemx {

FixedCovariance::FixedCovariance(std::vector<double> fixed_matrix, std::size_t dimension)
    : CovarianceStructure(dimension, 0), fixed_matrix_(std::move(fixed_matrix)) {
    if (fixed_matrix_.size() != dimension * dimension) {
        throw std::invalid_argument("Fixed matrix size does not match dimension squared");
    }
}

void FixedCovariance::fill_covariance(const std::vector<double>& /*parameters*/, std::vector<double>& matrix) const {
    matrix = fixed_matrix_;
}

}  // namespace libsemx
