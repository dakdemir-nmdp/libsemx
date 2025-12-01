#include "libsemx/covariance_structure.hpp"

#include <cmath>
#include <stdexcept>

namespace libsemx {

namespace {
[[nodiscard]] std::size_t triangular_number(std::size_t n) {
    return n * (n + 1) / 2;
}
}  // namespace

CovarianceStructure::CovarianceStructure(std::size_t dimension, std::size_t parameter_count)
    : dimension_(dimension), parameter_count_(parameter_count) {
    if (dimension_ == 0) {
        throw std::invalid_argument("covariance dimension must be positive");
    }
    // parameter_count can be 0 for fixed structures
}

std::vector<double> CovarianceStructure::materialize(const std::vector<double>& parameters) const {
    validate_parameters(parameters);
    std::vector<double> matrix(dimension_ * dimension_, 0.0);
    fill_covariance(parameters, matrix);
    return matrix;
}

void CovarianceStructure::validate_parameters(const std::vector<double>& parameters) const {
    if (parameters.size() != parameter_count_) {
        throw std::invalid_argument("parameter count mismatch for covariance structure");
    }
}

UnstructuredCovariance::UnstructuredCovariance(std::size_t dimension)
    : CovarianceStructure(dimension, triangular_number(dimension)) {}

void UnstructuredCovariance::fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const {
    const std::size_t dim = dimension();
    std::size_t param_index = 0;
    for (std::size_t row = 0; row < dim; ++row) {
        for (std::size_t col = 0; col <= row; ++col) {
            const double value = parameters[param_index++];
            matrix[row * dim + col] = value;
            matrix[col * dim + row] = value;
        }
    }
}

void UnstructuredCovariance::validate_parameters(const std::vector<double>& parameters) const {
    CovarianceStructure::validate_parameters(parameters);
    const std::size_t dim = dimension();
    std::size_t param_index = 0;
    for (std::size_t row = 0; row < dim; ++row) {
        for (std::size_t col = 0; col <= row; ++col) {
            const double value = parameters[param_index++];
            if (row == col && !(value > 0.0)) {
                throw std::invalid_argument("unstructured covariance requires positive diagonal elements");
            }
        }
    }
}

DiagonalCovariance::DiagonalCovariance(std::size_t dimension)
    : CovarianceStructure(dimension, dimension) {}

void DiagonalCovariance::fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const {
    const std::size_t dim = dimension();
    for (std::size_t i = 0; i < dim; ++i) {
        matrix[i * dim + i] = parameters[i];
    }
}

void DiagonalCovariance::validate_parameters(const std::vector<double>& parameters) const {
    CovarianceStructure::validate_parameters(parameters);
    for (double value : parameters) {
        if (!(value > 0.0)) {
            throw std::invalid_argument("diagonal covariance requires strictly positive entries");
        }
    }
}

}  // namespace libsemx
