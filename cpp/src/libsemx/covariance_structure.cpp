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

std::vector<std::vector<double>> CovarianceStructure::parameter_gradients(const std::vector<double>& /*parameters*/) const {
    throw std::runtime_error("parameter_gradients not implemented for this covariance structure");
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

std::vector<std::vector<double>> UnstructuredCovariance::parameter_gradients(const std::vector<double>& parameters) const {
    validate_parameters(parameters);
    const std::size_t dim = dimension();
    const std::size_t pc = parameter_count();
    std::vector<std::vector<double>> grads(pc, std::vector<double>(dim * dim, 0.0));

    std::size_t param_index = 0;
    for (std::size_t row = 0; row < dim; ++row) {
        for (std::size_t col = 0; col <= row; ++col) {
            // param_index corresponds to (row, col)
            grads[param_index][row * dim + col] = 1.0;
            if (row != col) {
                grads[param_index][col * dim + row] = 1.0;
            }
            param_index++;
        }
    }
    return grads;
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

std::vector<std::vector<double>> DiagonalCovariance::parameter_gradients(const std::vector<double>& parameters) const {
    validate_parameters(parameters);
    const std::size_t dim = dimension();
    const std::size_t pc = parameter_count();
    std::vector<std::vector<double>> grads(pc, std::vector<double>(dim * dim, 0.0));

    for (std::size_t i = 0; i < dim; ++i) {
        // param i corresponds to (i, i)
        grads[i][i * dim + i] = 1.0;
    }
    return grads;
}

}  // namespace libsemx
