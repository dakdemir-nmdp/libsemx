#include "libsemx/genomic_kernel.hpp"

#include <stdexcept>

namespace libsemx {

namespace {

[[nodiscard]] std::vector<double> compute_allele_frequencies(const std::vector<double>& markers,
                                                             std::size_t n_individuals,
                                                             std::size_t n_markers) {
    std::vector<double> freqs(n_markers, 0.0);
    if (markers.size() != n_individuals * n_markers) {
        throw std::invalid_argument("marker matrix size mismatch for GRM construction");
    }
    for (std::size_t j = 0; j < n_markers; ++j) {
        double sum = 0.0;
        for (std::size_t i = 0; i < n_individuals; ++i) {
            sum += markers[i * n_markers + j];
        }
        freqs[j] = sum / (2.0 * static_cast<double>(n_individuals));
    }
    return freqs;
}

[[nodiscard]] double scaling_denominator(const std::vector<double>& freqs) {
    double denom = 0.0;
    for (double p : freqs) {
        denom += p * (1.0 - p);
    }
    denom *= 2.0;
    if (!(denom > 0.0)) {
        throw std::invalid_argument("invalid allele frequencies: GRM denominator is non-positive");
    }
    return denom;
}

[[nodiscard]] double mean_diagonal(const std::vector<double>& matrix, std::size_t dimension) {
    if (dimension == 0 || matrix.size() != dimension * dimension) {
        throw std::invalid_argument("matrix size mismatch while normalizing GRM");
    }
    double trace = 0.0;
    for (std::size_t i = 0; i < dimension; ++i) {
        trace += matrix[i * dimension + i];
    }
    return trace / static_cast<double>(dimension);
}

[[nodiscard]] std::vector<double> accumulate_grm(const std::vector<double>& markers,
                                                 const std::vector<double>& freqs,
                                                 std::size_t n_individuals,
                                                 std::size_t n_markers,
                                                 bool center,
                                                 double denom) {
    std::vector<double> kernel(n_individuals * n_individuals, 0.0);

    for (std::size_t i = 0; i < n_individuals; ++i) {
        for (std::size_t k = i; k < n_individuals; ++k) {
            double sum = 0.0;
            for (std::size_t j = 0; j < n_markers; ++j) {
                double lhs = markers[i * n_markers + j];
                double rhs = markers[k * n_markers + j];
                if (center) {
                    const double adjustment = 2.0 * freqs[j];
                    lhs -= adjustment;
                    rhs -= adjustment;
                }
                sum += lhs * rhs;
            }
            const double value = sum / denom;
            kernel[i * n_individuals + k] = value;
            if (i != k) {
                kernel[k * n_individuals + i] = value;
            }
        }
    }

    return kernel;
}

void normalize_kernel(std::vector<double>& kernel, std::size_t dimension) {
    const double diag_mean = mean_diagonal(kernel, dimension);
    if (!(diag_mean > 0.0)) {
        throw std::invalid_argument("cannot normalize GRM: non-positive diagonal mean");
    }
    const double scale = 1.0 / diag_mean;
    for (double& v : kernel) {
        v *= scale;
    }
}

}  // namespace

std::vector<double> GenomicRelationshipMatrix::vanraden(const std::vector<double>& markers,
                                                        std::size_t n_individuals,
                                                        std::size_t n_markers,
                                                        GenomicKernelOptions options) {
    if (n_individuals == 0 || n_markers == 0) {
        throw std::invalid_argument("GRM requires non-zero individuals and markers");
    }

    const auto freqs = compute_allele_frequencies(markers, n_individuals, n_markers);
    const double denom = scaling_denominator(freqs);
    auto kernel = accumulate_grm(markers, freqs, n_individuals, n_markers, options.center, denom);

    if (options.normalize) {
        normalize_kernel(kernel, n_individuals);
    }

    return kernel;
}

std::vector<double> GenomicRelationshipMatrix::kronecker(const std::vector<double>& left,
                                                         std::size_t left_dim,
                                                         const std::vector<double>& right,
                                                         std::size_t right_dim) {
    if (left_dim == 0 || right_dim == 0) {
        throw std::invalid_argument("kronecker requires non-zero dimensions");
    }
    if (left.size() != left_dim * left_dim || right.size() != right_dim * right_dim) {
        throw std::invalid_argument("matrix size mismatch for kronecker product");
    }

    const std::size_t out_dim = left_dim * right_dim;
    std::vector<double> result(out_dim * out_dim, 0.0);

    for (std::size_t i = 0; i < left_dim; ++i) {
        for (std::size_t j = 0; j < left_dim; ++j) {
            const double scale = left[i * left_dim + j];
            for (std::size_t r = 0; r < right_dim; ++r) {
                for (std::size_t c = 0; c < right_dim; ++c) {
                    const std::size_t row = i * right_dim + r;
                    const std::size_t col = j * right_dim + c;
                    result[row * out_dim + col] = scale * right[r * right_dim + c];
                }
            }
        }
    }

    return result;
}

GenomicKernelCovariance::GenomicKernelCovariance(std::vector<double> kernel, std::size_t dimension)
    : CovarianceStructure(dimension, 1), kernel_(std::move(kernel)) {
    if (kernel_.size() != dimension * dimension) {
        throw std::invalid_argument("genomic kernel size mismatch");
    }
}

void GenomicKernelCovariance::fill_covariance(const std::vector<double>& parameters,
                                              std::vector<double>& matrix) const {
    validate_parameters(parameters);
    const double variance = parameters[0];
    if (!(variance > 0.0)) {
        throw std::invalid_argument("genomic kernel variance must be positive");
    }
    if (matrix.size() != kernel_.size()) {
        throw std::invalid_argument("matrix size mismatch in genomic kernel covariance");
    }
    const std::size_t dim = dimension();
    for (std::size_t i = 0; i < matrix.size(); ++i) {
        matrix[i] = variance * kernel_[i];
    }
    // Add a tiny jitter on the diagonal to guard against near-PSD kernels.
    const double jitter = 1e-6;
    for (std::size_t i = 0; i < dim; ++i) {
        matrix[i * dim + i] += jitter;
    }
}

std::vector<std::vector<double>> GenomicKernelCovariance::parameter_gradients(
    const std::vector<double>& parameters) const {
    validate_parameters(parameters);
    return {kernel_};
}

}  // namespace libsemx
