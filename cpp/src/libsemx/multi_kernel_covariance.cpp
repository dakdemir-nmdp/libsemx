#include "libsemx/multi_kernel_covariance.hpp"
#include <stdexcept>

namespace libsemx {

MultiKernelCovariance::MultiKernelCovariance(std::vector<std::vector<double>> kernels, std::size_t dimension)
    : CovarianceStructure(dimension, 1 + kernels.size()), kernels_(std::move(kernels)) {
    if (kernels_.empty()) {
        throw std::invalid_argument("At least one kernel matrix is required");
    }
    for (const auto& k : kernels_) {
        if (k.size() != dimension * dimension) {
            throw std::invalid_argument("Kernel matrix size does not match dimension squared");
        }
    }
}

void MultiKernelCovariance::fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const {
    // Parameters: [sigma_sq, w_1, w_2, ..., w_k]
    // Total parameters = 1 + k
    
    if (parameters.size() != 1 + kernels_.size()) {
        throw std::invalid_argument("Incorrect number of parameters for MultiKernelCovariance");
    }

    double sigma_sq = parameters[0];
    if (sigma_sq < 0) {
        throw std::invalid_argument("Variance parameter (sigma_sq) must be non-negative");
    }

    // Initialize matrix with zeros
    std::fill(matrix.begin(), matrix.end(), 0.0);

    for (std::size_t k = 0; k < kernels_.size(); ++k) {
        double w = parameters[1 + k];
        // Weights can be negative? Usually non-negative for valid covariance.
        // But we'll let the optimizer/user decide constraints.
        // However, if we want to ensure PSD, w_i >= 0 is typical if kernels are PSD.
        
        double scale = sigma_sq * w;
        const auto& kernel = kernels_[k];
        for (std::size_t i = 0; i < matrix.size(); ++i) {
            matrix[i] += scale * kernel[i];
        }
    }
}

std::vector<std::vector<double>> MultiKernelCovariance::parameter_gradients(const std::vector<double>& parameters) const {
    validate_parameters(parameters);
    double sigma_sq = parameters[0];
    
    std::vector<std::vector<double>> grads;
    grads.reserve(parameters.size());
    
    // Gradient w.r.t sigma_sq: sum(w_k * K_k)
    std::vector<double> d_sigma(dimension() * dimension(), 0.0);
    for (std::size_t k = 0; k < kernels_.size(); ++k) {
        double w = parameters[1 + k];
        const auto& kernel = kernels_[k];
        for (std::size_t i = 0; i < d_sigma.size(); ++i) {
            d_sigma[i] += w * kernel[i];
        }
    }
    grads.push_back(std::move(d_sigma));
    
    // Gradient w.r.t w_k: sigma_sq * K_k
    for (std::size_t k = 0; k < kernels_.size(); ++k) {
        std::vector<double> d_w(dimension() * dimension());
        const auto& kernel = kernels_[k];
        for (std::size_t i = 0; i < d_w.size(); ++i) {
            d_w[i] = sigma_sq * kernel[i];
        }
        grads.push_back(std::move(d_w));
    }
    
    return grads;
}

}  // namespace libsemx
