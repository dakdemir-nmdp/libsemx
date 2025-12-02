#include "libsemx/multi_kernel_covariance.hpp"
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace libsemx {

MultiKernelCovariance::MultiKernelCovariance(std::vector<std::vector<double>> kernels, std::size_t dimension, bool simplex_weights)
    : CovarianceStructure(dimension, 1 + kernels.size()), kernels_(std::move(kernels)), simplex_weights_(simplex_weights) {
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

    std::vector<double> weights;
    weights.reserve(kernels_.size());

    if (simplex_weights_) {
        // Softmax transform
        double max_theta = -1e9; // For numerical stability
        for (size_t k = 0; k < kernels_.size(); ++k) {
            if (parameters[1 + k] > max_theta) max_theta = parameters[1 + k];
        }
        
        double sum_exp = 0.0;
        for (size_t k = 0; k < kernels_.size(); ++k) {
            double val = std::exp(parameters[1 + k] - max_theta);
            weights.push_back(val);
            sum_exp += val;
        }
        for (auto& w : weights) w /= sum_exp;
    } else {
        for (size_t k = 0; k < kernels_.size(); ++k) {
            weights.push_back(parameters[1 + k]);
        }
    }

    // Initialize matrix with zeros
    std::fill(matrix.begin(), matrix.end(), 0.0);

    for (std::size_t k = 0; k < kernels_.size(); ++k) {
        double w = weights[k];
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
    
    std::vector<double> weights;
    weights.reserve(kernels_.size());
    
    if (simplex_weights_) {
        double max_theta = -1e9;
        for (size_t k = 0; k < kernels_.size(); ++k) {
            if (parameters[1 + k] > max_theta) max_theta = parameters[1 + k];
        }
        double sum_exp = 0.0;
        for (size_t k = 0; k < kernels_.size(); ++k) {
            double val = std::exp(parameters[1 + k] - max_theta);
            weights.push_back(val);
            sum_exp += val;
        }
        for (auto& w : weights) w /= sum_exp;
    } else {
        for (size_t k = 0; k < kernels_.size(); ++k) {
            weights.push_back(parameters[1 + k]);
        }
    }
    
    std::vector<std::vector<double>> grads;
    grads.reserve(parameters.size());
    
    // Gradient w.r.t sigma_sq: sum(w_k * K_k)
    std::vector<double> d_sigma(dimension() * dimension(), 0.0);
    for (std::size_t k = 0; k < kernels_.size(); ++k) {
        double w = weights[k];
        const auto& kernel = kernels_[k];
        for (std::size_t i = 0; i < d_sigma.size(); ++i) {
            d_sigma[i] += w * kernel[i];
        }
    }
    grads.push_back(std::move(d_sigma));
    
    // Gradients w.r.t weights/thetas
    if (simplex_weights_) {
        // dK/dtheta_j = sigma_sq * sum_i (dw_i/dtheta_j * K_i)
        // dw_i/dtheta_j = w_i * (delta_ij - w_j)
        
        for (std::size_t j = 0; j < kernels_.size(); ++j) {
            std::vector<double> d_theta(dimension() * dimension(), 0.0);
            double w_j = weights[j];
            
            for (std::size_t i = 0; i < kernels_.size(); ++i) {
                double w_i = weights[i];
                double jacobian = w_i * ((i == j ? 1.0 : 0.0) - w_j);
                
                // Add contribution: sigma_sq * jacobian * K_i
                double scale = sigma_sq * jacobian;
                const auto& kernel = kernels_[i];
                for (std::size_t idx = 0; idx < d_theta.size(); ++idx) {
                    d_theta[idx] += scale * kernel[idx];
                }
            }
            grads.push_back(std::move(d_theta));
        }
    } else {
        // Gradient w.r.t w_k: sigma_sq * K_k
        for (std::size_t k = 0; k < kernels_.size(); ++k) {
            std::vector<double> d_w(dimension() * dimension());
            const auto& kernel = kernels_[k];
            for (std::size_t i = 0; i < d_w.size(); ++i) {
                d_w[i] = sigma_sq * kernel[i];
            }
            grads.push_back(std::move(d_w));
        }
    }
    
    return grads;
}

}  // namespace libsemx
