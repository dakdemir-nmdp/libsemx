#include "libsemx/kronecker_covariance.hpp"
#include <stdexcept>

namespace libsemx {


namespace {
    std::size_t sum_params(const std::vector<std::unique_ptr<CovarianceStructure>>& components) {
        std::size_t sum = 0;
        for (const auto& c : components) sum += c->parameter_count();
        return sum;
    }

    std::size_t prod_dim(const std::vector<std::unique_ptr<CovarianceStructure>>& components) {
        if (components.empty()) return 0;
        std::size_t prod = 1;
        for (const auto& c : components) prod *= c->dimension();
        return prod;
    }
}

KroneckerCovariance::KroneckerCovariance(std::vector<std::unique_ptr<CovarianceStructure>> components, bool learn_scale)
    : CovarianceStructure(prod_dim(components), sum_params(components) + (learn_scale ? 1 : 0)),
      components_(std::move(components)), learn_scale_(learn_scale) {
    if (components_.empty()) {
        throw std::invalid_argument("KroneckerCovariance requires at least one component");
    }
}

void KroneckerCovariance::fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const {
    double global_scale = 1.0;
    std::size_t param_offset = 0;

    if (learn_scale_) {
        global_scale = parameters[0];
        if (global_scale < 0) throw std::invalid_argument("Scale parameter must be non-negative");
        param_offset = 1;
    }

    // Materialize first component
    std::size_t p0 = components_[0]->parameter_count();
    std::vector<double> params0(parameters.begin() + param_offset, parameters.begin() + param_offset + p0);
    std::vector<double> current_matrix = components_[0]->materialize(params0);
    param_offset += p0;

    std::size_t current_dim = components_[0]->dimension();

    if (learn_scale_) {
        double trace = 0.0;
        for (std::size_t i = 0; i < current_dim; ++i) {
            trace += current_matrix[i * current_dim + i];
        }
        double mean_diag = trace / current_dim;
        if (mean_diag > 1e-12) {
            double inv_scale = 1.0 / mean_diag;
            for (auto& val : current_matrix) val *= inv_scale;
        }
    }

    // Iteratively Kronecker product
    for (size_t i = 1; i < components_.size(); ++i) {
        std::size_t pi = components_[i]->parameter_count();
        std::vector<double> paramsi(parameters.begin() + param_offset, parameters.begin() + param_offset + pi);
        std::vector<double> next_matrix = components_[i]->materialize(paramsi);
        param_offset += pi;

        std::size_t next_dim = components_[i]->dimension();
        
        if (learn_scale_) {
            double trace = 0.0;
            for (std::size_t k = 0; k < next_dim; ++k) {
                trace += next_matrix[k * next_dim + k];
            }
            double mean_diag = trace / next_dim;
            if (mean_diag > 1e-12) {
                double inv_scale = 1.0 / mean_diag;
                for (auto& val : next_matrix) val *= inv_scale;
            }
        }

        // Compute Kronecker product: current (A) x next (B)
        std::size_t new_dim = current_dim * next_dim;
        std::vector<double> new_matrix(new_dim * new_dim);

        for (std::size_t rA = 0; rA < current_dim; ++rA) {
            for (std::size_t cA = 0; cA < current_dim; ++cA) {
                double valA = current_matrix[rA * current_dim + cA];
                for (std::size_t rB = 0; rB < next_dim; ++rB) {
                    for (std::size_t cB = 0; cB < next_dim; ++cB) {
                        double valB = next_matrix[rB * next_dim + cB];
                        
                        std::size_t r = rA * next_dim + rB;
                        std::size_t c = cA * next_dim + cB;
                        
                        new_matrix[r * new_dim + c] = valA * valB;
                    }
                }
            }
        }
        current_matrix = std::move(new_matrix);
        current_dim = new_dim;
    }

    // Apply global scale
    if (learn_scale_) {
        for (auto& val : current_matrix) val *= global_scale;
    }

    matrix = std::move(current_matrix);
}

}  // namespace libsemx
