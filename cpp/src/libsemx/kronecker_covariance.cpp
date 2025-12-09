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

bool KroneckerCovariance::is_sparse() const {
    for (const auto& c : components_) {
        if (c->is_sparse()) return true;
    }
    return false;
}

Eigen::SparseMatrix<double> KroneckerCovariance::materialize_sparse(const std::vector<double>& parameters) const {
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
    Eigen::SparseMatrix<double> current_matrix = components_[0]->materialize_sparse(params0);
    param_offset += p0;

    if (learn_scale_) {
        double trace = 0.0;
        for (int k = 0; k < current_matrix.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(current_matrix, k); it; ++it) {
                if (it.row() == it.col()) trace += it.value();
            }
        }
        double mean_diag = trace / current_matrix.rows();
        if (mean_diag > 1e-12) {
            current_matrix *= (1.0 / mean_diag);
        }
    }

    // Iteratively Kronecker product
    for (size_t i = 1; i < components_.size(); ++i) {
        std::size_t pi = components_[i]->parameter_count();
        std::vector<double> paramsi(parameters.begin() + param_offset, parameters.begin() + param_offset + pi);
        Eigen::SparseMatrix<double> next_matrix = components_[i]->materialize_sparse(paramsi);
        param_offset += pi;

        if (learn_scale_) {
            double trace = 0.0;
            for (int k = 0; k < next_matrix.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(next_matrix, k); it; ++it) {
                    if (it.row() == it.col()) trace += it.value();
                }
            }
            double mean_diag = trace / next_matrix.rows();
            if (mean_diag > 1e-12) {
                next_matrix *= (1.0 / mean_diag);
            }
        }

        // Compute Kronecker product: current (A) x next (B)
        std::size_t rowsA = current_matrix.rows();
        std::size_t colsA = current_matrix.cols();
        std::size_t rowsB = next_matrix.rows();
        std::size_t colsB = next_matrix.cols();
        
        Eigen::SparseMatrix<double> result(rowsA * rowsB, colsA * colsB);
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(current_matrix.nonZeros() * next_matrix.nonZeros());

        for (int kA = 0; kA < current_matrix.outerSize(); ++kA) {
            for (Eigen::SparseMatrix<double>::InnerIterator itA(current_matrix, kA); itA; ++itA) {
                double valA = itA.value();
                std::size_t rowA = itA.row();
                std::size_t colA = itA.col();

                for (int kB = 0; kB < next_matrix.outerSize(); ++kB) {
                    for (Eigen::SparseMatrix<double>::InnerIterator itB(next_matrix, kB); itB; ++itB) {
                        double valB = itB.value();
                        std::size_t rowB = itB.row();
                        std::size_t colB = itB.col();

                        triplets.emplace_back(rowA * rowsB + rowB, colA * colsB + colB, valA * valB);
                    }
                }
            }
        }
        result.setFromTriplets(triplets.begin(), triplets.end());
        current_matrix = std::move(result);
    }

    if (learn_scale_) {
        current_matrix *= global_scale;
    }

    return current_matrix;
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

std::vector<std::vector<double>> KroneckerCovariance::parameter_gradients(const std::vector<double>& parameters) const {
    double global_scale = 1.0;
    std::size_t param_offset = 0;

    if (learn_scale_) {
        global_scale = parameters[0];
        param_offset = 1;
    }

    // 1. Materialize all components and compute scaling factors
    std::vector<std::vector<double>> matrices;
    std::vector<std::size_t> dims;
    std::vector<double> scales; 
    std::vector<double> traces;

    std::size_t temp_offset = param_offset;
    for (const auto& comp : components_) {
        std::size_t p = comp->parameter_count();
        std::vector<double> params(parameters.begin() + temp_offset, parameters.begin() + temp_offset + p);
        std::vector<double> mat = comp->materialize(params);
        std::size_t d = comp->dimension();
        
        double trace = 0.0;
        if (learn_scale_) {
            for (std::size_t k = 0; k < d; ++k) trace += mat[k * d + k];
            double mean_diag = trace / d;
            if (mean_diag > 1e-12) {
                double s = 1.0 / mean_diag;
                scales.push_back(s);
                traces.push_back(trace);
                for (auto& val : mat) val *= s;
            } else {
                scales.push_back(1.0);
                traces.push_back(0.0);
            }
        } else {
            scales.push_back(1.0);
            traces.push_back(0.0);
        }
        
        matrices.push_back(std::move(mat));
        dims.push_back(d);
        temp_offset += p;
    }

    std::vector<std::vector<double>> grads;
    
    // 2. Gradient w.r.t global scale (if applicable)
    if (learn_scale_) {
        // dK/dsigma = K / sigma = K_norm
        std::vector<double> current = matrices[0];
        std::size_t current_dim = dims[0];
        
        for (size_t i = 1; i < matrices.size(); ++i) {
            const auto& next = matrices[i];
            std::size_t next_dim = dims[i];
            std::size_t new_dim = current_dim * next_dim;
            std::vector<double> next_K(new_dim * new_dim);
            
            for (std::size_t rA = 0; rA < current_dim; ++rA) {
                for (std::size_t cA = 0; cA < current_dim; ++cA) {
                    double valA = current[rA * current_dim + cA];
                    for (std::size_t rB = 0; rB < next_dim; ++rB) {
                        for (std::size_t cB = 0; cB < next_dim; ++cB) {
                            next_K[(rA * next_dim + rB) * new_dim + (cA * next_dim + cB)] = valA * next[rB * next_dim + cB];
                        }
                    }
                }
            }
            current = std::move(next_K);
            current_dim = new_dim;
        }
        grads.push_back(std::move(current));
    }

    // 3. Gradients w.r.t component parameters
    temp_offset = param_offset;
    for (size_t i = 0; i < components_.size(); ++i) {
        std::size_t p = components_[i]->parameter_count();
        std::vector<double> params(parameters.begin() + temp_offset, parameters.begin() + temp_offset + p);
        auto comp_grads = components_[i]->parameter_gradients(params);
        
        double s_i = scales[i];
        
        for (auto& dM : comp_grads) {
            // Apply trace correction if needed
            if (learn_scale_ && s_i != 1.0) {
                double tr_dM = 0.0;
                std::size_t d_i = dims[i];
                for (std::size_t k = 0; k < d_i; ++k) tr_dM += dM[k * d_i + k];
                
                double factor = tr_dM / traces[i];
                const auto& M_scaled = matrices[i];
                
                for (std::size_t k = 0; k < dM.size(); ++k) {
                    dM[k] = s_i * dM[k] - factor * M_scaled[k];
                }
            }
            
            // Kronecker product with other scaled matrices
            std::vector<double> current = (i == 0) ? dM : matrices[0];
            std::size_t current_dim = dims[0];
            
            for (size_t j = 1; j < components_.size(); ++j) {
                const std::vector<double>& next = (j == i) ? dM : matrices[j];
                std::size_t next_dim = dims[j];
                std::size_t new_dim = current_dim * next_dim;
                std::vector<double> next_K(new_dim * new_dim);
                
                for (std::size_t rA = 0; rA < current_dim; ++rA) {
                    for (std::size_t cA = 0; cA < current_dim; ++cA) {
                        double valA = current[rA * current_dim + cA];
                        for (std::size_t rB = 0; rB < next_dim; ++rB) {
                            for (std::size_t cB = 0; cB < next_dim; ++cB) {
                                next_K[(rA * next_dim + rB) * new_dim + (cA * next_dim + cB)] = valA * next[rB * next_dim + cB];
                            }
                        }
                    }
                }
                current = std::move(next_K);
                current_dim = new_dim;
            }
            
            // Apply global scale
            if (learn_scale_) {
                for (auto& val : current) val *= global_scale;
            }
            
            grads.push_back(std::move(current));
        }
        temp_offset += p;
    }
    
    return grads;
}

std::vector<Eigen::SparseMatrix<double>> KroneckerCovariance::parameter_gradients_sparse(const std::vector<double>& parameters) const {
    double global_scale = 1.0;
    std::size_t param_offset = 0;

    if (learn_scale_) {
        global_scale = parameters[0];
        param_offset = 1;
    }

    // 1. Materialize all components and compute scaling factors
    std::vector<Eigen::SparseMatrix<double>> matrices;
    std::vector<double> scales; 
    std::vector<double> traces;

    std::size_t temp_offset = param_offset;
    for (const auto& comp : components_) {
        std::size_t p = comp->parameter_count();
        std::vector<double> params(parameters.begin() + temp_offset, parameters.begin() + temp_offset + p);
        Eigen::SparseMatrix<double> mat = comp->materialize_sparse(params);
        
        double trace = 0.0;
        if (learn_scale_) {
            for (int k=0; k<mat.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it) {
                    if (it.row() == it.col()) trace += it.value();
                }
            }
            double mean_diag = trace / mat.rows();
            if (mean_diag > 1e-12) {
                double s = 1.0 / mean_diag;
                scales.push_back(s);
                traces.push_back(trace);
                mat *= s;
            } else {
                scales.push_back(1.0);
                traces.push_back(0.0);
            }
        } else {
            scales.push_back(1.0);
            traces.push_back(0.0);
        }
        
        matrices.push_back(std::move(mat));
        temp_offset += p;
    }

    std::vector<Eigen::SparseMatrix<double>> grads;
    
    // 2. Gradient w.r.t global scale (if applicable)
    if (learn_scale_) {
        // dK/dsigma = K / sigma = K_norm
        Eigen::SparseMatrix<double> current = matrices[0];
        
        for (size_t i = 1; i < matrices.size(); ++i) {
            const auto& next = matrices[i];
            // Kronecker product
            std::size_t rowsA = current.rows();
            std::size_t colsA = current.cols();
            std::size_t rowsB = next.rows();
            std::size_t colsB = next.cols();
            
            Eigen::SparseMatrix<double> result(rowsA * rowsB, colsA * colsB);
            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(current.nonZeros() * next.nonZeros());

            for (int kA = 0; kA < current.outerSize(); ++kA) {
                for (Eigen::SparseMatrix<double>::InnerIterator itA(current, kA); itA; ++itA) {
                    double valA = itA.value();
                    std::size_t rowA = itA.row();
                    std::size_t colA = itA.col();

                    for (int kB = 0; kB < next.outerSize(); ++kB) {
                        for (Eigen::SparseMatrix<double>::InnerIterator itB(next, kB); itB; ++itB) {
                            triplets.emplace_back(rowA * rowsB + itB.row(), colA * colsB + itB.col(), valA * itB.value());
                        }
                    }
                }
            }
            result.setFromTriplets(triplets.begin(), triplets.end());
            current = std::move(result);
        }
        grads.push_back(std::move(current));
    }

    // 3. Gradients w.r.t component parameters
    temp_offset = param_offset;
    for (size_t i = 0; i < components_.size(); ++i) {
        std::size_t p = components_[i]->parameter_count();
        std::vector<double> params(parameters.begin() + temp_offset, parameters.begin() + temp_offset + p);
        auto comp_grads = components_[i]->parameter_gradients_sparse(params);
        
        double s_i = scales[i];
        
        for (auto& dM : comp_grads) {
            // Apply trace correction if needed
            if (learn_scale_ && s_i != 1.0) {
                double tr_dM = 0.0;
                for (int k=0; k<dM.outerSize(); ++k) {
                    for (Eigen::SparseMatrix<double>::InnerIterator it(dM, k); it; ++it) {
                        if (it.row() == it.col()) tr_dM += it.value();
                    }
                }
                
                double factor = tr_dM / traces[i];
                // dM = s_i * dM - factor * M_scaled
                dM *= s_i;
                dM -= factor * matrices[i]; // Eigen handles sparse subtraction
            }
            
            // Kronecker product with other scaled matrices
            Eigen::SparseMatrix<double> current = (i == 0) ? dM : matrices[0];
            
            for (size_t j = 1; j < components_.size(); ++j) {
                const Eigen::SparseMatrix<double>& next = (j == i) ? dM : matrices[j];
                
                std::size_t rowsA = current.rows();
                std::size_t colsA = current.cols();
                std::size_t rowsB = next.rows();
                std::size_t colsB = next.cols();
                
                Eigen::SparseMatrix<double> result(rowsA * rowsB, colsA * colsB);
                std::vector<Eigen::Triplet<double>> triplets;
                triplets.reserve(current.nonZeros() * next.nonZeros());

                for (int kA = 0; kA < current.outerSize(); ++kA) {
                    for (Eigen::SparseMatrix<double>::InnerIterator itA(current, kA); itA; ++itA) {
                        double valA = itA.value();
                        std::size_t rowA = itA.row();
                        std::size_t colA = itA.col();

                        for (int kB = 0; kB < next.outerSize(); ++kB) {
                            for (Eigen::SparseMatrix<double>::InnerIterator itB(next, kB); itB; ++itB) {
                                triplets.emplace_back(rowA * rowsB + itB.row(), colA * colsB + itB.col(), valA * itB.value());
                            }
                        }
                    }
                }
                result.setFromTriplets(triplets.begin(), triplets.end());
                current = std::move(result);
            }
            
            // Apply global scale
            if (learn_scale_) {
                current *= global_scale;
            }
            
            grads.push_back(std::move(current));
        }
        temp_offset += p;
    }
    
    return grads;
}

}  // namespace libsemx
