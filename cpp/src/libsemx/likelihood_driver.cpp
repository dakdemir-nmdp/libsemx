#include <cstdio>
#include <iostream>
#include "libsemx/likelihood_driver.hpp"
#include "libsemx/outcome_family_factory.hpp"
#include "libsemx/covariance_structure.hpp"
#include "libsemx/model_objective.hpp"
#include "libsemx/parameter_transform.hpp"
#include "libsemx/parameter_catalog.hpp"
#include "libsemx/post_estimation.hpp"

#include "libsemx/scaled_fixed_covariance.hpp"
#include "libsemx/multi_kernel_covariance.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace libsemx {

namespace {



using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;


struct RandomEffectInfo {
    std::string id;
    std::string grouping_var;
    std::vector<std::string> design_vars;
    std::string covariance_id;
    const CovarianceSpec* cov_spec;
    std::vector<double> G_matrix;
    std::vector<double> G_inverse;
    double log_det_G = 0.0;
    std::vector<std::vector<double>> G_gradients;
};


void cholesky(std::size_t n, const std::vector<double>& matrix, std::vector<double>& lower) {
    if (matrix.size() != n * n) {
        throw std::invalid_argument("Matrix size mismatch for Cholesky factorization");
    }
    Eigen::Map<const RowMajorMatrix> mat(matrix.data(), n, n);
    Eigen::LLT<RowMajorMatrix> llt(mat);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("Cholesky factorization failed");
    }
    RowMajorMatrix L = llt.matrixL();
    lower.resize(n * n);
    Eigen::Map<RowMajorMatrix>(lower.data(), n, n) = L;
}

double log_det_cholesky(std::size_t n, const std::vector<double>& lower) {
    if (lower.size() != n * n) {
        throw std::invalid_argument("Matrix size mismatch for log-det computation");
    }
    Eigen::Map<const RowMajorMatrix> L(lower.data(), n, n);
    double log_det = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double val = L(i, i);
        if (!(val > 0.0)) {
            throw std::runtime_error("Non-positive diagonal encountered in Cholesky factor");
        }
        log_det += std::log(val);
    }
    return 2.0 * log_det;
}

std::vector<double> solve_cholesky(std::size_t n,
                                   const std::vector<double>& lower,
                                   const std::vector<double>& rhs) {
    if (lower.size() != n * n || rhs.size() != n) {
        throw std::invalid_argument("Dimension mismatch in solve_cholesky");
    }
    Eigen::Map<const RowMajorMatrix> L(lower.data(), n, n);
    Eigen::Map<const Eigen::VectorXd> b(rhs.data(), n);
    Eigen::VectorXd y = L.triangularView<Eigen::Lower>().solve(b);
    Eigen::VectorXd x = L.transpose().triangularView<Eigen::Upper>().solve(y);
    std::vector<double> result(n);
    Eigen::Map<Eigen::VectorXd>(result.data(), n) = x;
    return result;
}

std::vector<double> invert_from_cholesky(std::size_t n, const std::vector<double>& lower) {
    if (lower.size() != n * n) {
        throw std::invalid_argument("Matrix size mismatch when forming inverse");
    }
    Eigen::Map<const RowMajorMatrix> L(lower.data(), n, n);
    RowMajorMatrix identity = RowMajorMatrix::Identity(n, n);
    RowMajorMatrix Linv = L.triangularView<Eigen::Lower>().solve(identity);
    RowMajorMatrix inv = Linv.transpose() * Linv;
    std::vector<double> result(n * n);
    Eigen::Map<RowMajorMatrix>(result.data(), n, n) = inv;
    return result;
}

std::vector<RandomEffectInfo> build_random_effect_infos(
    const ModelIR& model,
    const std::unordered_map<std::string, std::vector<double>>& covariance_parameters,
    const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data,
    bool need_gradients) {
    std::unordered_map<std::string, const CovarianceSpec*> cov_lookup;
    for (const auto& cov : model.covariances) {
        cov_lookup.emplace(cov.id, &cov);
    }

    std::vector<RandomEffectInfo> infos;
    infos.reserve(model.random_effects.size());

    for (const auto& re : model.random_effects) {
        if (re.variables.empty()) {
            throw std::runtime_error("Random effect " + re.id + " is missing grouping/design variables");
        }
        auto cov_it = cov_lookup.find(re.covariance_id);
        if (cov_it == cov_lookup.end()) {
            throw std::runtime_error("Covariance spec not found for random effect: " + re.covariance_id);
        }

        auto structure = create_covariance_structure(*cov_it->second, fixed_covariance_data);
        std::vector<double> params;
        if (structure->parameter_count() > 0) {
            auto param_it = covariance_parameters.find(re.covariance_id);
            if (param_it == covariance_parameters.end()) {
                throw std::runtime_error("Missing covariance parameters for: " + re.covariance_id);
            }
            params = param_it->second;
        }

        RandomEffectInfo info;
        info.id = re.id;
        info.grouping_var = re.variables.front();
        info.design_vars.assign(re.variables.begin() + 1, re.variables.end());
        info.covariance_id = re.covariance_id;
        info.cov_spec = cov_it->second;
        info.G_matrix = structure->materialize(params);

        if (info.cov_spec->dimension == 0) {
            throw std::runtime_error("Random effect " + re.id + " has zero-dimensional covariance");
        }

        std::vector<double> chol(info.cov_spec->dimension * info.cov_spec->dimension);
        cholesky(info.cov_spec->dimension, info.G_matrix, chol);
        info.log_det_G = log_det_cholesky(info.cov_spec->dimension, chol);
        info.G_inverse = invert_from_cholesky(info.cov_spec->dimension, chol);

        if (need_gradients && structure->parameter_count() > 0) {
            info.G_gradients = structure->parameter_gradients(params);
        }
        infos.push_back(std::move(info));
    }

    return infos;
}

double trace_product(const std::vector<double>& A,
                     const std::vector<double>& B,
                     std::size_t dim);

double quadratic_form(const std::vector<double>& vec,
                      const std::vector<double>& matrix,
                      std::size_t dim);

std::map<double, std::vector<std::size_t>> build_group_map(
    const RandomEffectInfo& info,
    const std::unordered_map<std::string, std::vector<double>>& data,
    std::size_t n) {
    auto it = data.find(info.grouping_var);
    if (it == data.end()) {
        throw std::runtime_error("Grouping data missing for variable: " + info.grouping_var);
    }
    if (it->second.size() != n) {
        throw std::runtime_error("Grouping variable " + info.grouping_var + " size mismatch");
    }
    std::map<double, std::vector<std::size_t>> groups;
    for (std::size_t i = 0; i < n; ++i) {
        groups[it->second[i]].push_back(i);
    }
    return groups;
}

const std::vector<double>& resolve_design_values(
    const std::string& name,
    const std::unordered_map<std::string, std::vector<double>>& data,
    const std::unordered_map<std::string, std::vector<double>>& linear_predictors) {
    if (auto d_it = data.find(name); d_it != data.end()) {
        return d_it->second;
    }
    if (auto lp_it = linear_predictors.find(name); lp_it != linear_predictors.end()) {
        return lp_it->second;
    }
    throw std::runtime_error("Design variable not found: " + name);
}

std::vector<double> build_design_matrix(
    const RandomEffectInfo& info,
    const std::vector<std::size_t>& indices,
    const std::unordered_map<std::string, std::vector<double>>& data,
    const std::unordered_map<std::string, std::vector<double>>& linear_predictors) {
    std::size_t n_i = indices.size();
    std::size_t q = info.cov_spec->dimension;
    std::vector<double> Z(n_i * q, 0.0);

    if (info.design_vars.empty()) {
        if (q == 1) {
            for (std::size_t r = 0; r < n_i; ++r) {
                Z[r * q] = 1.0;
            }
            return Z;
        }

        if (q == n_i) {
            // Allow Z = I for GRM / scaled-fixed effects where the covariance dimension
            // matches the number of observations in the block.
            for (std::size_t r = 0; r < n_i; ++r) {
                Z[r * q + r] = 1.0;
            }
            return Z;
        }

        throw std::runtime_error("Random effect " + info.id + " expects " + std::to_string(q) +
                                 " design columns but none were specified");
    }

    if (info.design_vars.size() != q) {
        throw std::runtime_error("Design variable count does not match covariance dimension for " + info.id);
    }

    for (std::size_t col = 0; col < q; ++col) {
        const auto& values = resolve_design_values(info.design_vars[col], data, linear_predictors);
        for (std::size_t row = 0; row < n_i; ++row) {
            if (indices[row] >= values.size()) {
                throw std::runtime_error("Design variable " + info.design_vars[col] + " has insufficient data length");
            }
            Z[row * q + col] = values[indices[row]];
        }
    }
    return Z;
}

const std::vector<double>& get_extra_params_for_variable(
    const std::unordered_map<std::string, std::vector<double>>& extra_params,
    const std::string& var_name) {
    auto it = extra_params.find(var_name);
    if (it != extra_params.end()) {
        return it->second;
    }
    static const std::vector<double> kEmpty;
    return kEmpty;
}

struct LaplaceObservationEntry {
    std::size_t block_index;
    std::size_t row_index;
};

struct LaplaceBlock {
    const RandomEffectInfo* info;
    std::vector<std::size_t> obs_indices;
    std::vector<double> design_matrix;
    std::size_t q = 0;
    std::size_t offset = 0;
};

struct LaplaceSystem {
    std::vector<LaplaceBlock> blocks;
    std::vector<std::vector<LaplaceObservationEntry>> observation_entries;
    std::size_t total_dim = 0;
};

struct LaplaceSystemResult {
    std::vector<double> u;
    std::vector<double> neg_hessian;
    std::vector<double> chol_neg_hessian;
    std::vector<OutcomeEvaluation> evaluations;
};

LaplaceSystem build_laplace_system(
    const std::vector<RandomEffectInfo>& infos,
    const std::unordered_map<std::string, std::vector<double>>& data,
    const std::unordered_map<std::string, std::vector<double>>& linear_predictors,
    std::size_t n) {
    LaplaceSystem system;
    system.observation_entries.resize(n);
    std::size_t offset = 0;
    for (const auto& info : infos) {
        auto group_map = build_group_map(info, data, n);
        for (const auto& [_, indices] : group_map) {
            LaplaceBlock block;
            block.info = &info;
            block.obs_indices = indices;
            block.q = info.cov_spec->dimension;
            block.offset = offset;
            block.design_matrix = build_design_matrix(info, indices, data, linear_predictors);
            system.blocks.push_back(block);
            const std::size_t block_idx = system.blocks.size() - 1;
            for (std::size_t row = 0; row < indices.size(); ++row) {
                system.observation_entries[indices[row]].push_back({block_idx, row});
            }
            offset += block.q;
        }
    }
    system.total_dim = offset;
    return system;
}

void compute_system_grad_hess(const LaplaceSystem& system,
                              const std::vector<double>& u,
                              const std::vector<double>& obs_data,
                              const std::vector<double>& pred_data,
                              const std::vector<double>& disp_data,
                              const std::vector<double>* status_vec,
                              const std::vector<double>& extra_vec,
                              const OutcomeFamily& family,
                              std::vector<double>& grad,
                              std::vector<double>& hess,
                              std::vector<OutcomeEvaluation>* evals_out) {
    const std::size_t Q = system.total_dim;
    const std::size_t n = obs_data.size();
    std::fill(grad.begin(), grad.end(), 0.0);
    std::fill(hess.begin(), hess.end(), 0.0);

    for (const auto& block : system.blocks) {
        const auto& info = *block.info;
        const double* G_inv = info.G_inverse.data();
        const double* u_block = &u[block.offset];
        for (std::size_t r = 0; r < block.q; ++r) {
            double sum = 0.0;
            for (std::size_t c = 0; c < block.q; ++c) {
                const std::size_t row_idx = block.offset + r;
                const std::size_t col_idx = block.offset + c;
                hess[row_idx * Q + col_idx] -= G_inv[r * block.q + c];
                sum += G_inv[r * block.q + c] * u_block[c];
            }
            grad[block.offset + r] -= sum;
        }
    }

    if (evals_out) {
        evals_out->resize(n);
    }

    for (std::size_t obs_idx = 0; obs_idx < n; ++obs_idx) {
        if (std::isnan(obs_data[obs_idx])) {
            if (evals_out) {
                (*evals_out)[obs_idx] = OutcomeEvaluation{0.0, 0.0, 0.0, 0.0};
            }
            continue;
        }

        double eta = pred_data[obs_idx];
        const auto& entries = system.observation_entries[obs_idx];
        for (const auto& entry : entries) {
            const auto& block = system.blocks[entry.block_index];
            const double* z_row = &block.design_matrix[entry.row_index * block.q];
            const double* u_block = &u[block.offset];
            for (std::size_t j = 0; j < block.q; ++j) {
                eta += z_row[j] * u_block[j];
            }
        }

        const double s = status_vec ? (*status_vec)[obs_idx] : 1.0;
        auto eval = family.evaluate(obs_data[obs_idx], eta, disp_data[obs_idx], s, extra_vec);
        if (evals_out) {
            (*evals_out)[obs_idx] = eval;
        }

        for (const auto& entry : entries) {
            const auto& block = system.blocks[entry.block_index];
            const double* z_row = &block.design_matrix[entry.row_index * block.q];
            double* grad_block = &grad[block.offset];
            for (std::size_t j = 0; j < block.q; ++j) {
                grad_block[j] += z_row[j] * eval.first_derivative;
            }
        }

        for (std::size_t a = 0; a < entries.size(); ++a) {
            const auto& entry_a = entries[a];
            const auto& block_a = system.blocks[entry_a.block_index];
            const double* z_a = &block_a.design_matrix[entry_a.row_index * block_a.q];
            for (std::size_t b = a; b < entries.size(); ++b) {
                const auto& entry_b = entries[b];
                const auto& block_b = system.blocks[entry_b.block_index];
                const double* z_b = &block_b.design_matrix[entry_b.row_index * block_b.q];
                for (std::size_t r = 0; r < block_a.q; ++r) {
                    for (std::size_t c = 0; c < block_b.q; ++c) {
                        double value = z_a[r] * z_b[c] * eval.second_derivative;
                        const std::size_t row_idx = block_a.offset + r;
                        const std::size_t col_idx = block_b.offset + c;
                        hess[row_idx * Q + col_idx] += value;
                        if (entry_b.block_index != entry_a.block_index) {
                            hess[col_idx * Q + row_idx] += value;
                        }
                    }
                }
            }
        }
    }
}

double compute_system_objective(const LaplaceSystem& system,
                                const std::vector<double>& u,
                                const std::vector<double>& obs_data,
                                const std::vector<double>& pred_data,
                                const std::vector<double>& disp_data,
                                const std::vector<double>* status_vec,
                                const std::vector<double>& extra_vec,
                                const OutcomeFamily& family) {
    const std::size_t n = obs_data.size();
    double loglik = 0.0;

    // 1. Log-likelihood of data given u
    for (std::size_t obs_idx = 0; obs_idx < n; ++obs_idx) {
        if (std::isnan(obs_data[obs_idx])) continue;

        double eta = pred_data[obs_idx];
        const auto& entries = system.observation_entries[obs_idx];
        for (const auto& entry : entries) {
            const auto& block = system.blocks[entry.block_index];
            const double* z_row = &block.design_matrix[entry.row_index * block.q];
            const double* u_block = &u[block.offset];
            for (std::size_t j = 0; j < block.q; ++j) {
                eta += z_row[j] * u_block[j];
            }
        }

        const double s = status_vec ? (*status_vec)[obs_idx] : 1.0;
        loglik += family.evaluate(obs_data[obs_idx], eta, disp_data[obs_idx], s, extra_vec).log_likelihood;
    }

    // 2. Penalty term: -0.5 * u^T * G^-1 * u
    for (const auto& block : system.blocks) {
        const auto& info = *block.info;
        const double* G_inv = info.G_inverse.data();
        const double* u_block = &u[block.offset];
        // u_block^T * G_inv * u_block
        // G_inv is symmetric
        for (std::size_t r = 0; r < block.q; ++r) {
            double row_sum = 0.0;
            for (std::size_t c = 0; c < block.q; ++c) {
                row_sum += G_inv[r * block.q + c] * u_block[c];
            }
            loglik -= 0.5 * u_block[r] * row_sum;
        }
    }

    return loglik;
}

bool solve_laplace_system(const LaplaceSystem& system,
                          const std::vector<double>& obs_data,
                          const std::vector<double>& pred_data,
                          const std::vector<double>& disp_data,
                          const std::vector<double>* status_vec,
                          const std::vector<double>& extra_vec,
                          const OutcomeFamily& family,
                          LaplaceSystemResult& result) {
    const std::size_t Q = system.total_dim;
    std::vector<double> u(Q, 0.0);
    std::vector<double> grad(Q, 0.0);
    std::vector<double> hess(Q * Q, 0.0);
    std::vector<double> neg_hess(Q * Q, 0.0);
    std::vector<double> chol(Q * Q, 0.0);
    std::vector<double> delta(Q, 0.0);
    std::vector<double> u_new(Q, 0.0);

    constexpr int kMaxIter = 100;
    constexpr double kTol = 1e-6;
    constexpr double kDampingInit = 1e-3;
    constexpr double kDampingMax = 1e2;

    double current_obj = compute_system_objective(system, u, obs_data, pred_data, disp_data, status_vec, extra_vec, family);

    for (int iter = 0; iter < kMaxIter; ++iter) {
        compute_system_grad_hess(system, u, obs_data, pred_data, disp_data, status_vec, extra_vec, family, grad, hess, nullptr);
        
        // Prepare negative hessian
        for (std::size_t idx = 0; idx < Q * Q; ++idx) {
            neg_hess[idx] = -hess[idx];
        }

        // Damping loop for Cholesky
        double damping = 0.0;
        bool chol_success = false;
        
        // Try without damping first
        try {
            cholesky(Q, neg_hess, chol);
            chol_success = true;
        } catch (...) {
            damping = kDampingInit;
        }

        while (!chol_success && damping <= kDampingMax) {
             // Add damping to diagonal
             std::vector<double> damped_hess = neg_hess;
             for(size_t i=0; i<Q; ++i) damped_hess[i*Q+i] += damping;
             
             try {
                 cholesky(Q, damped_hess, chol);
                 chol_success = true;
             } catch (...) {
                 damping *= 10.0;
             }
        }

        if (!chol_success) {
            // Failed to factorize even with damping
            return false;
        }

        delta = solve_cholesky(Q, chol, grad);

        // Line search
        double step = 1.0;
        bool improved = false;
        while (step > 1e-4) {
            for(size_t i=0; i<Q; ++i) u_new[i] = u[i] + step * delta[i];
            double new_obj = compute_system_objective(system, u_new, obs_data, pred_data, disp_data, status_vec, extra_vec, family);
            
            // Simple improvement check
            if (new_obj > current_obj) {
                current_obj = new_obj;
                u = u_new;
                improved = true;
                break;
            }
            step *= 0.5;
        }

        if (!improved) {
            // Cannot improve objective, maybe converged or stuck
            // Check gradient size or delta size
            double max_delta = 0.0;
            for(double d : delta) max_delta = std::max(max_delta, std::abs(d));
            if (max_delta < kTol) break;
            
            // If delta was large but we couldn't improve, maybe we are at a saddle point or numerical issues
            break; 
        }

        // Check convergence based on step size * delta
        double max_change = 0.0;
        for(size_t i=0; i<Q; ++i) max_change = std::max(max_change, std::abs(step * delta[i]));
        if (max_change < kTol) {
            break;
        }
    }

    result.u = u;
    result.neg_hessian = std::vector<double>(Q * Q);
    result.chol_neg_hessian = std::vector<double>(Q * Q);
    compute_system_grad_hess(system, u, obs_data, pred_data, disp_data, status_vec, extra_vec, family, grad, hess, &result.evaluations);
    for (std::size_t idx = 0; idx < Q * Q; ++idx) {
        result.neg_hessian[idx] = -hess[idx];
    }
    try {
        cholesky(Q, result.neg_hessian, result.chol_neg_hessian);
    } catch (...) {
        return false;
    }
    return true;
}

std::vector<double> multiply_matrices(const std::vector<double>& A,
                                      const std::vector<double>& B,
                                      std::size_t dim) {
    std::vector<double> result(dim * dim, 0.0);
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t k = 0; k < dim; ++k) {
            const double a = A[i * dim + k];
            for (std::size_t j = 0; j < dim; ++j) {
                result[i * dim + j] += a * B[k * dim + j];
            }
        }
    }
    return result;
}

std::vector<double> multiply_matrix_vector(const std::vector<double>& matrix,
                                           const std::vector<double>& vec,
                                           std::size_t dim) {
    std::vector<double> result(dim, 0.0);
    for (std::size_t i = 0; i < dim; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < dim; ++j) {
            sum += matrix[i * dim + j] * vec[j];
        }
        result[i] = sum;
    }
    return result;
}

double compute_prior_loglik(const LaplaceSystem& system,
                            const LaplaceSystemResult& result,
                            double log_2pi) {
    double total = 0.0;
    for (const auto& block : system.blocks) {
        const auto& info = *block.info;
        std::vector<double> u_block(block.q);
        for (std::size_t j = 0; j < block.q; ++j) {
            u_block[j] = result.u[block.offset + j];
        }
        double quad = quadratic_form(u_block, info.G_inverse, block.q);
        total += -0.5 * (block.q * log_2pi + info.log_det_G + quad);
    }
    return total;
}

std::vector<double> compute_observation_quad_terms(const LaplaceSystem& system,
                                                   const std::vector<double>& neg_hess_inv) {
    const std::size_t Q = system.total_dim;
    const std::size_t n = system.observation_entries.size();
    std::vector<double> buffer(Q, 0.0);
    std::vector<double> quads(n, 0.0);
    for (std::size_t obs_idx = 0; obs_idx < n; ++obs_idx) {
        std::fill(buffer.begin(), buffer.end(), 0.0);
        for (const auto& entry : system.observation_entries[obs_idx]) {
            const auto& block = system.blocks[entry.block_index];
            const double* z_row = &block.design_matrix[entry.row_index * block.q];
            for (std::size_t j = 0; j < block.q; ++j) {
                buffer[block.offset + j] = z_row[j];
            }
        }
        double quad = 0.0;
        for (std::size_t i = 0; i < Q; ++i) {
            if (buffer[i] == 0.0) continue;
            for (std::size_t j = 0; j < Q; ++j) {
                if (buffer[j] == 0.0) continue;
                quad += buffer[i] * neg_hess_inv[i * Q + j] * buffer[j];
            }
        }
        quads[obs_idx] = quad;
    }
    return quads;
}

std::vector<double> project_observations(const LaplaceSystem& system,
                                         const std::vector<double>& direction) {
    const std::size_t n = system.observation_entries.size();
    std::vector<double> projected(n, 0.0);
    for (std::size_t obs_idx = 0; obs_idx < n; ++obs_idx) {
        double sum = 0.0;
        for (const auto& entry : system.observation_entries[obs_idx]) {
            const auto& block = system.blocks[entry.block_index];
            const double* z_row = &block.design_matrix[entry.row_index * block.q];
            for (std::size_t j = 0; j < block.q; ++j) {
                sum += z_row[j] * direction[block.offset + j];
            }
        }
        projected[obs_idx] = sum;
    }
    return projected;
}

void accumulate_weighted_forcing(const LaplaceSystem& system,
                                 const std::vector<double>& weights,
                                 std::vector<double>& forcing) {
    if (weights.size() != system.observation_entries.size()) {
        throw std::runtime_error("Weight vector size mismatch in accumulate_weighted_forcing");
    }
    std::fill(forcing.begin(), forcing.end(), 0.0);
    for (std::size_t obs_idx = 0; obs_idx < weights.size(); ++obs_idx) {
        double w = weights[obs_idx];
        if (w == 0.0) continue;
        for (const auto& entry : system.observation_entries[obs_idx]) {
            const auto& block = system.blocks[entry.block_index];
            const double* z_row = &block.design_matrix[entry.row_index * block.q];
            for (std::size_t j = 0; j < block.q; ++j) {
                forcing[block.offset + j] += z_row[j] * w;
            }
        }
    }
}

double trace_product(const std::vector<double>& A,
                     const std::vector<double>& B,
                     std::size_t dim) {
    double trace = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            trace += A[i * dim + j] * B[j * dim + i];
        }
    }
    return trace;
}

double quadratic_form(const std::vector<double>& vec,
                      const std::vector<double>& matrix,
                      std::size_t dim) {
    double total = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            total += vec[i] * matrix[i * dim + j] * vec[j];
        }
    }
    return total;
}

} // namespace

double LikelihoodDriver::evaluate_total_loglik(const std::vector<double>& observed,
                                               const std::vector<double>& linear_predictors,
                                               const std::vector<double>& dispersions,
                                               const OutcomeFamily& family,
                                               const std::vector<double>& status,
                                               const std::vector<double>& extra_params) const {
    const std::size_t n = observed.size();
    if (linear_predictors.size() != n) {
        throw std::invalid_argument("observed and predictor vectors must have the same size");
    }
    if (!status.empty() && status.size() != n) {
        throw std::invalid_argument("status vector must match observed size if provided");
    }

    if (n == 0) {
        if (!dispersions.empty() && dispersions.size() != 1) {
            throw std::invalid_argument("dispersions must be empty or size 1 when no observations are present");
        }
        return 0.0;
    }

    if (dispersions.empty()) {
        throw std::invalid_argument("dispersions vector must be non-empty when observations exist");
    }
    if (dispersions.size() != n && dispersions.size() != 1) {
        throw std::invalid_argument("dispersions vector must have size 1 or match the observation count");
    }
    const bool shared_dispersion = dispersions.size() == 1;
    const double shared_dispersion_value = shared_dispersion ? dispersions.front() : 0.0;

    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double dispersion_value = shared_dispersion ? shared_dispersion_value : dispersions[i];
        const double s = status.empty() ? 1.0 : status[i];
        const auto eval = family.evaluate(observed[i], linear_predictors[i], dispersion_value, s, extra_params);
        total += eval.log_likelihood;
    }
    return total;
}

double LikelihoodDriver::evaluate_total_loglik_mixed(const std::vector<double>& observed,
                                                      const std::vector<double>& linear_predictors,
                                                      const std::vector<double>& dispersions,
                                                      const std::vector<const OutcomeFamily*>& families,
                                                      const std::vector<double>& status,
                                                      const std::vector<std::vector<double>>& extra_params) const {
    const std::size_t n = observed.size();
    if (linear_predictors.size() != n || families.size() != n) {
        throw std::invalid_argument("observed, predictor, and family vectors must have the same size");
    }
    if (!status.empty() && status.size() != n) {
        throw std::invalid_argument("status vector must match observed size if provided");
    }
    if (!extra_params.empty() && extra_params.size() != n) {
        throw std::invalid_argument("extra_params vector must match observed size if provided");
    }
    if (dispersions.empty()) {
        throw std::invalid_argument("dispersions vector must be non-empty when observations exist");
    }
    if (dispersions.size() != n && dispersions.size() != 1) {
        throw std::invalid_argument("dispersions vector must have size 1 or match the observation count");
    }

    const bool shared_dispersion = dispersions.size() == 1;
    const double shared_dispersion_value = shared_dispersion ? dispersions.front() : 0.0;

    double total = 0.0;
    for (std::size_t i = 0; i < observed.size(); ++i) {
        const double dispersion_value = shared_dispersion ? shared_dispersion_value : dispersions[i];
        const double s = status.empty() ? 1.0 : status[i];
        const std::vector<double>& ep = extra_params.empty() ? std::vector<double>{} : extra_params[i];
        const auto eval = families[i]->evaluate(observed[i], linear_predictors[i], dispersion_value, s, ep);
        total += eval.log_likelihood;
    }
    return total;
}

double LikelihoodDriver::evaluate_model_loglik(const ModelIR& model,
                                                const std::unordered_map<std::string, std::vector<double>>& data,
                                                const std::unordered_map<std::string, std::vector<double>>& linear_predictors,
                                                const std::unordered_map<std::string, std::vector<double>>& dispersions,
                                                const std::unordered_map<std::string, std::vector<double>>& covariance_parameters,
                                                const std::unordered_map<std::string, std::vector<double>>& status,
                                                const std::unordered_map<std::string, std::vector<double>>& extra_params,
                                                const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data,
                                                EstimationMethod method) const {
    
    if (model.random_effects.empty()) {
        double total = 0.0;
        for (const auto& var : model.variables) {
            if (var.kind != VariableKind::Observed) {
                continue;  // Skip latent and grouping variables for now
            }
            auto data_it = data.find(var.name);
            if (data_it == data.end()) {
                throw std::invalid_argument("Missing data for variable: " + var.name);
            }

            auto pred_it = linear_predictors.find(var.name);
            auto disp_it = dispersions.find(var.name);

            if (pred_it == linear_predictors.end()) {
                throw std::invalid_argument("Missing linear predictors for variable: " + var.name);
            }
            if (disp_it == dispersions.end()) {
                throw std::invalid_argument("Missing dispersions for variable: " + var.name);
            }

            const auto& obs = data_it->second;
            const auto& preds = pred_it->second;
            const auto& disps = disp_it->second;
            if (obs.size() != preds.size()) {
                throw std::invalid_argument("Data vectors for variable " + var.name + " have mismatched sizes");
            }
            if (disps.empty()) {
                throw std::invalid_argument("Dispersion vector for variable " + var.name + " is empty");
            }
            if (disps.size() != obs.size() && disps.size() != 1) {
                throw std::invalid_argument("Dispersion vector for variable " + var.name + " must have size 1 or match observation count");
            }
            const bool shared_dispersion = disps.size() == 1;
            const double shared_dispersion_value = shared_dispersion ? disps.front() : 0.0;
            
            const std::vector<double>* status_vec = nullptr;
            auto status_it = status.find(var.name);
            if (status_it != status.end()) {
                status_vec = &status_it->second;
                if (status_vec->size() != obs.size()) {
                    throw std::invalid_argument("Status vector for variable " + var.name + " has mismatched size");
                }
            }

            const std::vector<double>* extra_vec = nullptr;
            auto extra_it = extra_params.find(var.name);
            if (extra_it != extra_params.end()) {
                extra_vec = &extra_it->second;
            }
            const std::vector<double>& ep = extra_vec ? *extra_vec : std::vector<double>{};

            auto family = OutcomeFamilyFactory::create(var.family);
            for (std::size_t i = 0; i < obs.size(); ++i) {
                if (std::isnan(obs[i])) {
                    continue;
                }
                double s = status_vec ? (*status_vec)[i] : 1.0;
                const double dispersion_value = shared_dispersion ? shared_dispersion_value : disps[i];
                const auto eval = family->evaluate(obs[i], preds[i], dispersion_value, s, ep);
                total += eval.log_likelihood;
            }

            if (method == EstimationMethod::REML && var.family == "gaussian") {
                // Identify predictors
                std::vector<std::string> predictors;
                for (const auto& edge : model.edges) {
                    if (edge.kind == EdgeKind::Regression && edge.target == var.name) {
                        predictors.push_back(edge.source);
                    }
                }
                std::sort(predictors.begin(), predictors.end()); // Ensure deterministic order

                if (!predictors.empty()) {
                    std::size_t n = obs.size();
                    std::size_t p = predictors.size();
                    std::vector<double> X(n * p);
                    
                    // Build X
                    for (std::size_t j = 0; j < p; ++j) {
                        auto it = data.find(predictors[j]);
                        if (it == data.end() || it->second.size() != n) {
                            // Should not happen if model is valid, but safe check
                            continue; 
                        }
                        for (std::size_t i = 0; i < n; ++i) {
                            X[i * p + j] = it->second[i];
                        }
                    }

                    // Compute Xt * V^-1 * X
                    // V is diagonal with elements 'dispersion_value'
                    std::vector<double> Xt_Vinv_X(p * p, 0.0);
                    
                    for (std::size_t r = 0; r < p; ++r) {
                        for (std::size_t c = 0; c < p; ++c) {
                            double sum = 0.0;
                            for (std::size_t i = 0; i < n; ++i) {
                                if (std::isnan(obs[i])) continue;
                                double disp = shared_dispersion ? shared_dispersion_value : disps[i];
                                sum += X[i * p + r] * X[i * p + c] / disp;
                            }
                            Xt_Vinv_X[r * p + c] = sum;
                        }
                    }

                    std::vector<double> L_reml(p * p);
                    try {
                        cholesky(p, Xt_Vinv_X, L_reml);
                        double log_det_reml = log_det_cholesky(p, L_reml);
                        double log_2pi = std::log(2.0 * 3.14159265358979323846);
                        total -= 0.5 * log_det_reml;
                        total += 0.5 * static_cast<double>(p) * log_2pi;
                    } catch (...) {
                        // If Cholesky fails (e.g. singular design matrix), we can't compute REML adjustment.
                        // Fallback to ML or return -inf? 
                        // For now, let's return -inf to signal failure
                        return -std::numeric_limits<double>::infinity();
                    }
                }
            }
        }
        return total;
    }

    // Mixed model logic
    auto random_effect_infos = build_random_effect_infos(model, covariance_parameters, fixed_covariance_data, false);
    if (random_effect_infos.empty()) {
        throw std::runtime_error("Random effect metadata missing");
    }

    std::string obs_var_name;
    std::string obs_family_name;
    for (const auto& var : model.variables) {
        if (var.kind == VariableKind::Observed) {
            bool is_target = false;
            for (const auto& edge : model.edges) {
                if (edge.target == var.name) {
                    is_target = true;
                    break;
                }
            }
            if (is_target || obs_var_name.empty()) {
                obs_var_name = var.name;
                obs_family_name = var.family;
            }
        }
    }
    if (obs_var_name.empty()) {
        throw std::runtime_error("No observed variable found");
    }

    auto pred_it = linear_predictors.find(obs_var_name);
    if (pred_it == linear_predictors.end()) {
        if (linear_predictors.size() == 1) {
            obs_var_name = linear_predictors.begin()->first;
            auto var_it = std::find_if(model.variables.begin(), model.variables.end(), [&](const auto& v) {
                return v.name == obs_var_name;
            });
            if (var_it == model.variables.end()) {
                throw std::runtime_error("Outcome variable metadata not found: " + obs_var_name);
            }
            obs_family_name = var_it->family;
            pred_it = linear_predictors.begin();
        } else {
            throw std::runtime_error("Linear predictors for outcome variable not found");
        }
    }
    const auto& pred_data = pred_it->second;

    auto obs_it = data.find(obs_var_name);
    if (obs_it == data.end()) {
        throw std::runtime_error("Observed variable not found in data: " + obs_var_name);
    }
    const auto& obs_data = obs_it->second;

    if (obs_data.size() != pred_data.size()) {
        throw std::runtime_error("Outcome data and predictors have mismatched sizes");
    }
    std::size_t n = obs_data.size();

    auto disp_it = dispersions.find(obs_var_name);
    if (disp_it == dispersions.end()) {
        throw std::runtime_error("Dispersions not found for: " + obs_var_name);
    }
    const auto& disp_raw = disp_it->second;
    if (disp_raw.empty()) {
        throw std::runtime_error("Dispersion vector is empty");
    }
    std::vector<double> disp_broadcast;
    const std::vector<double>* disp_ptr = &disp_raw;
    if (disp_raw.size() == 1) {
        disp_broadcast.assign(n, disp_raw.front());
        disp_ptr = &disp_broadcast;
    } else if (disp_raw.size() != n) {
        throw std::runtime_error("Dispersion vector has mismatched size");
    }
    const auto& disp_data = *disp_ptr;

    const double log_2pi = std::log(2.0 * 3.14159265358979323846);

    // Identify fixed effect predictors for REML adjustments
    std::vector<std::string> fixed_effect_vars;
    if (method == EstimationMethod::REML) {
        for (const auto& edge : model.edges) {
            if (edge.kind == EdgeKind::Regression && edge.target == obs_var_name) {
                fixed_effect_vars.push_back(edge.source);
            }
        }
        std::sort(fixed_effect_vars.begin(), fixed_effect_vars.end());
    }

    if (obs_family_name == "gaussian") {
        // Check for block-diagonal optimization
        bool can_optimize_blocks = false;
        std::string common_grouping_var;
        if (!random_effect_infos.empty()) {
            can_optimize_blocks = true;
            common_grouping_var = random_effect_infos[0].grouping_var;
            if (common_grouping_var.empty()) {
                 can_optimize_blocks = false;
            } else {
                for (size_t k = 1; k < random_effect_infos.size(); ++k) {
                    if (random_effect_infos[k].grouping_var != common_grouping_var) {
                        can_optimize_blocks = false;
                        break;
                    }
                }
            }
        }

        if (can_optimize_blocks) {
            // Optimized path for block-diagonal covariance
            auto group_map = build_group_map(random_effect_infos[0], data, n);
            
            // Prepare REML structures if needed
            std::vector<double> Xt_Vinv_X;
            std::size_t p_reml = 0;
            if (method == EstimationMethod::REML && !fixed_effect_vars.empty()) {
                p_reml = fixed_effect_vars.size();
                Xt_Vinv_X.assign(p_reml * p_reml, 0.0);
            }

            double total_loglik = 0.0;

            for (const auto& [group_id, all_indices] : group_map) {
                // Filter for observed data
                std::vector<std::size_t> indices;
                indices.reserve(all_indices.size());
                for(auto idx : all_indices) {
                    if (!std::isnan(obs_data[idx])) {
                        indices.push_back(idx);
                    }
                }
                
                if (indices.empty()) continue;
                
                std::size_t n_i = indices.size();
                std::vector<double> V_i(n_i * n_i, 0.0);
                std::vector<double> resid_i(n_i);
                
                // Initialize V_i with dispersion and resid_i
                for (std::size_t r = 0; r < n_i; ++r) {
                    std::size_t idx = indices[r];
                    resid_i[r] = obs_data[idx] - pred_data[idx];
                    V_i[r * n_i + r] = disp_data[idx];
                }
                
                // Add random effects
                for (const auto& info : random_effect_infos) {
                    auto Z_i_full = build_design_matrix(info, indices, data, linear_predictors);
                    std::size_t q = info.cov_spec->dimension;
                    
                    // Z_i_full is n_i x q
                    // Compute Z G Z^T
                    Eigen::Map<const RowMajorMatrix> Z_map(Z_i_full.data(), n_i, q);
                    Eigen::Map<const RowMajorMatrix> G_map(info.G_matrix.data(), q, q);
                    Eigen::Map<RowMajorMatrix> V_map(V_i.data(), n_i, n_i);
                    V_map += Z_map * G_map * Z_map.transpose();
                }                
                // Factorize V_i
                std::vector<double> L_i(n_i * n_i);
                cholesky(n_i, V_i, L_i);
                double log_det_i = log_det_cholesky(n_i, L_i);
                std::vector<double> alpha_i = solve_cholesky(n_i, L_i, resid_i);
                std::vector<double> V_inv_i = invert_from_cholesky(n_i, L_i);
                
                double quad_form_i = 0.0;
                for (std::size_t k = 0; k < n_i; ++k) {
                    quad_form_i += resid_i[k] * alpha_i[k];
                }
                
                total_loglik -= 0.5 * (n_i * log_2pi + log_det_i + quad_form_i);
                
                // REML Accumulation
                if (method == EstimationMethod::REML && !fixed_effect_vars.empty()) {
                    std::vector<double> X_i(n_i * p_reml);
                    for (std::size_t j = 0; j < p_reml; ++j) {
                        const auto& vec = data.at(fixed_effect_vars[j]);
                        for (std::size_t r = 0; r < n_i; ++r) {
                            X_i[r * p_reml + j] = vec[indices[r]];
                        }
                    }
                    
                    std::vector<double> Vinv_X_i(n_i * p_reml, 0.0);
                    for (std::size_t c = 0; c < p_reml; ++c) {
                        for (std::size_t r = 0; r < n_i; ++r) {
                            double sum = 0.0;
                            for (std::size_t k = 0; k < n_i; ++k) {
                                sum += V_inv_i[r * n_i + k] * X_i[k * p_reml + c];
                            }
                            Vinv_X_i[r * p_reml + c] = sum;
                        }
                    }
                    
                    for (std::size_t r = 0; r < p_reml; ++r) {
                        for (std::size_t c = 0; c < p_reml; ++c) {
                            double sum = 0.0;
                            for (std::size_t k = 0; k < n_i; ++k) {
                                sum += X_i[k * p_reml + r] * Vinv_X_i[k * p_reml + c];
                            }
                            Xt_Vinv_X[r * p_reml + c] += sum;
                        }
                    }
                }
            }
            
            // Finalize REML
            if (method == EstimationMethod::REML && !fixed_effect_vars.empty()) {
                std::vector<double> L_reml(p_reml * p_reml);
                try {
                    cholesky(p_reml, Xt_Vinv_X, L_reml);
                    double log_det_reml = log_det_cholesky(p_reml, L_reml);
                    total_loglik -= 0.5 * log_det_reml;
                    total_loglik += 0.5 * static_cast<double>(p_reml) * log_2pi;
                } catch (...) {
                    return -std::numeric_limits<double>::infinity();
                }
            }
            
            return total_loglik;
        }

        std::vector<std::map<double, std::vector<std::size_t>>> group_maps;
        group_maps.reserve(random_effect_infos.size());
        for (const auto& info : random_effect_infos) {
            group_maps.push_back(build_group_map(info, data, n));
        }

        std::vector<double> V_full(n * n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            V_full[i * n + i] = disp_data[i];
        }

        for (std::size_t re_idx = 0; re_idx < random_effect_infos.size(); ++re_idx) {
            const auto& info = random_effect_infos[re_idx];
            const auto& groups = group_maps[re_idx];
            for (const auto& [_, indices] : groups) {
                auto Z_i = build_design_matrix(info, indices, data, linear_predictors);
                std::size_t n_i = indices.size();
                std::size_t q = info.cov_spec->dimension;

                std::vector<double> ZG(n_i * q, 0.0);
                for (std::size_t r = 0; r < n_i; ++r) {
                    for (std::size_t c = 0; c < q; ++c) {
                        double sum = 0.0;
                        for (std::size_t k = 0; k < q; ++k) {
                            sum += Z_i[r * q + k] * info.G_matrix[k * q + c];
                        }
                        ZG[r * q + c] = sum;
                    }
                }

                for (std::size_t r = 0; r < n_i; ++r) {
                    for (std::size_t c = 0; c < n_i; ++c) {
                        double sum = 0.0;
                        for (std::size_t k = 0; k < q; ++k) {
                            sum += ZG[r * q + k] * Z_i[c * q + k];
                        }
                        V_full[indices[r] * n + indices[c]] += sum;
                    }
                }
            }
        }

        // Handle missing data by subsetting
        std::vector<std::size_t> observed_indices;
        observed_indices.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (!std::isnan(obs_data[i])) {
                observed_indices.push_back(i);
            }
        }
        std::size_t n_obs = observed_indices.size();

        std::vector<double> V_sub(n_obs * n_obs);
        std::vector<double> resid_sub(n_obs);
        for (std::size_t r = 0; r < n_obs; ++r) {
            resid_sub[r] = obs_data[observed_indices[r]] - pred_data[observed_indices[r]];
            for (std::size_t c = 0; c < n_obs; ++c) {
                V_sub[r * n_obs + c] = V_full[observed_indices[r] * n + observed_indices[c]];
            }
        }

        std::vector<double> L(n_obs * n_obs);
        cholesky(n_obs, V_sub, L);
        double log_det = log_det_cholesky(n_obs, L);
        std::vector<double> alpha = solve_cholesky(n_obs, L, resid_sub);

        double quad_form = 0.0;
        for (std::size_t i = 0; i < n_obs; ++i) {
            quad_form += resid_sub[i] * alpha[i];
        }

        double total_loglik = -0.5 * (n_obs * log_2pi + log_det + quad_form);

        if (method == EstimationMethod::REML && !fixed_effect_vars.empty()) {
            std::vector<double> V_inv = invert_from_cholesky(n_obs, L);
            std::size_t p = fixed_effect_vars.size();
            std::vector<double> X_sub(n_obs * p);
            for (std::size_t j = 0; j < p; ++j) {
                auto it = data.find(fixed_effect_vars[j]);
                if (it == data.end() || it->second.size() != n) {
                    throw std::runtime_error("Missing data for fixed effect: " + fixed_effect_vars[j]);
                }
                for (std::size_t i = 0; i < n_obs; ++i) {
                    X_sub[i * p + j] = it->second[observed_indices[i]];
                }
            }

            std::vector<double> Vinv_X(n_obs * p, 0.0);
            for (std::size_t col = 0; col < p; ++col) {
                for (std::size_t row = 0; row < n_obs; ++row) {
                    double sum = 0.0;
                    for (std::size_t k = 0; k < n_obs; ++k) {
                        sum += V_inv[row * n_obs + k] * X_sub[k * p + col];
                    }
                    Vinv_X[row * p + col] = sum;
                }
            }

            std::vector<double> Xt_Vinv_X(p * p, 0.0);
            for (std::size_t r = 0; r < p; ++r) {
                for (std::size_t c = 0; c < p; ++c) {
                    double sum = 0.0;
                    for (std::size_t row = 0; row < n_obs; ++row) {
                        sum += X_sub[row * p + r] * Vinv_X[row * p + c];
                    }
                    Xt_Vinv_X[r * p + c] = sum;
                }
            }

            std::vector<double> L_reml(p * p);
            cholesky(p, Xt_Vinv_X, L_reml);
            double log_det_reml = log_det_cholesky(p, L_reml);
            total_loglik -= 0.5 * log_det_reml;
            total_loglik += 0.5 * static_cast<double>(p) * log_2pi;
        }

        return total_loglik;
    }

    if (random_effect_infos.empty()) {
        throw std::runtime_error("Random effect metadata missing");
    }

    auto outcome_family = OutcomeFamilyFactory::create(obs_family_name);
    auto status_it = status.find(obs_var_name);
    const std::vector<double>* status_vec = nullptr;
    if (status_it != status.end()) {
        if (status_it->second.size() != n) {
            throw std::runtime_error("Status vector has mismatched size");
        }
        status_vec = &status_it->second;
    }
    const auto& extra_vec = get_extra_params_for_variable(extra_params, obs_var_name);

    LaplaceSystem system = build_laplace_system(random_effect_infos, data, linear_predictors, n);
    if (system.total_dim == 0) {
        throw std::runtime_error("Laplace system constructed without latent dimensions");
    }

    LaplaceSystemResult laplace_result;
    if (!solve_laplace_system(system,
                              obs_data,
                              pred_data,
                              disp_data,
                              status_vec,
                              extra_vec,
                              *outcome_family,
                              laplace_result)) {
        return -std::numeric_limits<double>::infinity();
    }

    double log_prior = compute_prior_loglik(system, laplace_result, log_2pi);
    double log_lik_data = 0.0;
    for (const auto& eval : laplace_result.evaluations) {
        log_lik_data += eval.log_likelihood;
    }
    double log_det_neg_hess = log_det_cholesky(system.total_dim, laplace_result.chol_neg_hessian);
    double total_loglik = log_prior + log_lik_data + 0.5 * system.total_dim * log_2pi - 0.5 * log_det_neg_hess;
    return total_loglik;
}

std::unordered_map<std::string, double> LikelihoodDriver::evaluate_model_gradient(const ModelIR& model,
                                                const std::unordered_map<std::string, std::vector<double>>& data,
                                                const std::unordered_map<std::string, std::vector<double>>& linear_predictors,
                                                const std::unordered_map<std::string, std::vector<double>>& dispersions,
                                                const std::unordered_map<std::string, std::vector<double>>& covariance_parameters,
                                                const std::unordered_map<std::string, std::vector<double>>& status,
                                                const std::unordered_map<std::string, std::vector<double>>& extra_params,
                                                const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data,
                                                EstimationMethod method,
                                                const std::unordered_map<std::string, DataParamMapping>& data_param_mappings,
                                                const std::unordered_map<std::string, DataParamMapping>& dispersion_param_mappings) const {
    std::unordered_map<std::string, double> gradients;
    (void)method;

    if (model.random_effects.empty()) {
        // Pre-process edges for efficient lookup: target -> [(source, param_id)]
        std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> incoming_edges;
        std::unordered_map<std::string, std::string> dispersion_params;

        for (const auto& edge : model.edges) {
            if (edge.kind == EdgeKind::Regression && !edge.parameter_id.empty()) {
                incoming_edges[edge.target].emplace_back(edge.source, edge.parameter_id);
            } else if (edge.kind == EdgeKind::Covariance && edge.source == edge.target && !edge.parameter_id.empty()) {
                dispersion_params[edge.source] = edge.parameter_id;
            }
        }

        for (const auto& var : model.variables) {
            if (var.kind != VariableKind::Observed) continue;

            auto data_it = data.find(var.name);
            auto pred_it = linear_predictors.find(var.name);
            auto disp_it = dispersions.find(var.name);
            
            if (data_it == data.end() || pred_it == linear_predictors.end() || disp_it == dispersions.end()) {
                 continue; 
            }

            const auto& obs = data_it->second;
            const auto& preds = pred_it->second;
            const auto& disps = disp_it->second;
            if (obs.size() != preds.size()) {
                throw std::invalid_argument("Data vectors for variable " + var.name + " have mismatched sizes");
            }
            if (disps.empty()) {
                throw std::invalid_argument("Dispersion vector for variable " + var.name + " is empty");
            }
            if (disps.size() != obs.size() && disps.size() != 1) {
                throw std::invalid_argument("Dispersion vector for variable " + var.name + " must have size 1 or match observation count");
            }
            const bool shared_dispersion = disps.size() == 1;
            const double shared_dispersion_value = shared_dispersion ? disps.front() : 0.0;
            
            const std::vector<double>* status_vec = nullptr;
            if (status.count(var.name)) status_vec = &status.at(var.name);

            const std::vector<double>* extra_vec = nullptr;
            if (extra_params.count(var.name)) extra_vec = &extra_params.at(var.name);
            const std::vector<double>& ep = extra_vec ? *extra_vec : std::vector<double>{};

            auto family = OutcomeFamilyFactory::create(var.family);
            
            // Get incoming edges for this variable
            const auto& edges = incoming_edges[var.name];
            std::string disp_param_id;
            if (dispersion_params.count(var.name)) {
                disp_param_id = dispersion_params.at(var.name);
            }

            for (size_t i = 0; i < obs.size(); ++i) {
                if (std::isnan(obs[i])) {
                    continue;
                }
                double s = status_vec ? (*status_vec)[i] : 1.0;
                const double dispersion_value = shared_dispersion ? shared_dispersion_value : disps[i];
                auto eval = family->evaluate(obs[i], preds[i], dispersion_value, s, ep);
                double d_lp = eval.first_derivative; // d(loglik)/d(eta)

                for (const auto& edge : edges) {
                    const std::string& source = edge.first;
                    const std::string& param_id = edge.second;
                    
                    double source_val = 0.0;
                    if (data.count(source)) {
                        source_val = data.at(source)[i];
                    }
                    
                    gradients[param_id] += d_lp * source_val;
                }

                if (!disp_param_id.empty()) {
                    gradients[disp_param_id] += eval.d_dispersion;
                }
            }
        }
        return gradients;
    } else {
        auto random_effect_infos = build_random_effect_infos(model, covariance_parameters, fixed_covariance_data, true);
        if (random_effect_infos.empty()) {
            throw std::runtime_error("Random effect metadata missing");
        }

        std::string obs_var_name;
        std::string obs_family_name;
        for (const auto& var : model.variables) {
            if (var.kind == VariableKind::Observed) {
                bool is_target = false;
                for (const auto& edge : model.edges) {
                    if (edge.target == var.name) {
                        is_target = true;
                        break;
                    }
                }
                if (is_target || obs_var_name.empty()) {
                    obs_var_name = var.name;
                    obs_family_name = var.family;
                }
            }
        }
        if (obs_var_name.empty()) {
            throw std::runtime_error("No observed variable found");
        }

        auto pred_it = linear_predictors.find(obs_var_name);
        if (pred_it == linear_predictors.end()) {
            if (linear_predictors.size() == 1) {
                obs_var_name = linear_predictors.begin()->first;
                auto it = std::find_if(model.variables.begin(), model.variables.end(), [&](const auto& v) {
                    return v.name == obs_var_name;
                });
                if (it == model.variables.end()) {
                    throw std::runtime_error("Outcome variable metadata not found: " + obs_var_name);
                }
                obs_family_name = it->family;
                pred_it = linear_predictors.begin();
            } else {
                throw std::runtime_error("Linear predictors for outcome variable not found");
            }
        }

        auto obs_it = data.find(obs_var_name);
        if (obs_it == data.end()) {
            throw std::runtime_error("Observed variable not found in data: " + obs_var_name);
        }
        const auto& obs_data = obs_it->second;
        const auto& pred_data = pred_it->second;
        if (obs_data.size() != pred_data.size()) {
            throw std::runtime_error("Outcome data and predictors have mismatched sizes");
        }
        std::size_t n = obs_data.size();

        auto disp_it = dispersions.find(obs_var_name);
        if (disp_it == dispersions.end()) {
            throw std::runtime_error("Dispersions not found for: " + obs_var_name);
        }
        const auto& disp_raw = disp_it->second;
        if (disp_raw.empty()) {
            throw std::runtime_error("Dispersion vector is empty");
        }
        std::vector<double> disp_broadcast;
        const std::vector<double>* disp_ptr = &disp_raw;
        if (disp_raw.size() == 1) {
            disp_broadcast.assign(n, disp_raw.front());
            disp_ptr = &disp_broadcast;
        } else if (disp_raw.size() != n) {
            throw std::runtime_error("Dispersion vector has mismatched size");
        }
        const auto& disp_data = *disp_ptr;

        if (obs_family_name == "gaussian") {
            std::vector<std::map<double, std::vector<std::size_t>>> group_maps;
            group_maps.reserve(random_effect_infos.size());
            for (const auto& info : random_effect_infos) {
                group_maps.push_back(build_group_map(info, data, n));
            }

            std::vector<double> V_full(n * n, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                V_full[i * n + i] = disp_data[i];
            }

            for (std::size_t re_idx = 0; re_idx < random_effect_infos.size(); ++re_idx) {
                const auto& info = random_effect_infos[re_idx];
                const auto& groups = group_maps[re_idx];
                for (const auto& [_, indices] : groups) {
                    auto Z_i = build_design_matrix(info, indices, data, linear_predictors);
                    std::size_t n_i = indices.size();
                    std::size_t q = info.cov_spec->dimension;

                    std::vector<double> ZG(n_i * q, 0.0);
                    for (std::size_t r = 0; r < n_i; ++r) {
                        for (std::size_t c = 0; c < q; ++c) {
                            double sum = 0.0;
                            for (std::size_t k = 0; k < q; ++k) {
                                sum += Z_i[r * q + k] * info.G_matrix[k * q + c];
                            }
                            ZG[r * q + c] = sum;
                        }
                    }

                    for (std::size_t r = 0; r < n_i; ++r) {
                        for (std::size_t c = 0; c < n_i; ++c) {
                            double sum = 0.0;
                            for (std::size_t k = 0; k < q; ++k) {
                                sum += ZG[r * q + k] * Z_i[c * q + k];
                            }
                            V_full[indices[r] * n + indices[c]] += sum;
                        }
                    }
                }
            }

            // Subset logic for gradient
            std::vector<std::size_t> observed_indices;
            std::vector<int> original_to_subset(n, -1);
            for (std::size_t i = 0; i < n; ++i) {
                if (!std::isnan(obs_data[i])) {
                    original_to_subset[i] = observed_indices.size();
                    observed_indices.push_back(i);
                }
            }
            std::size_t n_obs = observed_indices.size();

            std::vector<double> V_sub(n_obs * n_obs);
            std::vector<double> resid_sub(n_obs);
            for (std::size_t r = 0; r < n_obs; ++r) {
                resid_sub[r] = obs_data[observed_indices[r]] - pred_data[observed_indices[r]];
                for (std::size_t c = 0; c < n_obs; ++c) {
                    V_sub[r * n_obs + c] = V_full[observed_indices[r] * n + observed_indices[c]];
                }
            }

            std::vector<double> L(n_obs * n_obs);
            cholesky(n_obs, V_sub, L);
            std::vector<double> V_inv = invert_from_cholesky(n_obs, L);
            std::vector<double> alpha = solve_cholesky(n_obs, L, resid_sub);

            for (const auto& edge : model.edges) {
                if (edge.kind != EdgeKind::Regression || edge.target != obs_var_name || edge.parameter_id.empty()) {
                    continue;
                }
                if (!data.count(edge.source)) {
                    continue;
                }
                const auto& src_vec = data.at(edge.source);
                double accum = 0.0;
                for (std::size_t i = 0; i < n_obs; ++i) {
                    accum += src_vec[observed_indices[i]] * alpha[i];
                }
                gradients[edge.parameter_id] += accum;
            }

            // Gradients for dispersions (residual variances)
            if (!dispersion_param_mappings.empty()) {
                auto map_it = dispersion_param_mappings.find(obs_var_name);
                if (map_it != dispersion_param_mappings.end()) {
                    const auto& mapping = map_it->second;
                    
                    // dL/d_sigma2_k = 0.5 * (alpha_k^2 - (V^-1)_kk)
                    // We iterate over observed indices
                    for (std::size_t i = 0; i < n_obs; ++i) {
                        std::size_t original_idx = observed_indices[i];
                        
                        // Map original_idx to parameter
                        size_t idx = original_idx % mapping.stride;
                        if (idx < mapping.pattern.size()) {
                            const std::string& pid = mapping.pattern[idx];
                            if (!pid.empty()) {
                                double dL_dsigma2 = 0.5 * (alpha[i] * alpha[i] - V_inv[i * n_obs + i]);
                                gradients[pid] += dL_dsigma2;
                            }
                        }
                    }
                }
            }

            for (std::size_t re_idx = 0; re_idx < random_effect_infos.size(); ++re_idx) {
                const auto& info = random_effect_infos[re_idx];
                
                bool has_active_data = false;
                for(const auto& v : info.design_vars) {
                    if(data_param_mappings.count(v)) { has_active_data = true; break; }
                }

                if (info.G_gradients.empty() && !has_active_data) continue;
                
                const auto& groups = group_maps[re_idx];
                for (const auto& [_, indices] : groups) {
                    auto Z_i = build_design_matrix(info, indices, data, linear_predictors);
                    std::size_t n_i = indices.size();
                    std::size_t q = info.cov_spec->dimension;

                    std::vector<double> temp(n_i * q, 0.0);
                    for (std::size_t r = 0; r < n_i; ++r) {
                        int r_sub = original_to_subset[indices[r]];
                        if (r_sub == -1) continue;

                        for (std::size_t c = 0; c < q; ++c) {
                            double sum = 0.0;
                            for (std::size_t k = 0; k < n_i; ++k) {
                                int k_sub = original_to_subset[indices[k]];
                                if (k_sub != -1) {
                                    double term = alpha[r_sub] * alpha[k_sub] - V_inv[r_sub * n_obs + k_sub];
                                    sum += term * Z_i[k * q + c];
                                }
                            }
                            temp[r * q + c] = sum;
                        }
                    }

                    if (!info.G_gradients.empty()) {
                        std::vector<double> M(q * q, 0.0);
                        for (std::size_t r = 0; r < q; ++r) {
                            for (std::size_t c = 0; c < q; ++c) {
                                double sum = 0.0;
                                for (std::size_t k = 0; k < n_i; ++k) {
                                    sum += Z_i[k * q + r] * temp[k * q + c];
                                }
                                M[r * q + c] = sum;
                            }
                        }

                        for (std::size_t k = 0; k < info.G_gradients.size(); ++k) {
                            const auto& dG = info.G_gradients[k];
                            double trace = 0.0;
                            for (std::size_t r = 0; r < q; ++r) {
                                for (std::size_t c = 0; c < q; ++c) {
                                    trace += M[r * q + c] * dG[c * q + r];
                                }
                            }
                            std::string param_name = info.covariance_id + "_" + std::to_string(k);
                            gradients[param_name] += 0.5 * trace;
                        }
                    }
                    
                    if (has_active_data) {
                        // dL/dZ = P Z G = temp * G
                        for (std::size_t r = 0; r < n_i; ++r) {
                            for (std::size_t c = 0; c < q; ++c) {
                                const std::string& var_name = info.design_vars[c];
                                auto map_it = data_param_mappings.find(var_name);
                                if (map_it != data_param_mappings.end()) {
                                    const auto& mapping = map_it->second;
                                    size_t idx = indices[r] % mapping.stride;
                                    if (idx < mapping.pattern.size()) {
                                        const std::string& pid = mapping.pattern[idx];
                                        if (!pid.empty()) {
                                            double val = 0.0;
                                            for (std::size_t k = 0; k < q; ++k) {
                                                val += temp[r * q + k] * info.G_matrix[k * q + c];
                                            }
                                            gradients[pid] += val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return gradients;
        }
        auto outcome_family = OutcomeFamilyFactory::create(obs_family_name);
        auto status_it = status.find(obs_var_name);
        const std::vector<double>* status_vec = nullptr;
        if (status_it != status.end()) {
            if (status_it->second.size() != n) {
                throw std::runtime_error("Status vector has mismatched size");
            }
            status_vec = &status_it->second;
        }
        const auto& extra_vec = get_extra_params_for_variable(extra_params, obs_var_name);

        LaplaceSystem system = build_laplace_system(random_effect_infos, data, linear_predictors, n);
        if (system.total_dim == 0) {
            throw std::runtime_error("Laplace system constructed without latent dimensions");
        }

        LaplaceSystemResult laplace_result;
        if (!solve_laplace_system(system,
                                  obs_data,
                                  pred_data,
                                  disp_data,
                                  status_vec,
                                  extra_vec,
                                  *outcome_family,
                                  laplace_result)) {
            throw std::runtime_error("Laplace mode failed to converge for gradient computation");
        }

        if (!dispersion_param_mappings.empty()) {
            auto map_it = dispersion_param_mappings.find(obs_var_name);
            if (map_it != dispersion_param_mappings.end()) {
                const auto& mapping = map_it->second;
                for (std::size_t i = 0; i < n; ++i) {
                    size_t idx = i % mapping.stride;
                    if (idx < mapping.pattern.size()) {
                        const std::string& pid = mapping.pattern[idx];
                        if (!pid.empty()) {
                            gradients[pid] += laplace_result.evaluations[i].d_dispersion;
                        }
                    }
                }
            }
        }

        std::vector<double> neg_hess_inv = invert_from_cholesky(system.total_dim, laplace_result.chol_neg_hessian);
        auto quad_terms = compute_observation_quad_terms(system, neg_hess_inv);
        std::vector<double> forcing(system.total_dim, 0.0);

        for (const auto& edge : model.edges) {
            if (edge.kind != EdgeKind::Regression || edge.target != obs_var_name || edge.parameter_id.empty()) {
                continue;
            }
            auto src_it = data.find(edge.source);
            if (src_it == data.end() || src_it->second.size() != n) {
                continue;
            }

            const auto& src_vec = src_it->second;
            std::vector<double> weights(n, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                weights[i] = laplace_result.evaluations[i].second_derivative * src_vec[i];
            }
            accumulate_weighted_forcing(system, weights, forcing);
            auto du = solve_cholesky(system.total_dim, laplace_result.chol_neg_hessian, forcing);
            auto z_dot_du = project_observations(system, du);

            double accum = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const auto& eval = laplace_result.evaluations[i];
                double logdet_adjust = src_vec[i] + z_dot_du[i];
                accum += src_vec[i] * eval.first_derivative + 0.5 * eval.third_derivative * logdet_adjust * quad_terms[i];
            }
            gradients[edge.parameter_id] += accum;
        }

        struct CovarianceDerivativeCache {
            std::string param_name;
            std::vector<double> Ginv_dG_Ginv;
            double trace_Ginv_dG;
        };

        std::unordered_map<std::string, std::vector<CovarianceDerivativeCache>> covariance_caches;
        for (const auto& info : random_effect_infos) {
            if (info.G_gradients.empty()) {
                continue;
            }
            std::vector<CovarianceDerivativeCache> caches;
            std::size_t q = info.cov_spec->dimension;
            for (std::size_t idx = 0; idx < info.G_gradients.size(); ++idx) {
                CovarianceDerivativeCache cache;
                cache.param_name = info.covariance_id + "_" + std::to_string(idx);
                auto temp = multiply_matrices(info.G_inverse, info.G_gradients[idx], q);
                cache.Ginv_dG_Ginv = multiply_matrices(temp, info.G_inverse, q);
                cache.trace_Ginv_dG = trace_product(info.G_inverse, info.G_gradients[idx], q);
                caches.push_back(std::move(cache));
            }
            covariance_caches.emplace(info.covariance_id, std::move(caches));
        }

        const std::size_t Q = system.total_dim;
        for (const auto& block : system.blocks) {
            auto cache_it = covariance_caches.find(block.info->covariance_id);
            if (cache_it == covariance_caches.end()) {
                continue;
            }

            std::vector<double> u_block(block.q);
            for (std::size_t j = 0; j < block.q; ++j) {
                u_block[j] = laplace_result.u[block.offset + j];
            }

            for (const auto& cache : cache_it->second) {
                auto block_forcing = multiply_matrix_vector(cache.Ginv_dG_Ginv, u_block, block.q);
                std::fill(forcing.begin(), forcing.end(), 0.0);
                for (std::size_t j = 0; j < block.q; ++j) {
                    forcing[block.offset + j] = block_forcing[j];
                }
                auto du = solve_cholesky(system.total_dim, laplace_result.chol_neg_hessian, forcing);
                auto z_dot_du = project_observations(system, du);

                double logdet_adjust = 0.0;
                for (std::size_t obs_idx = 0; obs_idx < n; ++obs_idx) {
                    logdet_adjust += 0.5 * laplace_result.evaluations[obs_idx].third_derivative * z_dot_du[obs_idx] * quad_terms[obs_idx];
                }

                double prior_term = 0.5 * quadratic_form(u_block, cache.Ginv_dG_Ginv, block.q) - 0.5 * cache.trace_Ginv_dG;
                double trace_term = 0.0;
                for (std::size_t r = 0; r < block.q; ++r) {
                    for (std::size_t c = 0; c < block.q; ++c) {
                        const std::size_t row_idx = block.offset + r;
                        const std::size_t col_idx = block.offset + c;
                        trace_term += neg_hess_inv[row_idx * Q + col_idx] * cache.Ginv_dG_Ginv[c * block.q + r];
                    }
                }
                double logdet_term = 0.5 * trace_term + logdet_adjust;
                gradients[cache.param_name] += prior_term + logdet_term;
            }
        }

        return gradients;
    }
}

std::pair<double, std::unordered_map<std::string, double>> LikelihoodDriver::evaluate_model_loglik_and_gradient(
    const ModelIR& model,
    const std::unordered_map<std::string, std::vector<double>>& data,
    const std::unordered_map<std::string, std::vector<double>>& linear_predictors,
    const std::unordered_map<std::string, std::vector<double>>& dispersions,
    const std::unordered_map<std::string, std::vector<double>>& covariance_parameters,
    const std::unordered_map<std::string, std::vector<double>>& status,
    const std::unordered_map<std::string, std::vector<double>>& extra_params,
    const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data,
    EstimationMethod method,
    const std::unordered_map<std::string, DataParamMapping>& data_param_mappings,
    const std::unordered_map<std::string, DataParamMapping>& dispersion_param_mappings) const {

    std::unordered_map<std::string, double> gradients;
    double total_loglik = 0.0;

    if (model.random_effects.empty()) {
        // Pre-process edges for efficient lookup: target -> [(source, param_id)]
        std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> incoming_edges;
        std::unordered_map<std::string, std::string> dispersion_params;

        for (const auto& edge : model.edges) {
            if (edge.kind == EdgeKind::Regression && !edge.parameter_id.empty()) {
                incoming_edges[edge.target].emplace_back(edge.source, edge.parameter_id);
            } else if (edge.kind == EdgeKind::Covariance && edge.source == edge.target && !edge.parameter_id.empty()) {
                dispersion_params[edge.source] = edge.parameter_id;
            }
        }

        for (const auto& var : model.variables) {
            if (var.kind != VariableKind::Observed) continue;

            auto data_it = data.find(var.name);
            auto pred_it = linear_predictors.find(var.name);
            auto disp_it = dispersions.find(var.name);
            if (data_it == data.end() || pred_it == linear_predictors.end() || disp_it == dispersions.end()) {
                continue;
            }

            const auto& obs = data_it->second;
            const auto& preds = pred_it->second;
            const auto& disps = disp_it->second;
            if (obs.size() != preds.size()) {
                throw std::invalid_argument("Data vectors for variable " + var.name + " have mismatched sizes");
            }
            if (disps.empty()) {
                throw std::invalid_argument("Dispersion vector for variable " + var.name + " is empty");
            }
            if (disps.size() != obs.size() && disps.size() != 1) {
                throw std::invalid_argument("Dispersion vector for variable " + var.name + " must have size 1 or match observation count");
            }
            const bool shared_dispersion = disps.size() == 1;
            const double shared_dispersion_value = shared_dispersion ? disps.front() : 0.0;

            const std::vector<double>* status_vec = nullptr;
            if (status.count(var.name)) status_vec = &status.at(var.name);

            const std::vector<double>* extra_vec = nullptr;
            if (extra_params.count(var.name)) extra_vec = &extra_params.at(var.name);
            const std::vector<double>& ep = extra_vec ? *extra_vec : std::vector<double>{};

            auto family = OutcomeFamilyFactory::create(var.family);
            const auto& edges = incoming_edges[var.name];
            std::string disp_param_id;
            if (dispersion_params.count(var.name)) {
                disp_param_id = dispersion_params.at(var.name);
            }

            for (std::size_t i = 0; i < obs.size(); ++i) {
                if (std::isnan(obs[i])) {
                    continue;
                }
                double s = status_vec ? (*status_vec)[i] : 1.0;
                const double dispersion_value = shared_dispersion ? shared_dispersion_value : disps[i];
                auto eval = family->evaluate(obs[i], preds[i], dispersion_value, s, ep);
                total_loglik += eval.log_likelihood;
                double d_lp = eval.first_derivative; // d(loglik)/d(eta)

                for (const auto& edge : edges) {
                    const std::string& source = edge.first;
                    const std::string& param_id = edge.second;

                    double source_val = 0.0;
                    if (data.count(source)) {
                        source_val = data.at(source)[i];
                    }

                    gradients[param_id] += d_lp * source_val;
                }

                if (!disp_param_id.empty()) {
                    gradients[disp_param_id] += eval.d_dispersion;
                }
            }

            // REML adjustment (log-likelihood only)
            if (method == EstimationMethod::REML && var.family == "gaussian") {
                std::vector<std::string> predictors;
                for (const auto& edge : model.edges) {
                    if (edge.kind == EdgeKind::Regression && edge.target == var.name) {
                        predictors.push_back(edge.source);
                    }
                }
                std::sort(predictors.begin(), predictors.end());
                if (!predictors.empty()) {
                    std::size_t n = obs.size();
                    std::size_t p = predictors.size();
                    std::vector<double> X(n * p);
                    for (std::size_t j = 0; j < p; ++j) {
                        auto it = data.find(predictors[j]);
                        if (it == data.end() || it->second.size() != n) {
                            continue;
                        }
                        for (std::size_t i = 0; i < n; ++i) {
                            X[i * p + j] = it->second[i];
                        }
                    }

                    std::vector<double> Xt_Vinv_X(p * p, 0.0);
                    for (std::size_t r = 0; r < p; ++r) {
                        for (std::size_t c = 0; c < p; ++c) {
                            double sum = 0.0;
                            for (std::size_t i = 0; i < n; ++i) {
                                if (std::isnan(obs[i])) continue;
                                double disp = shared_dispersion ? shared_dispersion_value : disps[i];
                                sum += X[i * p + r] * X[i * p + c] / disp;
                            }
                            Xt_Vinv_X[r * p + c] = sum;
                        }
                    }

                    std::vector<double> L_reml(p * p);
                    cholesky(p, Xt_Vinv_X, L_reml);
                    double log_det_reml = log_det_cholesky(p, L_reml);
                    double log_2pi = std::log(2.0 * 3.14159265358979323846);
                    total_loglik -= 0.5 * log_det_reml;
                    total_loglik += 0.5 * static_cast<double>(p) * log_2pi;
                }
            }
        }
        return {total_loglik, gradients};
    }

    // Mixed model logic
    auto random_effect_infos = build_random_effect_infos(model, covariance_parameters, fixed_covariance_data, true);
    if (random_effect_infos.empty()) {
        throw std::runtime_error("Random effect metadata missing");
    }

    std::string obs_var_name;
    std::string obs_family_name;
    for (const auto& var : model.variables) {
        if (var.kind == VariableKind::Observed) {
            bool is_target = false;
            for (const auto& edge : model.edges) {
                if (edge.target == var.name) {
                    is_target = true;
                    break;
                }
            }
            if (is_target || obs_var_name.empty()) {
                obs_var_name = var.name;
                obs_family_name = var.family;
            }
        }
    }
    if (obs_var_name.empty()) {
        throw std::runtime_error("No observed variable found");
    }

    auto pred_it = linear_predictors.find(obs_var_name);
    if (pred_it == linear_predictors.end()) {
        if (linear_predictors.size() == 1) {
            obs_var_name = linear_predictors.begin()->first;
            auto var_it = std::find_if(model.variables.begin(), model.variables.end(), [&](const auto& v) {
                return v.name == obs_var_name;
            });
            if (var_it == model.variables.end()) {
                throw std::runtime_error("Outcome variable metadata not found: " + obs_var_name);
            }
            obs_family_name = var_it->family;
            pred_it = linear_predictors.begin();
        } else {
            throw std::runtime_error("Linear predictors for outcome variable not found");
        }
    }
    const auto& pred_data = pred_it->second;

    auto obs_it = data.find(obs_var_name);
    if (obs_it == data.end()) {
        throw std::runtime_error("Observed variable not found in data: " + obs_var_name);
    }
    const auto& obs_data = obs_it->second;

    if (obs_data.size() != pred_data.size()) {
        throw std::runtime_error("Outcome data and predictors have mismatched sizes");
    }
    std::size_t n = obs_data.size();

    auto disp_it = dispersions.find(obs_var_name);
    if (disp_it == dispersions.end()) {
        throw std::runtime_error("Dispersions not found for: " + obs_var_name);
    }
    const auto& disp_raw = disp_it->second;
    if (disp_raw.empty()) {
        throw std::runtime_error("Dispersion vector is empty");
    }
    std::vector<double> disp_broadcast;
    const std::vector<double>* disp_ptr = &disp_raw;
    if (disp_raw.size() == 1) {
        disp_broadcast.assign(n, disp_raw.front());
        disp_ptr = &disp_broadcast;
    } else if (disp_raw.size() != n) {
        throw std::runtime_error("Dispersion vector has mismatched size");
    }
    const auto& disp_data = *disp_ptr;

    const double log_2pi = std::log(2.0 * 3.14159265358979323846);

    // Identify fixed effect predictors for REML adjustments
    std::vector<std::string> fixed_effect_vars;
    if (method == EstimationMethod::REML) {
        for (const auto& edge : model.edges) {
            if (edge.kind == EdgeKind::Regression && edge.target == obs_var_name) {
                fixed_effect_vars.push_back(edge.source);
            }
        }
        std::sort(fixed_effect_vars.begin(), fixed_effect_vars.end());
    }

    if (obs_family_name == "gaussian") {
        // Check for block-diagonal optimization
        bool can_optimize_blocks = false;
        std::string common_grouping_var;
        if (!random_effect_infos.empty()) {
            can_optimize_blocks = true;
            common_grouping_var = random_effect_infos[0].grouping_var;
            if (common_grouping_var.empty()) {
                 can_optimize_blocks = false;
            } else {
                for (size_t k = 1; k < random_effect_infos.size(); ++k) {
                    if (random_effect_infos[k].grouping_var != common_grouping_var) {
                        can_optimize_blocks = false;
                        break;
                    }
                }
            }
        }

        if (can_optimize_blocks) {
            // Optimized path for block-diagonal covariance with pattern grouping
            auto group_map = build_group_map(random_effect_infos[0], data, n);
            
            // Prepare REML structures if needed
            std::vector<double> Xt_Vinv_X;
            std::size_t p_reml = 0;
            if (method == EstimationMethod::REML && !fixed_effect_vars.empty()) {
                p_reml = fixed_effect_vars.size();
                Xt_Vinv_X.assign(p_reml * p_reml, 0.0);
            }

            // Group by missingness pattern
            std::map<std::vector<std::size_t>, std::vector<std::size_t>> patterns;
            for (const auto& [group_id, all_indices] : group_map) {
                std::vector<std::size_t> indices;
                indices.reserve(all_indices.size());
                for(auto idx : all_indices) {
                    if (!std::isnan(obs_data[idx])) {
                        indices.push_back(idx);
                    }
                }
                if (indices.empty()) continue;
                
                std::vector<std::size_t> relative_indices;
                relative_indices.reserve(indices.size());
                std::size_t group_start = all_indices[0];
                for (auto idx : indices) {
                    relative_indices.push_back(idx - group_start);
                }
                patterns[relative_indices].push_back(group_id);
            }

            for (const auto& [rel_indices, group_ids] : patterns) {
                std::size_t n_i = rel_indices.size();
                std::size_t rep_group_id = group_ids[0];
                const auto& rep_all_indices = group_map[rep_group_id];
                std::vector<std::size_t> rep_indices;
                rep_indices.reserve(n_i);
                for (auto rel_idx : rel_indices) {
                    rep_indices.push_back(rep_all_indices[rel_idx]);
                }
                
                // Build V_i for representative
                std::vector<double> V_i(n_i * n_i, 0.0);
                for (std::size_t r = 0; r < n_i; ++r) {
                    std::size_t idx = rep_indices[r];
                    V_i[r * n_i + r] = disp_data[idx];
                }
                
                for (const auto& info : random_effect_infos) {
                    auto Z_i_full = build_design_matrix(info, rep_indices, data, linear_predictors);
                    std::size_t q = info.cov_spec->dimension;
                    Eigen::Map<const RowMajorMatrix> Z_map(Z_i_full.data(), n_i, q);
                    Eigen::Map<const RowMajorMatrix> G_map(info.G_matrix.data(), q, q);
                    Eigen::Map<RowMajorMatrix> V_map(V_i.data(), n_i, n_i);
                    V_map += Z_map * G_map * Z_map.transpose();
                }
                
                std::vector<double> L_i(n_i * n_i);
                cholesky(n_i, V_i, L_i);
                double log_det_i = log_det_cholesky(n_i, L_i);
                std::vector<double> V_inv_i = invert_from_cholesky(n_i, L_i);
                Eigen::Map<const RowMajorMatrix> V_inv_map(V_inv_i.data(), n_i, n_i);
                
                // Precompute matrices for gradients if needed
                struct PatternCache {
                    std::vector<double> Vinv_Z; // n_i * q
                    std::vector<double> A;      // q * q
                };
                std::vector<PatternCache> caches(random_effect_infos.size());
                
                for (size_t k=0; k<random_effect_infos.size(); ++k) {
                    const auto& info = random_effect_infos[k];
                    if (info.G_gradients.empty() && info.design_vars.empty()) continue;
                    
                    auto Z_i_full = build_design_matrix(info, rep_indices, data, linear_predictors);
                    std::size_t q = info.cov_spec->dimension;
                    
                    caches[k].Vinv_Z.resize(n_i * q);
                    Eigen::Map<RowMajorMatrix> Vinv_Z_map(caches[k].Vinv_Z.data(), n_i, q);
                    Eigen::Map<const RowMajorMatrix> Z_map(Z_i_full.data(), n_i, q);
                    Vinv_Z_map = V_inv_map * Z_map;
                    
                    caches[k].A.resize(q * q);
                    Eigen::Map<RowMajorMatrix> A_map(caches[k].A.data(), q, q);
                    A_map = Z_map.transpose() * Vinv_Z_map;
                }

                // Accumulate over groups in pattern
                for (auto gid : group_ids) {
                    const auto& all_ind = group_map[gid];
                    std::vector<double> resid(n_i);
                    for (std::size_t r = 0; r < n_i; ++r) {
                        std::size_t idx = all_ind[rel_indices[r]];
                        resid[r] = obs_data[idx] - pred_data[idx];
                    }
                    Eigen::Map<Eigen::VectorXd> r_vec(resid.data(), n_i);
                    
                    // alpha = V^-1 * r
                    Eigen::VectorXd alpha = V_inv_map * r_vec;
                    
                    double quad_form = r_vec.dot(alpha);
                    total_loglik -= 0.5 * (n_i * log_2pi + log_det_i + quad_form);
                    
                    // Gradients
                    
                    // 1. Dispersion
                    if (!dispersion_param_mappings.empty()) {
                        auto map_it = dispersion_param_mappings.find(obs_var_name);
                        if (map_it != dispersion_param_mappings.end()) {
                            const auto& mapping = map_it->second;
                            for (std::size_t r = 0; r < n_i; ++r) {
                                std::size_t global_idx = all_ind[rel_indices[r]];
                                size_t pat_idx = global_idx % mapping.stride;
                                if (pat_idx < mapping.pattern.size()) {
                                    const std::string& pid = mapping.pattern[pat_idx];
                                    if (!pid.empty()) {
                                        double term1 = V_inv_i[r * n_i + r];
                                        double term2 = alpha(r) * alpha(r);
                                        gradients[pid] += 0.5 * (term2 - term1);
                                    }
                                }
                            }
                        }
                    }
                    
                    // 2. Fixed Effects
                    for (const auto& edge : model.edges) {
                        if (edge.kind == EdgeKind::Regression && edge.target == obs_var_name && !edge.parameter_id.empty()) {
                            if (data.count(edge.source)) {
                                const auto& src_vec = data.at(edge.source);
                                double grad_accum = 0.0;
                                for (std::size_t r = 0; r < n_i; ++r) {
                                    grad_accum += alpha(r) * src_vec[all_ind[rel_indices[r]]];
                                }
                                gradients[edge.parameter_id] += grad_accum;
                            }
                        }
                    }
                    
                    // 3. Covariance & Loadings
                    for (size_t k=0; k<random_effect_infos.size(); ++k) {
                        const auto& info = random_effect_infos[k];
                        std::size_t q = info.cov_spec->dimension;
                        
                        Eigen::Map<const RowMajorMatrix> Vinv_Z_map(caches[k].Vinv_Z.data(), n_i, q);
                        Eigen::VectorXd b = Vinv_Z_map.transpose() * r_vec;
                        
                        // dL/dG
                        for (std::size_t idx = 0; idx < info.G_gradients.size(); ++idx) {
                            std::string param_name = info.covariance_id + "_" + std::to_string(idx);
                            const auto& dG = info.G_gradients[idx];
                            Eigen::Map<const RowMajorMatrix> dG_map(dG.data(), q, q);
                            Eigen::Map<const RowMajorMatrix> A_map(caches[k].A.data(), q, q);
                            
                            double term1 = b.transpose() * dG_map * b;
                            double term2 = (A_map * dG_map).trace();
                            gradients[param_name] += 0.5 * (term1 - term2);
                        }
                        
                        // dL/dZ
                        bool need_Z_grad = false;
                        for (const auto& var_name : info.design_vars) {
                            if (data_param_mappings.count(var_name)) {
                                need_Z_grad = true;
                                break;
                            }
                        }
                        
                        if (need_Z_grad) {
                            Eigen::MatrixXd M = alpha * b.transpose() - Vinv_Z_map;
                            Eigen::Map<const RowMajorMatrix> G_map(info.G_matrix.data(), q, q);
                            Eigen::MatrixXd dLdZ = M * G_map;
                            
                            for (std::size_t c = 0; c < q; ++c) {
                                const std::string& var_name = info.design_vars[c];
                                auto map_it = data_param_mappings.find(var_name);
                                if (map_it != data_param_mappings.end()) {
                                    const auto& mapping = map_it->second;
                                    for (std::size_t r = 0; r < n_i; ++r) {
                                        std::size_t global_idx = all_ind[rel_indices[r]];
                                        size_t pat_idx = global_idx % mapping.stride;
                                        if (pat_idx < mapping.pattern.size()) {
                                            const std::string& pid = mapping.pattern[pat_idx];
                                            if (!pid.empty()) {
                                                gradients[pid] += dLdZ(r, c);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    // REML
                    if (method == EstimationMethod::REML && !fixed_effect_vars.empty()) {
                        std::vector<double> X_i(n_i * p_reml);
                        for (std::size_t j = 0; j < p_reml; ++j) {
                            const auto& vec = data.at(fixed_effect_vars[j]);
                            for (std::size_t r = 0; r < n_i; ++r) {
                                X_i[r * p_reml + j] = vec[all_ind[rel_indices[r]]];
                            }
                        }
                        Eigen::Map<const RowMajorMatrix> X_map(X_i.data(), n_i, p_reml);
                        Eigen::Map<RowMajorMatrix> Xt_Vinv_X_map(Xt_Vinv_X.data(), p_reml, p_reml);
                        Xt_Vinv_X_map += X_map.transpose() * V_inv_map * X_map;
                    }
                }
            }
            
            if (method == EstimationMethod::REML && !fixed_effect_vars.empty()) {
                std::vector<double> L_reml(p_reml * p_reml);
                try {
                    cholesky(p_reml, Xt_Vinv_X, L_reml);
                    double log_det_reml = log_det_cholesky(p_reml, L_reml);
                    total_loglik -= 0.5 * log_det_reml;
                    total_loglik += 0.5 * static_cast<double>(p_reml) * log_2pi;
                } catch (...) {
                    return {-std::numeric_limits<double>::infinity(), gradients};
                }
            }
            
            return {total_loglik, gradients};
        }

        std::vector<std::map<double, std::vector<std::size_t>>> group_maps;
        group_maps.reserve(random_effect_infos.size());
        for (const auto& info : random_effect_infos) {
            group_maps.push_back(build_group_map(info, data, n));
        }

        std::vector<double> V_full(n * n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            V_full[i * n + i] = disp_data[i];
        }

        for (std::size_t re_idx = 0; re_idx < random_effect_infos.size(); ++re_idx) {
            const auto& info = random_effect_infos[re_idx];
            const auto& groups = group_maps[re_idx];
            for (const auto& [_, indices] : groups) {
                auto Z_i = build_design_matrix(info, indices, data, linear_predictors);
                std::size_t n_i = indices.size();
                std::size_t q = info.cov_spec->dimension;

                std::vector<double> ZG(n_i * q, 0.0);
                for (std::size_t r = 0; r < n_i; ++r) {
                    for (std::size_t c = 0; c < q; ++c) {
                        double sum = 0.0;
                        for (std::size_t k = 0; k < q; ++k) {
                            sum += Z_i[r * q + k] * info.G_matrix[k * q + c];
                        }
                        ZG[r * q + c] = sum;
                    }
                }

                for (std::size_t r = 0; r < n_i; ++r) {
                    for (std::size_t c = 0; c < n_i; ++c) {
                        double sum = 0.0;
                        for (std::size_t k = 0; k < q; ++k) {
                            sum += ZG[r * q + k] * Z_i[c * q + k];
                        }
                        V_full[indices[r] * n + indices[c]] += sum;
                    }
                }
            }
        }

        // Handle missing data by subsetting
        std::vector<std::size_t> observed_indices;
        observed_indices.reserve(n);
        std::vector<int> original_to_subset(n, -1);
        for (std::size_t i = 0; i < n; ++i) {
            if (!std::isnan(obs_data[i])) {
                original_to_subset[i] = observed_indices.size();
                observed_indices.push_back(i);
            }
        }
        std::size_t n_obs = observed_indices.size();

        std::vector<double> V_sub(n_obs * n_obs);
        std::vector<double> resid_sub(n_obs);
        for (std::size_t r = 0; r < n_obs; ++r) {
            resid_sub[r] = obs_data[observed_indices[r]] - pred_data[observed_indices[r]];
            for (std::size_t c = 0; c < n_obs; ++c) {
                V_sub[r * n_obs + c] = V_full[observed_indices[r] * n + observed_indices[c]];
            }
        }

        std::vector<double> L(n_obs * n_obs);
        cholesky(n_obs, V_sub, L);
        double log_det = log_det_cholesky(n_obs, L);
        std::vector<double> alpha = solve_cholesky(n_obs, L, resid_sub);
        std::vector<double> V_inv = invert_from_cholesky(n_obs, L);

        double quad_form = 0.0;
        for (std::size_t i = 0; i < n_obs; ++i) {
            quad_form += resid_sub[i] * alpha[i];
        }

        total_loglik = -0.5 * (n_obs * log_2pi + log_det + quad_form);

        if (method == EstimationMethod::REML && !fixed_effect_vars.empty()) {
            std::size_t p = fixed_effect_vars.size();
            std::vector<double> X_sub(n_obs * p);
            for (std::size_t j = 0; j < p; ++j) {
                auto it = data.find(fixed_effect_vars[j]);
                if (it == data.end() || it->second.size() != n) {
                    throw std::runtime_error("Missing data for fixed effect: " + fixed_effect_vars[j]);
                }
                for (std::size_t i = 0; i < n_obs; ++i) {
                    X_sub[i * p + j] = it->second[observed_indices[i]];
                }
            }

            std::vector<double> Vinv_X(n_obs * p, 0.0);
            for (std::size_t col = 0; col < p; ++col) {
                for (std::size_t row = 0; row < n_obs; ++row) {
                    double sum = 0.0;
                    for (std::size_t k = 0; k < n_obs; ++k) {
                        sum += V_inv[row * n_obs + k] * X_sub[k * p + col];
                    }
                    Vinv_X[row * p + col] = sum;
                }
            }

            std::vector<double> Xt_Vinv_X(p * p, 0.0);
            for (std::size_t r = 0; r < p; ++r) {
                for (std::size_t c = 0; c < p; ++c) {
                    double sum = 0.0;
                    for (std::size_t row = 0; row < n_obs; ++row) {
                        sum += X_sub[row * p + r] * Vinv_X[row * p + c];
                    }
                    Xt_Vinv_X[r * p + c] = sum;
                }
            }

            std::vector<double> L_reml(p * p);
            cholesky(p, Xt_Vinv_X, L_reml);
            double log_det_reml = log_det_cholesky(p, L_reml);
            total_loglik -= 0.5 * log_det_reml;
            total_loglik += 0.5 * static_cast<double>(p) * log_2pi;
        }

        // Gradients
        for (const auto& edge : model.edges) {
            if (edge.kind != EdgeKind::Regression || edge.target != obs_var_name || edge.parameter_id.empty()) {
                continue;
            }
            if (!data.count(edge.source)) {
                continue;
            }
            const auto& src_vec = data.at(edge.source);
            double accum = 0.0;
            for (std::size_t i = 0; i < n_obs; ++i) {
                accum += src_vec[observed_indices[i]] * alpha[i];
            }
            gradients[edge.parameter_id] += accum;
        }

        // Gradients for dispersions (residual variances)
        if (!dispersion_param_mappings.empty()) {
            auto map_it = dispersion_param_mappings.find(obs_var_name);
            if (map_it != dispersion_param_mappings.end()) {
                const auto& mapping = map_it->second;
                for (std::size_t i = 0; i < n_obs; ++i) {
                    std::size_t original_idx = observed_indices[i];
                    size_t idx = original_idx % mapping.stride;
                    if (idx < mapping.pattern.size()) {
                        const std::string& pid = mapping.pattern[idx];
                        if (!pid.empty()) {
                            double dL_dsigma2 = 0.5 * (alpha[i] * alpha[i] - V_inv[i * n_obs + i]);
                            gradients[pid] += dL_dsigma2;
                        }
                    }
                }
            }
        }

        for (std::size_t re_idx = 0; re_idx < random_effect_infos.size(); ++re_idx) {
            const auto& info = random_effect_infos[re_idx];

            bool has_active_data = false;
            for(const auto& v : info.design_vars) {
                if(data_param_mappings.count(v)) { has_active_data = true; break; }
            }

            if (info.G_gradients.empty() && !has_active_data) continue;

            const auto& groups = group_maps[re_idx];
            for (const auto& [_, indices] : groups) {
                auto Z_i = build_design_matrix(info, indices, data, linear_predictors);
                std::size_t n_i = indices.size();
                std::size_t q = info.cov_spec->dimension;

                std::vector<double> temp(n_i * q, 0.0);
                for (std::size_t r = 0; r < n_i; ++r) {
                    int r_sub = original_to_subset[indices[r]];
                    if (r_sub == -1) continue;

                    for (std::size_t c = 0; c < q; ++c) {
                        double sum = 0.0;
                        for (std::size_t k = 0; k < n_i; ++k) {
                            int k_sub = original_to_subset[indices[k]];
                            if (k_sub != -1) {
                                double term = alpha[r_sub] * alpha[k_sub] - V_inv[r_sub * n_obs + k_sub];
                                sum += term * Z_i[k * q + c];
                            }
                        }
                        temp[r * q + c] = sum;
                    }
                }

                if (!info.G_gradients.empty()) {
                    std::vector<double> M(q * q, 0.0);
                    for (std::size_t r = 0; r < q; ++r) {
                        for (std::size_t c = 0; c < q; ++c) {
                            double sum = 0.0;
                            for (std::size_t k = 0; k < n_i; ++k) {
                                sum += Z_i[k * q + r] * temp[k * q + c];
                            }
                            M[r * q + c] = sum;
                        }
                    }

                    for (std::size_t k = 0; k < info.G_gradients.size(); ++k) {
                        const auto& dG = info.G_gradients[k];
                        double trace = 0.0;
                        for (std::size_t r = 0; r < q; ++r) {
                            for (std::size_t c = 0; c < q; ++c) {
                                trace += M[r * q + c] * dG[c * q + r];
                            }
                        }
                        std::string param_name = info.covariance_id + "_" + std::to_string(k);
                        gradients[param_name] += 0.5 * trace;
                    }
                }

                if (has_active_data) {
                    for (std::size_t r = 0; r < n_i; ++r) {
                        for (std::size_t c = 0; c < q; ++c) {
                            const std::string& var_name = info.design_vars[c];
                            auto map_it = data_param_mappings.find(var_name);
                            if (map_it != data_param_mappings.end()) {
                                const auto& mapping = map_it->second;
                                size_t idx = indices[r] % mapping.stride;
                                if (idx < mapping.pattern.size()) {
                                    const std::string& pid = mapping.pattern[idx];
                                    if (!pid.empty()) {
                                        double val = 0.0;
                                        for (std::size_t k = 0; k < q; ++k) {
                                            val += temp[r * q + k] * info.G_matrix[k * q + c];
                                        }
                                        gradients[pid] += val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return {total_loglik, gradients};
    }

    auto outcome_family = OutcomeFamilyFactory::create(obs_family_name);
    auto status_it = status.find(obs_var_name);
    const std::vector<double>* status_vec = nullptr;
    if (status_it != status.end()) {
        if (status_it->second.size() != n) {
            throw std::runtime_error("Status vector has mismatched size");
        }
        status_vec = &status_it->second;
    }
    const auto& extra_vec = get_extra_params_for_variable(extra_params, obs_var_name);

    LaplaceSystem system = build_laplace_system(random_effect_infos, data, linear_predictors, n);
    if (system.total_dim == 0) {
        throw std::runtime_error("Laplace system constructed without latent dimensions");
    }

    LaplaceSystemResult laplace_result;
    if (!solve_laplace_system(system,
                              obs_data,
                              pred_data,
                              disp_data,
                              status_vec,
                              extra_vec,
                              *outcome_family,
                              laplace_result)) {
        return { -std::numeric_limits<double>::infinity(), gradients };
    }

    double log_prior = compute_prior_loglik(system, laplace_result, log_2pi);
    double log_lik_data = 0.0;
    for (const auto& eval : laplace_result.evaluations) {
        log_lik_data += eval.log_likelihood;
    }
    double log_det_neg_hess = log_det_cholesky(system.total_dim, laplace_result.chol_neg_hessian);
    total_loglik = log_prior + log_lik_data + 0.5 * system.total_dim * log_2pi - 0.5 * log_det_neg_hess;

    if (!dispersion_param_mappings.empty()) {
        auto map_it = dispersion_param_mappings.find(obs_var_name);
        if (map_it != dispersion_param_mappings.end()) {
            const auto& mapping = map_it->second;
            for (std::size_t i = 0; i < n; ++i) {
                size_t idx = i % mapping.stride;
                if (idx < mapping.pattern.size()) {
                    const std::string& pid = mapping.pattern[idx];
                    if (!pid.empty()) {
                        gradients[pid] += laplace_result.evaluations[i].d_dispersion;
                    }
                }
            }
        }
    }

    std::vector<double> neg_hess_inv = invert_from_cholesky(system.total_dim, laplace_result.chol_neg_hessian);
    auto quad_terms = compute_observation_quad_terms(system, neg_hess_inv);
    std::vector<double> forcing(system.total_dim, 0.0);

    for (const auto& edge : model.edges) {
        if (edge.kind != EdgeKind::Regression || edge.target != obs_var_name || edge.parameter_id.empty()) {
            continue;
        }
        auto src_it = data.find(edge.source);
        if (src_it == data.end() || src_it->second.size() != n) {
            continue;
        }

        const auto& src_vec = src_it->second;
        std::vector<double> weights(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            weights[i] = laplace_result.evaluations[i].second_derivative * src_vec[i];
        }
        accumulate_weighted_forcing(system, weights, forcing);
        auto du = solve_cholesky(system.total_dim, laplace_result.chol_neg_hessian, forcing);
        auto z_dot_du = project_observations(system, du);

        double accum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const auto& eval = laplace_result.evaluations[i];
            double logdet_adjust = src_vec[i] + z_dot_du[i];
            accum += src_vec[i] * eval.first_derivative + 0.5 * eval.third_derivative * logdet_adjust * quad_terms[i];
        }
        gradients[edge.parameter_id] += accum;
    }

    struct CovarianceDerivativeCache {
        std::string param_name;
        std::vector<double> Ginv_dG_Ginv;
        double trace_Ginv_dG;
    };

    std::unordered_map<std::string, std::vector<CovarianceDerivativeCache>> covariance_caches;
    for (const auto& info : random_effect_infos) {
        if (info.G_gradients.empty()) {
            continue;
        }
        std::vector<CovarianceDerivativeCache> caches;
        std::size_t q = info.cov_spec->dimension;
        for (std::size_t idx = 0; idx < info.G_gradients.size(); ++idx) {
            CovarianceDerivativeCache cache;
            cache.param_name = info.covariance_id + "_" + std::to_string(idx);
            auto temp = multiply_matrices(info.G_inverse, info.G_gradients[idx], q);
            cache.Ginv_dG_Ginv = multiply_matrices(temp, info.G_inverse, q);
            cache.trace_Ginv_dG = trace_product(info.G_inverse, info.G_gradients[idx], q);
            caches.push_back(std::move(cache));
        }
        covariance_caches.emplace(info.covariance_id, std::move(caches));
    }

    const std::size_t Q = system.total_dim;
    for (const auto& block : system.blocks) {
        auto cache_it = covariance_caches.find(block.info->covariance_id);
        if (cache_it == covariance_caches.end()) {
            continue;
        }

        std::vector<double> u_block(block.q);
        for (std::size_t j = 0; j < block.q; ++j) {
            u_block[j] = laplace_result.u[block.offset + j];
        }

        for (const auto& cache : cache_it->second) {
            auto block_forcing = multiply_matrix_vector(cache.Ginv_dG_Ginv, u_block, block.q);
            std::fill(forcing.begin(), forcing.end(), 0.0);
            for (std::size_t j = 0; j < block.q; ++j) {
                forcing[block.offset + j] = block_forcing[j];
            }
            auto du = solve_cholesky(system.total_dim, laplace_result.chol_neg_hessian, forcing);
            auto z_dot_du = project_observations(system, du);

            double logdet_adjust = 0.0;
            for (std::size_t obs_idx = 0; obs_idx < n; ++obs_idx) {
                logdet_adjust += 0.5 * laplace_result.evaluations[obs_idx].third_derivative * z_dot_du[obs_idx] * quad_terms[obs_idx];
            }

            double prior_term = 0.5 * quadratic_form(u_block, cache.Ginv_dG_Ginv, block.q) - 0.5 * cache.trace_Ginv_dG;
            double trace_term = 0.0;
            for (std::size_t r = 0; r < block.q; ++r) {
                for (std::size_t c = 0; c < block.q; ++c) {
                    const std::size_t row_idx = block.offset + r;
                    const std::size_t col_idx = block.offset + c;
                    trace_term += neg_hess_inv[row_idx * Q + col_idx] * cache.Ginv_dG_Ginv[c * block.q + r];
                }
            }
            double logdet_term = 0.5 * trace_term + logdet_adjust;
            gradients[cache.param_name] += prior_term + logdet_term;
        }
    }

    return {total_loglik, gradients};
}

namespace {



Eigen::MatrixXd compute_hessian(const ObjectiveFunction& objective, const std::vector<double>& parameters) {
    const double epsilon = 1e-5;
    const size_t n = parameters.size();
    Eigen::MatrixXd hessian(n, n);
    std::vector<double> x = parameters;
    
    for (size_t j = 0; j < n; ++j) {
        double original_val = x[j];
        
        x[j] = original_val + epsilon;
        std::vector<double> grad_plus = objective.gradient(x);
        
        x[j] = original_val - epsilon;
        std::vector<double> grad_minus = objective.gradient(x);
        
        x[j] = original_val; // Restore
        
        for (size_t i = 0; i < n; ++i) {
            hessian(i, j) = (grad_plus[i] - grad_minus[i]) / (2.0 * epsilon);
        }
    }
    
    // Symmetrize
    hessian = 0.5 * (hessian + hessian.transpose());
    return hessian;
}

} // namespace

FitResult LikelihoodDriver::fit(const ModelIR& model,
                                         const std::unordered_map<std::string, std::vector<double>>& data,
                                         const OptimizationOptions& options,
                                         const std::string& optimizer_name,
                                         const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data,
                                         const std::unordered_map<std::string, std::vector<double>>& status,
                                         EstimationMethod method) const {
    
    ModelObjective objective(*this, model, data, fixed_covariance_data, status, method);
    auto initial_params = objective.initial_parameters();
    
    std::unique_ptr<Optimizer> optimizer;
    if (optimizer_name == "lbfgs") {
        optimizer = make_lbfgs_optimizer();
    } else {
        optimizer = make_gradient_descent_optimizer();
    }
    
    auto result = optimizer->optimize(objective, initial_params, options);
    
    FitResult fit_result;
    fit_result.optimization_result = result;
    fit_result.optimization_result.parameters = objective.to_constrained(result.parameters);
    fit_result.parameter_names = objective.parameter_names();
    fit_result.covariance_matrices = objective.get_covariance_matrices(fit_result.optimization_result.parameters);

    if (result.converged) {
        try {
            Eigen::MatrixXd hessian = compute_hessian(objective, result.parameters);
            // Hessian of NLL is Fisher Information. Covariance is Inverse.
            Eigen::MatrixXd vcov_unconstrained = hessian.inverse();
            
            std::vector<double> derivs = objective.constrained_derivatives(result.parameters);
            size_t n = result.parameters.size();
            Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n, n);
            for(size_t i=0; i<n; ++i) J(i,i) = derivs[i];
            
            Eigen::MatrixXd vcov_constrained = J * vcov_unconstrained * J.transpose();
            
            fit_result.standard_errors.resize(n);
            for(size_t i=0; i<n; ++i) {
                double var = vcov_constrained(i,i);
                fit_result.standard_errors[i] = (var > 0) ? std::sqrt(var) : std::numeric_limits<double>::quiet_NaN();
            }
            
            fit_result.vcov.resize(n*n);
            Eigen::Map<Eigen::MatrixXd>(fit_result.vcov.data(), n, n) = vcov_constrained;
            
            double nll = result.objective_value;
            double k = static_cast<double>(n);
            
            size_t sample_size = 0;
            if (!data.empty()) {
                sample_size = data.begin()->second.size();
            }
            
            fit_result.aic = 2.0 * k + 2.0 * nll;
            if (sample_size > 0) {
                fit_result.bic = k * std::log(static_cast<double>(sample_size)) + 2.0 * nll;
            } else {
                fit_result.bic = std::numeric_limits<double>::quiet_NaN();
            }

            // Fit Indices (SEM only)
            if (model.random_effects.empty() && sample_size > 1) {
                std::vector<size_t> obs_indices;
                std::vector<std::string> obs_names;
                size_t n_vars = model.variables.size();
                for(size_t i=0; i<n_vars; ++i) {
                    if (model.variables[i].kind == VariableKind::Observed) {
                        if (model.variables[i].name == "_intercept") continue;
                        obs_indices.push_back(i);
                        obs_names.push_back(model.variables[i].name);
                    }
                }
                size_t p = obs_indices.size();
                
                bool has_data = true;
                for(const auto& name : obs_names) {
                    if (data.find(name) == data.end()) { has_data = false; break; }
                }

                if (p > 0 && has_data) {
                    Eigen::MatrixXd S(p, p);
                    Eigen::VectorXd y_bar(p);
                    
                    for(size_t i=0; i<p; ++i) {
                        const auto& vec = data.at(obs_names[i]);
                        double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
                        y_bar(i) = sum / sample_size;
                    }
                    
                    for(size_t i=0; i<p; ++i) {
                        for(size_t j=i; j<p; ++j) {
                            double sum_sq = 0.0;
                            const auto& vec_i = data.at(obs_names[i]);
                            const auto& vec_j = data.at(obs_names[j]);
                            double mu_i = y_bar(i);
                            double mu_j = y_bar(j);
                            for(size_t kk=0; kk<sample_size; ++kk) {
                                sum_sq += (vec_i[kk] - mu_i) * (vec_j[kk] - mu_j);
                            }
                            S(i, j) = S(j, i) = sum_sq / sample_size;
                        }
                    }
                    
                    double log_det_S = std::log(S.determinant());
                    double pi = std::acos(-1.0);
                    double log_2pi = std::log(2.0 * pi);
                    double loglik_sat = -0.5 * sample_size * (p * log_2pi + log_det_S + p);
                    
                    double sum_log_var = 0.0;
                    for(size_t i=0; i<p; ++i) sum_log_var += std::log(S(i, i));
                    double loglik_base = -0.5 * sample_size * (p * log_2pi + sum_log_var + p);
                    
                    double loglik_user = -nll;
                    
                    double df_sat = p * (p + 1) / 2.0 + p;
                    double df_user = df_sat - k;
                    double df_base = df_sat - 2.0 * p;
                    
                    double chi2_user = 2.0 * (loglik_sat - loglik_user);
                    double chi2_base = 2.0 * (loglik_sat - loglik_base);
                    if (chi2_user < 0) chi2_user = 0.0;
                    
                    fit_result.chi_square = chi2_user;
                    fit_result.df = df_user;
                    
                    if (df_user > 0) {
                        double d = std::max(0.0, chi2_user - df_user);
                        fit_result.rmsea = std::sqrt(d / (df_user * sample_size));
                    } else {
                        fit_result.rmsea = 0.0;
                    }
                    
                    double d_user = std::max(0.0, chi2_user - df_user);
                    double d_base = std::max(0.0, chi2_base - df_base);
                    if (d_base > 0) {
                        fit_result.cfi = 1.0 - d_user / d_base;
                    } else {
                        fit_result.cfi = 1.0;
                    }
                    
                    if (df_base > 0 && df_user > 0) {
                        double ratio_base = chi2_base / df_base;
                        double ratio_user = chi2_user / df_user;
                        fit_result.tli = (ratio_base - ratio_user) / (ratio_base - 1.0);
                    } else {
                        fit_result.tli = 1.0;
                    }
                    
                    std::vector<double> full_means(n_vars, 0.0);
                    std::vector<double> full_cov(n_vars * n_vars, 0.0);
                    
                    for(size_t i=0; i<p; ++i) {
                        size_t idx = obs_indices[i];
                        full_means[idx] = y_bar(i);
                        for(size_t j=0; j<p; ++j) {
                            size_t jdx = obs_indices[j];
                            full_cov[idx * n_vars + jdx] = S(i, j);
                        }
                    }
                    
                    auto diag = compute_model_diagnostics(
                        model, 
                        objective.parameter_names(), 
                        fit_result.optimization_result.parameters,
                        full_means,
                        full_cov,
                        fixed_covariance_data
                    );
                    
                    fit_result.srmr = diag.srmr;
                }
            }
            
        } catch (...) {
             size_t n = result.parameters.size();
             fit_result.standard_errors.assign(n, std::numeric_limits<double>::quiet_NaN());
             fit_result.vcov.assign(n*n, std::numeric_limits<double>::quiet_NaN());
             fit_result.aic = std::numeric_limits<double>::quiet_NaN();
             fit_result.bic = std::numeric_limits<double>::quiet_NaN();
        }
    } else {
         size_t n = result.parameters.size();
         fit_result.standard_errors.assign(n, std::numeric_limits<double>::quiet_NaN());
         fit_result.vcov.assign(n*n, std::numeric_limits<double>::quiet_NaN());
         fit_result.aic = std::numeric_limits<double>::quiet_NaN();
         fit_result.bic = std::numeric_limits<double>::quiet_NaN();
    }

    return fit_result;
}

}  // namespace libsemx
