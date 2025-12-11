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
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
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
#include <list>
#include <memory>

namespace libsemx {




namespace {



using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;


struct RandomEffectInfo {
    std::string id;
    std::string grouping_var;
    std::vector<std::string> design_vars;
    std::string covariance_id;
    const CovarianceSpec* cov_spec;
    
    // Dense storage
    std::vector<double> G_matrix;
    std::vector<double> G_inverse;
    std::vector<std::vector<double>> G_gradients;
    
    // Sparse storage
    bool is_sparse = false;
    std::shared_ptr<Eigen::SparseMatrix<double>> G_sparse;
    std::shared_ptr<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>> sparse_solver;
    std::vector<Eigen::SparseMatrix<double>> G_gradients_sparse;

    double log_det_G = 0.0;
    std::vector<std::string> target_vars;
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

void build_random_effect_infos(
    const ModelIR& model,
    const std::unordered_map<std::string, std::vector<double>>& covariance_parameters,
    const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data,
    bool need_gradients,
    std::vector<RandomEffectInfo>& infos) {
    
    infos.clear();
    infos.reserve(model.random_effects.size());

    std::unordered_map<std::string, const CovarianceSpec*> cov_lookup;
    for (const auto& cov : model.covariances) {
        cov_lookup.emplace(cov.id, &cov);
    }

    std::unordered_map<std::string, std::vector<std::string>> re_targets;
    for (const auto& edge : model.edges) {
        if (edge.kind == EdgeKind::Regression) {
            re_targets[edge.source].push_back(edge.target);
        }
    }

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

        infos.emplace_back();
        RandomEffectInfo& info = infos.back();

        info.id = re.id;
        info.grouping_var = re.variables.front();
        info.design_vars.assign(re.variables.begin() + 1, re.variables.end());
        info.covariance_id = re.covariance_id;
        info.cov_spec = cov_it->second;
        
        if (structure->is_sparse()) {
            info.is_sparse = true;
            
            info.G_sparse = std::make_shared<Eigen::SparseMatrix<double>>(structure->materialize_sparse(params));
            
            
            // info.sparse_solver = std::make_shared<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>();
            auto solver = std::make_shared<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>();
            
            solver->analyzePattern(*info.G_sparse);
            solver->factorize(*info.G_sparse);
            
            if (solver->info() != Eigen::Success) {
                throw std::runtime_error("Sparse Cholesky factorization failed for " + re.id);
            }
            
            info.sparse_solver = solver;

            // Log determinant from sparse solver (LDLT)
            // log|G| = sum(log(D_ii))
            double log_det = 0.0;
            
            Eigen::VectorXd D = info.sparse_solver->vectorD();
            for (int k=0; k<D.size(); ++k) {
                log_det += std::log(D[k]);
            }
            
            info.log_det_G = log_det;
            
            
            if (need_gradients && structure->parameter_count() > 0) {
                info.G_gradients_sparse = structure->parameter_gradients_sparse(params);
            }
            
        } else {
            info.is_sparse = false;
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
        }
        
        auto target_it = re_targets.find(re.id);
        if (target_it != re_targets.end()) {
            info.target_vars = target_it->second;
        }
    }
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

        // Fallback: If q != n_i and no design vars, check if we can infer Z from grouping variable
        // This happens when we have repeated records (n_i > q) and the grouping variable
        // maps observations to levels of the random effect (1..q).
        // We assume the grouping variable values are 1-based indices or 0-based?
        // In R, factors are 1-based integers.
        // Let's try to use the grouping variable itself as the index.
        try {
             const auto& group_vals = resolve_design_values(info.grouping_var, data, linear_predictors);
             // Check if values are within range [1, q] or [0, q-1]
             // We'll assume 1-based if min >= 1, else 0-based?
             // Or just check bounds.
             
             for (std::size_t r = 0; r < n_i; ++r) {
                 double val = group_vals[indices[r]];
                 long col_idx = static_cast<long>(val);
                 
                 if (col_idx < 0 || static_cast<std::size_t>(col_idx) >= q) {
                     throw std::runtime_error("Index " + std::to_string(col_idx) + " out of bounds for covariance dimension " + std::to_string(q));
                 }
                 
                 Z[r * q + col_idx] = 1.0;
             }
             return Z;
        } catch (...) {
             // Fall through to error
        }

        throw std::runtime_error("Random effect " + info.id + " expects " + std::to_string(q) +
                                 " design columns but none were specified, and automatic inference failed.");
    }

    if (info.design_vars.size() != q) {
        // Special case: Single design variable with q > 1 implies categorical/index mapping
        if (info.design_vars.size() == 1 && q > 1) {
            const auto& values = resolve_design_values(info.design_vars[0], data, linear_predictors);
            for (std::size_t row = 0; row < n_i; ++row) {
                if (indices[row] >= values.size()) {
                    throw std::runtime_error("Design variable " + info.design_vars[0] + " has insufficient data length");
                }
                double val = values[indices[row]];
                // Check if integer
                if (std::floor(val) != val) {
                     throw std::runtime_error("Design variable " + info.design_vars[0] + " must be integer-valued for index mapping");
                }
                long col_idx = static_cast<long>(val);
                if (col_idx < 0 || static_cast<std::size_t>(col_idx) >= q) {
                     throw std::runtime_error("Index " + std::to_string(col_idx) + " out of bounds for covariance dimension " + std::to_string(q));
                }
                Z[row * q + col_idx] = 1.0;
            }
            return Z;
        }
        throw std::runtime_error("Design variable count (" + std::to_string(info.design_vars.size()) + 
                                 ") does not match covariance dimension (" + std::to_string(q) + ") for " + info.id);
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
    std::size_t index = 0;
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
    
    // Sparse support
    using SparseSolver = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>;
    std::shared_ptr<SparseSolver> sparse_solver;
    bool use_sparse = false;
    double log_det_neg_hess = 0.0;

    // Decoupled support
    std::vector<std::vector<double>> block_inverses; 
};

LaplaceSystem build_laplace_system(
    const std::vector<RandomEffectInfo>& infos,
    const std::unordered_map<std::string, std::vector<double>>& data,
    const std::unordered_map<std::string, std::vector<double>>& linear_predictors,
    std::size_t n);

struct LaplaceCache {
    LaplaceSystem system;
    std::vector<bool> block_static;
    std::vector<std::size_t> block_info_index;
    std::vector<std::string> re_ids;
    std::vector<std::string> grouping_vars;
    std::vector<std::size_t> q_dims;
    std::size_t n = 0;

    void reset() {
        system = LaplaceSystem{};
        block_static.clear();
        block_info_index.clear();
        re_ids.clear();
        grouping_vars.clear();
        q_dims.clear();
        n = 0;
    }
};

bool design_matrix_static(const RandomEffectInfo& info,
                          const std::unordered_map<std::string, std::vector<double>>& data,
                          const std::unordered_map<std::string, std::vector<double>>& linear_predictors) {
    if (info.design_vars.empty()) {
        return true;
    }

    for (const auto& var : info.design_vars) {
        // Latent loading data are mutated between evaluations, so keep them dynamic.
        if (var.rfind("_loading_", 0) == 0) {
            return false;
        }
        if (data.count(var) == 0) {
            // Pulled from linear predictors or missing; treat as dynamic.
            return false;
        }
    }
    return true;
}

LaplaceSystem& get_or_build_laplace_system(const std::vector<RandomEffectInfo>& infos,
                                           const std::unordered_map<std::string, std::vector<double>>& data,
                                           const std::unordered_map<std::string, std::vector<double>>& linear_predictors,
                                           std::size_t n,
                                           std::shared_ptr<void>& cache_handle) {
    auto cache = std::static_pointer_cast<LaplaceCache>(cache_handle);
    if (!cache) {
        cache = std::make_shared<LaplaceCache>();
        cache_handle = cache;
    }

    bool rebuild = cache->n != n || cache->re_ids.size() != infos.size();
    if (!rebuild) {
        for (std::size_t i = 0; i < infos.size(); ++i) {
            if (cache->re_ids[i] != infos[i].id ||
                cache->grouping_vars[i] != infos[i].grouping_var ||
                cache->q_dims[i] != infos[i].cov_spec->dimension) {
                rebuild = true;
                break;
            }
        }
    }

    if (rebuild) {
        cache->reset();
        cache->n = n;
        cache->re_ids.reserve(infos.size());
        cache->grouping_vars.reserve(infos.size());
        cache->q_dims.reserve(infos.size());

        // Build a fresh system using the existing helper.
        LaplaceSystem fresh = build_laplace_system(infos, data, linear_predictors, n);
        cache->system = std::move(fresh);

        // Record static/dynamic design metadata and block->info mapping.
        cache->system.total_dim = 0;
        for (const auto& block : cache->system.blocks) {
            cache->system.total_dim = std::max(cache->system.total_dim, block.offset + block.q);
        }

        cache->block_static.reserve(cache->system.blocks.size());
        cache->block_info_index.reserve(cache->system.blocks.size());

        for (std::size_t info_idx = 0; info_idx < infos.size(); ++info_idx) {
            const auto& info = infos[info_idx];
            cache->re_ids.push_back(info.id);
            cache->grouping_vars.push_back(info.grouping_var);
            cache->q_dims.push_back(info.cov_spec->dimension);

            bool static_design = design_matrix_static(info, data, linear_predictors);
            auto group_map = build_group_map(info, data, n);
            for ([[maybe_unused]] const auto& entry : group_map) {
                cache->block_static.push_back(static_design);
                cache->block_info_index.push_back(info_idx);
            }
        }

        if (cache->block_static.size() != cache->system.blocks.size()) {
            throw std::runtime_error("Laplace cache metadata mismatch");
        }

        return cache->system;
    }

    // Reuse layout; refresh pointers and any dynamic design matrices.
    for (std::size_t block_idx = 0; block_idx < cache->system.blocks.size(); ++block_idx) {
        const auto info_idx = cache->block_info_index[block_idx];
        auto& block = cache->system.blocks[block_idx];
        block.info = &infos[info_idx];

        if (!cache->block_static[block_idx]) {
            block.design_matrix = build_design_matrix(*block.info, block.obs_indices, data, linear_predictors);
        }
    }

    return cache->system;
}

LaplaceSystem build_laplace_system(
    const std::vector<RandomEffectInfo>& infos,
    const std::unordered_map<std::string, std::vector<double>>& data,
    const std::unordered_map<std::string, std::vector<double>>& linear_predictors,
    std::size_t n) {
    LaplaceSystem system;
    system.observation_entries.resize(n);
    std::size_t offset = 0;
    for (size_t i = 0; i < infos.size(); ++i) {
        const auto& info = infos[i];
        auto group_map = build_group_map(info, data, n);
        int group_count = 0;
        for (const auto& [_, indices] : group_map) {
            LaplaceBlock block;
            block.info = &info;
            block.obs_indices = indices;
            block.q = info.cov_spec->dimension;
            block.offset = offset;
            block.design_matrix = build_design_matrix(info, indices, data, linear_predictors);
            block.index = system.blocks.size();
            system.blocks.push_back(block);
            const std::size_t block_idx = system.blocks.size() - 1;
            for (std::size_t row = 0; row < indices.size(); ++row) {
                system.observation_entries[indices[row]].push_back({block_idx, row});
            }
            offset += block.q;
            group_count++;
        }
    }
    system.total_dim = offset;
    return system;
}

struct OutcomeData {
    std::string name;
    const std::vector<double>& obs;
    const std::vector<double>& pred;
    const std::vector<double>& disp;
    const std::vector<double>* status;
    const std::vector<double>& extra;
    const OutcomeFamily* family;
};

void compute_system_grad_hess(const LaplaceSystem& system,
                              const std::vector<double>& u,
                              const std::vector<OutcomeData>& outcomes,
                              std::vector<double>& grad,
                              std::vector<double>& hess,
                              std::vector<OutcomeEvaluation>* evals_out) {
    const std::size_t Q = system.total_dim;
    std::fill(grad.begin(), grad.end(), 0.0);
    std::fill(hess.begin(), hess.end(), 0.0);

    for (const auto& block : system.blocks) {
        const auto& info = *block.info;
        
        if (info.is_sparse) {
             if (!info.sparse_solver) {
                 throw std::runtime_error("Sparse solver is null");
             }
             Eigen::Map<const Eigen::VectorXd> u_vec(&u[block.offset], block.q);
             Eigen::VectorXd g_inv_u = info.sparse_solver->solve(u_vec);
             for (int i=0; i<block.q; ++i) {
                 grad[block.offset + i] -= g_inv_u[i];
             }
             
             for (int i=0; i<block.q; ++i) {
                 Eigen::VectorXd rhs = Eigen::VectorXd::Zero(block.q);
                 rhs[i] = 1.0;
                 Eigen::VectorXd col = info.sparse_solver->solve(rhs);
                 for (int j=0; j<block.q; ++j) {
                     hess[(block.offset + j) * Q + (block.offset + i)] -= col[j];
                 }
             }
        } else {
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
    }

    if (evals_out) {
        evals_out->clear();
    }

    for (const auto& outcome : outcomes) {
        std::size_t n = outcome.obs.size();
        for (std::size_t obs_idx = 0; obs_idx < n; ++obs_idx) {
            if (std::isnan(outcome.obs[obs_idx])) {
                if (evals_out) {
                    evals_out->push_back(OutcomeEvaluation{0.0, 0.0, 0.0, 0.0, 0.0, {}});
                }
                continue;
            }

            double eta = outcome.pred[obs_idx];
            const auto& entries = system.observation_entries[obs_idx];
            
            std::vector<const LaplaceBlock*> active_blocks;
            std::vector<std::size_t> active_rows;
            
            for (const auto& entry : entries) {
                const auto& block = system.blocks[entry.block_index];
                bool targets = false;
                for (const auto& t : block.info->target_vars) {
                    if (t == outcome.name) {
                        targets = true;
                        break;
                    }
                }
                if (targets) {
                    active_blocks.push_back(&block);
                    active_rows.push_back(entry.row_index);
                    const double* z_row = &block.design_matrix[entry.row_index * block.q];
                    const double* u_block = &u[block.offset];
                    for (std::size_t j = 0; j < block.q; ++j) {
                        eta += z_row[j] * u_block[j];
                    }
                }
            }

            const double s = outcome.status ? (*outcome.status)[obs_idx] : 1.0;
            auto eval = outcome.family->evaluate(outcome.obs[obs_idx], eta, outcome.disp[obs_idx], s, outcome.extra);
            if (evals_out) {
                evals_out->push_back(eval);
            }

            for (std::size_t i = 0; i < active_blocks.size(); ++i) {
                const auto* block = active_blocks[i];
                std::size_t row = active_rows[i];
                const double* z_row = &block->design_matrix[row * block->q];
                double* grad_block = &grad[block->offset];
                for (std::size_t j = 0; j < block->q; ++j) {
                    grad_block[j] += z_row[j] * eval.first_derivative;
                }
            }

            for (std::size_t a = 0; a < active_blocks.size(); ++a) {
                const auto* block_a = active_blocks[a];
                std::size_t row_a = active_rows[a];
                const double* z_a = &block_a->design_matrix[row_a * block_a->q];
                
                for (std::size_t b = a; b < active_blocks.size(); ++b) {
                    const auto* block_b = active_blocks[b];
                    std::size_t row_b = active_rows[b];
                    const double* z_b = &block_b->design_matrix[row_b * block_b->q];
                    
                    for (std::size_t r = 0; r < block_a->q; ++r) {
                        for (std::size_t c = 0; c < block_b->q; ++c) {
                            double value = z_a[r] * z_b[c] * eval.second_derivative;
                            const std::size_t row_idx = block_a->offset + r;
                            const std::size_t col_idx = block_b->offset + c;
                            hess[row_idx * Q + col_idx] += value;
                            if (row_idx != col_idx) {
                                hess[col_idx * Q + row_idx] += value;
                            }
                        }
                    }
                }
            }
        }
    }
}

double compute_system_objective(const LaplaceSystem& system,
                                const std::vector<double>& u,
                                const std::vector<OutcomeData>& outcomes) {
    double loglik = 0.0;

    // 1. Log-likelihood of data given u
    for (const auto& outcome : outcomes) {
        std::size_t n = outcome.obs.size();
        for (std::size_t obs_idx = 0; obs_idx < n; ++obs_idx) {
            if (std::isnan(outcome.obs[obs_idx])) continue;

            double eta = outcome.pred[obs_idx];

            if (obs_idx >= system.observation_entries.size()) {
                 std::abort();
            }

            const auto& entries = system.observation_entries[obs_idx];
            for (const auto& entry : entries) {
                if (entry.block_index >= system.blocks.size()) {
                     std::abort();
                }
                const auto& block = system.blocks[entry.block_index];
                bool targets = false;
                for (const auto& t : block.info->target_vars) {
                    if (t == outcome.name) {
                        targets = true;
                        break;
                    }
                }
                if (!targets) continue;

                const double* z_row = &block.design_matrix[entry.row_index * block.q];
                const double* u_block = &u[block.offset];
                for (std::size_t j = 0; j < block.q; ++j) {
                    eta += z_row[j] * u_block[j];
                }
            }

            const double s = outcome.status ? (*outcome.status)[obs_idx] : 1.0;
            
            if (!outcome.family) {
                 std::abort();
            }
            loglik += outcome.family->evaluate(outcome.obs[obs_idx], eta, outcome.disp[obs_idx], s, outcome.extra).log_likelihood;
        }
    }

    // 2. Penalty term: -0.5 * u^T * G^-1 * u
    for (const auto& block : system.blocks) {
        const auto& info = *block.info;
        const double* u_block = &u[block.offset];
        
        if (info.is_sparse) {
            if (!info.sparse_solver) {
                throw std::runtime_error("sparse_solver is null in compute_system_objective");
            }
            // u^T G^-1 u
            Eigen::Map<const Eigen::VectorXd> u_vec(u_block, block.q);
            Eigen::VectorXd G_inv_u = info.sparse_solver->solve(u_vec);
            loglik -= 0.5 * u_vec.dot(G_inv_u);
        } else {
            const double* G_inv = info.G_inverse.data();
            for (std::size_t r = 0; r < block.q; ++r) {
                double row_sum = 0.0;
                for (std::size_t c = 0; c < block.q; ++c) {
                    row_sum += G_inv[r * block.q + c] * u_block[c];
                }
                loglik -= 0.5 * u_block[r] * row_sum;
            }
        }
    }

    return loglik;
}

void compute_system_grad_hess_sparse(const LaplaceSystem& system,
                                     const std::vector<double>& u,
                                     const std::vector<OutcomeData>& outcomes,
                                     std::vector<double>& grad,
                                     Eigen::SparseMatrix<double>& neg_hess,
                                     std::vector<OutcomeEvaluation>* evals_out,
                                     SparseHessianAccumulator* accumulator = nullptr) {
    const std::size_t Q = system.total_dim;
    std::fill(grad.begin(), grad.end(), 0.0);
    
    if (accumulator && accumulator->initialized) {
        // Reset Hessian values to zero
        for (int k=0; k<accumulator->hessian.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(accumulator->hessian, k); it; ++it) {
                it.valueRef() = 0.0;
            }
        }
        
        // Add Prior contributions (G^{-1})
        for (const auto& block : system.blocks) {
            if (block.info->is_sparse) {
                // Gradient: G^-1 u
                Eigen::Map<const Eigen::VectorXd> u_vec(&u[block.offset], block.q);
                Eigen::VectorXd g_inv_u = block.info->sparse_solver->solve(u_vec);
                for (int i=0; i<block.q; ++i) {
                    grad[block.offset + i] -= g_inv_u[i];
                }

                // Hessian: Add G^-1
                if (block.info->G_sparse->nonZeros() == block.info->G_sparse->rows()) {
                     // Diagonal optimization
                     for (int k=0; k<block.info->G_sparse->outerSize(); ++k) {
                        for (Eigen::SparseMatrix<double>::InnerIterator it(*block.info->G_sparse, k); it; ++it) {
                            if (it.row() == it.col()) {
                                double val = 1.0 / it.value();
                                accumulator->hessian.coeffRef(block.offset + it.row(), block.offset + it.col()) += val;
                            }
                        }
                     }
                } else {
                     // General sparse inverse
                     for (int i=0; i<block.q; ++i) {
                         Eigen::VectorXd rhs = Eigen::VectorXd::Zero(block.q);
                         rhs[i] = 1.0;
                         Eigen::VectorXd col = block.info->sparse_solver->solve(rhs);
                         for (int j=0; j<block.q; ++j) {
                             double val = col[j];
                             if (val != 0.0) {
                                 accumulator->hessian.coeffRef(block.offset + j, block.offset + i) += val;
                             }
                         }
                     }
                }
            } else {
                const auto& G_inv = block.info->G_inverse;
                for (std::size_t r = 0; r < block.q; ++r) {
                    double sum = 0.0;
                    for (std::size_t c = 0; c < block.q; ++c) {
                        double val = G_inv[r * block.q + c];
                        if (val != 0.0) {
                            accumulator->hessian.coeffRef(block.offset + r, block.offset + c) += val;
                        }
                        sum += val * u[block.offset + c];
                    }
                    grad[block.offset + r] -= sum;
                }
            }
        }
        
        if (evals_out) evals_out->clear();
        
        for (size_t out_idx = 0; out_idx < outcomes.size(); ++out_idx) {
            const auto& outcome = outcomes[out_idx];
            const auto& recipe = accumulator->outcome_recipes[out_idx];
            std::size_t n = outcome.obs.size();
            
            for (std::size_t obs_idx = 0; obs_idx < n; ++obs_idx) {
                if (std::isnan(outcome.obs[obs_idx])) {
                    if (evals_out) evals_out->push_back(OutcomeEvaluation{0.0, 0.0, 0.0, 0.0, 0.0, {}});
                    continue;
                }
                
                double eta = outcome.pred[obs_idx];
                const auto& entries = system.observation_entries[obs_idx];
                
                std::vector<const LaplaceBlock*> active_blocks;
                std::vector<std::size_t> active_rows;
                for (const auto& entry : entries) {
                    const auto& block = system.blocks[entry.block_index];
                    bool targets = false;
                    for (const auto& t : block.info->target_vars) {
                        if (t == outcome.name) {
                            targets = true;
                            break;
                        }
                    }
                    if (targets) {
                        active_blocks.push_back(&block);
                        active_rows.push_back(entry.row_index);
                        const double* z_row = &block.design_matrix[entry.row_index * block.q];
                        const double* u_block = &u[block.offset];
                        for (std::size_t j = 0; j < block.q; ++j) {
                            eta += z_row[j] * u_block[j];
                        }
                    }
                }

                const double s = outcome.status ? (*outcome.status)[obs_idx] : 1.0;
                auto eval = outcome.family->evaluate(outcome.obs[obs_idx], eta, outcome.disp[obs_idx], s, outcome.extra);
                if (evals_out) evals_out->push_back(eval);

                for (std::size_t i = 0; i < active_blocks.size(); ++i) {
                    const auto* block = active_blocks[i];
                    std::size_t row = active_rows[i];
                    const double* z_row = &block->design_matrix[row * block->q];
                    double* grad_block = &grad[block->offset];
                    for (std::size_t j = 0; j < block->q; ++j) {
                        grad_block[j] += z_row[j] * eval.first_derivative;
                    }
                }

                double w = -eval.second_derivative;
                const auto& obs_recipe = recipe.obs_recipes[obs_idx];
                for (const auto& item : obs_recipe) {
                    double val = w * item.z_product;
                    accumulator->hessian.coeffRef(item.row_idx, item.col_idx) += val;
                    if (item.row_idx != item.col_idx) {
                        accumulator->hessian.coeffRef(item.col_idx, item.row_idx) += val;
                    }
                }
            }
        }
        
        neg_hess = accumulator->hessian;
        return;
    }

    neg_hess.setZero();
    std::vector<Eigen::Triplet<double>> triplets;
    
    {
        for (const auto& block : system.blocks) {
            if (block.info->is_sparse) {
                // Gradient: G^-1 u
                Eigen::Map<const Eigen::VectorXd> u_vec(&u[block.offset], block.q);
                Eigen::VectorXd g_inv_u = block.info->sparse_solver->solve(u_vec);
                for (int i=0; i<block.q; ++i) {
                    grad[block.offset + i] -= g_inv_u[i];
                }

                // Hessian: Add G^-1
                if (block.info->G_sparse->nonZeros() == block.info->G_sparse->rows()) {
                     // Diagonal optimization
                     for (int k=0; k<block.info->G_sparse->outerSize(); ++k) {
                        for (Eigen::SparseMatrix<double>::InnerIterator it(*block.info->G_sparse, k); it; ++it) {
                            if (it.row() == it.col()) {
                                double val = 1.0 / it.value();
                                triplets.emplace_back(block.offset + it.row(), block.offset + it.col(), val);
                            }
                        }
                     }
                } else {
                     // General sparse inverse
                     for (int i=0; i<block.q; ++i) {
                         Eigen::VectorXd rhs = Eigen::VectorXd::Zero(block.q);
                         rhs[i] = 1.0;
                         Eigen::VectorXd col = block.info->sparse_solver->solve(rhs);
                         for (int j=0; j<block.q; ++j) {
                             double val = col[j];
                             if (val != 0.0) {
                                 triplets.emplace_back(block.offset + j, block.offset + i, val);
                             }
                         }
                     }
                }
            } else {
                const auto& info = *block.info;
                const double* G_inv = info.G_inverse.data();
                const double* u_block = &u[block.offset];
                for (std::size_t r = 0; r < block.q; ++r) {
                    double sum = 0.0;
                    for (std::size_t c = 0; c < block.q; ++c) {
                        double val = G_inv[r * block.q + c];
                        if (val != 0.0) triplets.emplace_back(block.offset + r, block.offset + c, val);
                        sum += val * u_block[c];
                    }
                    grad[block.offset + r] -= sum;
                }
            }
        }

        if (evals_out) evals_out->clear();

        for (size_t out_idx = 0; out_idx < outcomes.size(); ++out_idx) {
            const auto& outcome = outcomes[out_idx];
            std::size_t n = outcome.obs.size();
            for (std::size_t obs_idx = 0; obs_idx < n; ++obs_idx) {
                if (std::isnan(outcome.obs[obs_idx])) {
                    if (evals_out) {
                        evals_out->push_back(OutcomeEvaluation{0.0, 0.0, 0.0, 0.0, 0.0, {}});
                    }
                    continue;
                }

                double eta = outcome.pred[obs_idx];
                const auto& entries = system.observation_entries[obs_idx];
                
                std::vector<const LaplaceBlock*> active_blocks;
                std::vector<std::size_t> active_rows;
                for (const auto& entry : entries) {
                    const auto& block = system.blocks[entry.block_index];
                    bool targets = false;
                    for (const auto& t : block.info->target_vars) {
                        if (t == outcome.name) {
                            targets = true;
                            break;
                        }
                    }
                    if (targets) {
                        active_blocks.push_back(&block);
                        active_rows.push_back(entry.row_index);
                        const double* z_row = &block.design_matrix[entry.row_index * block.q];
                        const double* u_block = &u[block.offset];
                        for (std::size_t j = 0; j < block.q; ++j) {
                            eta += z_row[j] * u_block[j];
                        }
                    }
                }

                const double s = outcome.status ? (*outcome.status)[obs_idx] : 1.0;
                auto eval = outcome.family->evaluate(outcome.obs[obs_idx], eta, outcome.disp[obs_idx], s, outcome.extra);
                if (evals_out) evals_out->push_back(eval);

                for (std::size_t i = 0; i < active_blocks.size(); ++i) {
                    const auto* block = active_blocks[i];
                    std::size_t row = active_rows[i];
                    const double* z_row = &block->design_matrix[row * block->q];
                    double* grad_block = &grad[block->offset];
                    for (std::size_t j = 0; j < block->q; ++j) {
                        grad_block[j] += z_row[j] * eval.first_derivative;
                    }
                }

                double w = -eval.second_derivative;
                for (std::size_t a = 0; a < active_blocks.size(); ++a) {
                    const auto* block_a = active_blocks[a];
                    std::size_t row_a = active_rows[a];
                    const double* z_a = &block_a->design_matrix[row_a * block_a->q];
                    
                    for (std::size_t b = a; b < active_blocks.size(); ++b) {
                        const auto* block_b = active_blocks[b];
                        std::size_t row_b = active_rows[b];
                        const double* z_b = &block_b->design_matrix[row_b * block_b->q];
                        
                        for (std::size_t r = 0; r < block_a->q; ++r) {
                            for (std::size_t c = 0; c < block_b->q; ++c) {
                                double val = w * z_a[r] * z_b[c];
                                int row_idx = block_a->offset + r;
                                int col_idx = block_b->offset + c;
                                triplets.emplace_back(row_idx, col_idx, val);
                                if (row_idx != col_idx) {
                                    triplets.emplace_back(col_idx, row_idx, val);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    neg_hess.setFromTriplets(triplets.begin(), triplets.end());
}


bool solve_laplace_system_sparse(const LaplaceSystem& system,
                                 const std::vector<OutcomeData>& outcomes,
                                 LaplaceSystemResult& result,
                                 SparseHessianAccumulator* accumulator = nullptr) {
    const std::size_t Q = system.total_dim;
    std::vector<double> u(Q, 0.0);
    std::vector<double> grad(Q, 0.0);
    Eigen::SparseMatrix<double> neg_hess(Q, Q);
    std::vector<double> delta(Q, 0.0);
    std::vector<double> u_new(Q, 0.0);

    constexpr int kMaxIter = 100;
    constexpr double kDampingInit = 1e-3;
    constexpr double kDampingMax = 1e2;

    if (result.u.size() == Q) {
        u = result.u;
    }

    // Initialize accumulator structure if needed
    if (accumulator && !accumulator->initialized) {
        // std::cout << "Initializing accumulator..." << std::endl;
        std::vector<Eigen::Triplet<double>> triplets;
        
        // Prior structure (G^{-1})
        for (const auto& block : system.blocks) {
            if (block.info->is_sparse) {
                /*
                const auto& G = block.info->G_sparse;
                for (int k=0; k<G.outerSize(); ++k) {
                    for (Eigen::SparseMatrix<double>::InnerIterator it(G, k); it; ++it) {
                        if (it.row() == it.col()) {
                            triplets.emplace_back(block.offset + it.row(), block.offset + it.col(), 1.0);
                        }
                    }
                }
                */
            } else {
                const auto& G_inv = block.info->G_inverse;
                for (std::size_t r = 0; r < block.q; ++r) {
                    for (std::size_t c = 0; c < block.q; ++c) {
                        if (G_inv[r * block.q + c] != 0.0) {
                            triplets.emplace_back(block.offset + r, block.offset + c, 1.0);
                        }
                    }
                }
            }
        }
        
        // Observation structure
        accumulator->outcome_recipes.resize(outcomes.size());
        for (size_t out_idx = 0; out_idx < outcomes.size(); ++out_idx) {
            const auto& outcome = outcomes[out_idx];
            auto& recipe = accumulator->outcome_recipes[out_idx];
            std::size_t n = outcome.obs.size();
            recipe.obs_recipes.resize(n);
            
            for (std::size_t obs_idx = 0; obs_idx < n; ++obs_idx) {
                if (std::isnan(outcome.obs[obs_idx])) continue;
                
                const auto& entries = system.observation_entries[obs_idx];
                std::vector<const LaplaceBlock*> active_blocks;
                std::vector<std::size_t> active_rows;
                for (const auto& entry : entries) {
                    const auto& block = system.blocks[entry.block_index];
                    bool targets = false;
                    for (const auto& t : block.info->target_vars) {
                        if (t == outcome.name) {
                            targets = true;
                            break;
                        }
                    }
                    if (targets) {
                        active_blocks.push_back(&block);
                        active_rows.push_back(entry.row_index);
                    }
                }
                
                for (std::size_t a = 0; a < active_blocks.size(); ++a) {
                    const auto* block_a = active_blocks[a];
                    std::size_t row_a = active_rows[a];
                    const double* z_a = &block_a->design_matrix[row_a * block_a->q];
                    
                    for (std::size_t b = a; b < active_blocks.size(); ++b) {
                        const auto* block_b = active_blocks[b];
                        std::size_t row_b = active_rows[b];
                        const double* z_b = &block_b->design_matrix[row_b * block_b->q];
                        
                        for (std::size_t r = 0; r < block_a->q; ++r) {
                            for (std::size_t c = 0; c < block_b->q; ++c) {
                                double z_prod = z_a[r] * z_b[c];
                                int row_idx = block_a->offset + r;
                                int col_idx = block_b->offset + c;
                                
                                recipe.obs_recipes[obs_idx].push_back({row_idx, col_idx, z_prod});
                                triplets.emplace_back(row_idx, col_idx, 1.0);
                                if (row_idx != col_idx) {
                                    triplets.emplace_back(col_idx, row_idx, 1.0);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        accumulator->hessian.resize(Q, Q);
        accumulator->hessian.setFromTriplets(triplets.begin(), triplets.end());
        accumulator->initialized = true;
        
        // Analyze pattern once
        accumulator->solver.analyzePattern(accumulator->hessian);
    }

    double current_obj = compute_system_objective(system, u, outcomes);
    
    // Use local solver if accumulator not provided (or use accumulator's solver)
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>* solver_ptr = nullptr;
    
    if (accumulator) {
        // Create a shared_ptr that doesn't delete the object, as it's owned by accumulator
        result.sparse_solver = std::shared_ptr<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>(&accumulator->solver, [](Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>*){});
        solver_ptr = &accumulator->solver;
    } else {
        if (!result.sparse_solver) {
            result.sparse_solver = std::make_shared<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>();
        }
        solver_ptr = result.sparse_solver.get();
    }
    auto& solver = *solver_ptr;

    for (int iter = 0; iter < kMaxIter; ++iter) {
        compute_system_grad_hess_sparse(system, u, outcomes, grad, neg_hess, (iter == 0) ? nullptr : &result.evaluations, accumulator);
        
        double damping = 0.0;
        bool success = false;
        
        {
            if (accumulator) {
                solver.factorize(neg_hess);
            } else {
                solver.compute(neg_hess);
            }
        }
        if (solver.info() == Eigen::Success) {
            success = true;
        } else {
            damping = kDampingInit;
        }
        
        while (!success && damping <= kDampingMax) {
            Eigen::SparseMatrix<double> damped = neg_hess;
            for (int k=0; k<damped.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(damped, k); it; ++it) {
                    if (it.row() == it.col()) {
                        it.valueRef() += damping;
                    }
                }
            }
            
            {
                solver.factorize(damped);
            }
            if (solver.info() == Eigen::Success) {
                success = true;
            } else {
                damping *= 10.0;
            }
        }

        if (!success) return false;

        Eigen::Map<Eigen::VectorXd> grad_vec(grad.data(), Q);
        Eigen::VectorXd delta_vec = solver.solve(grad_vec);
        Eigen::Map<Eigen::VectorXd>(delta.data(), Q) = delta_vec;

        double step = 1.0;
        bool improved = false;
        while (step > 1e-4) {
            for (std::size_t i = 0; i < Q; ++i) {
                u_new[i] = u[i] + step * delta[i];
            }
            double new_obj = compute_system_objective(system, u_new, outcomes);
            if (new_obj > current_obj + 1e-8) {
                u = u_new;
                current_obj = new_obj;
                improved = true;
                break;
            }
            step *= 0.5;
        }

        if (!improved && step <= 1e-4) {
            break;
        }
    }
    
    result.u = u;
    result.use_sparse = true;
    
    // Ensure evaluations are populated for the final u
    compute_system_grad_hess_sparse(system, u, outcomes, grad, neg_hess, &result.evaluations, accumulator);
    
    // Compute log determinant
    double log_det = 0.0;
    if (solver.info() == Eigen::Success) {
        Eigen::VectorXd D = solver.vectorD();
        for (int i = 0; i < D.size(); ++i) {
            log_det += std::log(D[i]);
        }
    } else {
        log_det = std::numeric_limits<double>::infinity();
    }
    result.log_det_neg_hess = log_det;
    
    return true;
}


bool solve_laplace_block(const LaplaceBlock& block,
                         const std::vector<OutcomeData>& outcomes,
                         std::vector<double>& u_block,
                         std::vector<double>& hess_block) {
    const std::size_t q = block.q;
    std::vector<double> grad(q);
    std::vector<double> neg_hess(q * q);
    std::vector<double> chol(q * q);
    std::vector<double> delta(q);
    std::vector<double> u_new(q);

    constexpr int kMaxIter = 5; // Reduced for debugging
    constexpr double kTol = 1e-6;
    constexpr double kDampingInit = 1e-3;
    constexpr double kDampingMax = 1e2;

    // static int call_count = 0;
    // if (call_count % 100 == 0) fprintf(stderr, "Block %d size %zu\n", call_count, block.obs_indices.size());
    // call_count++;
    // fprintf(stderr, "Processing block size %zu\n", block.obs_indices.size());

    auto compute_obj = [&](const std::vector<double>& u_curr) {
        double obj = 0.0;
        // Prior
        if (!block.info->is_sparse) {
            const double* G_inv = block.info->G_inverse.data();
            for (size_t r = 0; r < q; ++r) {
                for (size_t c = 0; c < q; ++c) {
                    obj -= 0.5 * u_curr[r] * G_inv[r * q + c] * u_curr[c];
                }
            }
        }
        
        // Likelihood
        for (size_t i = 0; i < block.obs_indices.size(); ++i) {
            size_t global_idx = block.obs_indices[i];
            const double* z_row = &block.design_matrix[i * q];
            double z_u = 0.0;
            for(size_t k=0; k<q; ++k) z_u += z_row[k] * u_curr[k];
            
            for (const auto& outcome : outcomes) {
                if (global_idx >= outcome.obs.size()) continue;
                if (std::isnan(outcome.obs[global_idx])) continue;
                
                bool targets = false;
                for(const auto& t : block.info->target_vars) {
                    if (t == outcome.name) { targets = true; break; }
                }
                if (!targets) continue;

                double eta = outcome.pred[global_idx] + z_u;
                double s = outcome.status ? (*outcome.status)[global_idx] : 1.0;
                auto eval = outcome.family->evaluate(outcome.obs[global_idx], eta, outcome.disp[global_idx], s, outcome.extra);
                obj += eval.log_likelihood;
            }
        }
        return obj;
    };

    double current_obj = compute_obj(u_block);

    for (int iter = 0; iter < kMaxIter; ++iter) {
        std::fill(grad.begin(), grad.end(), 0.0);
        std::fill(neg_hess.begin(), neg_hess.end(), 0.0);

        // Prior Grad/Hess
        if (!block.info->is_sparse) {
            const double* G_inv = block.info->G_inverse.data();
            for (size_t r = 0; r < q; ++r) {
                for (size_t c = 0; c < q; ++c) {
                    neg_hess[r * q + c] += G_inv[r * q + c];
                    grad[r] -= G_inv[r * q + c] * u_block[c];
                }
            }
        }

        // Likelihood Grad/Hess
        for (size_t i = 0; i < block.obs_indices.size(); ++i) {
            size_t global_idx = block.obs_indices[i];
            const double* z_row = &block.design_matrix[i * q];
            double z_u = 0.0;
            for(size_t k=0; k<q; ++k) z_u += z_row[k] * u_block[k];
            
            for (const auto& outcome : outcomes) {
                if (global_idx >= outcome.obs.size()) continue;
                if (std::isnan(outcome.obs[global_idx])) continue;
                
                bool targets = false;
                for(const auto& t : block.info->target_vars) {
                    if (t == outcome.name) { targets = true; break; }
                }
                if (!targets) continue;

                double eta = outcome.pred[global_idx] + z_u;
                double s = outcome.status ? (*outcome.status)[global_idx] : 1.0;
                auto eval = outcome.family->evaluate(outcome.obs[global_idx], eta, outcome.disp[global_idx], s, outcome.extra);
                
                for(size_t r=0; r<q; ++r) {
                    grad[r] += z_row[r] * eval.first_derivative;
                    for(size_t c=0; c<q; ++c) {
                        neg_hess[r * q + c] -= z_row[r] * z_row[c] * eval.second_derivative;
                    }
                }
            }
        }

        // Solve Newton step
        double damping = 0.0;
        bool chol_success = false;
        try {
            cholesky(q, neg_hess, chol);
            chol_success = true;
        } catch (...) {
            damping = kDampingInit;
        }

        while (!chol_success && damping <= kDampingMax) {
             std::vector<double> damped_hess = neg_hess;
             for(size_t i=0; i<q; ++i) damped_hess[i*q+i] += damping;
             try {
                 cholesky(q, damped_hess, chol);
                 chol_success = true;
             } catch (...) {
                 damping *= 10.0;
             }
        }

        if (!chol_success) return false;

        delta = solve_cholesky(q, chol, grad);

        // Line search
        double step = 1.0;
        bool improved = false;
        while (step > 1e-4) {
            for(size_t i=0; i<q; ++i) u_new[i] = u_block[i] + step * delta[i];
            double new_obj = compute_obj(u_new);
            if (new_obj > current_obj) {
                current_obj = new_obj;
                u_block = u_new;
                improved = true;
                break;
            }
            step *= 0.5;
        }

        if (!improved) {
            double max_delta = 0.0;
            for(double d : delta) max_delta = std::max(max_delta, std::abs(d));
            if (max_delta < kTol) break;
            break; 
        }
    }
    
    // Recompute final Hessian for return
    std::fill(neg_hess.begin(), neg_hess.end(), 0.0);
    if (!block.info->is_sparse) {
        const double* G_inv = block.info->G_inverse.data();
        for (size_t r = 0; r < q; ++r) {
            for (size_t c = 0; c < q; ++c) {
                neg_hess[r * q + c] += G_inv[r * q + c];
            }
        }
    }
    for (size_t i = 0; i < block.obs_indices.size(); ++i) {
        size_t global_idx = block.obs_indices[i];
        const double* z_row = &block.design_matrix[i * q];
        double z_u = 0.0;
        for(size_t k=0; k<q; ++k) z_u += z_row[k] * u_block[k];
        
        for (const auto& outcome : outcomes) {
            if (global_idx >= outcome.obs.size()) continue;
            if (std::isnan(outcome.obs[global_idx])) continue;
            bool targets = false;
            for(const auto& t : block.info->target_vars) {
                if (t == outcome.name) { targets = true; break; }
            }
            if (!targets) continue;

            double eta = outcome.pred[global_idx] + z_u;
            double s = outcome.status ? (*outcome.status)[global_idx] : 1.0;
            auto eval = outcome.family->evaluate(outcome.obs[global_idx], eta, outcome.disp[global_idx], s, outcome.extra);
            for(size_t r=0; r<q; ++r) {
                for(size_t c=0; c<q; ++c) {
                    neg_hess[r * q + c] -= z_row[r] * z_row[c] * eval.second_derivative;
                }
            }
        }
    }
    hess_block = neg_hess;
    return true;
}

bool solve_laplace_system_decoupled(const LaplaceSystem& system,
                                    const std::vector<OutcomeData>& outcomes,
                                    LaplaceSystemResult& result) {
    const std::size_t Q = system.total_dim;
    result.u.assign(Q, 0.0);
    
    result.block_inverses.resize(system.blocks.size());

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(Q * 5); 
    
    double total_log_det = 0.0;
    
    for (size_t i = 0; i < system.blocks.size(); ++i) {
        const auto& block = system.blocks[i];
        std::vector<double> u_block(block.q, 0.0);
        std::vector<double> hess_block(block.q * block.q);
        
        if (!solve_laplace_block(block, outcomes, u_block, hess_block)) {
            return false;
        }

        std::vector<double> chol(block.q * block.q);
        try {
            cholesky(block.q, hess_block, chol);
            result.block_inverses[i] = invert_from_cholesky(block.q, chol);
        } catch (...) {
            return false;
        }
        
        for(size_t k=0; k<block.q; ++k) {
            result.u[block.offset + k] = u_block[k];
        }
        
        for(size_t r=0; r<block.q; ++r) {
            for(size_t c=0; c<block.q; ++c) {
                double val = hess_block[r * block.q + c];
                if (val != 0.0) {
                    triplets.emplace_back(block.offset + r, block.offset + c, val);
                }
            }
        }
    }
    
    result.use_sparse = true;
    result.sparse_solver = std::make_shared<LaplaceSystemResult::SparseSolver>();
    Eigen::SparseMatrix<double> mat(Q, Q);
    mat.setFromTriplets(triplets.begin(), triplets.end());
    
    result.sparse_solver->analyzePattern(mat);
    result.sparse_solver->factorize(mat);
    
    if (result.sparse_solver->info() != Eigen::Success) {
        return false;
    }
    
    Eigen::VectorXd D = result.sparse_solver->vectorD();
    for (int i=0; i<D.size(); ++i) {
        total_log_det += std::log(D[i]);
    }
    result.log_det_neg_hess = total_log_det;
    
    result.evaluations.clear();
    for (const auto& outcome : outcomes) {
        std::size_t n = outcome.obs.size();
        for (std::size_t obs_idx = 0; obs_idx < n; ++obs_idx) {
            if (std::isnan(outcome.obs[obs_idx])) {
                result.evaluations.push_back({0.0, 0.0, 0.0, 0.0, 0.0, {}});
                continue;
            }
            
            double eta = outcome.pred[obs_idx];
            const auto& entries = system.observation_entries[obs_idx];
            for (const auto& entry : entries) {
                const auto& block = system.blocks[entry.block_index];
                bool targets = false;
                for(const auto& t : block.info->target_vars) {
                    if (t == outcome.name) { targets = true; break; }
                }
                if (targets) {
                    const double* z_row = &block.design_matrix[entry.row_index * block.q];
                    const double* u_ptr = &result.u[block.offset];
                    for(size_t k=0; k<block.q; ++k) eta += z_row[k] * u_ptr[k];
                }
            }
            
            double s = outcome.status ? (*outcome.status)[obs_idx] : 1.0;
            result.evaluations.push_back(outcome.family->evaluate(outcome.obs[obs_idx], eta, outcome.disp[obs_idx], s, outcome.extra));
        }
    }
    
    return true;
}

bool solve_laplace_system(const LaplaceSystem& system,
                          const std::vector<OutcomeData>& outcomes,
                          LaplaceSystemResult& result,
                          SparseHessianAccumulator* accumulator = nullptr) {
    const std::size_t Q = system.total_dim;

    bool decoupled = true;
    size_t max_entry_size = 0;
    for (const auto& entry : system.observation_entries) {
        max_entry_size = std::max(max_entry_size, entry.size());
        if (entry.size() > 1) {
            decoupled = false;
            break;
        }
    }

    if (decoupled && !accumulator) {
        return solve_laplace_system_decoupled(system, outcomes, result);
    }
    
    bool any_sparse = false;

    for (const auto& block : system.blocks) {
        if (block.info->is_sparse) {
            any_sparse = true;
            break;
        }
    }

    if (accumulator || any_sparse) {
        return solve_laplace_system_sparse(system, outcomes, result, accumulator);
    }

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

    double current_obj = compute_system_objective(system, u, outcomes);

    for (int iter = 0; iter < kMaxIter; ++iter) {
        compute_system_grad_hess(system, u, outcomes, grad, hess, nullptr);
        
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
            double new_obj = compute_system_objective(system, u_new, outcomes);
            
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
    compute_system_grad_hess(system, u, outcomes, grad, hess, &result.evaluations);
    for (std::size_t idx = 0; idx < Q * Q; ++idx) {
        result.neg_hessian[idx] = -hess[idx];
    }
    try {
        cholesky(Q, result.neg_hessian, result.chol_neg_hessian);
    } catch (...) {
        return false;
    }
    
    result.use_sparse = false;
    result.log_det_neg_hess = log_det_cholesky(Q, result.chol_neg_hessian);
    
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

        double quad;
        if (info.is_sparse) {
            if (!info.sparse_solver) {
                throw std::runtime_error("sparse_solver is null in compute_prior_loglik");
            }
            Eigen::Map<const Eigen::VectorXd> u_vec(u_block.data(), block.q);
            Eigen::VectorXd x = info.sparse_solver->solve(u_vec);
            quad = u_vec.dot(x);
        } else {
            quad = quadratic_form(u_block, info.G_inverse, block.q);
        }
        total += -0.5 * (block.q * log_2pi + info.log_det_G + quad);
    }
    return total;
}

std::vector<double> compute_observation_quad_terms(const LaplaceSystem& system,
                                                   const LaplaceSystemResult& result,
                                                   std::string_view target_var) {
    const std::size_t Q = system.total_dim;
    const std::size_t n = system.observation_entries.size();
    std::vector<double> quads(n, 0.0);

    if (!result.block_inverses.empty()) {
        for (std::size_t obs_idx = 0; obs_idx < n; ++obs_idx) {
            double quad = 0.0;
            for (const auto& entry : system.observation_entries[obs_idx]) {
                if (entry.block_index >= result.block_inverses.size()) continue;
                const auto& block = system.blocks[entry.block_index];
                
                bool targets = false;
                for (const auto& t : block.info->target_vars) {
                    if (t == target_var) { targets = true; break; }
                }
                if (!targets) continue;

                const auto& inv_block = result.block_inverses[entry.block_index];
                const double* z_row = &block.design_matrix[entry.row_index * block.q];
                
                for (size_t r = 0; r < block.q; ++r) {
                    for (size_t c = 0; c < block.q; ++c) {
                        quad += z_row[r] * inv_block[r * block.q + c] * z_row[c];
                    }
                }
            }
            quads[obs_idx] = quad;
        }
        return quads;
    }

    std::vector<double> buffer(Q, 0.0);
    
    std::vector<double> neg_hess_inv;
    if (!result.use_sparse) {
        neg_hess_inv = invert_from_cholesky(Q, result.chol_neg_hessian);
    }

    for (std::size_t obs_idx = 0; obs_idx < n; ++obs_idx) {
        std::fill(buffer.begin(), buffer.end(), 0.0);
        bool has_entries = false;
        for (const auto& entry : system.observation_entries[obs_idx]) {
            if (entry.block_index >= system.blocks.size()) {
                continue;
            }
            const auto& block = system.blocks[entry.block_index];
            
            if (!block.info) {
                continue;
            }

            bool targets = false;
            for (const auto& t : block.info->target_vars) {
                if (t == target_var) {
                    targets = true;
                    break;
                }
            }
            if (!targets) continue;

            if (entry.row_index * block.q >= block.design_matrix.size()) {
                 continue;
            }

            const double* z_row = &block.design_matrix[entry.row_index * block.q];
            for (std::size_t j = 0; j < block.q; ++j) {
                if (block.offset + j >= buffer.size()) {
                    continue;
                }
                buffer[block.offset + j] = z_row[j];
            }
            has_entries = true;
        }
        
        if (!has_entries) continue;

        if (result.use_sparse) {
            Eigen::Map<Eigen::VectorXd> z_vec(buffer.data(), Q);
            Eigen::VectorXd v = result.sparse_solver->solve(z_vec);
            quads[obs_idx] = z_vec.dot(v);
        } else {
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
    }
    return quads;
}

std::vector<double> project_observations(const LaplaceSystem& system,
                                         const std::vector<double>& direction,
                                         std::string_view target_var) {
    const std::size_t n = system.observation_entries.size();
    std::vector<double> projected(n, 0.0);
    for (std::size_t obs_idx = 0; obs_idx < n; ++obs_idx) {
        double sum = 0.0;
        for (const auto& entry : system.observation_entries[obs_idx]) {
            const auto& block = system.blocks[entry.block_index];

            bool targets = false;
            for (const auto& t : block.info->target_vars) {
                if (t == target_var) {
                    targets = true;
                    break;
                }
            }
            if (!targets) continue;

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
                                                EstimationMethod method,
                                                bool force_laplace) const {
    
    if (model.random_effects.empty()) {
        double total = 0.0;
        // std::cout << "Evaluating LL..." << std::endl;
        for (const auto& var : model.variables) {
            if (var.kind != VariableKind::Observed) {
                continue;  // Skip latent and grouping variables for now
            }
            if (var.family.empty() || var.family == "fixed") {
                continue; // Skip variables without a family (treated as exogenous)
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
    std::vector<RandomEffectInfo> random_effect_infos;
    build_random_effect_infos(model, covariance_parameters, fixed_covariance_data, false, random_effect_infos);
    if (random_effect_infos.empty()) {
        throw std::runtime_error("Random effect metadata missing");
    }

    std::vector<std::string> outcome_vars;
    std::unordered_set<std::string> outcome_set;
    
    for (const auto& edge : model.edges) {
        if (edge.kind == EdgeKind::Regression) {
            auto it = std::find_if(model.variables.begin(), model.variables.end(), [&](const auto& v) {
                return v.name == edge.target && v.kind == VariableKind::Observed;
            });
            if (it != model.variables.end()) {
                if (outcome_set.find(edge.target) == outcome_set.end()) {
                    outcome_vars.push_back(edge.target);
                    outcome_set.insert(edge.target);
                }
            }
        }
    }
    
    for (const auto& var : model.variables) {
        if (var.kind == VariableKind::Observed && !var.family.empty() && var.family != "fixed") {
            if (outcome_set.find(var.name) == outcome_set.end()) {
                outcome_vars.push_back(var.name);
                outcome_set.insert(var.name);
            }
        }
    }

    if (outcome_vars.empty()) {
        throw std::runtime_error("No outcome variables found");
    }

    std::vector<OutcomeData> outcomes;
    std::list<std::vector<double>> disp_storage;
    std::list<std::unique_ptr<OutcomeFamily>> family_storage;
    std::size_t n = 0;
    bool n_set = false;

    for (const auto& name : outcome_vars) {
        auto var_it = std::find_if(model.variables.begin(), model.variables.end(), [&](const auto& v) {
            return v.name == name;
        });
        if (var_it == model.variables.end()) continue;

        auto data_it = data.find(name);
        if (data_it == data.end()) throw std::runtime_error("Missing data for outcome: " + name);
        const auto& obs_data = data_it->second;
        
        if (!n_set) {
            n = obs_data.size();
            n_set = true;
        } else if (obs_data.size() != n) {
            throw std::runtime_error("Mismatched data size for outcome: " + name);
        }

        auto pred_it = linear_predictors.find(name);
        if (pred_it == linear_predictors.end()) throw std::runtime_error("Missing linear predictors for: " + name);
        const auto& pred_data = pred_it->second;

        auto disp_it = dispersions.find(name);
        if (disp_it == dispersions.end()) throw std::runtime_error("Missing dispersions for: " + name);
        
        const std::vector<double>* disp_ptr = &disp_it->second;
        if (disp_ptr->size() == 1) {
            disp_storage.emplace_back(n, disp_ptr->front());
            disp_ptr = &disp_storage.back();
        } else if (disp_ptr->size() != n) {
             throw std::runtime_error("Dispersion size mismatch");
        }

        family_storage.push_back(OutcomeFamilyFactory::create(var_it->family));
        
        const std::vector<double>* status_ptr = nullptr;
        auto status_it = status.find(name);
        if (status_it != status.end()) status_ptr = &status_it->second;

        const std::vector<double>* extra_ptr = nullptr;
        auto extra_it = extra_params.find(name);
        if (extra_it != extra_params.end()) extra_ptr = &extra_it->second;
        static const std::vector<double> empty_extra;

        outcomes.push_back({name, obs_data, pred_data, *disp_ptr, status_ptr, extra_ptr ? *extra_ptr : empty_extra, family_storage.back().get()});
    }

    const double log_2pi = std::log(2.0 * 3.14159265358979323846);

    if (outcomes.size() == 1 && !force_laplace) {
        auto var_it = std::find_if(model.variables.begin(), model.variables.end(), [&](const auto& v) {
            return v.name == outcomes[0].name;
        });
        if (var_it != model.variables.end() && var_it->family == "gaussian") {
            const auto& outcome = outcomes[0];
            std::vector<std::map<double, std::vector<std::size_t>>> group_maps;
            group_maps.reserve(random_effect_infos.size());
            for (const auto& info : random_effect_infos) {
                group_maps.push_back(build_group_map(info, data, n));
            }

            // Check for block diagonal structure (single grouping variable)
            bool block_diagonal = !random_effect_infos.empty();
            std::string common_grouping;
            if (block_diagonal) {
                common_grouping = random_effect_infos[0].grouping_var;
                for (size_t i = 1; i < random_effect_infos.size(); ++i) {
                    if (random_effect_infos[i].grouping_var != common_grouping) {
                        block_diagonal = false;
                        break;
                    }
                }
            }

            if (block_diagonal) {
                double total_ll = 0.0;
                const auto& groups = group_maps[0];
                double log_2pi = std::log(2.0 * 3.14159265358979323846);

                for (const auto& [group_id, indices] : groups) {
                    size_t n_i = indices.size();
                    std::vector<double> V_i(n_i * n_i, 0.0);
                    
                    // Diagonal (Residuals)
                    for(size_t r=0; r<n_i; ++r) {
                        V_i[r * n_i + r] = outcome.disp[indices[r]];
                    }

                    // Random Effects
                    for (size_t re_idx = 0; re_idx < random_effect_infos.size(); ++re_idx) {
                        const auto& info = random_effect_infos[re_idx];
                        auto Z_i = build_design_matrix(info, indices, data, linear_predictors);
                        size_t q = info.cov_spec->dimension;

                        std::vector<double> ZG(n_i * q, 0.0);
                        if (info.is_sparse) {
                            if (!info.G_sparse) throw std::runtime_error("G_sparse is null");
                            Eigen::Map<const RowMajorMatrix> Z_map(Z_i.data(), n_i, q);
                            Eigen::MatrixXd Z_dense = Z_map;
                            Eigen::MatrixXd ZG_eigen = Z_dense * *info.G_sparse;
                            Eigen::Map<RowMajorMatrix>(ZG.data(), n_i, q) = ZG_eigen;
                        } else {
                            for (std::size_t r = 0; r < n_i; ++r) {
                                for (std::size_t c = 0; c < q; ++c) {
                                    double sum = 0.0;
                                    for (std::size_t k = 0; k < q; ++k) {
                                        sum += Z_i[r * q + k] * info.G_matrix[k * q + c];
                                    }
                                    ZG[r * q + c] = sum;
                                }
                            }
                        }

                        for (std::size_t r = 0; r < n_i; ++r) {
                            for (std::size_t c = 0; c < n_i; ++c) {
                                double sum = 0.0;
                                for (std::size_t k = 0; k < q; ++k) {
                                    sum += ZG[r * q + k] * Z_i[c * q + k];
                                }
                                V_i[r * n_i + c] += sum;
                            }
                        }
                    }

                    // Compute LL_i
                    std::vector<double> resid_i(n_i);
                    bool has_nan = false;
                    for(size_t r=0; r<n_i; ++r) {
                        if (std::isnan(outcome.obs[indices[r]])) {
                            has_nan = true; 
                            break; 
                        }
                        resid_i[r] = outcome.obs[indices[r]] - outcome.pred[indices[r]];
                    }
                    if (has_nan) continue; 

                    std::vector<double> L_i(n_i * n_i);
                    try {
                        cholesky(n_i, V_i, L_i);
                    } catch(...) {
                        return -std::numeric_limits<double>::infinity();
                    }

                    double log_det = log_det_cholesky(n_i, L_i);
                    auto alpha = solve_cholesky(n_i, L_i, resid_i);
                    double quad = 0.0;
                    for(size_t r=0; r<n_i; ++r) quad += resid_i[r] * alpha[r];

                    total_ll += -0.5 * (n_i * log_2pi + log_det + quad);
                }
                
                return total_ll;
            }

            std::vector<double> V_full(n * n, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                V_full[i * n + i] = outcome.disp[i];
            }

            for (std::size_t re_idx = 0; re_idx < random_effect_infos.size(); ++re_idx) {
                const auto& info = random_effect_infos[re_idx];
                const auto& groups = group_maps[re_idx];
                for (const auto& [_, indices] : groups) {
                    auto Z_i = build_design_matrix(info, indices, data, linear_predictors);
                    std::size_t n_i = indices.size();
                    std::size_t q = info.cov_spec->dimension;

                    std::vector<double> ZG(n_i * q, 0.0);
                    if (info.is_sparse) {
                        if (!info.G_sparse) throw std::runtime_error("G_sparse is null");
                        Eigen::Map<const RowMajorMatrix> Z_map(Z_i.data(), n_i, q);
                        Eigen::MatrixXd Z_dense = Z_map;
                        Eigen::MatrixXd ZG_eigen = Z_dense * *info.G_sparse;
                        Eigen::Map<RowMajorMatrix>(ZG.data(), n_i, q) = ZG_eigen;
                    } else {
                        for (std::size_t r = 0; r < n_i; ++r) {
                            for (std::size_t c = 0; c < q; ++c) {
                                double sum = 0.0;
                                for (std::size_t k = 0; k < q; ++k) {
                                    sum += Z_i[r * q + k] * info.G_matrix[k * q + c];
                                }
                                ZG[r * q + c] = sum;
                            }
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

            std::vector<std::size_t> observed_indices;
            for (std::size_t i = 0; i < n; ++i) {
                if (!std::isnan(outcome.obs[i])) {
                    observed_indices.push_back(i);
                }
            }
            std::size_t n_obs = observed_indices.size();

            std::vector<double> V_sub(n_obs * n_obs);
            std::vector<double> resid_sub(n_obs);
            for (std::size_t r = 0; r < n_obs; ++r) {
                resid_sub[r] = outcome.obs[observed_indices[r]] - outcome.pred[observed_indices[r]];
                for (std::size_t c = 0; c < n_obs; ++c) {
                    V_sub[r * n_obs + c] = V_full[observed_indices[r] * n + observed_indices[c]];
                }
            }

            std::vector<double> L(n_obs * n_obs);
            try {
                cholesky(n_obs, V_sub, L);
            } catch (...) {
                return -std::numeric_limits<double>::infinity();
            }
            
            double log_det_V = log_det_cholesky(n_obs, L);
            std::vector<double> alpha = solve_cholesky(n_obs, L, resid_sub);
            double quad_form = 0.0;
            for(size_t i=0; i<n_obs; ++i) quad_form += resid_sub[i] * alpha[i];

            double total = -0.5 * (n_obs * log_2pi + log_det_V + quad_form);

            if (method == EstimationMethod::REML) {
                 std::vector<std::string> predictors;
                 for (const auto& edge : model.edges) {
                     if (edge.kind == EdgeKind::Regression && edge.target == outcome.name) {
                         predictors.push_back(edge.source);
                     }
                 }
                 std::sort(predictors.begin(), predictors.end());

                 if (!predictors.empty()) {
                     std::size_t p = predictors.size();
                     std::vector<double> X(n_obs * p);
                     
                     for (std::size_t j = 0; j < p; ++j) {
                         auto it = data.find(predictors[j]);
                         if (it == data.end()) continue;
                         for (std::size_t i = 0; i < n_obs; ++i) {
                             X[i * p + j] = it->second[observed_indices[i]];
                         }
                     }

                     std::vector<double> Xt_Vinv_X(p * p, 0.0);
                     for (std::size_t j = 0; j < p; ++j) {
                         std::vector<double> col_j(n_obs);
                         for(size_t i=0; i<n_obs; ++i) col_j[i] = X[i*p + j];
                         std::vector<double> Vinv_X_j = solve_cholesky(n_obs, L, col_j);
                         
                         for (std::size_t k = 0; k <= j; ++k) {
                             double sum = 0.0;
                             for(size_t i=0; i<n_obs; ++i) sum += X[i*p + k] * Vinv_X_j[i];
                             Xt_Vinv_X[j*p + k] = sum;
                             Xt_Vinv_X[k*p + j] = sum;
                         }
                     }
                     
                     std::vector<double> L_reml(p * p);
                     try {
                         cholesky(p, Xt_Vinv_X, L_reml);
                         double log_det_reml = log_det_cholesky(p, L_reml);
                         total -= 0.5 * log_det_reml;
                         total += 0.5 * static_cast<double>(p) * log_2pi;
                     } catch (...) {
                         return -std::numeric_limits<double>::infinity();
                     }
                 }
            }
            return total;
        }
    }

    LaplaceSystem& system = get_or_build_laplace_system(random_effect_infos, data, linear_predictors, n, laplace_cache_);
    if (system.total_dim == 0) {
        throw std::runtime_error("Laplace system constructed without latent dimensions");
    }

    LaplaceSystemResult laplace_result;
    if (!solve_laplace_system(system, outcomes, laplace_result)) {
        std::cerr << "solve_laplace_system failed!" << std::endl;
        return -std::numeric_limits<double>::infinity();
    }

    double log_prior = compute_prior_loglik(system, laplace_result, log_2pi);
    double log_lik_data = 0.0;
    for (const auto& eval : laplace_result.evaluations) {
        log_lik_data += eval.log_likelihood;
    }
    double log_det_neg_hess = laplace_result.log_det_neg_hess;
    double total_loglik = log_prior + log_lik_data + 0.5 * system.total_dim * log_2pi - 0.5 * log_det_neg_hess;

    return total_loglik;
}

std::vector<double> get_inverse_block(const LaplaceSystemResult& result, const LaplaceBlock& block, std::size_t Q) {
    if (!result.block_inverses.empty() && block.index < result.block_inverses.size()) {
        return result.block_inverses[block.index];
    }
    std::size_t sz = block.q * block.q;
    std::vector<double> inv_block(sz);
    
    bool sparse = result.use_sparse;

    if (sparse) {
        for (std::size_t k = 0; k < block.q; ++k) {
            Eigen::VectorXd rhs(Q);
            rhs.setZero();
            rhs[block.offset + k] = 1.0;
            Eigen::VectorXd sol = result.sparse_solver->solve(rhs);
            for (std::size_t r = 0; r < block.q; ++r) {
                inv_block[r * block.q + k] = sol[block.offset + r];
            }
        }
    } else {
        for (std::size_t k = 0; k < block.q; ++k) {
            std::vector<double> rhs(Q, 0.0);
            rhs[block.offset + k] = 1.0;
            std::vector<double> sol = solve_cholesky(Q, result.chol_neg_hessian, rhs);
            for (std::size_t r = 0; r < block.q; ++r) {
                inv_block[r * block.q + k] = sol[block.offset + r];
            }
        }
    }
    return inv_block;
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
                                                const std::unordered_map<std::string, DataParamMapping>& dispersion_param_mappings,
                                                const std::unordered_map<std::string, std::vector<std::string>>& extra_param_mappings,
                                                bool force_laplace) const {
    
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
            if (var.family.empty() || var.family == "fixed") continue;

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

            const std::vector<std::string>* extra_mapping_vec = nullptr;
            if (extra_param_mappings.count(var.name)) {
                extra_mapping_vec = &extra_param_mappings.at(var.name);
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

                if (extra_mapping_vec && !eval.d_extra_params.empty()) {
                    for (size_t k = 0; k < eval.d_extra_params.size(); ++k) {
                        if (k < extra_mapping_vec->size()) {
                            const std::string& param_id = (*extra_mapping_vec)[k];
                            if (!param_id.empty()) {
                                gradients[param_id] += eval.d_extra_params[k];
                            }
                        }
                    }
                }
            }
        }
        return gradients;
    } else {
        std::vector<RandomEffectInfo> random_effect_infos;
        build_random_effect_infos(model, covariance_parameters, fixed_covariance_data, true, random_effect_infos);
        if (random_effect_infos.empty()) {
            throw std::runtime_error("Random effect metadata missing");
        }

        std::vector<std::string> outcome_vars;
        std::unordered_set<std::string> outcome_set;
        
        for (const auto& edge : model.edges) {
            if (edge.kind == EdgeKind::Regression) {
                auto it = std::find_if(model.variables.begin(), model.variables.end(), [&](const auto& v) {
                    return v.name == edge.target && v.kind == VariableKind::Observed;
                });
                if (it != model.variables.end()) {
                    if (outcome_set.find(edge.target) == outcome_set.end()) {
                        outcome_vars.push_back(edge.target);
                        outcome_set.insert(edge.target);
                    }
                }
            }
        }
        
        for (const auto& var : model.variables) {
            if (var.kind == VariableKind::Observed && !var.family.empty() && var.family != "fixed") {
                if (outcome_set.find(var.name) == outcome_set.end()) {
                    outcome_vars.push_back(var.name);
                    outcome_set.insert(var.name);
                }
            }
        }

        if (outcome_vars.empty()) {
            throw std::runtime_error("No outcome variables found");
        }

        std::vector<OutcomeData> outcomes;
        std::list<std::vector<double>> disp_storage;
        std::list<std::unique_ptr<OutcomeFamily>> family_storage;
        std::size_t n = 0;
        bool n_set = false;

        for (const auto& name : outcome_vars) {
            auto var_it = std::find_if(model.variables.begin(), model.variables.end(), [&](const auto& v) {
                return v.name == name;
            });
            if (var_it == model.variables.end()) continue;

            auto data_it = data.find(name);
            if (data_it == data.end()) throw std::runtime_error("Missing data for outcome: " + name);
            const auto& obs_data = data_it->second;
            
            if (!n_set) {
                n = obs_data.size();
                n_set = true;
            } else if (obs_data.size() != n) {
                throw std::runtime_error("Mismatched data size for outcome: " + name);
            }

            auto pred_it = linear_predictors.find(name);
            if (pred_it == linear_predictors.end()) throw std::runtime_error("Missing linear predictors for: " + name);
            const auto& pred_data = pred_it->second;

            auto disp_it = dispersions.find(name);
            if (disp_it == dispersions.end()) throw std::runtime_error("Missing dispersions for: " + name);
            
            const std::vector<double>* disp_ptr = &disp_it->second;
            if (disp_ptr->size() == 1) {
                disp_storage.emplace_back(n, disp_ptr->front());
                disp_ptr = &disp_storage.back();
            } else if (disp_ptr->size() != n) {
                 throw std::runtime_error("Dispersion size mismatch");
            }

            family_storage.push_back(OutcomeFamilyFactory::create(var_it->family));
            
            const std::vector<double>* status_ptr = nullptr;
            auto status_it = status.find(name);
            if (status_it != status.end()) status_ptr = &status_it->second;

            const std::vector<double>* extra_ptr = nullptr;
            auto extra_it = extra_params.find(name);
            if (extra_it != extra_params.end()) extra_ptr = &extra_it->second;
            static const std::vector<double> empty_extra;

            outcomes.push_back({name, obs_data, pred_data, *disp_ptr, status_ptr, extra_ptr ? *extra_ptr : empty_extra, family_storage.back().get()});
        }

        bool use_analytic = false;
        if (outcomes.size() == 1 && !force_laplace) {
             auto it = std::find_if(model.variables.begin(), model.variables.end(), [&](const auto& v) {
                 return v.name == outcomes[0].name;
             });
             if (it != model.variables.end() && it->family == "gaussian") {
                 use_analytic = true;
             }
        }

        if (use_analytic) {
            const auto& outcome = outcomes[0];
            const auto& obs_data = outcome.obs;
            const auto& pred_data = outcome.pred;
            const auto& disp_data = outcome.disp;
            std::string obs_var_name = outcome.name;

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
                    if (info.is_sparse) {
                        if (!info.G_sparse) throw std::runtime_error("G_sparse is null");
                        Eigen::Map<const RowMajorMatrix> Z_map(Z_i.data(), n_i, q);
                        Eigen::MatrixXd Z_dense = Z_map;
                        Eigen::MatrixXd ZG_eigen = Z_dense * *info.G_sparse;
                        Eigen::Map<RowMajorMatrix>(ZG.data(), n_i, q) = ZG_eigen;
                    } else {
                        for (std::size_t r = 0; r < n_i; ++r) {
                            for (std::size_t c = 0; c < q; ++c) {
                                double sum = 0.0;
                                for (std::size_t k = 0; k < q; ++k) {
                                    sum += Z_i[r * q + k] * info.G_matrix[k * q + c];
                                }
                                ZG[r * q + c] = sum;
                            }
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

                if (info.G_gradients.empty() && info.G_gradients_sparse.empty() && !has_active_data) continue;
                
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

                    if (!info.G_gradients.empty() || !info.G_gradients_sparse.empty()) {
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

                        if (info.is_sparse) {
                             for (std::size_t k = 0; k < info.G_gradients_sparse.size(); ++k) {
                                const auto& dG = info.G_gradients_sparse[k];
                                double trace = 0.0;
                                for (int j=0; j<dG.outerSize(); ++j) {
                                    for (Eigen::SparseMatrix<double>::InnerIterator it(dG, j); it; ++it) {
                                        trace += M[it.col() * q + it.row()] * it.value();
                                    }
                                }
                                std::string param_name = info.covariance_id + "_" + std::to_string(k);
                                gradients[param_name] += 0.5 * trace;
                             }
                        } else {
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
                                            if (info.is_sparse) {
                                                for (Eigen::SparseMatrix<double>::InnerIterator it(*info.G_sparse, c); it; ++it) {
                                                    val += temp[r * q + it.row()] * it.value();
                                                }
                                            } else {
                                                for (std::size_t k = 0; k < q; ++k) {
                                                    val += temp[r * q + k] * info.G_matrix[k * q + c];
                                                }
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

        LaplaceSystem& system = get_or_build_laplace_system(random_effect_infos, data, linear_predictors, n, laplace_cache_);
        if (system.total_dim == 0) {
            throw std::runtime_error("Laplace system constructed without latent dimensions");
        }

        LaplaceSystemResult laplace_result;
        if (!solve_laplace_system(system,
                                  outcomes,
                                  laplace_result)) {
            throw std::runtime_error("Laplace mode failed to converge for gradient computation");
        }

        std::size_t eval_offset = 0;
        for (const auto& outcome : outcomes) {
            std::string obs_var_name = outcome.name;
            std::size_t n_obs = outcome.obs.size();

            if (!dispersion_param_mappings.empty()) {
                auto map_it = dispersion_param_mappings.find(obs_var_name);
                if (map_it != dispersion_param_mappings.end()) {
                    const auto& mapping = map_it->second;
                    for (std::size_t i = 0; i < n_obs; ++i) {
                        size_t idx = i % mapping.stride;
                        if (idx < mapping.pattern.size()) {
                            const std::string& pid = mapping.pattern[idx];
                            if (!pid.empty()) {
                                gradients[pid] += laplace_result.evaluations[eval_offset + i].d_dispersion;
                            }
                        }
                    }
                }
            }

            auto quad_terms = compute_observation_quad_terms(system, laplace_result, obs_var_name);
            std::vector<double> forcing(system.total_dim, 0.0);

            for (const auto& edge : model.edges) {
                if (edge.kind != EdgeKind::Regression || edge.target != obs_var_name || edge.parameter_id.empty()) {
                    continue;
                }
                auto src_it = data.find(edge.source);
                if (src_it == data.end() || src_it->second.size() != n_obs) {
                    continue;
                }

                const auto& src_vec = src_it->second;
                std::vector<double> weights(n_obs, 0.0);
                for (std::size_t i = 0; i < n_obs; ++i) {
                    weights[i] = laplace_result.evaluations[eval_offset + i].second_derivative * src_vec[i];
                }
                accumulate_weighted_forcing(system, weights, forcing);
                std::vector<double> du(system.total_dim);
                if (laplace_result.use_sparse) {
                    Eigen::Map<Eigen::VectorXd> f_vec(forcing.data(), system.total_dim);
                    Eigen::VectorXd du_vec = laplace_result.sparse_solver->solve(f_vec);
                    Eigen::Map<Eigen::VectorXd>(du.data(), system.total_dim) = du_vec;
                } else {
                    du = solve_cholesky(system.total_dim, laplace_result.chol_neg_hessian, forcing);
                }
                auto z_dot_du = project_observations(system, du, obs_var_name);

                double accum = 0.0;
                for (std::size_t i = 0; i < n_obs; ++i) {
                    const auto& eval = laplace_result.evaluations[eval_offset + i];
                    double logdet_adjust = src_vec[i] + z_dot_du[i];
                    accum += src_vec[i] * eval.first_derivative + 0.5 * eval.third_derivative * logdet_adjust * quad_terms[i];
                }
                gradients[edge.parameter_id] += accum;
            }
            eval_offset += n_obs;
        }

        struct CovarianceDerivativeCache {
            std::string param_name;
            std::vector<double> Ginv_dG_Ginv;
            double trace_Ginv_dG;
        };

        std::unordered_map<std::string, std::vector<CovarianceDerivativeCache>> covariance_caches;
        for (const auto& info : random_effect_infos) {
            if (info.G_gradients.empty() && info.G_gradients_sparse.empty()) {
                continue;
            }
            std::vector<CovarianceDerivativeCache> caches;
            std::size_t q = info.cov_spec->dimension;

            if (info.is_sparse) {
                 if (!info.sparse_solver) throw std::runtime_error("Sparse solver missing for " + info.id);
                 
                 // Densify G_inv for gradient computation
                 Eigen::MatrixXd I = Eigen::MatrixXd::Identity(q, q);
                 Eigen::MatrixXd G_inv_eigen = info.sparse_solver->solve(I);
                 
                 for (std::size_t idx = 0; idx < info.G_gradients_sparse.size(); ++idx) {
                     CovarianceDerivativeCache cache;
                     cache.param_name = info.covariance_id + "_" + std::to_string(idx);
                     
                     const auto& dG_sparse = info.G_gradients_sparse[idx];
                     
                     // Ginv * dG * Ginv
                     Eigen::MatrixXd dG_Ginv = dG_sparse * G_inv_eigen;
                     Eigen::MatrixXd term = G_inv_eigen * dG_Ginv;
                     
                     cache.Ginv_dG_Ginv.resize(q * q);
                     Eigen::Map<RowMajorMatrix>(cache.Ginv_dG_Ginv.data(), q, q) = term;
                     
                     // trace(Ginv * dG)
                     double trace = 0.0;
                     for (int k=0; k<dG_sparse.outerSize(); ++k) {
                        for (Eigen::SparseMatrix<double>::InnerIterator it(dG_sparse, k); it; ++it) {
                            trace += G_inv_eigen(it.col(), it.row()) * it.value();
                        }
                     }
                     cache.trace_Ginv_dG = trace;
                     caches.push_back(std::move(cache));
                 }
            } else {
                for (std::size_t idx = 0; idx < info.G_gradients.size(); ++idx) {
                    CovarianceDerivativeCache cache;
                    cache.param_name = info.covariance_id + "_" + std::to_string(idx);
                    auto temp = multiply_matrices(info.G_inverse, info.G_gradients[idx], q);
                    cache.Ginv_dG_Ginv = multiply_matrices(temp, info.G_inverse, q);
                    cache.trace_Ginv_dG = trace_product(info.G_inverse, info.G_gradients[idx], q);
                    caches.push_back(std::move(cache));
                }
            }
            covariance_caches.emplace(info.covariance_id, std::move(caches));
        }

        const std::size_t Q = system.total_dim;
        std::vector<double> forcing(system.total_dim, 0.0);
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
                std::vector<double> du(system.total_dim);
                if (laplace_result.use_sparse) {
                    Eigen::Map<Eigen::VectorXd> f_vec(forcing.data(), system.total_dim);
                    Eigen::VectorXd du_vec = laplace_result.sparse_solver->solve(f_vec);
                    Eigen::Map<Eigen::VectorXd>(du.data(), system.total_dim) = du_vec;
                } else {
                    du = solve_cholesky(system.total_dim, laplace_result.chol_neg_hessian, forcing);
                }
                
                double logdet_adjust = 0.0;
                std::size_t eval_offset = 0;
                for (const auto& outcome : outcomes) {
                    auto z_dot_du = project_observations(system, du, outcome.name);
                    auto quad_terms = compute_observation_quad_terms(system, laplace_result, outcome.name);
                    std::size_t n_obs = outcome.obs.size();
                    for (std::size_t obs_idx = 0; obs_idx < n_obs; ++obs_idx) {
                        logdet_adjust += 0.5 * laplace_result.evaluations[eval_offset + obs_idx].third_derivative * z_dot_du[obs_idx] * quad_terms[obs_idx];
                    }
                    eval_offset += n_obs;
                }

                double prior_term = 0.5 * quadratic_form(u_block, cache.Ginv_dG_Ginv, block.q) - 0.5 * cache.trace_Ginv_dG;
                double trace_term = 0.0;
                std::vector<double> inv_block = get_inverse_block(laplace_result, block, Q);
                for (std::size_t r = 0; r < block.q; ++r) {
                    for (std::size_t c = 0; c < block.q; ++c) {
                        trace_term += inv_block[r * block.q + c] * cache.Ginv_dG_Ginv[c * block.q + r];
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
                                                const std::unordered_map<std::string, DataParamMapping>& dispersion_param_mappings,
                                                const std::unordered_map<std::string, std::vector<std::string>>& extra_param_mappings,
                                                bool force_laplace,
                                                SparseHessianAccumulator* sparse_accumulator) const {
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
            if (var.family.empty()) continue;

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
    std::vector<RandomEffectInfo> random_effect_infos;
    build_random_effect_infos(model, covariance_parameters, fixed_covariance_data, true, random_effect_infos);
    
    if (random_effect_infos.empty()) {
        throw std::runtime_error("Random effect metadata missing");
    }

    std::vector<std::string> all_outcome_vars;
    for (const auto& var : model.variables) {
        if (var.kind == VariableKind::Observed) {
            bool is_target = false;
            for (const auto& edge : model.edges) {
                if (edge.target == var.name) {
                    is_target = true;
                    break;
                }
            }
            if (is_target) {
                all_outcome_vars.push_back(var.name);
            }
        }
    }

    std::string obs_var_name;
    std::string obs_family_name;
    if (!all_outcome_vars.empty()) {
        obs_var_name = all_outcome_vars[0];
        auto it = std::find_if(model.variables.begin(), model.variables.end(), [&](const auto& v){ return v.name == obs_var_name; });
        
        if (it == model.variables.end()) {
             throw std::runtime_error("Outcome variable not found in model variables: " + obs_var_name);
        }
        obs_family_name = it->family;
    } else {
        for (const auto& var : model.variables) {
            if (var.kind == VariableKind::Observed) {
                obs_var_name = var.name;
                obs_family_name = var.family;
                break;
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
        std::unordered_set<std::string> latent_vars;
        for (const auto& var : model.variables) {
            if (var.kind == VariableKind::Latent) {
                latent_vars.insert(var.name);
            }
        }

        for (const auto& edge : model.edges) {
            if (edge.kind == EdgeKind::Regression && edge.target == obs_var_name) {
                if (latent_vars.find(edge.source) == latent_vars.end()) {
                    fixed_effect_vars.push_back(edge.source);
                }
            }
        }
        std::sort(fixed_effect_vars.begin(), fixed_effect_vars.end());
    }

    if (obs_family_name == "gaussian" && all_outcome_vars.size() <= 1 && !force_laplace) {
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
                    Eigen::Map<RowMajorMatrix> V_map(V_i.data(), n_i, n_i);
                    
                    if (info.is_sparse) {
                        if (!info.G_sparse) {
                             throw std::runtime_error("G_sparse is null");
                        }
                        Eigen::MatrixXd Z_dense = Z_map;
                        Eigen::MatrixXd temp = Z_dense * *info.G_sparse;
                        V_map += temp * Z_dense.transpose();
                    } else {
                        Eigen::Map<const RowMajorMatrix> G_map(info.G_matrix.data(), q, q);
                        V_map += Z_map * G_map * Z_map.transpose();
                    }
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
                    if (info.G_gradients.empty() && info.G_gradients_sparse.empty() && info.design_vars.empty()) continue;
                    
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
                                        double grad_val = 0.5 * (term2 - term1);
                                        gradients[pid] += grad_val;
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
                        
                        if (caches[k].Vinv_Z.empty()) {
                            continue;
                        }

                        Eigen::Map<const RowMajorMatrix> Vinv_Z_map(caches[k].Vinv_Z.data(), n_i, q);
                        Eigen::VectorXd b = Vinv_Z_map.transpose() * r_vec;
                        
                        // dL/dG
                        if (info.is_sparse) {
                            if (caches[k].A.empty()) {
                                // If A is empty, we shouldn't be here if we have gradients
                                if (!info.G_gradients_sparse.empty()) {
                                     std::cerr << "Error: caches[" << k << "].A is empty but G_gradients_sparse is not!" << std::endl;
                                }
                                continue;
                            }
                            Eigen::Map<const RowMajorMatrix> A_map(caches[k].A.data(), q, q);
                            for (std::size_t idx = 0; idx < info.G_gradients_sparse.size(); ++idx) {
                                std::string param_name = info.covariance_id + "_" + std::to_string(idx);
                                const auto& dG = info.G_gradients_sparse[idx];
                                
                                double term1 = b.transpose() * dG * b;
                                double term2 = 0.0;
                                
                                // trace(A * dG) = sum_{i,j} A_ij * dG_ji
                                for (int k_outer=0; k_outer<dG.outerSize(); ++k_outer) {
                                    for (Eigen::SparseMatrix<double>::InnerIterator it(dG, k_outer); it; ++it) {
                                        if (it.col() >= q || it.row() >= q) {
                                            std::cerr << "Index out of bounds in sparse trace: " << it.row() << "," << it.col() << " q=" << q << std::endl;
                                            continue;
                                        }
                                        term2 += A_map(it.col(), it.row()) * it.value();
                                    }
                                }
                                gradients[param_name] += 0.5 * (term1 - term2);
                            }
                        } else {
                            for (std::size_t idx = 0; idx < info.G_gradients.size(); ++idx) {
                                std::string param_name = info.covariance_id + "_" + std::to_string(idx);
                                const auto& dG = info.G_gradients[idx];
                                Eigen::Map<const RowMajorMatrix> dG_map(dG.data(), q, q);
                                Eigen::Map<const RowMajorMatrix> A_map(caches[k].A.data(), q, q);
                                
                                double term1 = b.transpose() * dG_map * b;
                                double term2 = (A_map * dG_map).trace();
                                gradients[param_name] += 0.5 * (term1 - term2);
                            }
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
                            Eigen::MatrixXd dLdZ;
                            if (info.is_sparse) {
                                dLdZ = M * *info.G_sparse;
                            } else {
                                Eigen::Map<const RowMajorMatrix> G_map(info.G_matrix.data(), q, q);
                                dLdZ = M * G_map;
                            }
                            
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

    LaplaceSystem& system = get_or_build_laplace_system(random_effect_infos, data, linear_predictors, n, laplace_cache_);
    if (system.total_dim == 0) {
        throw std::runtime_error("Laplace system constructed without latent dimensions");
    }

    std::vector<std::string> target_outcomes = all_outcome_vars;
    if (target_outcomes.empty()) {
        target_outcomes.push_back(obs_var_name);
    }

    std::vector<std::unique_ptr<OutcomeFamily>> families;
    std::vector<OutcomeData> outcomes;
    std::vector<std::vector<double>> broadcasted_dispersions;
    broadcasted_dispersions.reserve(target_outcomes.size());

    for(const auto& name : target_outcomes) {
        auto var_it = std::find_if(model.variables.begin(), model.variables.end(), [&](const auto& v){ return v.name == name; });
        auto fam = OutcomeFamilyFactory::create(var_it->family);
        
        const auto& obs_vec = data.at(name);
        const auto& pred_vec = linear_predictors.at(name);
        const auto& disp_vec = dispersions.at(name);
        
        if (obs_vec.size() != n) throw std::runtime_error("Mismatched observation count for " + name);

        const std::vector<double>* s_vec = nullptr;
        if (status.count(name)) s_vec = &status.at(name);
        
        const auto& ex_vec = get_extra_params_for_variable(extra_params, name);
        
        const std::vector<double>* p_disp = &disp_vec;
        if (disp_vec.size() == 1 && n > 1) {
            broadcasted_dispersions.emplace_back(n, disp_vec[0]);
            p_disp = &broadcasted_dispersions.back();
        }
        
        families.push_back(std::move(fam));
        outcomes.push_back({name, obs_vec, pred_vec, *p_disp, s_vec, ex_vec, families.back().get()});
    }

    LaplaceSystemResult laplace_result;
    
    if (!solve_laplace_system(system,
                              outcomes,
                              laplace_result,
                              sparse_accumulator)) {
        return { -std::numeric_limits<double>::infinity(), gradients };
    }

    double log_prior = compute_prior_loglik(system, laplace_result, log_2pi);

    double log_lik_data = 0.0;
    for (const auto& eval : laplace_result.evaluations) {
        log_lik_data += eval.log_likelihood;
    }
    double log_det_neg_hess = laplace_result.log_det_neg_hess;
    total_loglik = log_prior + log_lik_data + 0.5 * system.total_dim * log_2pi - 0.5 * log_det_neg_hess;

    std::vector<double> forcing(system.total_dim, 0.0);

    std::size_t eval_offset = 0;
    for(size_t k=0; k<outcomes.size(); ++k) {
        const auto& outcome = outcomes[k];
        std::size_t n_outcome = outcome.obs.size();
        std::string_view sv = outcome.name;
        auto quad_terms = compute_observation_quad_terms(system, laplace_result, sv);

        if (!dispersion_param_mappings.empty()) {
            auto map_it = dispersion_param_mappings.find(outcome.name);
            if (map_it != dispersion_param_mappings.end()) {
                const auto& mapping = map_it->second;
                for (std::size_t i = 0; i < n_outcome; ++i) {
                    size_t idx = i % mapping.stride;
                    if (idx < mapping.pattern.size()) {
                        const std::string& pid = mapping.pattern[idx];
                        if (!pid.empty()) {
                            gradients[pid] += laplace_result.evaluations[eval_offset + i].d_dispersion;
                            gradients[pid] += 0.5 * laplace_result.evaluations[eval_offset + i].d_hessian_d_dispersion * quad_terms[i];
                        }
                    }
                }
            }
        }

        for (const auto& edge : model.edges) {
            if (edge.kind != EdgeKind::Regression || edge.target != outcome.name || edge.parameter_id.empty()) {
                continue;
            }
            auto src_it = data.find(edge.source);
            if (src_it == data.end() || src_it->second.size() != n_outcome) {
                continue;
            }

            const auto& src_vec = src_it->second;
            std::vector<double> weights(n_outcome, 0.0);
            for (std::size_t i = 0; i < n_outcome; ++i) {
                weights[i] = laplace_result.evaluations[eval_offset + i].second_derivative * src_vec[i];
            }
            accumulate_weighted_forcing(system, weights, forcing);
            std::vector<double> du(system.total_dim);
            
            bool decoupled_path = !laplace_result.block_inverses.empty();
            if (decoupled_path) {
                 for (const auto& block : system.blocks) {
                     const auto& inv_block = laplace_result.block_inverses[block.index];
                     for (size_t r = 0; r < block.q; ++r) {
                         double sum = 0.0;
                         for (size_t c = 0; c < block.q; ++c) {
                             sum += inv_block[r * block.q + c] * forcing[block.offset + c];
                         }
                         du[block.offset + r] = sum;
                     }
                 }
            } else if (laplace_result.use_sparse) {
                Eigen::Map<Eigen::VectorXd> f_vec(forcing.data(), system.total_dim);
                Eigen::VectorXd du_vec = laplace_result.sparse_solver->solve(f_vec);
                Eigen::Map<Eigen::VectorXd>(du.data(), system.total_dim) = du_vec;
            } else {
                du = solve_cholesky(system.total_dim, laplace_result.chol_neg_hessian, forcing);
            }
            auto z_dot_du = project_observations(system, du, outcome.name);

            double accum = 0.0;
            for (std::size_t i = 0; i < n_outcome; ++i) {
                const auto& eval = laplace_result.evaluations[eval_offset + i];
                double logdet_adjust = src_vec[i] + z_dot_du[i];
                accum += src_vec[i] * eval.first_derivative + 0.5 * eval.third_derivative * logdet_adjust * quad_terms[i];
            }
            gradients[edge.parameter_id] += accum;
        }
        eval_offset += n_outcome;
    }

    struct CovarianceDerivativeCache {
        std::string param_name;
        std::vector<double> Ginv_dG_Ginv;
        double trace_Ginv_dG;
    };

    std::unordered_map<std::string, std::vector<CovarianceDerivativeCache>> covariance_caches;
    for (const auto& info : random_effect_infos) {
        if (info.G_gradients.empty() && info.G_gradients_sparse.empty()) {
            continue;
        }
        std::vector<CovarianceDerivativeCache> caches;
        std::size_t q = info.cov_spec->dimension;

        if (info.is_sparse) {
             if (!info.sparse_solver) throw std::runtime_error("Sparse solver missing for " + info.id);
             
             // Densify G_inv for gradient computation
             Eigen::MatrixXd I = Eigen::MatrixXd::Identity(q, q);
             Eigen::MatrixXd G_inv_eigen = info.sparse_solver->solve(I);
             
             for (std::size_t idx = 0; idx < info.G_gradients_sparse.size(); ++idx) {
                 CovarianceDerivativeCache cache;
                 cache.param_name = info.covariance_id + "_" + std::to_string(idx);
                 
                 const auto& dG_sparse = info.G_gradients_sparse[idx];
                 
                 // Ginv * dG * Ginv
                 Eigen::MatrixXd dG_Ginv = dG_sparse * G_inv_eigen;
                 Eigen::MatrixXd term = G_inv_eigen * dG_Ginv;
                 
                 cache.Ginv_dG_Ginv.resize(q * q);
                 Eigen::Map<RowMajorMatrix>(cache.Ginv_dG_Ginv.data(), q, q) = term;
                 
                 // trace(Ginv * dG)
                 double trace = 0.0;
                 for (int k=0; k<dG_sparse.outerSize(); ++k) {
                    for (Eigen::SparseMatrix<double>::InnerIterator it(dG_sparse, k); it; ++it) {
                        trace += G_inv_eigen(it.col(), it.row()) * it.value();
                    }
                 }
                 cache.trace_Ginv_dG = trace;
                 caches.push_back(std::move(cache));
             }
        } else {
            for (std::size_t idx = 0; idx < info.G_gradients.size(); ++idx) {
                CovarianceDerivativeCache cache;
                cache.param_name = info.covariance_id + "_" + std::to_string(idx);
                auto temp = multiply_matrices(info.G_inverse, info.G_gradients[idx], q);
                cache.Ginv_dG_Ginv = multiply_matrices(temp, info.G_inverse, q);
                cache.trace_Ginv_dG = trace_product(info.G_inverse, info.G_gradients[idx], q);
                caches.push_back(std::move(cache));
            }
        }
        covariance_caches.emplace(info.covariance_id, std::move(caches));
    }

    // Precompute quad_terms for all outcomes
    std::vector<std::vector<double>> all_quad_terms(outcomes.size());
    for(size_t k=0; k<outcomes.size(); ++k) {
        all_quad_terms[k] = compute_observation_quad_terms(system, laplace_result, outcomes[k].name);
    }

    const std::size_t Q = system.total_dim;
    std::vector<double> cov_forcing(system.total_dim, 0.0);

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
            
            std::vector<double> du_block(block.q);
            std::vector<double> du; 
            bool decoupled_path = !laplace_result.block_inverses.empty();

            if (decoupled_path) {
                 const auto& inv_block = laplace_result.block_inverses[block.index];
                 for (size_t r = 0; r < block.q; ++r) {
                     double sum = 0.0;
                     for (size_t c = 0; c < block.q; ++c) {
                         sum += inv_block[r * block.q + c] * block_forcing[c];
                     }
                     du_block[r] = sum;
                 }
            } else {
                std::fill(cov_forcing.begin(), cov_forcing.end(), 0.0);
                for (std::size_t j = 0; j < block.q; ++j) {
                    cov_forcing[block.offset + j] = block_forcing[j];
                }
                du.resize(system.total_dim);
                if (laplace_result.use_sparse) {
                    Eigen::Map<Eigen::VectorXd> f_vec(cov_forcing.data(), system.total_dim);
                    Eigen::VectorXd du_vec = laplace_result.sparse_solver->solve(f_vec);
                    Eigen::Map<Eigen::VectorXd>(du.data(), system.total_dim) = du_vec;
                } else {
                    du = solve_cholesky(system.total_dim, laplace_result.chol_neg_hessian, cov_forcing);
                }
            }
            
            double logdet_adjust_total = 0.0;
            std::size_t eval_offset_cov = 0;
            for(size_t k=0; k<outcomes.size(); ++k) {
                const auto& outcome = outcomes[k];
                const auto& quad_terms = all_quad_terms[k];
                
                if (decoupled_path) {
                    bool targets = false;
                    for(const auto& t : block.info->target_vars) {
                        if (t == outcome.name) { targets = true; break; }
                    }
                    if (targets) {
                        for (size_t r = 0; r < block.obs_indices.size(); ++r) {
                            size_t obs_idx = block.obs_indices[r];
                            if (std::isnan(outcome.obs[obs_idx])) continue;
                            
                            const double* z_row = &block.design_matrix[r * block.q];
                            double z_dot_du_val = 0.0;
                            for(size_t j=0; j<block.q; ++j) z_dot_du_val += z_row[j] * du_block[j];
                            
                            const auto& eval = laplace_result.evaluations[eval_offset_cov + obs_idx];
                            logdet_adjust_total += 0.5 * eval.third_derivative * z_dot_du_val * quad_terms[obs_idx];
                        }
                    }
                } else {
                    auto z_dot_du = project_observations(system, du, outcome.name);
                    std::size_t n_outcome = outcome.obs.size();
                    for (std::size_t obs_idx = 0; obs_idx < n_outcome; ++obs_idx) {
                        const auto& eval = laplace_result.evaluations[eval_offset_cov + obs_idx];
                        logdet_adjust_total += 0.5 * eval.third_derivative * z_dot_du[obs_idx] * quad_terms[obs_idx];
                    }
                }
                eval_offset_cov += outcome.obs.size();
            }

            double prior_term = 0.5 * quadratic_form(u_block, cache.Ginv_dG_Ginv, block.q) - 0.5 * cache.trace_Ginv_dG;

            double trace_term = 0.0;
            std::vector<double> inv_block = get_inverse_block(laplace_result, block, Q);

            for (std::size_t r = 0; r < block.q; ++r) {
                for (std::size_t c = 0; c < block.q; ++c) {
                    trace_term += inv_block[r * block.q + c] * cache.Ginv_dG_Ginv[c * block.q + r];
                }
            }
            double logdet_term = 0.5 * trace_term + logdet_adjust_total;
            gradients[cache.param_name] += prior_term + logdet_term;
        }
    }

    // Gradients for extra parameters (thresholds)
    if (!extra_param_mappings.empty()) {
        
        std::size_t eval_offset_extra = 0;
        for (const auto& outcome : outcomes) {
            std::size_t n_outcome = outcome.obs.size();
            auto map_it = extra_param_mappings.find(outcome.name);
            
            if (map_it != extra_param_mappings.end()) {
                const auto& param_ids = map_it->second;
                
                // We need quad_terms
                auto quad_terms = compute_observation_quad_terms(system, laplace_result, outcome.name);
                
                for (std::size_t i = 0; i < n_outcome; ++i) {
                    const auto& eval = laplace_result.evaluations[eval_offset_extra + i];
                    if (eval.d_extra_params.empty()) continue;
                    
                    for (std::size_t k = 0; k < param_ids.size(); ++k) {
                        const std::string& pid = param_ids[k];
                        if (pid.empty()) continue;
                        
                        double val = 0.0;
                        if (k < eval.d_extra_params.size()) {
                            val += eval.d_extra_params[k];
                        }
                        if (k < eval.d_hessian_d_extra_params.size()) {
                            val += 0.5 * eval.d_hessian_d_extra_params[k] * quad_terms[i];
                        }
                        gradients[pid] += val;
                    }
                }
            }
            eval_offset_extra += n_outcome;
        }
    }

    // Gradients for design matrix (loadings)
    if (!data_param_mappings.empty()) {
        std::size_t eval_offset_data = 0;
        const std::size_t Q = system.total_dim;
        
        for (const auto& outcome : outcomes) {
            std::size_t n_outcome = outcome.obs.size();
            
            for (std::size_t i = 0; i < n_outcome; ++i) {
                const auto& eval = laplace_result.evaluations[eval_offset_data + i];
                std::size_t global_obs_idx = eval_offset_data + i;
                
                if (global_obs_idx >= system.observation_entries.size()) continue;
                
                for (const auto& entry : system.observation_entries[global_obs_idx]) {
                    const auto& block = system.blocks[entry.block_index];
                    const auto& info = *block.info;
                    
                    bool has_mapped = false;
                    for(const auto& v : info.design_vars) {
                        if(data_param_mappings.count(v)) { has_mapped = true; break; }
                    }
                    if (!has_mapped) continue;
                    
                    const double* z_row = &block.design_matrix[entry.row_index * block.q];
                    std::vector<double> inv_block = get_inverse_block(laplace_result, block, Q);
                    
                    for (std::size_t c = 0; c < block.q; ++c) {
                        const std::string& var_name = info.design_vars[c];
                        auto map_it = data_param_mappings.find(var_name);
                        if (map_it != data_param_mappings.end()) {
                            const auto& mapping = map_it->second;
                            size_t idx = i % mapping.stride;
                            if (idx < mapping.pattern.size()) {
                                const std::string& pid = mapping.pattern[idx];
                                if (!pid.empty()) {
                                    double term1 = eval.first_derivative * laplace_result.u[block.offset + c];
                                    
                                    double term2_inner = 0.0;
                                    for (std::size_t r = 0; r < block.q; ++r) {
                                        term2_inner += inv_block[c * block.q + r] * z_row[r];
                                    }
                                    double term2 = eval.second_derivative * term2_inner;
                                    
                                    gradients[pid] += term1 + term2;
                                }
                            }
                        }
                    }
                }
            }
            eval_offset_data += n_outcome;
        }
    }

    return {total_loglik, gradients};
}

std::unordered_map<std::string, std::vector<double>> LikelihoodDriver::compute_random_effects(
    const ModelIR& model,
    const std::unordered_map<std::string, std::vector<double>>& data,
    const std::unordered_map<std::string, std::vector<double>>& linear_predictors,
    const std::unordered_map<std::string, std::vector<double>>& dispersions,
    const std::unordered_map<std::string, std::vector<double>>& covariance_parameters,
    const std::unordered_map<std::string, std::vector<double>>& status,
    const std::unordered_map<std::string, std::vector<double>>& extra_params,
    const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data) const {

    std::unordered_map<std::string, std::vector<double>> random_effects;

    if (model.random_effects.empty()) {
        return random_effects;
    }

    // 1. Identify observed variable and family
    std::string obs_var_name;
    std::string obs_family_name;
    for (const auto& var : model.variables) {
        if (var.kind == VariableKind::Observed) {
            if (var.family.empty()) continue;
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
        return random_effects;
    }

    auto pred_it = linear_predictors.find(obs_var_name);
    if (pred_it == linear_predictors.end()) {
        if (linear_predictors.size() == 1) {
            obs_var_name = linear_predictors.begin()->first;
            auto var_it = std::find_if(model.variables.begin(), model.variables.end(), [&](const auto& v) {
                return v.name == obs_var_name;
            });
            if (var_it == model.variables.end()) {
                return random_effects;
            }
            obs_family_name = var_it->family;
            pred_it = linear_predictors.begin();
        } else {
            return random_effects;
        }
    }
    const auto& pred_data = pred_it->second;

    auto obs_it = data.find(obs_var_name);
    if (obs_it == data.end()) {
        return random_effects;
    }
    const auto& obs_data = obs_it->second;
    size_t n = obs_data.size();

    auto disp_it = dispersions.find(obs_var_name);
    if (disp_it == dispersions.end()) {
        return random_effects;
    }
    const auto& disp_data = disp_it->second;

    // 2. Build RandomEffectInfos
    std::vector<RandomEffectInfo> random_effect_infos;
    build_random_effect_infos(model, covariance_parameters, fixed_covariance_data, false, random_effect_infos);
    if (random_effect_infos.empty()) return random_effects;

    // 3. Build/Get Laplace System
    std::shared_ptr<void> cache_handle;
    LaplaceSystem& system = get_or_build_laplace_system(random_effect_infos, data, linear_predictors, n, cache_handle);

    if (system.total_dim == 0) return random_effects;

    // 4. Solve Laplace System
    auto outcome_family = OutcomeFamilyFactory::create(obs_family_name);
    LaplaceSystemResult laplace_result;
    
    const std::vector<double>* status_vec = nullptr;
    if (status.count(obs_var_name)) status_vec = &status.at(obs_var_name);
    
    const std::vector<double>* extra_vec = nullptr;
    if (extra_params.count(obs_var_name)) extra_vec = &extra_params.at(obs_var_name);
    const std::vector<double>& ep = extra_vec ? *extra_vec : std::vector<double>{};

    std::vector<OutcomeData> outcomes;
    outcomes.push_back({obs_var_name, obs_data, pred_data, disp_data, status_vec, ep, outcome_family.get()});

    if (!solve_laplace_system(system, outcomes, laplace_result)) {
        return random_effects;
    }

    // 5. Unpack 'u' into random_effects map
    for(const auto& info : random_effect_infos) {
        random_effects[info.id] = {};
    }

    for(const auto& block : system.blocks) {
        if (!block.info) continue;
        std::string id = block.info->id;
        if (random_effects.find(id) == random_effects.end()) continue;
        
        std::vector<double>& vec = random_effects[id];
        size_t start = block.offset;
        size_t len = block.q;
        if (start + len <= laplace_result.u.size()) {
            vec.insert(vec.end(), laplace_result.u.begin() + start, laplace_result.u.begin() + start + len);
        }
    }

    return random_effects;
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
            double val = (grad_plus[i] - grad_minus[i]) / (2.0 * epsilon);
            if (std::isnan(val)) {
                // std::cout << "NaN in Hessian at (" << i << "," << j << ")" << std::endl;
                // std::cout << "grad_plus[" << i << "] = " << grad_plus[i] << std::endl;
                // std::cout << "grad_minus[" << i << "] = " << grad_minus[i] << std::endl;
            }
            hessian(i, j) = val;
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
                                         EstimationMethod method,
                                         const std::unordered_map<std::string, std::vector<std::string>>& extra_param_mappings) const {
    
    ModelObjective objective(*this, model, data, fixed_covariance_data, status, method, extra_param_mappings, options.force_laplace);
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
    
    // Convert parameters if necessary (e.g. Cholesky -> Covariance)
    std::vector<double> constrained_params = objective.to_constrained(result.parameters);
    fit_result.optimization_result.parameters = objective.convert_to_model_parameters(constrained_params);
    
    fit_result.parameter_names = objective.parameter_names();
    fit_result.covariance_matrices = objective.get_covariance_matrices(fit_result.optimization_result.parameters);

    // Compute Random Effects (BLUPs)
    if (!model.random_effects.empty()) {
        std::unordered_map<std::string, std::vector<double>> linear_predictors;
        std::unordered_map<std::string, std::vector<double>> dispersions;
        std::unordered_map<std::string, std::vector<double>> covariance_parameters;
        
        // Use original constrained parameters (L) for internal prediction workspaces
        objective.build_prediction_workspaces(constrained_params, 
                                              linear_predictors, 
                                              dispersions, 
                                              covariance_parameters);
                                              
        fit_result.random_effects = compute_random_effects(
            model,
            data,
            linear_predictors,
            dispersions,
            covariance_parameters,
            status,
            {}, // extra_params
            fixed_covariance_data
        );
    }

    // Compute AIC/BIC regardless of convergence status (using final values)
    double nll = result.objective_value;
    size_t n = result.parameters.size();
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

    if (result.converged) {
        try {
            Eigen::MatrixXd hessian = compute_hessian(objective, result.parameters);
            
            // Hessian of NLL is Fisher Information. Covariance is Inverse.
            Eigen::MatrixXd vcov_unconstrained = hessian.inverse();
            
            std::vector<double> derivs = objective.constrained_derivatives(result.parameters);
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
             // Keep AIC/BIC as computed
        }
    } else {
         size_t n = result.parameters.size();
         fit_result.standard_errors.assign(n, std::numeric_limits<double>::quiet_NaN());
         fit_result.vcov.assign(n*n, std::numeric_limits<double>::quiet_NaN());
         // Keep AIC/BIC as computed
    }

    // Transform parameters to constrained space for reporting
    fit_result.optimization_result.parameters = objective.to_constrained(result.parameters);
    fit_result.parameter_names = objective.parameter_names();

    return fit_result;
}

FitResult LikelihoodDriver::fit_multi_group(const std::vector<ModelIR>& models,
                                            const std::vector<std::unordered_map<std::string, std::vector<double>>>& data,
                                            const OptimizationOptions& options,
                                            const std::string& optimizer_name,
                                            const std::vector<std::unordered_map<std::string, std::vector<std::vector<double>>>>& fixed_covariance_data,
                                            const std::vector<std::unordered_map<std::string, std::vector<double>>>& status,
                                            EstimationMethod method,
                                            const std::vector<std::unordered_map<std::string, std::vector<std::string>>>& extra_param_mappings) const {
    
    if (models.size() != data.size()) {
        throw std::invalid_argument("Number of models must match number of data sets");
    }
    
    std::vector<std::unique_ptr<ModelObjective>> objectives;
    objectives.reserve(models.size());
    
    size_t total_obs = 0;
    
    // Default empty maps to ensure references remain valid
    std::unordered_map<std::string, std::vector<std::vector<double>>> empty_fixed_cov;
    std::unordered_map<std::string, std::vector<double>> empty_status;
    std::unordered_map<std::string, std::vector<std::string>> empty_extra;
    
    for (size_t i = 0; i < models.size(); ++i) {
        const auto& fixed_cov = (i < fixed_covariance_data.size()) ? fixed_covariance_data[i] : empty_fixed_cov;
        const auto& stat = (i < status.size()) ? status[i] : empty_status;
        const auto& extra = (i < extra_param_mappings.size()) ? extra_param_mappings[i] : empty_extra;
        
        objectives.push_back(std::make_unique<ModelObjective>(*this, models[i], data[i], fixed_cov, stat, method, extra, options.force_laplace));
        
        // Count observations (approximate, using first variable)
        if (!data[i].empty()) {
            total_obs += data[i].begin()->second.size();
        }
    }
    
    MultiGroupModelObjective multi_obj(std::move(objectives));
    
    std::unique_ptr<Optimizer> optimizer;
    if (optimizer_name == "lbfgs") {
        optimizer = std::make_unique<LBFGSOptimizer>();
    } else if (optimizer_name == "gd") {
        optimizer = std::make_unique<GradientDescentOptimizer>();
    } else {
        throw std::invalid_argument("Unknown optimizer: " + optimizer_name);
    }
    
    auto initial_params = multi_obj.initial_parameters();
    auto result = optimizer->optimize(multi_obj, initial_params, options);
    
    FitResult fit_result;
    fit_result.optimization_result = result;
    fit_result.parameter_names = multi_obj.parameter_names();
    
    // Compute AIC/BIC
    size_t k = result.parameters.size();
    double loglik = -result.objective_value;
    fit_result.aic = 2.0 * k - 2.0 * loglik;
    fit_result.bic = k * std::log(static_cast<double>(total_obs)) - 2.0 * loglik;
    
    // Compute Hessian and SEs
    if (result.converged) {
        try {
            size_t n_params = result.parameters.size();
            std::vector<double> hessian(n_params * n_params);
            
            // Numerical Hessian
            double epsilon = 1e-4;
            std::vector<double> x = result.parameters;
            std::vector<double> grad0 = multi_obj.gradient(x);
            
            for (size_t i = 0; i < n_params; ++i) {
                double original = x[i];
                x[i] += epsilon;
                std::vector<double> grad_plus = multi_obj.gradient(x);
                x[i] = original;
                
                for (size_t j = 0; j < n_params; ++j) {
                    hessian[i * n_params + j] = (grad_plus[j] - grad0[j]) / epsilon;
                }
            }
            
            // Symmetrize
            for (size_t i = 0; i < n_params; ++i) {
                for (size_t j = i + 1; j < n_params; ++j) {
                    double avg = 0.5 * (hessian[i * n_params + j] + hessian[j * n_params + i]);
                    hessian[i * n_params + j] = avg;
                    hessian[j * n_params + i] = avg;
                }
            }
            
            // Invert
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> H(hessian.data(), n_params, n_params);
            Eigen::MatrixXd H_inv = H.inverse();
            
            fit_result.vcov.resize(n_params * n_params);
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(fit_result.vcov.data(), n_params, n_params) = H_inv;
            
            fit_result.standard_errors.resize(n_params);
            for (size_t i = 0; i < n_params; ++i) {
                double var = fit_result.vcov[i * n_params + i];
                if (var > 0) {
                    fit_result.standard_errors[i] = std::sqrt(var);
                } else {
                    fit_result.standard_errors[i] = std::numeric_limits<double>::quiet_NaN();
                }
            }
            
            // Aggregate covariance matrices and random effects
            // We need to split parameters back to groups
            auto group_params = multi_obj.split_parameters(result.parameters);
            const auto& objs = multi_obj.objectives();
            
            for (size_t i = 0; i < objs.size(); ++i) {
                // We need to convert unconstrained to constrained for each group
                // But ModelObjective::get_covariance_matrices takes constrained parameters.
                // And ModelObjective::to_constrained takes unconstrained.
                
                auto constrained = objs[i]->to_constrained(group_params[i]);
                
                auto covs = objs[i]->get_covariance_matrices(constrained);
                for (const auto& [key, val] : covs) {
                    // If key exists, we overwrite? Or check consistency?
                    // For now overwrite.
                    fit_result.covariance_matrices[key] = val;
                }
                
                // Random effects
                // We need to call compute_random_effects logic.
                // But compute_random_effects is on LikelihoodDriver and takes ModelIR etc.
                // We can use the helper on LikelihoodDriver if we expose it, or duplicate logic.
                // Actually, ModelObjective doesn't expose random effects computation directly.
                // But LikelihoodDriver::compute_random_effects does.
                // We can call it.
                
                // We need linear predictors and dispersions first.
                std::unordered_map<std::string, std::vector<double>> lin_preds;
                std::unordered_map<std::string, std::vector<double>> disps;
                std::unordered_map<std::string, std::vector<double>> cov_params;
                
                objs[i]->build_prediction_workspaces(constrained, lin_preds, disps, cov_params);
                auto extra_p = objs[i]->build_extra_params(constrained);
                
                const auto& fixed_cov = (i < fixed_covariance_data.size()) ? fixed_covariance_data[i] : empty_fixed_cov;
                const auto& stat = (i < status.size()) ? status[i] : empty_status;

                auto res = compute_random_effects(models[i], data[i], lin_preds, disps, cov_params, 
                                                  stat,
                                                  extra_p,
                                                  fixed_cov);
                
                for (const auto& [key, val] : res) {
                    fit_result.random_effects[key] = val;
                }
            }
            
        } catch (...) {
             size_t n = result.parameters.size();
             fit_result.standard_errors.assign(n, std::numeric_limits<double>::quiet_NaN());
             fit_result.vcov.assign(n*n, std::numeric_limits<double>::quiet_NaN());
        }
    } else {
         size_t n = result.parameters.size();
         fit_result.standard_errors.assign(n, std::numeric_limits<double>::quiet_NaN());
         fit_result.vcov.assign(n*n, std::numeric_limits<double>::quiet_NaN());
    }
    
    // Transform parameters to constrained space for reporting
    fit_result.optimization_result.parameters = multi_obj.to_constrained(result.parameters);
    fit_result.parameter_names = multi_obj.parameter_names();

    return fit_result;
}

}  // namespace libsemx
