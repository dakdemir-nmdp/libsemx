#include "libsemx/likelihood_driver.hpp"
#include "libsemx/outcome_family_factory.hpp"
#include "libsemx/covariance_structure.hpp"
#include "libsemx/parameter_transform.hpp"
#include "libsemx/parameter_catalog.hpp"

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

constexpr double kDefaultCoefficientInit = 0.0;
constexpr double kDefaultVarianceInit = 0.5;

using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

std::string normalize_structure_id(const std::string& id) {
    std::string lowered;
    lowered.reserve(id.size());
    for (unsigned char ch : id) {
        if (ch == '-') {
            lowered.push_back('_');
        } else {
            lowered.push_back(static_cast<char>(std::tolower(ch)));
        }
    }
    return lowered;
}

std::optional<std::size_t> parse_factor_rank(const std::string& normalized) {
    auto parse_tail = [](std::string_view tail) -> std::optional<std::size_t> {
        if (tail.empty()) {
            return std::nullopt;
        }
        std::size_t idx = 0;
        while (idx < tail.size() && (tail[idx] == '_' || tail[idx] == '-' || tail[idx] == '(' || tail[idx] == 'q')) {
            ++idx;
        }
        if (idx >= tail.size()) {
            return std::nullopt;
        }
        std::size_t value = 0;
        for (; idx < tail.size(); ++idx) {
            const char ch = tail[idx];
            if (ch == ')' || ch == ' ') {
                break;
            }
            if (!std::isdigit(static_cast<unsigned char>(ch))) {
                return std::nullopt;
            }
            value = value * 10 + static_cast<std::size_t>(ch - '0');
        }
        if (value == 0) {
            return std::nullopt;
        }
        return value;
    };

    if (normalized == "factor_analytic" || normalized == "fa") {
        return 1;
    }
    if (normalized.rfind("fa", 0) == 0) {
        return parse_tail(normalized.substr(2));
    }
    constexpr std::string_view kPrefix = "factor_analytic";
    if (normalized.rfind(kPrefix, 0) == 0) {
        auto rank = parse_tail(normalized.substr(kPrefix.size()));
        if (rank) {
            return rank;
        }
        return 1;
    }
    return std::nullopt;
}

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

std::unique_ptr<CovarianceStructure> create_covariance_structure(
    const CovarianceSpec& spec,
    const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data) {

    const std::string normalized = normalize_structure_id(spec.structure);

    if (normalized == "unstructured") {
        return std::make_unique<UnstructuredCovariance>(spec.dimension);
    } else if (normalized == "diagonal") {
        return std::make_unique<DiagonalCovariance>(spec.dimension);
    } else if (normalized == "scaled_fixed") {
        auto fixed_it = fixed_covariance_data.find(spec.id);
        if (fixed_it == fixed_covariance_data.end()) {
            throw std::runtime_error("Missing fixed covariance data for: " + spec.id);
        }
        if (fixed_it->second.empty()) {
            throw std::runtime_error("Fixed covariance data is empty for: " + spec.id);
        }
        return std::make_unique<ScaledFixedCovariance>(fixed_it->second[0], spec.dimension);
    } else if (normalized == "multi_kernel") {
        auto fixed_it = fixed_covariance_data.find(spec.id);
        if (fixed_it == fixed_covariance_data.end()) {
            throw std::runtime_error("Missing fixed covariance data for: " + spec.id);
        }
        return std::make_unique<MultiKernelCovariance>(fixed_it->second, spec.dimension);
    } else if (normalized == "compound_symmetry" || normalized == "cs") {
        return std::make_unique<CompoundSymmetryCovariance>(spec.dimension);
    } else if (normalized == "ar1") {
        return std::make_unique<AR1Covariance>(spec.dimension);
    } else if (normalized == "toeplitz") {
        return std::make_unique<ToeplitzCovariance>(spec.dimension);
    } else if (auto rank = parse_factor_rank(normalized)) {
        if (*rank >= spec.dimension || *rank == 0) {
            throw std::runtime_error("Factor-analytic rank must satisfy 0 < rank < dimension for covariance: " + spec.id);
        }
        return std::make_unique<FactorAnalyticCovariance>(spec.dimension, *rank);
    } else {
        throw std::runtime_error("Unknown covariance structure: " + spec.structure);
    }
}

std::vector<bool> build_covariance_positive_mask(const CovarianceSpec& spec,
                                                 const CovarianceStructure& structure) {
    std::size_t count = structure.parameter_count();
    std::vector<bool> mask(count, false);
    if (count == 0) {
        return mask;
    }

    const std::string normalized = normalize_structure_id(spec.structure);

    if (normalized == "diagonal") {
        std::fill(mask.begin(), mask.end(), true);
        return mask;
    }

    if (normalized == "unstructured") {
        std::size_t idx = 0;
        for (std::size_t row = 0; row < spec.dimension; ++row) {
            for (std::size_t col = 0; col <= row; ++col) {
                if (idx < mask.size()) {
                    mask[idx] = (row == col);
                }
                ++idx;
            }
        }
        return mask;
    }

    if (normalized == "scaled_fixed") {
        mask[0] = true;
        return mask;
    }

    if (normalized == "multi_kernel") {
        std::fill(mask.begin(), mask.end(), true);
        return mask;
    }

    if (normalized == "compound_symmetry" || normalized == "cs") {
        std::fill(mask.begin(), mask.end(), true);
        return mask;
    }

    if (normalized == "ar1") {
        mask[0] = true;
        if (mask.size() > 1) {
            mask[1] = true;
        }
        return mask;
    }

    if (normalized == "toeplitz") {
        std::fill(mask.begin(), mask.end(), true);
        return mask;
    }

    if (auto rank = parse_factor_rank(normalized)) {
        const std::size_t loadings = spec.dimension * rank.value();
        for (std::size_t idx = loadings; idx < mask.size(); ++idx) {
            mask[idx] = true;
        }
        return mask;
    }

    return mask;
}

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

    constexpr int kMaxIter = 30;
    constexpr double kTol = 1e-6;

    for (int iter = 0; iter < kMaxIter; ++iter) {
        compute_system_grad_hess(system, u, obs_data, pred_data, disp_data, status_vec, extra_vec, family, grad, hess, nullptr);
        for (std::size_t idx = 0; idx < Q * Q; ++idx) {
            neg_hess[idx] = -hess[idx];
        }
        try {
            cholesky(Q, neg_hess, chol);
        } catch (...) {
            return false;
        }
        auto delta = solve_cholesky(Q, chol, grad);
        double max_delta = 0.0;
        for (std::size_t j = 0; j < Q; ++j) {
            u[j] += delta[j];
            max_delta = std::max(max_delta, std::abs(delta[j]));
        }
        if (max_delta < kTol) {
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
    if (observed.size() != linear_predictors.size() || observed.size() != dispersions.size()) {
        throw std::invalid_argument("all input vectors must have the same size");
    }
    if (!status.empty() && status.size() != observed.size()) {
        throw std::invalid_argument("status vector must match observed size if provided");
    }

    double total = 0.0;
    for (std::size_t i = 0; i < observed.size(); ++i) {
        double s = status.empty() ? 1.0 : status[i];
        const auto eval = family.evaluate(observed[i], linear_predictors[i], dispersions[i], s, extra_params);
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
    if (observed.size() != linear_predictors.size() || observed.size() != dispersions.size() ||
        observed.size() != families.size()) {
        throw std::invalid_argument("all input vectors must have the same size");
    }
    if (!status.empty() && status.size() != observed.size()) {
        throw std::invalid_argument("status vector must match observed size if provided");
    }
    if (!extra_params.empty() && extra_params.size() != observed.size()) {
        throw std::invalid_argument("extra_params vector must match observed size if provided");
    }
    double total = 0.0;
    for (std::size_t i = 0; i < observed.size(); ++i) {
        double s = status.empty() ? 1.0 : status[i];
        const std::vector<double>& ep = extra_params.empty() ? std::vector<double>{} : extra_params[i];
        const auto eval = families[i]->evaluate(observed[i], linear_predictors[i], dispersions[i], s, ep);
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
            auto pred_it = linear_predictors.find(var.name);
            auto disp_it = dispersions.find(var.name);
            if (data_it == data.end() || pred_it == linear_predictors.end() || disp_it == dispersions.end()) {
                throw std::invalid_argument("Missing data for variable: " + var.name);
            }
            const auto& obs = data_it->second;
            const auto& preds = pred_it->second;
            const auto& disps = disp_it->second;
            if (obs.size() != preds.size() || obs.size() != disps.size()) {
                throw std::invalid_argument("Data vectors for variable " + var.name + " have mismatched sizes");
            }
            
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
                double s = status_vec ? (*status_vec)[i] : 1.0;
                const auto eval = family->evaluate(obs[i], preds[i], disps[i], s, ep);
                total += eval.log_likelihood;
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
            obs_var_name = var.name;
            obs_family_name = var.family;
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
    const auto& disp_data = disp_it->second;
    if (disp_data.size() != n) {
        throw std::runtime_error("Dispersion vector has mismatched size");
    }

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

        std::vector<double> L(n * n);
        cholesky(n, V_full, L);
        double log_det = log_det_cholesky(n, L);

        std::vector<double> resid(n);
        for (std::size_t i = 0; i < n; ++i) {
            resid[i] = obs_data[i] - pred_data[i];
        }
        std::vector<double> alpha = solve_cholesky(n, L, resid);

        double quad_form = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            quad_form += resid[i] * alpha[i];
        }

        double total_loglik = -0.5 * (n * log_2pi + log_det + quad_form);

        if (method == EstimationMethod::REML && !fixed_effect_vars.empty()) {
            std::vector<double> V_inv = invert_from_cholesky(n, L);
            std::size_t p = fixed_effect_vars.size();
            std::vector<double> X(n * p);
            for (std::size_t j = 0; j < p; ++j) {
                auto it = data.find(fixed_effect_vars[j]);
                if (it == data.end() || it->second.size() != n) {
                    throw std::runtime_error("Missing data for fixed effect: " + fixed_effect_vars[j]);
                }
                for (std::size_t i = 0; i < n; ++i) {
                    X[i * p + j] = it->second[i];
                }
            }

            std::vector<double> Vinv_X(n * p, 0.0);
            for (std::size_t col = 0; col < p; ++col) {
                for (std::size_t row = 0; row < n; ++row) {
                    double sum = 0.0;
                    for (std::size_t k = 0; k < n; ++k) {
                        sum += V_inv[row * n + k] * X[k * p + col];
                    }
                    Vinv_X[row * p + col] = sum;
                }
            }

            std::vector<double> Xt_Vinv_X(p * p, 0.0);
            for (std::size_t r = 0; r < p; ++r) {
                for (std::size_t c = 0; c < p; ++c) {
                    double sum = 0.0;
                    for (std::size_t row = 0; row < n; ++row) {
                        sum += X[row * p + r] * Vinv_X[row * p + c];
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
                                                EstimationMethod method) const {
    std::unordered_map<std::string, double> gradients;
    (void)method;

    if (model.random_effects.empty()) {
        // Pre-process edges for efficient lookup: target -> [(source, param_id)]
        std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> incoming_edges;
        for (const auto& edge : model.edges) {
            if (edge.kind == EdgeKind::Regression && !edge.parameter_id.empty()) {
                incoming_edges[edge.target].emplace_back(edge.source, edge.parameter_id);
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
            
            const std::vector<double>* status_vec = nullptr;
            if (status.count(var.name)) status_vec = &status.at(var.name);

            const std::vector<double>* extra_vec = nullptr;
            if (extra_params.count(var.name)) extra_vec = &extra_params.at(var.name);
            const std::vector<double>& ep = extra_vec ? *extra_vec : std::vector<double>{};

            auto family = OutcomeFamilyFactory::create(var.family);
            
            // Get incoming edges for this variable
            const auto& edges = incoming_edges[var.name];

            for (size_t i = 0; i < obs.size(); ++i) {
                double s = status_vec ? (*status_vec)[i] : 1.0;
                auto eval = family->evaluate(obs[i], preds[i], disps[i], s, ep);
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
                obs_var_name = var.name;
                obs_family_name = var.family;
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
        const auto& disp_data = disp_it->second;
        if (disp_data.size() != n) {
            throw std::runtime_error("Dispersion vector has mismatched size");
        }

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

            std::vector<double> L(n * n);
            cholesky(n, V_full, L);
            std::vector<double> V_inv = invert_from_cholesky(n, L);

            std::vector<double> resid(n);
            for (std::size_t i = 0; i < n; ++i) {
                resid[i] = obs_data[i] - pred_data[i];
            }
            std::vector<double> alpha = solve_cholesky(n, L, resid);

            for (const auto& edge : model.edges) {
                if (edge.kind != EdgeKind::Regression || edge.target != obs_var_name || edge.parameter_id.empty()) {
                    continue;
                }
                if (!data.count(edge.source)) {
                    continue;
                }
                const auto& src_vec = data.at(edge.source);
                double accum = 0.0;
                for (std::size_t i = 0; i < n; ++i) {
                    accum += src_vec[i] * alpha[i];
                }
                gradients[edge.parameter_id] += accum;
            }

            for (std::size_t re_idx = 0; re_idx < random_effect_infos.size(); ++re_idx) {
                const auto& info = random_effect_infos[re_idx];
                if (info.G_gradients.empty()) continue;
                const auto& groups = group_maps[re_idx];
                for (const auto& [_, indices] : groups) {
                    auto Z_i = build_design_matrix(info, indices, data, linear_predictors);
                    std::size_t n_i = indices.size();
                    std::size_t q = info.cov_spec->dimension;

                    std::vector<double> temp(n_i * q, 0.0);
                    for (std::size_t r = 0; r < n_i; ++r) {
                        for (std::size_t c = 0; c < q; ++c) {
                            double sum = 0.0;
                            for (std::size_t k = 0; k < n_i; ++k) {
                                std::size_t row = indices[r];
                                std::size_t col = indices[k];
                                double term = alpha[row] * alpha[col] - V_inv[row * n + col];
                                sum += term * Z_i[k * q + c];
                            }
                            temp[r * q + c] = sum;
                        }
                    }

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

namespace {

class ModelObjective : public ObjectiveFunction {
public:
    ModelObjective(const LikelihoodDriver& driver,
                   const ModelIR& model,
                   const std::unordered_map<std::string, std::vector<double>>& data,
                   const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data = {})
        : driver_(driver), model_(model), data_(data), fixed_covariance_data_(fixed_covariance_data) {
        
        if (!model_.parameters.empty()) {
            for (const auto& param : model_.parameters) {
                std::shared_ptr<const ParameterTransform> transform;
                switch (param.constraint) {
                    case ParameterConstraint::Positive:
                        transform = make_log_transform();
                        break;
                    case ParameterConstraint::Free:
                    default:
                        transform = make_identity_transform();
                        break;
                }
                catalog_.register_parameter(param.id, param.initial_value, std::move(transform));
            }
        } else {
            for (const auto& edge : model_.edges) {
                if (edge.parameter_id.empty()) {
                    continue;
                }
                char* end;
                std::strtod(edge.parameter_id.c_str(), &end);
                if (end != edge.parameter_id.c_str() && *end == '\0') {
                    continue;  // numeric literal
                }
                catalog_.register_parameter(edge.parameter_id, kDefaultCoefficientInit, make_identity_transform());
            }
        }

        // Add covariance parameters
        for (const auto& cov : model_.covariances) {
            auto structure = create_covariance_structure(cov, fixed_covariance_data_);
            size_t count = structure->parameter_count();
            if (count == 0) {
                continue;
            }
            size_t start_idx = catalog_.size();
            auto mask = build_covariance_positive_mask(cov, *structure);
            if (mask.size() < count) {
                mask.resize(count, false);
            }
            for (size_t i = 0; i < count; ++i) {
                std::string param_name = cov.id + "_" + std::to_string(i);
                bool positive = mask[i];
                auto transform = positive ? make_log_transform() : make_identity_transform();
                double init_val = positive ? kDefaultVarianceInit : kDefaultCoefficientInit;
                catalog_.register_parameter(param_name, init_val, transform);
            }
            covariance_param_ranges_[cov.id] = {start_idx, count};
        }
    }

    [[nodiscard]] double value(const std::vector<double>& parameters) const override {
        const auto constrained = to_constrained(parameters);
        std::unordered_map<std::string, std::vector<double>> linear_predictors;
        std::unordered_map<std::string, std::vector<double>> dispersions;
        std::unordered_map<std::string, std::vector<double>> covariance_parameters;

        build_prediction_workspaces(constrained, linear_predictors, dispersions, covariance_parameters);

        return -driver_.evaluate_model_loglik(model_, data_, linear_predictors, dispersions, covariance_parameters, {}, {}, fixed_covariance_data_);
    }

    [[nodiscard]] std::vector<double> gradient(const std::vector<double>& parameters) const override {
        const auto constrained = to_constrained(parameters);
        std::unordered_map<std::string, std::vector<double>> linear_predictors;
        std::unordered_map<std::string, std::vector<double>> dispersions;
        std::unordered_map<std::string, std::vector<double>> covariance_parameters;

        build_prediction_workspaces(constrained, linear_predictors, dispersions, covariance_parameters);

        auto grad_map = driver_.evaluate_model_gradient(
            model_,
            data_,
            linear_predictors,
            dispersions,
            covariance_parameters,
            {},
            {},
            fixed_covariance_data_);
        
        std::vector<double> grad(parameters.size(), 0.0);
        auto chain = catalog_.constrained_derivatives(parameters);
        for (const auto& [param_id, g] : grad_map) {
            auto idx = catalog_.find_index(param_id);
            if (idx != ParameterCatalog::npos) {
                grad[idx] -= g * chain[idx];
            }
        }
        return grad;
    }
    
    const std::vector<std::string>& parameter_names() const { return catalog_.names(); }

    [[nodiscard]] std::vector<double> initial_parameters() const {
        return catalog_.initial_unconstrained();
    }

    [[nodiscard]] std::vector<double> to_constrained(const std::vector<double>& unconstrained) const {
        return catalog_.constrain(unconstrained);
    }

private:
    const LikelihoodDriver& driver_;
    const ModelIR& model_;
    const std::unordered_map<std::string, std::vector<double>>& data_;
    const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data_;
    ParameterCatalog catalog_;
    std::unordered_map<std::string, std::pair<size_t, size_t>> covariance_param_ranges_;

    void build_prediction_workspaces(const std::vector<double>& constrained_parameters,
                                     std::unordered_map<std::string, std::vector<double>>& linear_predictors,
                                     std::unordered_map<std::string, std::vector<double>>& dispersions,
                                     std::unordered_map<std::string, std::vector<double>>& covariance_parameters) const {
        linear_predictors.clear();
        dispersions.clear();
        covariance_parameters.clear();

        std::unordered_set<std::string> response_vars;
        for (const auto& edge : model_.edges) {
            if (edge.kind == EdgeKind::Regression) {
                response_vars.insert(edge.target);
            }
        }

        auto register_response = [&](const std::string& name) {
            if (!data_.contains(name)) return;
            linear_predictors[name] = std::vector<double>(data_.at(name).size(), 0.0);
            dispersions[name] = std::vector<double>(data_.at(name).size(), 1.0);
        };

        if (!response_vars.empty()) {
            for (const auto& name : response_vars) {
                register_response(name);
            }
        } else {
            for (const auto& var : model_.variables) {
                if (var.kind == VariableKind::Observed) {
                    register_response(var.name);
                }
            }
        }

        for (const auto& edge : model_.edges) {
            if (edge.kind != EdgeKind::Regression) continue;

            double weight = 0.0;
            if (!edge.parameter_id.empty()) {
                auto idx = catalog_.find_index(edge.parameter_id);
                if (idx != ParameterCatalog::npos) {
                    weight = constrained_parameters[idx];
                } else {
                    try {
                        weight = std::stod(edge.parameter_id);
                    } catch (...) {
                        weight = 0.0;
                    }
                }
            }

            if (data_.count(edge.source) && linear_predictors.count(edge.target)) {
                const auto& src_data = data_.at(edge.source);
                auto& tgt_lp = linear_predictors.at(edge.target);
                for (size_t i = 0; i < tgt_lp.size(); ++i) {
                    tgt_lp[i] += src_data[i] * weight;
                }
            }
        }

        for (const auto& [id, range] : covariance_param_ranges_) {
            std::vector<double> params;
            params.reserve(range.second);
            for (size_t i = 0; i < range.second; ++i) {
                params.push_back(constrained_parameters[range.first + i]);
            }
            covariance_parameters[id] = std::move(params);
        }
    }
};

} // namespace

OptimizationResult LikelihoodDriver::fit(const ModelIR& model,
                                         const std::unordered_map<std::string, std::vector<double>>& data,
                                         const OptimizationOptions& options,
                                         const std::string& optimizer_name,
                                         const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data) const {
    
    ModelObjective objective(*this, model, data, fixed_covariance_data);
    auto initial_params = objective.initial_parameters();
    
    std::unique_ptr<Optimizer> optimizer;
    if (optimizer_name == "lbfgs") {
        optimizer = make_lbfgs_optimizer();
    } else {
        optimizer = make_gradient_descent_optimizer();
    }
    
    auto result = optimizer->optimize(objective, initial_params, options);
    result.parameters = objective.to_constrained(result.parameters);
    return result;
}

}  // namespace libsemx
