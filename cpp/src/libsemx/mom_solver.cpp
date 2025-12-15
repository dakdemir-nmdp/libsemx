#include "libsemx/mom_solver.hpp"
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <functional>
#include <utility>

namespace libsemx {

MoMSolver::Result MoMSolver::fit(
    const Eigen::VectorXd& y,
    const Eigen::MatrixXd& X,
    const std::vector<size_t>& u_indices,
    const std::vector<size_t>& v_indices,
    const Options& options) {

    const size_t n = y.size();
    const size_t p = X.cols();

    // Validate inputs
    if (static_cast<size_t>(X.rows()) != n) {
        throw std::invalid_argument("X rows must match y size");
    }
    if (u_indices.size() != n) {
        throw std::invalid_argument("u_indices size must match y size");
    }
    if (v_indices.size() != n) {
        throw std::invalid_argument("v_indices size must match y size");
    }

    Result result;

    // Step 1: Compute OLS estimate for β
    Eigen::VectorXd beta_ols;
    try {
        Eigen::MatrixXd XtX = X.transpose() * X;
        Eigen::VectorXd Xty = X.transpose() * y;
        beta_ols = XtX.ldlt().solve(Xty);
    } catch (...) {
        result.beta = Eigen::VectorXd::Zero(p);
        result.variance_components = Eigen::Vector3d::Zero();
        result.converged = false;
        result.message = "OLS failed during initialization";
        return result;
    }

    // Step 2: Compute OLS residuals
    Eigen::VectorXd residuals = y - X * beta_ols;

    // Step 3: Compute sufficient statistics
    SufficientStatistics stats = compute_sufficient_statistics(residuals, u_indices, v_indices);

    if (options.verbose) {
        std::cout << "MoM Sufficient Statistics:" << std::endl;
        std::cout << "  S_0 = " << stats.S_0 << ", N = " << stats.N << std::endl;
        std::cout << "  S_R = " << stats.S_R << ", N_u = " << stats.N_u << std::endl;
        std::cout << "  S_C = " << stats.S_C << ", N_v = " << stats.N_v << std::endl;
        std::cout << "  S_RC = " << stats.S_RC << ", N_uv = " << stats.N_uv << std::endl;
    }

    // Count number of unique groups
    std::unordered_map<size_t, bool> unique_u, unique_v;
    for (size_t i = 0; i < n; ++i) {
        unique_u[u_indices[i]] = true;
        unique_v[v_indices[i]] = true;
    }
    result.n_groups_u = unique_u.size();
    result.n_groups_v = unique_v.size();

    // Step 4: Estimate variance components
    Eigen::Vector3d variances = estimate_variances(stats, options.min_variance);

    if (options.verbose) {
        double T0 = stats.N > 0 ? stats.S_0 / stats.N : 0.0;
        double TR = stats.N_u > 0 ? stats.S_R / static_cast<double>(stats.N_u) : 0.0;
        double TC = stats.N_v > 0 ? stats.S_C / static_cast<double>(stats.N_v) : 0.0;
        double TRC = stats.N_uv > 0 ? stats.S_RC / static_cast<double>(stats.N_uv) : 0.0;
        std::cout << "MoM U-statistics: T0=" << T0 << " TR=" << TR << " TC=" << TC << " TRC=" << TRC << std::endl;
        std::cout << "MoM Initial estimates: "
                  << "σ²_u=" << variances(0) << " "
                  << "σ²_v=" << variances(1) << " "
                  << "σ²_e=" << variances(2) << std::endl;
    }

    // Step 5: Optional GLS refinement for β
    Eigen::VectorXd beta_final = beta_ols;
    if (options.use_gls) {
        try {
            beta_final = compute_gls_beta(y, X, u_indices, v_indices, variances);

            if (options.verbose) {
                std::cout << "MoM: Refined β using GLS" << std::endl;
            }

            // Step 6: Optional second MoM step with GLS residuals
            if (options.second_step) {
                Eigen::VectorXd residuals_gls = y - X * beta_final;
                SufficientStatistics stats_gls = compute_sufficient_statistics(
                    residuals_gls, u_indices, v_indices);
                variances = estimate_variances(stats_gls, options.min_variance);

                if (options.verbose) {
                    std::cout << "MoM Second step estimates: "
                              << "σ²_u=" << variances(0) << " "
                              << "σ²_v=" << variances(1) << " "
                              << "σ²_e=" << variances(2) << std::endl;
                }
            }
        } catch (...) {
            if (options.verbose) {
                std::cout << "MoM: GLS refinement failed, using OLS estimate" << std::endl;
            }
            beta_final = beta_ols;
        }
    }

    result.beta = beta_final;
    result.variance_components = variances;
    result.converged = true;
    result.message = "MoM estimation completed successfully";

    return result;
}

namespace {
struct PairHash {
    std::size_t operator()(const std::pair<std::size_t, std::size_t>& p) const noexcept {
        std::size_t h1 = std::hash<std::size_t>{}(p.first);
        std::size_t h2 = std::hash<std::size_t>{}(p.second);
        // Stronger mix than xor to avoid obvious collisions
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};
}  // namespace

MoMSolver::SufficientStatistics MoMSolver::compute_sufficient_statistics(
    const Eigen::VectorXd& residuals,
    const std::vector<size_t>& u_indices,
    const std::vector<size_t>& v_indices) {

    const size_t N = residuals.size();
    SufficientStatistics stats;
    stats.N = N;

    // S_0 = Σ r_i²
    stats.S_0 = residuals.squaredNorm();

    // Accumulate row (u) statistics
    std::unordered_map<size_t, double> sum_r_row;
    std::unordered_map<size_t, double> sum_r_sq_row;
    std::unordered_map<size_t, size_t> n_row;

    for (size_t i = 0; i < N; ++i) {
        double r_i = residuals(i);
        size_t u_idx = u_indices[i];
        sum_r_row[u_idx] += r_i;
        sum_r_sq_row[u_idx] += r_i * r_i;
        n_row[u_idx]++;
    }

    // S_R = Σ_{i≠j, u_i=u_j} r_i r_j
    //     = Σ_u [(Σ_{i∈u} r_i)² - Σ_{i∈u} r_i²]
    stats.S_R = 0.0;
    stats.N_u = 0;
    for (const auto& kv : sum_r_row) {
        size_t u_idx = kv.first;
        size_t n_u = n_row[u_idx];
        if (n_u > 1) {
            double sum_r = sum_r_row[u_idx];
            double sum_r_sq = sum_r_sq_row[u_idx];
            stats.S_R += sum_r * sum_r - sum_r_sq;
            stats.N_u += n_u * (n_u - 1);
        }
    }

    // Accumulate column (v) statistics
    std::unordered_map<size_t, double> sum_r_col;
    std::unordered_map<size_t, double> sum_r_sq_col;
    std::unordered_map<size_t, size_t> n_col;

    for (size_t i = 0; i < N; ++i) {
        double r_i = residuals(i);
        size_t v_idx = v_indices[i];
        sum_r_col[v_idx] += r_i;
        sum_r_sq_col[v_idx] += r_i * r_i;
        n_col[v_idx]++;
    }

    // S_C = Σ_{i≠j, v_i=v_j} r_i r_j
    stats.S_C = 0.0;
    stats.N_v = 0;
    for (const auto& kv : sum_r_col) {
        size_t v_idx = kv.first;
        size_t n_v = n_col[v_idx];
        if (n_v > 1) {
            double sum_r = sum_r_col[v_idx];
            double sum_r_sq = sum_r_sq_col[v_idx];
            stats.S_C += sum_r * sum_r - sum_r_sq;
            stats.N_v += n_v * (n_v - 1);
        }
    }

    // Accumulate cell (u, v) statistics
    std::unordered_map<std::pair<size_t, size_t>, double, PairHash> sum_r_cell;
    std::unordered_map<std::pair<size_t, size_t>, double, PairHash> sum_r_sq_cell;
    std::unordered_map<std::pair<size_t, size_t>, size_t, PairHash> n_cell;

    for (size_t i = 0; i < N; ++i) {
        double r_i = residuals(i);
        size_t u_idx = u_indices[i];
        size_t v_idx = v_indices[i];
        std::pair<size_t, size_t> cell_key{u_idx, v_idx};

        sum_r_cell[cell_key] += r_i;
        sum_r_sq_cell[cell_key] += r_i * r_i;
        n_cell[cell_key]++;
    }

    // S_RC = Σ_{i≠j, u_i=u_j, v_i=v_j} r_i r_j
    stats.S_RC = 0.0;
    stats.N_uv = 0;
    for (const auto& kv : sum_r_cell) {
        const auto& cell_key = kv.first;
        size_t n_uv = n_cell.at(cell_key);
        if (n_uv > 1) {
            double sum_r = sum_r_cell.at(cell_key);
            double sum_r_sq = sum_r_sq_cell.at(cell_key);
            stats.S_RC += sum_r * sum_r - sum_r_sq;
            stats.N_uv += n_uv * (n_uv - 1);
        }
    }

    return stats;
}

Eigen::Vector3d MoMSolver::estimate_variances(
    const SufficientStatistics& stats,
    double min_variance) {

    // U-statistic estimators
    double T0 = stats.N > 0 ? stats.S_0 / stats.N : 0.0;
    double TR = stats.N_u > 0 ? stats.S_R / stats.N_u : 0.0;
    double TC = stats.N_v > 0 ? stats.S_C / stats.N_v : 0.0;
    double TRC = stats.N_uv > 0 ? stats.S_RC / stats.N_uv : 0.0;

    // Variance component estimates
    double sigma_u_sq = TR - TRC;
    double sigma_v_sq = TC - TRC;
    double sigma_e_sq = T0 - TR - TC + TRC;

    // Enforce non-negativity
    sigma_u_sq = std::max(sigma_u_sq, min_variance);
    sigma_v_sq = std::max(sigma_v_sq, min_variance);
    sigma_e_sq = std::max(sigma_e_sq, min_variance);

    Eigen::Vector3d variances;
    variances << sigma_u_sq, sigma_v_sq, sigma_e_sq;
    return variances;
}

Eigen::VectorXd MoMSolver::compute_gls_beta(
    const Eigen::VectorXd& y,
    const Eigen::MatrixXd& X,
    const std::vector<size_t>& u_indices,
    const std::vector<size_t>& v_indices,
    const Eigen::Vector3d& variances) {

    const size_t n = y.size();
    const double sigma_u_sq = variances(0);
    const double sigma_v_sq = variances(1);
    const double sigma_e_sq = variances(2);

    // Find maximum group indices
    size_t n_groups_u = 0;
    size_t n_groups_v = 0;
    for (size_t i = 0; i < n; ++i) {
        n_groups_u = std::max(n_groups_u, u_indices[i] + 1);
        n_groups_v = std::max(n_groups_v, v_indices[i] + 1);
    }

    // Construct design matrices Z and W
    Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(n, n_groups_u);
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(n, n_groups_v);

    for (size_t i = 0; i < n; ++i) {
        Z(i, u_indices[i]) = 1.0;
        W(i, v_indices[i]) = 1.0;
    }

    // Construct V = σ²_u Z Z^T + σ²_v W W^T + σ²_e I
    Eigen::MatrixXd V = sigma_u_sq * (Z * Z.transpose()) +
                        sigma_v_sq * (W * W.transpose()) +
                        sigma_e_sq * Eigen::MatrixXd::Identity(n, n);

    // Compute V^{-1}
    Eigen::MatrixXd V_inv = V.inverse();

    // Compute GLS estimate: β = (X^T V^{-1} X)^{-1} X^T V^{-1} y
    Eigen::MatrixXd XtVinvX = X.transpose() * V_inv * X;
    Eigen::VectorXd XtVinvy = X.transpose() * V_inv * y;

    return XtVinvX.ldlt().solve(XtVinvy);
}

} // namespace libsemx
