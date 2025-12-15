#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <unordered_map>

namespace libsemx {

/**
 * Method of Moments (MoM) solver for linear mixed models with crossed random effects.
 *
 * Implements fast O(N) variance component estimation based on Gao & Owen (2017/2018).
 * Uses U-statistics to estimate variance components without iteration, making it ideal
 * for very large-scale problems where iterative methods like AI or EM are too slow.
 *
 * Algorithm:
 *   1. Compute OLS estimate for fixed effects β
 *   2. Compute residuals r = y - Xβ
 *   3. Accumulate sufficient statistics in O(N) time:
 *      - S_0 = Σ r_i²
 *      - S_R = Σ_{i≠j, u_i=u_j} r_i r_j  (within first random effect)
 *      - S_C = Σ_{i≠j, v_i=v_j} r_i r_j  (within second random effect)
 *      - S_RC = Σ_{i≠j, u_i=u_j, v_i=v_j} r_i r_j  (within cells)
 *   4. Estimate variances using U-statistics:
 *      - T0 = S_0 / N
 *      - TR = S_R / N_u,  TC = S_C / N_v,  TRC = S_RC / N_uv
 *      - σ²_u = TR - TRC
 *      - σ²_v = TC - TRC
 *      - σ²_e = T0 - TR - TC + TRC
 *
 * Reference:
 *   Gao & Owen (2017). "Efficient moment calculations for variance components"
 */
class MoMSolver {
public:
    struct Options {
        bool use_gls;         // Refine β using GLS (computationally expensive)
        bool second_step;     // Apply second MoM step using GLS residuals
        bool verbose;         // Print diagnostic information
        double min_variance;  // Minimum allowed variance (floor)

        Options()
            : use_gls(false),
              second_step(false),
              verbose(false),
              min_variance(1e-10) {}
    };

    struct Result {
        Eigen::VectorXd beta;              // Fixed effects estimates
        Eigen::VectorXd variance_components; // [σ²_u, σ²_v, σ²_e]
        size_t n_groups_u;                 // Number of levels in first factor
        size_t n_groups_v;                 // Number of levels in second factor
        bool converged;                    // Always true for MoM (non-iterative)
        std::string message;               // Status message
    };

    /**
     * Fits a linear mixed model with two crossed random effects using Method of Moments.
     *
     * Model: y = Xβ + Z_u u + Z_v v + e
     *        where u ~ N(0, σ²_u I), v ~ N(0, σ²_v I), e ~ N(0, σ²_e I)
     *
     * The design matrices Z_u and Z_v are specified implicitly through group indices.
     *
     * @param y Response vector (n × 1)
     * @param X Fixed effects design matrix (n × p)
     * @param u_indices Group indices for first random effect (n × 1), values in [0, n_u-1]
     * @param v_indices Group indices for second random effect (n × 1), values in [0, n_v-1]
     * @param options Solver options
     * @return Result containing β and variance component estimates
     */
    static Result fit(
        const Eigen::VectorXd& y,
        const Eigen::MatrixXd& X,
        const std::vector<size_t>& u_indices,
        const std::vector<size_t>& v_indices,
        const Options& options = Options());

private:
    struct SufficientStatistics {
        double S_0;      // Σ r_i²
        double S_R;      // Σ_{i≠j, u_i=u_j} r_i r_j
        double S_C;      // Σ_{i≠j, v_i=v_j} r_i r_j
        double S_RC;     // Σ_{i≠j, u_i=u_j, v_i=v_j} r_i r_j
        size_t N_u;      // Σ_u n_u(n_u - 1)
        size_t N_v;      // Σ_v n_v(n_v - 1)
        size_t N_uv;     // Σ_{u,v} n_{uv}(n_{uv} - 1)
        size_t N;        // Total observations
    };

    /**
     * Computes sufficient statistics for MoM estimation in O(N) time.
     *
     * Uses hash maps to accumulate sums within groups efficiently without
     * explicitly constructing the potentially large design matrices.
     */
    static SufficientStatistics compute_sufficient_statistics(
        const Eigen::VectorXd& residuals,
        const std::vector<size_t>& u_indices,
        const std::vector<size_t>& v_indices);

    /**
     * Estimates variance components from sufficient statistics using U-statistics.
     *
     * Returns [σ²_u, σ²_v, σ²_e] with non-negativity enforced.
     */
    static Eigen::Vector3d estimate_variances(
        const SufficientStatistics& stats,
        double min_variance);

    /**
     * Computes GLS estimate of β given variance components.
     *
     * Constructs V = σ²_u Z_u Z_u^T + σ²_v Z_v Z_v^T + σ²_e I and solves:
     *   β = (X^T V^{-1} X)^{-1} X^T V^{-1} y
     *
     * This is computationally expensive for large N and should only be used
     * when options.use_gls is true.
     */
    static Eigen::VectorXd compute_gls_beta(
        const Eigen::VectorXd& y,
        const Eigen::MatrixXd& X,
        const std::vector<size_t>& u_indices,
        const std::vector<size_t>& v_indices,
        const Eigen::Vector3d& variances);
};

} // namespace libsemx
