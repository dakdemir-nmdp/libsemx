#pragma once

#include <Eigen/Dense>
#include <vector>
#include <functional>
#include <string>

namespace libsemx {

/**
 * Average Information (AI) optimizer for variance component estimation in mixed models.
 *
 * The AI algorithm uses the expected Fisher information matrix (Average Information matrix)
 * instead of the observed Hessian, providing robust second-order convergence for variance
 * components while avoiding the computational cost of calculating the full observed Hessian.
 *
 * Algorithm:
 *   1. Compute Score vector: S_k = 0.5 * (y^T M V_k M y - tr(M V_k))
 *   2. Compute AI matrix: AI_{k,l} = 0.5 * tr(V_k M V_l M)
 *   3. Update: δ = AI^{-1} @ Score
 *   4. Apply with step-halving: σ²_new = σ²_old + step * δ
 *
 * where M = P (projection matrix) for REML or M = V^{-1} for ML.
 *
 * Reference:
 *   Johnson & Thompson (1995). "Average information REML"
 *   Gilmour et al. (1995). "Average information algorithm"
 */
class AverageInformationOptimizer {
public:
    struct Options {
        size_t max_iterations;
        double tolerance;
        double min_variance;      // Minimum allowed variance
        double initial_step_size;    // Initial step size for updates
        double min_step_size;       // Minimum step size before giving up
        bool use_reml;              // Use REML (true) or ML (false)
        bool verbose;              // Print iteration info

        // Constructor with default values
        Options()
            : max_iterations(100),
              tolerance(1e-6),
              min_variance(1e-10),
              initial_step_size(1.0),
              min_step_size(1e-4),
              use_reml(true),
              verbose(false) {}
    };

    struct Result {
        Eigen::VectorXd parameters;        // Final variance components
        double log_likelihood;              // Final log-likelihood value
        size_t iterations;                  // Number of iterations performed
        bool converged;                     // Whether the algorithm converged
        std::string message;                // Status message
        Eigen::VectorXd final_score;       // Final score vector (for diagnostics)
        Eigen::MatrixXd final_ai_matrix;   // Final AI matrix (for std errors)
    };

    /**
     * Optimizes variance components for a linear mixed model.
     *
     * Model: y ~ N(X β, V) where V = Σ_k σ²_k V_k
     *
     * @param y Response vector (n × 1)
     * @param X Fixed effects design matrix (n × p)
     * @param V_matrices List of variance component matrices [V_0, V_1, ..., V_K]
     *                   The last matrix (V_K) should be the residual identity matrix
     * @param initial_params Initial variance component values (K+1 × 1)
     * @param fixed_components Boolean vector indicating which components are fixed (K+1 × 1)
     * @param options Optimization options
     * @return Optimization result with final variance components
     */
    static Result optimize(
        const Eigen::VectorXd& y,
        const Eigen::MatrixXd& X,
        const std::vector<Eigen::MatrixXd>& V_matrices,
        const Eigen::VectorXd& initial_params,
        const std::vector<bool>& fixed_components = {},
        const Options& options = Options());

private:
    /**
     * Computes the GLS estimate of β: β = (X^T V^{-1} X)^{-1} X^T V^{-1} y
     */
    static Eigen::VectorXd compute_beta(
        const Eigen::VectorXd& y,
        const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& V_inv);

    /**
     * Computes the projection matrix P = V^{-1} - V^{-1} X (X^T V^{-1} X)^{-1} X^T V^{-1}
     * Used for REML estimation.
     */
    static Eigen::MatrixXd compute_projection_matrix(
        const Eigen::MatrixXd& V_inv,
        const Eigen::MatrixXd& X);

    /**
     * Computes the score vector for variance components.
     * Score_k = 0.5 * (y^T M V_k M y - tr(M V_k))
     */
    static Eigen::VectorXd compute_score(
        const Eigen::VectorXd& y,
        const Eigen::MatrixXd& M,
        const std::vector<Eigen::MatrixXd>& V_matrices,
        const std::vector<size_t>& est_indices);

    /**
     * Computes the Average Information matrix.
     * AI_{k,l} = 0.5 * tr(V_k M V_l M)
     */
    static Eigen::MatrixXd compute_ai_matrix(
        const Eigen::MatrixXd& M,
        const std::vector<Eigen::MatrixXd>& V_matrices,
        const std::vector<size_t>& est_indices);

    /**
     * Computes REML log-likelihood: -0.5 * (log|V| + log|X^T V^{-1} X| + y^T P y)
     */
    static double compute_reml_loglik(
        const Eigen::VectorXd& y,
        const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXd& V_inv);

    /**
     * Computes ML log-likelihood: -0.5 * (log|V| + (y - Xβ)^T V^{-1} (y - Xβ))
     */
    static double compute_ml_loglik(
        const Eigen::VectorXd& y,
        const Eigen::MatrixXd& X,
        const Eigen::VectorXd& beta,
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXd& V_inv);

    /**
     * Computes log-determinant of a matrix using Cholesky decomposition.
     */
    static double compute_log_det(const Eigen::MatrixXd& M);
};

} // namespace libsemx
