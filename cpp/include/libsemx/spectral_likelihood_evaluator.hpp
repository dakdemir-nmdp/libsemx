#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace libsemx {

/**
 * Spectral Decomposition Likelihood Evaluator for Dense Kernel Models.
 *
 * Implements O(n) likelihood evaluation for models of the form:
 *   V = σ²_g K + σ²_e I
 *
 * where K is a precomputed kernel matrix (e.g., genomic relationship matrix).
 *
 * The key optimization is to eigendecompose K = U Λ U^T once, then:
 *   V = U(σ²_g Λ + σ²_e I)U^T
 *   V^{-1} = U(σ²_g Λ + σ²_e I)^{-1}U^T
 *   log|V| = log|σ²_g Λ + σ²_e I|
 *
 * This reduces O(n³) operations to O(n) per optimization iteration.
 *
 * Reference: Kang et al. (2008), Gilmour et al. (1995), Meyer (1989)
 * "Efficient Control of Population Structure in Model Organism Association Mapping"
 */
class SpectralLikelihoodEvaluator {
public:
    /**
     * Constructs evaluator from precomputed kernel matrix K.
     * Performs eigendecomposition: K = U Λ U^T
     *
     * @param kernel Symmetric kernel matrix (n x n)
     * @param min_eigenvalue Minimum eigenvalue to retain (regularization threshold)
     */
    explicit SpectralLikelihoodEvaluator(const Eigen::MatrixXd& kernel,
                                          double min_eigenvalue = 1e-10);

    /**
     * Evaluates Gaussian log-likelihood for model: y ~ N(Xβ, V)
     * where V = σ²_g K + σ²_e I
     *
     * @param y Response vector (n x 1)
     * @param X Fixed effects design matrix (n x p)
     * @param sigma_g_sq Genetic variance component
     * @param sigma_e_sq Residual variance component
     * @param use_reml If true, compute REML likelihood; otherwise ML
     * @return Log-likelihood value (negative for minimization)
     */
    [[nodiscard]] double evaluate_loglik(const Eigen::VectorXd& y,
                                          const Eigen::MatrixXd& X,
                                          double sigma_g_sq,
                                          double sigma_e_sq,
                                          bool use_reml = false) const;

    /**
     * Evaluates gradient of log-likelihood w.r.t. variance components.
     *
     * @param y Response vector (n x 1)
     * @param X Fixed effects design matrix (n x p)
     * @param sigma_g_sq Genetic variance component
     * @param sigma_e_sq Residual variance component
     * @param use_reml If true, compute REML gradient; otherwise ML
     * @return Gradient vector [d/dσ²_g, d/dσ²_e]
     */
    [[nodiscard]] Eigen::Vector2d evaluate_gradient(const Eigen::VectorXd& y,
                                                      const Eigen::MatrixXd& X,
                                                      double sigma_g_sq,
                                                      double sigma_e_sq,
                                                      bool use_reml = false) const;

    /**
     * Computes Best Linear Unbiased Predictor (BLUP) for random effects.
     * u = σ²_g K V^{-1} (y - Xβ)
     *
     * @param y Response vector (n x 1)
     * @param X Fixed effects design matrix (n x p)
     * @param sigma_g_sq Genetic variance component
     * @param sigma_e_sq Residual variance component
     * @return BLUP vector (n x 1)
     */
    [[nodiscard]] Eigen::VectorXd compute_blup(const Eigen::VectorXd& y,
                                                 const Eigen::MatrixXd& X,
                                                 double sigma_g_sq,
                                                 double sigma_e_sq) const;

    /**
     * Returns eigenvalues of kernel matrix K.
     */
    [[nodiscard]] const Eigen::VectorXd& eigenvalues() const { return lambda_; }

    /**
     * Returns eigenvectors of kernel matrix K.
     */
    [[nodiscard]] const Eigen::MatrixXd& eigenvectors() const { return U_; }

    /**
     * Returns number of observations.
     */
    [[nodiscard]] size_t size() const { return n_; }

private:
    size_t n_;                   // Number of observations
    Eigen::MatrixXd U_;          // Eigenvectors of K (n x n)
    Eigen::VectorXd lambda_;     // Eigenvalues of K (n x 1)
    double min_eigenvalue_;      // Minimum eigenvalue threshold

    /**
     * Computes GLS estimate β = (X^T V^{-1} X)^{-1} X^T V^{-1} y
     * using spectral decomposition.
     */
    [[nodiscard]] Eigen::VectorXd compute_beta(const Eigen::VectorXd& y,
                                                 const Eigen::MatrixXd& X,
                                                 double sigma_g_sq,
                                                 double sigma_e_sq) const;

    /**
     * Computes log-determinant of V = σ²_g K + σ²_e I using eigenvalues.
     */
    [[nodiscard]] double compute_logdet(double sigma_g_sq, double sigma_e_sq) const;

    /**
     * Computes V^{-1} y using spectral decomposition.
     */
    [[nodiscard]] Eigen::VectorXd compute_V_inv_y(const Eigen::VectorXd& y,
                                                    double sigma_g_sq,
                                                    double sigma_e_sq) const;
};

} // namespace libsemx
