#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace libsemx {

/**
 * Multivariate Spectral Decomposition Likelihood Evaluator for Dense Kernel Models.
 *
 * Implements efficient likelihood evaluation for multivariate mixed models:
 *   vec(Y) = (I_d ⊗ X)vec(B) + vec(U) + vec(E)
 *   Cov[vec(Y)] = G ⊗ K + R ⊗ I_n
 *
 * where:
 *   - Y is n × d (n observations, d traits)
 *   - K is n × n kernel matrix (e.g., genomic relationship matrix)
 *   - G is d × d genetic covariance matrix
 *   - R is d × d residual covariance matrix
 *   - X is n × p fixed effects design matrix
 *   - B is p × d fixed effects coefficients
 *
 * The key optimization is eigendecomposing K = Q Λ Q^T once, which transforms
 * the problem into n independent d×d blocks:
 *   Y* = Q^T Y  (transform data)
 *   X* = Q^T X  (transform design)
 *   Cov[Y*_i,:] = λ_i G + R  (independent across i)
 *
 * This reduces O(n³d³) operations to O(n d³) per optimization iteration.
 *
 * Reference: Zhou & Stephens (2014), "Efficient multivariate linear mixed model"
 *            Gilmour et al. (1995), "Average information REML"
 */
class MultivariateSpectralLikelihoodEvaluator {
public:
    /**
     * Constructs evaluator from precomputed kernel matrix K.
     * Performs eigendecomposition: K = Q Λ Q^T
     *
     * @param kernel Symmetric kernel matrix (n × n)
     * @param n_traits Number of traits (dimension d)
     * @param min_eigenvalue Minimum eigenvalue to retain (regularization threshold)
     */
    explicit MultivariateSpectralLikelihoodEvaluator(const Eigen::MatrixXd& kernel,
                                                       size_t n_traits,
                                                       double min_eigenvalue = 1e-10);

    /**
     * Evaluates Gaussian log-likelihood for multivariate model:
     *   vec(Y) ~ N((I_d ⊗ X)vec(B), G ⊗ K + R ⊗ I_n)
     *
     * @param Y Response matrix (n × d)
     * @param X Fixed effects design matrix (n × p)
     * @param G Genetic covariance matrix (d × d)
     * @param R Residual covariance matrix (d × d)
     * @param use_reml If true, compute REML likelihood; otherwise ML
     * @return Log-likelihood value (negative for minimization)
     */
    [[nodiscard]] double evaluate_loglik(const Eigen::MatrixXd& Y,
                                          const Eigen::MatrixXd& X,
                                          const Eigen::MatrixXd& G,
                                          const Eigen::MatrixXd& R,
                                          bool use_reml = false) const;

    /**
     * Evaluates gradient of log-likelihood w.r.t. covariance matrices.
     *
     * Returns gradients as flattened vectors for optimization:
     *   - grad_G: vectorized gradient w.r.t. G (d² × 1)
     *   - grad_R: vectorized gradient w.r.t. R (d² × 1)
     *
     * @param Y Response matrix (n × d)
     * @param X Fixed effects design matrix (n × p)
     * @param G Genetic covariance matrix (d × d)
     * @param R Residual covariance matrix (d × d)
     * @param use_reml If true, compute REML gradient; otherwise ML
     * @return Pair of (grad_G_vec, grad_R_vec)
     */
    [[nodiscard]] std::pair<Eigen::VectorXd, Eigen::VectorXd> evaluate_gradient(
        const Eigen::MatrixXd& Y,
        const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& G,
        const Eigen::MatrixXd& R,
        bool use_reml = false) const;

    /**
     * Computes Best Linear Unbiased Predictor (BLUP) for random effects.
     * U = G K V^{-1} (Y - XB)
     *
     * where V = G ⊗ K + R ⊗ I_n (in Kronecker product notation)
     *
     * @param Y Response matrix (n × d)
     * @param X Fixed effects design matrix (n × p)
     * @param G Genetic covariance matrix (d × d)
     * @param R Residual covariance matrix (d × d)
     * @return BLUP matrix (n × d)
     */
    [[nodiscard]] Eigen::MatrixXd compute_blup(const Eigen::MatrixXd& Y,
                                                 const Eigen::MatrixXd& X,
                                                 const Eigen::MatrixXd& G,
                                                 const Eigen::MatrixXd& R) const;

    /**
     * Returns eigenvalues of kernel matrix K.
     */
    [[nodiscard]] const Eigen::VectorXd& eigenvalues() const { return lambda_; }

    /**
     * Returns eigenvectors of kernel matrix K.
     */
    [[nodiscard]] const Eigen::MatrixXd& eigenvectors() const { return Q_; }

    /**
     * Returns number of observations.
     */
    [[nodiscard]] size_t n_obs() const { return n_; }

    /**
     * Returns number of traits.
     */
    [[nodiscard]] size_t n_traits() const { return d_; }

private:
    size_t n_;                   // Number of observations
    size_t d_;                   // Number of traits
    Eigen::MatrixXd Q_;          // Eigenvectors of K (n × n)
    Eigen::VectorXd lambda_;     // Eigenvalues of K (n × 1)
    double min_eigenvalue_;      // Minimum eigenvalue threshold

    /**
     * Computes GLS estimate B = (X^T V^{-1} X)^{-1} X^T V^{-1} Y
     * using spectral decomposition and Kronecker structure.
     *
     * @return Fixed effects matrix (p × d)
     */
    [[nodiscard]] Eigen::MatrixXd compute_beta(const Eigen::MatrixXd& Y,
                                                 const Eigen::MatrixXd& X,
                                                 const Eigen::MatrixXd& G,
                                                 const Eigen::MatrixXd& R) const;

    /**
     * Computes log-determinant of covariance matrix using eigenvalues:
     *   log|V| = log|G ⊗ K + R ⊗ I| = Σ_i log|λ_i G + R|
     */
    [[nodiscard]] double compute_logdet(const Eigen::MatrixXd& G,
                                         const Eigen::MatrixXd& R) const;

    /**
     * Computes matrix square root using eigendecomposition (for MM algorithm updates).
     * For positive definite M, returns sqrt(M) such that sqrt(M) @ sqrt(M) = M
     */
    [[nodiscard]] Eigen::MatrixXd compute_sqrtm(const Eigen::MatrixXd& M) const;

    /**
     * Validates that a matrix is positive definite.
     */
    [[nodiscard]] bool is_positive_definite(const Eigen::MatrixXd& M) const;
};

} // namespace libsemx
