#include "libsemx/multivariate_spectral_likelihood_evaluator.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace libsemx {

MultivariateSpectralLikelihoodEvaluator::MultivariateSpectralLikelihoodEvaluator(
    const Eigen::MatrixXd& kernel,
    size_t n_traits,
    double min_eigenvalue)
    : n_(kernel.rows()), d_(n_traits), min_eigenvalue_(min_eigenvalue) {

    if (kernel.rows() != kernel.cols()) {
        throw std::invalid_argument("Kernel matrix must be square");
    }
    if (n_traits == 0) {
        throw std::invalid_argument("Number of traits must be positive");
    }

    // Perform eigendecomposition of K = Q Λ Q^T
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(kernel);

    if (eigen_solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigendecomposition failed for kernel matrix");
    }

    // Store eigenvalues and eigenvectors
    lambda_ = eigen_solver.eigenvalues();
    Q_ = eigen_solver.eigenvectors();

    // Apply minimum eigenvalue threshold for numerical stability
    for (size_t i = 0; i < n_; ++i) {
        if (lambda_(i) < min_eigenvalue_) {
            lambda_(i) = min_eigenvalue_;
        }
    }
}

double MultivariateSpectralLikelihoodEvaluator::evaluate_loglik(
    const Eigen::MatrixXd& Y,
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& G,
    const Eigen::MatrixXd& R,
    bool use_reml) const {

    // Validate inputs
    if (Y.rows() != static_cast<long>(n_) || Y.cols() != static_cast<long>(d_)) {
        throw std::invalid_argument("Response matrix Y size mismatch");
    }
    if (X.rows() != static_cast<long>(n_)) {
        throw std::invalid_argument("Design matrix X row count mismatch");
    }
    if (G.rows() != static_cast<long>(d_) || G.cols() != static_cast<long>(d_)) {
        throw std::invalid_argument("Genetic covariance matrix G size mismatch");
    }
    if (R.rows() != static_cast<long>(d_) || R.cols() != static_cast<long>(d_)) {
        throw std::invalid_argument("Residual covariance matrix R size mismatch");
    }
    if (!is_positive_definite(G)) {
        throw std::invalid_argument("Genetic covariance matrix G must be positive definite");
    }
    if (!is_positive_definite(R)) {
        throw std::invalid_argument("Residual covariance matrix R must be positive definite");
    }

    const size_t p = X.cols();

    // Compute GLS estimate of B (p × d)
    const Eigen::MatrixXd B = compute_beta(Y, X, G, R);

    // Compute residuals: Resid = Y - X B  (n × d)
    const Eigen::MatrixXd Resid = Y - X * B;

    // Transform residuals: Resid_star = Q^T Resid  (n × d)
    const Eigen::MatrixXd Resid_star = Q_.transpose() * Resid;

    // Compute log-determinant of V
    const double logdet_V = compute_logdet(G, R);

    // Compute quadratic form: tr[Resid^T V^{-1} Resid]
    // Using eigendecomposition: sum_i tr[Resid_star[i,:]^T (λ_i G + R)^{-1} Resid_star[i,:]]
    double quad_form = 0.0;
    for (size_t i = 0; i < n_; ++i) {
        // V_i = λ_i G + R  (d × d)
        const Eigen::MatrixXd V_i = lambda_(i) * G + R;

        // Solve V_i^{-1} Resid_star[i,:]
        const Eigen::VectorXd r_i = Resid_star.row(i).transpose();
        const Eigen::VectorXd V_i_inv_r_i = V_i.ldlt().solve(r_i);

        // Add tr[r_i^T V_i^{-1} r_i] = r_i^T V_i^{-1} r_i
        quad_form += r_i.dot(V_i_inv_r_i);
    }

    // ML log-likelihood: -0.5 * [log|V| + tr(Resid^T V^{-1} Resid) + nd log(2π)]
    double loglik = -0.5 * (logdet_V + quad_form + n_ * d_ * std::log(2.0 * M_PI));

    // REML correction: -0.5 * log|X^T V^{-1} X|
    if (use_reml && p > 0) {
        // Transform X: X_star = Q^T X  (n × p)
        const Eigen::MatrixXd X_star = Q_.transpose() * X;

        // Compute X^T V^{-1} X using block structure
        // X^T V^{-1} X = sum_i X_star[i,:]^T ⊗ (λ_i G + R)^{-1} X_star[i,:]
        // This is a (pd × pd) matrix, but we can compute it as p×p blocks of d×d

        Eigen::MatrixXd XtVinvX = Eigen::MatrixXd::Zero(p * d_, p * d_);

        for (size_t i = 0; i < n_; ++i) {
            const Eigen::MatrixXd V_i = lambda_(i) * G + R;
            const Eigen::MatrixXd V_i_inv = V_i.inverse();

            // Kronecker product: X_star.row(i).transpose() * X_star.row(i) ⊗ V_i_inv
            const Eigen::VectorXd x_i = X_star.row(i).transpose();
            const Eigen::MatrixXd x_i_outer = x_i * x_i.transpose();

            // Add Kronecker product contribution
            for (size_t j1 = 0; j1 < p; ++j1) {
                for (size_t j2 = 0; j2 < p; ++j2) {
                    XtVinvX.block(j1 * d_, j2 * d_, d_, d_) += x_i_outer(j1, j2) * V_i_inv;
                }
            }
        }

        // Compute log-determinant using Cholesky decomposition
        Eigen::LLT<Eigen::MatrixXd> llt(XtVinvX);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("X^T V^{-1} X is not positive definite");
        }

        double logdet_XtVinvX = 0.0;
        for (size_t i = 0; i < p * d_; ++i) {
            logdet_XtVinvX += std::log(llt.matrixL()(i, i));
        }
        logdet_XtVinvX *= 2.0; // Determinant = product of diagonal^2

        loglik -= 0.5 * logdet_XtVinvX;
    }

    return loglik;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> MultivariateSpectralLikelihoodEvaluator::evaluate_gradient(
    const Eigen::MatrixXd& Y,
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& G,
    const Eigen::MatrixXd& R,
    bool use_reml) const {

    // Validate inputs
    if (Y.rows() != static_cast<long>(n_) || Y.cols() != static_cast<long>(d_)) {
        throw std::invalid_argument("Response matrix Y size mismatch");
    }
    if (X.rows() != static_cast<long>(n_)) {
        throw std::invalid_argument("Design matrix X row count mismatch");
    }
    if (G.rows() != static_cast<long>(d_) || G.cols() != static_cast<long>(d_)) {
        throw std::invalid_argument("Genetic covariance matrix G size mismatch");
    }
    if (R.rows() != static_cast<long>(d_) || R.cols() != static_cast<long>(d_)) {
        throw std::invalid_argument("Residual covariance matrix R size mismatch");
    }

    const size_t p = X.cols();

    // Compute GLS estimate of B (p × d)
    const Eigen::MatrixXd B = compute_beta(Y, X, G, R);

    // Compute residuals: Resid = Y - X B  (n × d)
    const Eigen::MatrixXd Resid = Y - X * B;

    // Transform residuals: Resid_star = Q^T Resid  (n × d)
    const Eigen::MatrixXd Resid_star = Q_.transpose() * Resid;

    // Initialize gradient matrices
    Eigen::MatrixXd grad_G = Eigen::MatrixXd::Zero(d_, d_);
    Eigen::MatrixXd grad_R = Eigen::MatrixXd::Zero(d_, d_);

    // Compute gradient contributions from log-determinant and quadratic form
    // d/dG log|V| = sum_i λ_i (λ_i G + R)^{-1}
    // d/dR log|V| = sum_i (λ_i G + R)^{-1}
    // d/dG tr[Resid^T V^{-1} Resid] = -sum_i λ_i (λ_i G + R)^{-1} Resid_star[i,:] Resid_star[i,:]^T (λ_i G + R)^{-1}
    // d/dR tr[Resid^T V^{-1} Resid] = -sum_i (λ_i G + R)^{-1} Resid_star[i,:] Resid_star[i,:]^T (λ_i G + R)^{-1}

    for (size_t i = 0; i < n_; ++i) {
        const Eigen::MatrixXd V_i = lambda_(i) * G + R;
        const Eigen::MatrixXd V_i_inv = V_i.inverse();

        const Eigen::VectorXd r_i = Resid_star.row(i).transpose();
        const Eigen::MatrixXd r_i_outer = r_i * r_i.transpose();

        // Gradient of log-determinant
        grad_G += lambda_(i) * V_i_inv;
        grad_R += V_i_inv;

        // Gradient of quadratic form (note the negative sign)
        const Eigen::MatrixXd V_i_inv_r_outer_V_i_inv = V_i_inv * r_i_outer * V_i_inv;
        grad_G -= lambda_(i) * V_i_inv_r_outer_V_i_inv;
        grad_R -= V_i_inv_r_outer_V_i_inv;
    }

    // Apply factor of -0.5 from log-likelihood
    grad_G *= -0.5;
    grad_R *= -0.5;

    // REML correction: gradient of -0.5 * log|X^T V^{-1} X|
    if (use_reml && p > 0) {
        const Eigen::MatrixXd X_star = Q_.transpose() * X;

        // Compute X^T V^{-1} X
        Eigen::MatrixXd XtVinvX = Eigen::MatrixXd::Zero(p * d_, p * d_);

        for (size_t i = 0; i < n_; ++i) {
            const Eigen::MatrixXd V_i = lambda_(i) * G + R;
            const Eigen::MatrixXd V_i_inv = V_i.inverse();

            const Eigen::VectorXd x_i = X_star.row(i).transpose();
            const Eigen::MatrixXd x_i_outer = x_i * x_i.transpose();

            for (size_t j1 = 0; j1 < p; ++j1) {
                for (size_t j2 = 0; j2 < p; ++j2) {
                    XtVinvX.block(j1 * d_, j2 * d_, d_, d_) += x_i_outer(j1, j2) * V_i_inv;
                }
            }
        }

        const Eigen::MatrixXd XtVinvX_inv = XtVinvX.inverse();

        // Compute REML gradient correction
        Eigen::MatrixXd reml_grad_G = Eigen::MatrixXd::Zero(d_, d_);
        Eigen::MatrixXd reml_grad_R = Eigen::MatrixXd::Zero(d_, d_);

        for (size_t i = 0; i < n_; ++i) {
            const Eigen::MatrixXd V_i = lambda_(i) * G + R;
            const Eigen::MatrixXd V_i_inv = V_i.inverse();

            const Eigen::VectorXd x_i = X_star.row(i).transpose();

            // Compute contribution: tr[(X^T V^{-1} X)^{-1} (x_i x_i^T ⊗ V_i_inv) V_i_inv_deriv]
            // This is complex; simplified version sums block traces
            for (size_t j1 = 0; j1 < p; ++j1) {
                for (size_t j2 = 0; j2 < p; ++j2) {
                    const Eigen::MatrixXd block = XtVinvX_inv.block(j1 * d_, j2 * d_, d_, d_);
                    const double x_contrib = x_i(j1) * x_i(j2);

                    const Eigen::MatrixXd temp = block * V_i_inv;
                    reml_grad_G += lambda_(i) * x_contrib * temp * V_i_inv;
                    reml_grad_R += x_contrib * temp * V_i_inv;
                }
            }
        }

        // Apply factor of 0.5 (note: we add because REML has negative log-determinant)
        grad_G += 0.5 * reml_grad_G;
        grad_R += 0.5 * reml_grad_R;
    }

    // Flatten gradients to vectors (column-major order)
    Eigen::VectorXd grad_G_vec = Eigen::Map<Eigen::VectorXd>(grad_G.data(), d_ * d_);
    Eigen::VectorXd grad_R_vec = Eigen::Map<Eigen::VectorXd>(grad_R.data(), d_ * d_);

    return std::make_pair(grad_G_vec, grad_R_vec);
}

Eigen::MatrixXd MultivariateSpectralLikelihoodEvaluator::compute_blup(
    const Eigen::MatrixXd& Y,
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& G,
    const Eigen::MatrixXd& R) const {

    if (Y.rows() != static_cast<long>(n_) || Y.cols() != static_cast<long>(d_)) {
        throw std::invalid_argument("Response matrix Y size mismatch");
    }
    if (X.rows() != static_cast<long>(n_)) {
        throw std::invalid_argument("Design matrix X row count mismatch");
    }

    // Compute GLS estimate of B (p × d)
    const Eigen::MatrixXd B = compute_beta(Y, X, G, R);

    // Compute residuals: Resid = Y - X B  (n × d)
    const Eigen::MatrixXd Resid = Y - X * B;

    // BLUP: U = G K V^{-1} Resid (conceptually, using Kronecker structure)
    // Using eigendecomposition:
    //   U = Q [λ_i G (λ_i G + R)^{-1} Resid_star[i,:]]_{i=1}^n Q^T
    //     = Q U_star

    // Transform residuals: Resid_star = Q^T Resid  (n × d)
    const Eigen::MatrixXd Resid_star = Q_.transpose() * Resid;

    // Compute U_star (n × d)
    Eigen::MatrixXd U_star(n_, d_);
    for (size_t i = 0; i < n_; ++i) {
        const Eigen::MatrixXd V_i = lambda_(i) * G + R;
        const Eigen::VectorXd r_i = Resid_star.row(i).transpose();

        // u_star_i = λ_i G (λ_i G + R)^{-1} r_i
        const Eigen::VectorXd V_i_inv_r_i = V_i.ldlt().solve(r_i);
        U_star.row(i) = (lambda_(i) * G * V_i_inv_r_i).transpose();
    }

    // Rotate back: U = Q U_star  (n × d)
    return Q_ * U_star;
}

Eigen::MatrixXd MultivariateSpectralLikelihoodEvaluator::compute_beta(
    const Eigen::MatrixXd& Y,
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& G,
    const Eigen::MatrixXd& R) const {

    const size_t p = X.cols();

    if (p == 0) {
        return Eigen::MatrixXd::Zero(0, d_);
    }

    // Transform Y and X: Y_star = Q^T Y, X_star = Q^T X
    const Eigen::MatrixXd Y_star = Q_.transpose() * Y;
    const Eigen::MatrixXd X_star = Q_.transpose() * X;

    // Compute (X^T V^{-1} X) and X^T V^{-1} Y using block structure
    // Result is (pd × pd) and (pd × 1) respectively, but we work in d×d blocks

    Eigen::MatrixXd XtVinvX = Eigen::MatrixXd::Zero(p * d_, p * d_);
    Eigen::MatrixXd XtVinvY = Eigen::MatrixXd::Zero(p * d_, 1);

    for (size_t i = 0; i < n_; ++i) {
        const Eigen::MatrixXd V_i = lambda_(i) * G + R;
        const Eigen::MatrixXd V_i_inv = V_i.inverse();

        const Eigen::VectorXd x_i = X_star.row(i).transpose();
        const Eigen::VectorXd y_i = Y_star.row(i).transpose();

        // X^T V^{-1} X contribution: x_i x_i^T ⊗ V_i_inv
        const Eigen::MatrixXd x_i_outer = x_i * x_i.transpose();
        for (size_t j1 = 0; j1 < p; ++j1) {
            for (size_t j2 = 0; j2 < p; ++j2) {
                XtVinvX.block(j1 * d_, j2 * d_, d_, d_) += x_i_outer(j1, j2) * V_i_inv;
            }
        }

        // X^T V^{-1} Y contribution: x_i ⊗ (V_i_inv y_i)
        const Eigen::VectorXd V_i_inv_y_i = V_i_inv * y_i;
        for (size_t j = 0; j < p; ++j) {
            XtVinvY.block(j * d_, 0, d_, 1) += x_i(j) * V_i_inv_y_i;
        }
    }

    // Solve: vec(B) = (X^T V^{-1} X)^{-1} X^T V^{-1} vec(Y)
    Eigen::VectorXd B_vec = XtVinvX.ldlt().solve(XtVinvY);

    // Reshape vec(B) back to (p × d)
    Eigen::MatrixXd B(p, d_);
    for (size_t j = 0; j < p; ++j) {
        B.row(j) = B_vec.segment(j * d_, d_).transpose();
    }

    return B;
}

double MultivariateSpectralLikelihoodEvaluator::compute_logdet(
    const Eigen::MatrixXd& G,
    const Eigen::MatrixXd& R) const {

    // log|V| = log|G ⊗ K + R ⊗ I| = sum_i log|λ_i G + R|
    double logdet = 0.0;
    for (size_t i = 0; i < n_; ++i) {
        const Eigen::MatrixXd V_i = lambda_(i) * G + R;

        // Compute log-determinant using Cholesky decomposition
        Eigen::LLT<Eigen::MatrixXd> llt(V_i);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("V_i = λ_i G + R is not positive definite");
        }

        double logdet_i = 0.0;
        for (size_t j = 0; j < d_; ++j) {
            logdet_i += std::log(llt.matrixL()(j, j));
        }
        logdet += 2.0 * logdet_i; // Determinant = product of diagonal^2
    }

    return logdet;
}

Eigen::MatrixXd MultivariateSpectralLikelihoodEvaluator::compute_sqrtm(
    const Eigen::MatrixXd& M) const {

    // Compute matrix square root using eigendecomposition
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("Eigendecomposition failed for matrix square root");
    }

    // sqrt(M) = U sqrt(Λ) U^T
    const Eigen::VectorXd sqrt_eigenvalues = es.eigenvalues().array().sqrt();
    return es.eigenvectors() * sqrt_eigenvalues.asDiagonal() * es.eigenvectors().transpose();
}

bool MultivariateSpectralLikelihoodEvaluator::is_positive_definite(
    const Eigen::MatrixXd& M) const {

    Eigen::LLT<Eigen::MatrixXd> llt(M);
    return llt.info() == Eigen::Success;
}

} // namespace libsemx
