#include "libsemx/spectral_likelihood_evaluator.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>

namespace libsemx {

SpectralLikelihoodEvaluator::SpectralLikelihoodEvaluator(const Eigen::MatrixXd& kernel,
                                                           double min_eigenvalue)
    : n_(kernel.rows()), min_eigenvalue_(min_eigenvalue) {

    if (kernel.rows() != kernel.cols()) {
        throw std::invalid_argument("Kernel matrix must be square");
    }

    // Perform eigendecomposition of K = U Λ U^T
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(kernel);

    if (eigen_solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigendecomposition failed for kernel matrix");
    }

    // Store eigenvalues and eigenvectors
    lambda_ = eigen_solver.eigenvalues();
    U_ = eigen_solver.eigenvectors();

    // Apply minimum eigenvalue threshold for numerical stability
    for (size_t i = 0; i < n_; ++i) {
        if (lambda_(i) < min_eigenvalue_) {
            lambda_(i) = min_eigenvalue_;
        }
    }
}

double SpectralLikelihoodEvaluator::evaluate_loglik(const Eigen::VectorXd& y,
                                                      const Eigen::MatrixXd& X,
                                                      double sigma_g_sq,
                                                      double sigma_e_sq,
                                                      bool use_reml) const {
    if (y.size() != static_cast<long>(n_)) {
        throw std::invalid_argument("Response vector size mismatch");
    }
    if (X.rows() != static_cast<long>(n_)) {
        throw std::invalid_argument("Design matrix row count mismatch");
    }
    if (sigma_g_sq < 0 || sigma_e_sq <= 0) {
        throw std::invalid_argument("Variance components must be non-negative (σ²_e > 0)");
    }

    const size_t p = X.cols();

    // Compute GLS estimate of β
    const Eigen::VectorXd beta = compute_beta(y, X, sigma_g_sq, sigma_e_sq);

    // Compute residuals r = y - Xβ
    const Eigen::VectorXd r = y - X * beta;

    // Rotate residuals: r_tilde = U^T r
    const Eigen::VectorXd r_tilde = U_.transpose() * r;

    // Compute log-determinant of V
    const double logdet_V = compute_logdet(sigma_g_sq, sigma_e_sq);

    // Compute quadratic form r^T V^{-1} r using eigenvalues
    // r^T V^{-1} r = r_tilde^T (σ²_g Λ + σ²_e I)^{-1} r_tilde
    double quad_form = 0.0;
    for (size_t i = 0; i < n_; ++i) {
        const double v_i = sigma_g_sq * lambda_(i) + sigma_e_sq;
        quad_form += r_tilde(i) * r_tilde(i) / v_i;
    }

    // ML log-likelihood: -0.5 * [log|V| + r^T V^{-1} r + n log(2π)]
    double loglik = -0.5 * (logdet_V + quad_form + n_ * std::log(2.0 * M_PI));

    // REML correction: -0.5 * log|X^T V^{-1} X|
    if (use_reml && p > 0) {
        // Rotate X: X_tilde = U^T X
        const Eigen::MatrixXd X_tilde = U_.transpose() * X;

        // Compute X^T V^{-1} X using rotated coordinates
        Eigen::MatrixXd XtVinvX = Eigen::MatrixXd::Zero(p, p);
        for (size_t i = 0; i < n_; ++i) {
            const double v_i_inv = 1.0 / (sigma_g_sq * lambda_(i) + sigma_e_sq);
            XtVinvX += v_i_inv * X_tilde.row(i).transpose() * X_tilde.row(i);
        }

        // Compute log-determinant using Cholesky decomposition
        Eigen::LLT<Eigen::MatrixXd> llt(XtVinvX);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("X^T V^{-1} X is not positive definite");
        }

        double logdet_XtVinvX = 0.0;
        for (size_t i = 0; i < p; ++i) {
            logdet_XtVinvX += std::log(llt.matrixL()(i, i));
        }
        logdet_XtVinvX *= 2.0; // Determinant = product of diagonal^2

        loglik -= 0.5 * logdet_XtVinvX;
    }

    return loglik;
}

Eigen::Vector2d SpectralLikelihoodEvaluator::evaluate_gradient(const Eigen::VectorXd& y,
                                                                 const Eigen::MatrixXd& X,
                                                                 double sigma_g_sq,
                                                                 double sigma_e_sq,
                                                                 bool use_reml) const {
    if (y.size() != static_cast<long>(n_)) {
        throw std::invalid_argument("Response vector size mismatch");
    }
    if (X.rows() != static_cast<long>(n_)) {
        throw std::invalid_argument("Design matrix row count mismatch");
    }
    if (sigma_g_sq < 0 || sigma_e_sq <= 0) {
        throw std::invalid_argument("Variance components must be non-negative (σ²_e > 0)");
    }

    const size_t p = X.cols();

    // Compute GLS estimate of β
    const Eigen::VectorXd beta = compute_beta(y, X, sigma_g_sq, sigma_e_sq);

    // Compute residuals r = y - Xβ
    const Eigen::VectorXd r = y - X * beta;

    // Rotate residuals: r_tilde = U^T r
    const Eigen::VectorXd r_tilde = U_.transpose() * r;

    // Initialize gradients
    double d_sigma_g = 0.0;
    double d_sigma_e = 0.0;

    // Compute gradient contributions from log-determinant and quadratic form
    // d/dσ²_g log|V| = tr(V^{-1} K) = sum_i λ_i / v_i
    // d/dσ²_e log|V| = tr(V^{-1}) = sum_i 1 / v_i
    // d/dσ²_g r^T V^{-1} r = -r^T V^{-1} K V^{-1} r = -sum_i λ_i r_i^2 / v_i^2
    // d/dσ²_e r^T V^{-1} r = -r^T V^{-1} V^{-1} r = -sum_i r_i^2 / v_i^2

    for (size_t i = 0; i < n_; ++i) {
        const double v_i = sigma_g_sq * lambda_(i) + sigma_e_sq;
        const double v_i_inv = 1.0 / v_i;
        const double v_i_inv_sq = v_i_inv * v_i_inv;
        const double r_i_sq = r_tilde(i) * r_tilde(i);

        // Gradient of log|V|
        d_sigma_g += lambda_(i) * v_i_inv;
        d_sigma_e += v_i_inv;

        // Gradient of quadratic form (note the negative sign)
        d_sigma_g -= lambda_(i) * r_i_sq * v_i_inv_sq;
        d_sigma_e -= r_i_sq * v_i_inv_sq;
    }

    // Apply factor of -0.5 from log-likelihood
    d_sigma_g *= -0.5;
    d_sigma_e *= -0.5;

    // REML correction: gradient of -0.5 * log|X^T V^{-1} X|
    if (use_reml && p > 0) {
        // Rotate X: X_tilde = U^T X
        const Eigen::MatrixXd X_tilde = U_.transpose() * X;

        // Compute X^T V^{-1} X
        Eigen::MatrixXd XtVinvX = Eigen::MatrixXd::Zero(p, p);
        for (size_t i = 0; i < n_; ++i) {
            const double v_i_inv = 1.0 / (sigma_g_sq * lambda_(i) + sigma_e_sq);
            XtVinvX += v_i_inv * X_tilde.row(i).transpose() * X_tilde.row(i);
        }

        // Compute (X^T V^{-1} X)^{-1} with fallback for rank-deficient designs
        Eigen::MatrixXd XtVinvX_inv;
        Eigen::LLT<Eigen::MatrixXd> llt(XtVinvX);
        if (llt.info() == Eigen::Success) {
            XtVinvX_inv = llt.solve(Eigen::MatrixXd::Identity(p, p));
        } else {
            Eigen::BDCSVD<Eigen::MatrixXd> svd(XtVinvX, Eigen::ComputeThinU | Eigen::ComputeThinV);
            const Eigen::VectorXd& sing = svd.singularValues();
            const double tol = std::numeric_limits<double>::epsilon() * std::max<std::size_t>(p, p) * sing.array().abs().maxCoeff();

            Eigen::MatrixXd S_inv = Eigen::MatrixXd::Zero(p, p);
            for (Eigen::Index i = 0; i < sing.size(); ++i) {
                if (sing(i) > tol) {
                    S_inv(i, i) = 1.0 / sing(i);
                }
            }
            XtVinvX_inv = svd.matrixV() * S_inv * svd.matrixU().transpose();
        }

        // Gradient correction for log|X^T V^{-1} X|
        // d/dσ²_g log|X^T V^{-1} X| = tr[(X^T V^{-1} X)^{-1} X^T V^{-1} K V^{-1} X]
        // d/dσ²_e log|X^T V^{-1} X| = tr[(X^T V^{-1} X)^{-1} X^T V^{-1} V^{-1} X]

        double reml_g = 0.0;
        double reml_e = 0.0;

        for (size_t i = 0; i < n_; ++i) {
            const double v_i = sigma_g_sq * lambda_(i) + sigma_e_sq;
            const double v_i_inv_sq = 1.0 / (v_i * v_i);

            // Compute X_tilde[i,:] * XtVinvX_inv * X_tilde[i,:]^T
            const Eigen::VectorXd temp = XtVinvX_inv * X_tilde.row(i).transpose();
            const double trace_contrib = X_tilde.row(i).dot(temp);

            reml_g += lambda_(i) * v_i_inv_sq * trace_contrib;
            reml_e += v_i_inv_sq * trace_contrib;
        }

        // Apply factor of -0.5 (note: we subtract because REML has negative log-determinant)
        d_sigma_g += 0.5 * reml_g;
        d_sigma_e += 0.5 * reml_e;
    }

    return Eigen::Vector2d(d_sigma_g, d_sigma_e);
}

Eigen::VectorXd SpectralLikelihoodEvaluator::compute_blup(const Eigen::VectorXd& y,
                                                            const Eigen::MatrixXd& X,
                                                            double sigma_g_sq,
                                                            double sigma_e_sq) const {
    if (y.size() != static_cast<long>(n_)) {
        throw std::invalid_argument("Response vector size mismatch");
    }
    if (X.rows() != static_cast<long>(n_)) {
        throw std::invalid_argument("Design matrix row count mismatch");
    }

    // Compute GLS estimate of β
    const Eigen::VectorXd beta = compute_beta(y, X, sigma_g_sq, sigma_e_sq);

    // Compute residuals r = y - Xβ
    const Eigen::VectorXd r = y - X * beta;

    // BLUP: u = σ²_g K V^{-1} r = σ²_g U Λ U^T V^{-1} r
    // = σ²_g U Λ (σ²_g Λ + σ²_e I)^{-1} U^T r

    // Rotate residuals: r_tilde = U^T r
    const Eigen::VectorXd r_tilde = U_.transpose() * r;

    // Apply diagonal scaling: u_tilde = σ²_g Λ (σ²_g Λ + σ²_e I)^{-1} r_tilde
    Eigen::VectorXd u_tilde(n_);
    for (size_t i = 0; i < n_; ++i) {
        const double v_i = sigma_g_sq * lambda_(i) + sigma_e_sq;
        u_tilde(i) = (sigma_g_sq * lambda_(i) / v_i) * r_tilde(i);
    }

    // Rotate back: u = U u_tilde
    return U_ * u_tilde;
}

Eigen::VectorXd SpectralLikelihoodEvaluator::compute_beta(const Eigen::VectorXd& y,
                                                            const Eigen::MatrixXd& X,
                                                            double sigma_g_sq,
                                                            double sigma_e_sq) const {
    const size_t p = X.cols();

    if (p == 0) {
        return Eigen::VectorXd::Zero(0);
    }

    // Rotate X and y: X_tilde = U^T X, y_tilde = U^T y
    const Eigen::MatrixXd X_tilde = U_.transpose() * X;
    const Eigen::VectorXd y_tilde = U_.transpose() * y;

    // Compute X^T V^{-1} X and X^T V^{-1} y using rotated coordinates
    Eigen::MatrixXd XtVinvX = Eigen::MatrixXd::Zero(p, p);
    Eigen::VectorXd XtVinvy = Eigen::VectorXd::Zero(p);

    for (size_t i = 0; i < n_; ++i) {
        const double v_i_inv = 1.0 / (sigma_g_sq * lambda_(i) + sigma_e_sq);
        XtVinvX += v_i_inv * X_tilde.row(i).transpose() * X_tilde.row(i);
        XtVinvy += v_i_inv * y_tilde(i) * X_tilde.row(i).transpose();
    }

    // Solve: β = (X^T V^{-1} X)^{-1} X^T V^{-1} y
    Eigen::LLT<Eigen::MatrixXd> llt(XtVinvX);
    if (llt.info() != Eigen::Success) {
        // Fall back to SVD for rank-deficient case
        return XtVinvX.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(XtVinvy);
    }

    return llt.solve(XtVinvy);
}

double SpectralLikelihoodEvaluator::compute_logdet(double sigma_g_sq, double sigma_e_sq) const {
    // log|V| = log|σ²_g Λ + σ²_e I| = sum_i log(σ²_g λ_i + σ²_e)
    double logdet = 0.0;
    for (size_t i = 0; i < n_; ++i) {
        const double v_i = sigma_g_sq * lambda_(i) + sigma_e_sq;
        logdet += std::log(v_i);
    }
    return logdet;
}

Eigen::VectorXd SpectralLikelihoodEvaluator::compute_V_inv_y(const Eigen::VectorXd& y,
                                                               double sigma_g_sq,
                                                               double sigma_e_sq) const {
    // V^{-1} y = U (σ²_g Λ + σ²_e I)^{-1} U^T y

    // Rotate y: y_tilde = U^T y
    const Eigen::VectorXd y_tilde = U_.transpose() * y;

    // Apply diagonal scaling: z_tilde = (σ²_g Λ + σ²_e I)^{-1} y_tilde
    Eigen::VectorXd z_tilde(n_);
    for (size_t i = 0; i < n_; ++i) {
        const double v_i_inv = 1.0 / (sigma_g_sq * lambda_(i) + sigma_e_sq);
        z_tilde(i) = v_i_inv * y_tilde(i);
    }

    // Rotate back: z = U z_tilde
    return U_ * z_tilde;
}

} // namespace libsemx
