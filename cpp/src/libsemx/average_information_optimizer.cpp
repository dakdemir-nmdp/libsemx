#include "libsemx/average_information_optimizer.hpp"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <algorithm>

namespace libsemx {

namespace {
bool invert_spd(const Eigen::MatrixXd& M, Eigen::MatrixXd& M_inv) {
    Eigen::LLT<Eigen::MatrixXd> llt(M);
    if (llt.info() == Eigen::Success) {
        M_inv = llt.solve(Eigen::MatrixXd::Identity(M.rows(), M.cols()));
        return true;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M);
    if (es.info() != Eigen::Success) {
        return false;
    }
    const auto& vals = es.eigenvalues();
    const double max_val = vals.maxCoeff();
    const double tol = std::numeric_limits<double>::epsilon() * std::max(M.rows(), M.cols()) * std::max(1.0, max_val);

    Eigen::VectorXd inv_vals(vals.size());
    for (Eigen::Index i = 0; i < vals.size(); ++i) {
        if (vals(i) <= tol) {
            return false;
        }
        inv_vals(i) = 1.0 / vals(i);
    }

    M_inv = es.eigenvectors() * inv_vals.asDiagonal() * es.eigenvectors().transpose();
    return true;
}
}  // namespace

AverageInformationOptimizer::Result AverageInformationOptimizer::optimize(
    const Eigen::VectorXd& y,
    const Eigen::MatrixXd& X,
    const std::vector<Eigen::MatrixXd>& V_matrices,
    const Eigen::VectorXd& initial_params,
    const std::vector<bool>& fixed_components,
    const Options& options) {

    const size_t n = y.size();
    const size_t K = V_matrices.size();

    // Validate inputs
    if (static_cast<size_t>(X.rows()) != n) {
        throw std::invalid_argument("X rows must match y size");
    }
    if (static_cast<size_t>(initial_params.size()) != K) {
        throw std::invalid_argument("Initial params size must match number of V matrices");
    }
    if (!fixed_components.empty() && fixed_components.size() != K) {
        throw std::invalid_argument("Fixed components size must match number of V matrices");
    }

    // Initialize fixed components vector (default: all components are estimated)
    std::vector<bool> fixed = fixed_components.empty()
        ? std::vector<bool>(K, false)
        : fixed_components;

    // Identify which components to estimate
    std::vector<size_t> est_indices;
    for (size_t k = 0; k < K; ++k) {
        if (!fixed[k]) {
            est_indices.push_back(k);
        }
    }

    const size_t num_est = est_indices.size();

    if (num_est == 0) {
        // All components are fixed, nothing to optimize
        Result result;
        result.parameters = initial_params;
        result.iterations = 0;
        result.converged = true;
        result.message = "All variance components are fixed";

        // Compute final log-likelihood
        Eigen::MatrixXd V = Eigen::MatrixXd::Zero(n, n);
        for (size_t k = 0; k < K; ++k) {
            V += initial_params(k) * V_matrices[k];
        }
        Eigen::MatrixXd V_inv;
        if (!invert_spd(V, V_inv)) {
            result.converged = false;
            result.message = "Variance matrix is not positive definite";
            return result;
        }

        if (options.use_reml) {
            result.log_likelihood = compute_reml_loglik(y, X, V, V_inv);
        } else {
            Eigen::VectorXd beta = compute_beta(y, X, V_inv);
            result.log_likelihood = compute_ml_loglik(y, X, beta, V, V_inv);
        }

        return result;
    }

    // Initialize parameters
    Eigen::VectorXd params = initial_params;
    Eigen::VectorXd params_old = params;

    bool converged = false;
    size_t iter = 0;
    double loglik = -std::numeric_limits<double>::infinity();

    // Main optimization loop
    for (iter = 0; iter < options.max_iterations; ++iter) {
        params_old = params;

        // Construct current variance matrix V = Σ_k σ²_k V_k
        Eigen::MatrixXd V = Eigen::MatrixXd::Zero(n, n);
        for (size_t k = 0; k < K; ++k) {
            V += params(k) * V_matrices[k];
        }

        Eigen::MatrixXd V_inv;
        if (!invert_spd(V, V_inv)) {
            Result result;
            result.parameters = params;
            result.iterations = iter;
            result.converged = false;
            result.message = "Variance matrix is not positive definite at iteration " + std::to_string(iter);
            result.log_likelihood = loglik;
            return result;
        }

        // Compute GLS estimate of β
        Eigen::VectorXd beta = compute_beta(y, X, V_inv);

        // Compute M matrix (projection for REML, inverse for ML)
        Eigen::MatrixXd M;
        if (options.use_reml) {
            M = compute_projection_matrix(V_inv, X);
        } else {
            M = V_inv;
        }

        // Compute score vector and AI matrix for estimated components
        Eigen::VectorXd score = compute_score(y, M, V_matrices, est_indices);
        Eigen::MatrixXd ai_matrix = compute_ai_matrix(M, V_matrices, est_indices);

        // Solve for update: delta = AI^{-1} @ score
        Eigen::VectorXd delta;
        try {
            delta = ai_matrix.ldlt().solve(score);
        } catch (...) {
            Result result;
            result.parameters = params;
            result.iterations = iter;
            result.converged = false;
            result.message = "Singular AI matrix encountered at iteration " + std::to_string(iter);
            result.log_likelihood = loglik;
            return result;
        }

        // Apply update with step-halving to ensure non-negative variances
        double step = options.initial_step_size;
        Eigen::VectorXd params_new;

        while (step >= options.min_step_size) {
            // Reset params_new and apply update with current step size
            params_new = params_old;
            for (size_t i = 0; i < num_est; ++i) {
                params_new(est_indices[i]) = params_old(est_indices[i]) + step * delta(i);
            }

            // Check if all variances are non-negative
            bool all_positive = true;
            for (size_t i = 0; i < num_est; ++i) {
                if (params_new(est_indices[i]) < 0) {
                    all_positive = false;
                    break;
                }
            }

            if (all_positive) {
                break;
            }

            // Halve the step
            step /= 2.0;
        }

        if (step < options.min_step_size) {
            Result result;
            result.parameters = params;
            result.iterations = iter;
            result.converged = false;
            result.message = "Step size became too small (negative variances)";
            result.log_likelihood = loglik;
            return result;
        }

        // Apply variance floor
        for (size_t k = 0; k < K; ++k) {
            if (params_new(k) < options.min_variance) {
                params_new(k) = options.min_variance;
            }
        }

        // Re-apply fixed values
        for (size_t k = 0; k < K; ++k) {
            if (fixed[k]) {
                params_new(k) = params_old(k);
            }
        }

        params = params_new;

        // Compute log-likelihood for monitoring
        Eigen::MatrixXd V_new = Eigen::MatrixXd::Zero(n, n);
        for (size_t k = 0; k < K; ++k) {
            V_new += params(k) * V_matrices[k];
        }

        Eigen::MatrixXd V_new_inv;
        if (!invert_spd(V_new, V_new_inv)) {
            Result result;
            result.parameters = params;
            result.iterations = iter + 1;
            result.converged = false;
            result.message = "Variance matrix is not positive definite after update";
            result.log_likelihood = loglik;
            return result;
        }

        if (options.use_reml) {
            loglik = compute_reml_loglik(y, X, V_new, V_new_inv);
        } else {
            Eigen::VectorXd beta_new = compute_beta(y, X, V_new_inv);
            loglik = compute_ml_loglik(y, X, beta_new, V_new, V_new_inv);
        }

        if (options.verbose) {
            std::cout << "AI Iter " << iter + 1 << ": ";
            for (size_t k = 0; k < K; ++k) {
                std::cout << "σ²_" << k << "=" << params(k) << " ";
            }
            std::cout << " LogLik=" << loglik << " Step=" << step << std::endl;
        }

        // Check convergence
        double max_change = 0.0;
        for (size_t i = 0; i < num_est; ++i) {
            size_t k = est_indices[i];
            max_change = std::max(max_change, std::abs(params(k) - params_old(k)));
        }

        if (max_change < options.tolerance) {
            converged = true;
            break;
        }
    }

    // Prepare result
    Result result;
    result.parameters = params;
    result.iterations = iter + 1;
    result.converged = converged;
    result.log_likelihood = loglik;

    if (converged) {
        result.message = "Converged successfully";
    } else {
        result.message = "Maximum iterations reached without convergence";
    }

    // Compute final score and AI matrix for diagnostics/standard errors
    Eigen::MatrixXd V_final = Eigen::MatrixXd::Zero(n, n);
    for (size_t k = 0; k < K; ++k) {
        V_final += params(k) * V_matrices[k];
    }

    Eigen::MatrixXd V_final_inv;
    if (!invert_spd(V_final, V_final_inv)) {
        result.message = "Variance matrix is not positive definite at final step";
        result.converged = false;
        return result;
    }
    Eigen::MatrixXd M_final;
    if (options.use_reml) {
        M_final = compute_projection_matrix(V_final_inv, X);
    } else {
        M_final = V_final_inv;
    }

    result.final_score = compute_score(y, M_final, V_matrices, est_indices);
    result.final_ai_matrix = compute_ai_matrix(M_final, V_matrices, est_indices);

    return result;
}

Eigen::VectorXd AverageInformationOptimizer::compute_beta(
    const Eigen::VectorXd& y,
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& V_inv) {

    const size_t p = X.cols();
    if (p == 0) {
        return Eigen::VectorXd::Zero(0);
    }

    // β = (X^T V^{-1} X)^{-1} X^T V^{-1} y
    Eigen::MatrixXd XtVinvX = X.transpose() * V_inv * X;
    Eigen::VectorXd XtVinvy = X.transpose() * V_inv * y;

    return XtVinvX.ldlt().solve(XtVinvy);
}

Eigen::MatrixXd AverageInformationOptimizer::compute_projection_matrix(
    const Eigen::MatrixXd& V_inv,
    const Eigen::MatrixXd& X) {

    const size_t p = X.cols();
    if (p == 0) {
        return V_inv;
    }

    // P = V^{-1} - V^{-1} X (X^T V^{-1} X)^{-1} X^T V^{-1}
    Eigen::MatrixXd XtVinvX = X.transpose() * V_inv * X;
    Eigen::MatrixXd XtVinvX_inv;
    if (!invert_spd(XtVinvX, XtVinvX_inv)) {
        throw std::runtime_error("X^T V^{-1} X is not positive definite");
    }
    Eigen::MatrixXd V_inv_X = V_inv * X;

    return V_inv - V_inv_X * XtVinvX_inv * V_inv_X.transpose();
}

Eigen::VectorXd AverageInformationOptimizer::compute_score(
    const Eigen::VectorXd& y,
    const Eigen::MatrixXd& M,
    const std::vector<Eigen::MatrixXd>& V_matrices,
    const std::vector<size_t>& est_indices) {

    const size_t num_est = est_indices.size();
    Eigen::VectorXd score(num_est);

    // Precompute y^T M
    Eigen::VectorXd yM = M.transpose() * y;

    for (size_t i = 0; i < num_est; ++i) {
        size_t k = est_indices[i];
        const Eigen::MatrixXd& V_k = V_matrices[k];

        // Score_k = 0.5 * (y^T M V_k M y - tr(M V_k))
        Eigen::MatrixXd V_k_M = V_k * M;
        double quad_form = yM.dot(V_k_M * y);
        double trace_term = (M * V_k).trace();

        score(i) = 0.5 * (quad_form - trace_term);
    }

    return score;
}

Eigen::MatrixXd AverageInformationOptimizer::compute_ai_matrix(
    const Eigen::MatrixXd& M,
    const std::vector<Eigen::MatrixXd>& V_matrices,
    const std::vector<size_t>& est_indices) {

    const size_t num_est = est_indices.size();
    Eigen::MatrixXd ai_matrix(num_est, num_est);

    for (size_t i = 0; i < num_est; ++i) {
        size_t k = est_indices[i];
        const Eigen::MatrixXd& V_k = V_matrices[k];
        Eigen::MatrixXd V_k_M = V_k * M;

        for (size_t j = i; j < num_est; ++j) {
            size_t l = est_indices[j];
            const Eigen::MatrixXd& V_l = V_matrices[l];

            // AI_{k,l} = 0.5 * tr(V_k M V_l M)
            double trace_val = (V_k_M * V_l * M).trace();
            ai_matrix(i, j) = 0.5 * trace_val;

            // Symmetric
            if (i != j) {
                ai_matrix(j, i) = ai_matrix(i, j);
            }
        }
    }

    return ai_matrix;
}

double AverageInformationOptimizer::compute_reml_loglik(
    const Eigen::VectorXd& y,
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXd& V_inv) {

    // log|V|
    double log_det_V = compute_log_det(V);

    // log|X^T V^{-1} X|
    Eigen::MatrixXd XtVinvX = X.transpose() * V_inv * X;
    double log_det_XtVinvX = compute_log_det(XtVinvX);

    // Compute projection matrix P
    Eigen::MatrixXd P = compute_projection_matrix(V_inv, X);

    // y^T P y
    double yPy = y.dot(P * y);

    // REML log-likelihood: -0.5 * (log|V| + log|X^T V^{-1} X| + y^T P y)
    return -0.5 * (log_det_V + log_det_XtVinvX + yPy);
}

double AverageInformationOptimizer::compute_ml_loglik(
    const Eigen::VectorXd& y,
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& beta,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXd& V_inv) {

    // log|V|
    double log_det_V = compute_log_det(V);

    // (y - Xβ)^T V^{-1} (y - Xβ)
    Eigen::VectorXd residuals = y - X * beta;
    double quad_form = residuals.dot(V_inv * residuals);

    // ML log-likelihood: -0.5 * (log|V| + (y-Xβ)^T V^{-1} (y-Xβ))
    return -0.5 * (log_det_V + quad_form);
}

double AverageInformationOptimizer::compute_log_det(const Eigen::MatrixXd& M) {
    // Use Cholesky decomposition for positive definite matrices
    Eigen::LLT<Eigen::MatrixXd> llt(M);

    if (llt.info() != Eigen::Success) {
        // Fall back to eigenvalue decomposition
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M);
        return es.eigenvalues().array().log().sum();
    }

    // log|M| = 2 * sum(log(diag(L)))
    double log_det = 0.0;
    for (Eigen::Index i = 0; i < M.rows(); ++i) {
        log_det += std::log(llt.matrixL()(i, i));
    }
    return 2.0 * log_det;
}

} // namespace libsemx
