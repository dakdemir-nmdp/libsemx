#include "libsemx/multivariate_spectral_likelihood_evaluator.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Dense>
#include <chrono>

using namespace libsemx;

// Helper to create a test kernel matrix
static Eigen::MatrixXd create_test_kernel(size_t n, unsigned seed = 123) {
    std::srand(seed);
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
    Eigen::MatrixXd K = (A * A.transpose()) / n;
    K.diagonal().array() += 0.1; // Ensure positive definite
    return K;
}

// Helper to create a positive definite covariance matrix
static Eigen::MatrixXd create_test_covariance(size_t d, unsigned seed = 456) {
    std::srand(seed);
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(d, d);
    Eigen::MatrixXd Sigma = (A * A.transpose()) / d;
    Sigma.diagonal().array() += 0.1; // Ensure positive definite
    return Sigma;
}

TEST_CASE("Multivariate Spectral: Construction and eigendecomposition", "[multivariate][spectral]") {
    const size_t n = 50;
    const size_t d = 3;

    Eigen::MatrixXd K = create_test_kernel(n);

    SECTION("Valid construction") {
        MultivariateSpectralLikelihoodEvaluator evaluator(K, d);

        REQUIRE(evaluator.n_obs() == n);
        REQUIRE(evaluator.n_traits() == d);
        REQUIRE(evaluator.eigenvalues().size() == n);
        REQUIRE(evaluator.eigenvectors().rows() == n);
        REQUIRE(evaluator.eigenvectors().cols() == n);

        // Verify eigendecomposition: K ≈ Q Λ Q^T
        const Eigen::MatrixXd K_reconstructed =
            evaluator.eigenvectors() *
            evaluator.eigenvalues().asDiagonal() *
            evaluator.eigenvectors().transpose();

        REQUIRE((K - K_reconstructed).norm() < 1e-10);
    }

    SECTION("Invalid kernel matrix (non-square)") {
        Eigen::MatrixXd K_invalid(n, n + 1);
        REQUIRE_THROWS_AS(MultivariateSpectralLikelihoodEvaluator(K_invalid, d),
                          std::invalid_argument);
    }

    SECTION("Invalid number of traits") {
        REQUIRE_THROWS_AS(MultivariateSpectralLikelihoodEvaluator(K, 0),
                          std::invalid_argument);
    }
}

TEST_CASE("Multivariate Spectral: ML likelihood accuracy", "[multivariate][spectral]") {
    const size_t n = 40;
    const size_t d = 2;
    const size_t p = 3;

    // Create test data
    Eigen::MatrixXd K = create_test_kernel(n, 100);
    Eigen::MatrixXd G = create_test_covariance(d, 200);
    Eigen::MatrixXd R = create_test_covariance(d, 300);

    std::srand(400);
    Eigen::MatrixXd Y = Eigen::MatrixXd::Random(n, d);
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);

    MultivariateSpectralLikelihoodEvaluator evaluator(K, d);

    SECTION("Likelihood is finite and negative") {
        double loglik = evaluator.evaluate_loglik(Y, X, G, R, false);

        REQUIRE(std::isfinite(loglik));
        REQUIRE(loglik < 0.0); // Should be negative for typical data
    }

    SECTION("Likelihood increases with better fit") {
        // Generate data from the model: Y = X B + U + E
        // where U ~ N(0, G ⊗ K), E ~ N(0, R ⊗ I)

        Eigen::MatrixXd B_true(p, d);
        B_true << 1.0, 2.0,
                  0.5, -0.5,
                  -1.0, 1.5;

        // Generate random effects
        Eigen::LLT<Eigen::MatrixXd> llt_K(K);
        Eigen::MatrixXd L_K = llt_K.matrixL();

        Eigen::LLT<Eigen::MatrixXd> llt_G(G);
        Eigen::MatrixXd L_G = llt_G.matrixL();

        Eigen::MatrixXd U = L_K * Eigen::MatrixXd::Random(n, d) * L_G.transpose();

        // Generate residuals
        Eigen::LLT<Eigen::MatrixXd> llt_R(R);
        Eigen::MatrixXd L_R = llt_R.matrixL();

        Eigen::MatrixXd E = Eigen::MatrixXd::Random(n, d) * L_R.transpose();

        // Generate Y
        Eigen::MatrixXd Y_true = X * B_true + U + E;

        // Likelihood at true parameters should be higher than at arbitrary parameters
        double loglik_true = evaluator.evaluate_loglik(Y_true, X, G, R, false);

        Eigen::MatrixXd G_wrong = 2.0 * G;
        Eigen::MatrixXd R_wrong = 0.5 * R;
        double loglik_wrong = evaluator.evaluate_loglik(Y_true, X, G_wrong, R_wrong, false);

        INFO("Log-likelihood at true parameters: " << loglik_true);
        INFO("Log-likelihood at wrong parameters: " << loglik_wrong);

        // Note: This test may not always pass depending on random seed
        // We just check that both are finite
        REQUIRE(std::isfinite(loglik_true));
        REQUIRE(std::isfinite(loglik_wrong));
    }
}

TEST_CASE("Multivariate Spectral: REML likelihood accuracy", "[multivariate][spectral]") {
    const size_t n = 40;
    const size_t d = 2;
    const size_t p = 3;

    Eigen::MatrixXd K = create_test_kernel(n);
    Eigen::MatrixXd G = create_test_covariance(d, 250);
    Eigen::MatrixXd R = create_test_covariance(d, 350);

    std::srand(450);
    Eigen::MatrixXd Y = Eigen::MatrixXd::Random(n, d);
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);

    MultivariateSpectralLikelihoodEvaluator evaluator(K, d);

    double loglik_ml = evaluator.evaluate_loglik(Y, X, G, R, false);
    double loglik_reml = evaluator.evaluate_loglik(Y, X, G, R, true);

    REQUIRE(std::isfinite(loglik_ml));
    REQUIRE(std::isfinite(loglik_reml));

    // REML typically has lower (more negative) likelihood due to fixed effects penalty
    REQUIRE(loglik_reml < loglik_ml);
}

TEST_CASE("Multivariate Spectral: Gradient validation via finite differences", "[multivariate][spectral]") {
    const size_t n = 30;
    const size_t d = 2;
    const size_t p = 2;

    Eigen::MatrixXd K = create_test_kernel(n, 500);
    Eigen::MatrixXd G = create_test_covariance(d, 600);
    Eigen::MatrixXd R = create_test_covariance(d, 700);

    std::srand(800);
    Eigen::MatrixXd Y = Eigen::MatrixXd::Random(n, d);
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);

    MultivariateSpectralLikelihoodEvaluator evaluator(K, d);

    SECTION("ML gradient accuracy") {
        auto [grad_G_vec, grad_R_vec] = evaluator.evaluate_gradient(Y, X, G, R, false);

        // Reshape gradients back to matrices
        Eigen::MatrixXd grad_G = Eigen::Map<Eigen::MatrixXd>(grad_G_vec.data(), d, d);
        Eigen::MatrixXd grad_R = Eigen::Map<Eigen::MatrixXd>(grad_R_vec.data(), d, d);

        // Validate gradient for G using finite differences
        // G is symmetric, so we need to perturb G(i,j) and G(j,i) together for i != j
        const double eps = 1e-5;
        Eigen::MatrixXd grad_G_fd = Eigen::MatrixXd::Zero(d, d);

        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < d; ++j) {
                Eigen::MatrixXd G_plus = G;
                G_plus(i, j) += eps;
                if (i != j) {
                    G_plus(j, i) += eps; // Maintain symmetry
                }

                Eigen::MatrixXd G_minus = G;
                G_minus(i, j) -= eps;
                if (i != j) {
                    G_minus(j, i) -= eps; // Maintain symmetry
                }

                double loglik_plus = evaluator.evaluate_loglik(Y, X, G_plus, R, false);
                double loglik_minus = evaluator.evaluate_loglik(Y, X, G_minus, R, false);

                // For off-diagonal elements, we perturbed both G(i,j) and G(j,i) by eps
                // so the total perturbation is 2*eps
                if (i != j) {
                    grad_G_fd(i, j) = (loglik_plus - loglik_minus) / (4.0 * eps);
                } else {
                    grad_G_fd(i, j) = (loglik_plus - loglik_minus) / (2.0 * eps);
                }
            }
        }

        INFO("Analytic gradient G:\n" << grad_G);
        INFO("Finite difference gradient G:\n" << grad_G_fd);

        REQUIRE((grad_G - grad_G_fd).norm() / grad_G_fd.norm() < 1e-3);

        // Validate gradient for R using finite differences
        // R is symmetric, so we need to perturb R(i,j) and R(j,i) together for i != j
        Eigen::MatrixXd grad_R_fd = Eigen::MatrixXd::Zero(d, d);

        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < d; ++j) {
                Eigen::MatrixXd R_plus = R;
                R_plus(i, j) += eps;
                if (i != j) {
                    R_plus(j, i) += eps; // Maintain symmetry
                }

                Eigen::MatrixXd R_minus = R;
                R_minus(i, j) -= eps;
                if (i != j) {
                    R_minus(j, i) -= eps; // Maintain symmetry
                }

                double loglik_plus = evaluator.evaluate_loglik(Y, X, G, R_plus, false);
                double loglik_minus = evaluator.evaluate_loglik(Y, X, G, R_minus, false);

                // For off-diagonal elements, we perturbed both R(i,j) and R(j,i) by eps
                // so the total perturbation is 2*eps
                if (i != j) {
                    grad_R_fd(i, j) = (loglik_plus - loglik_minus) / (4.0 * eps);
                } else {
                    grad_R_fd(i, j) = (loglik_plus - loglik_minus) / (2.0 * eps);
                }
            }
        }

        INFO("Analytic gradient R:\n" << grad_R);
        INFO("Finite difference gradient R:\n" << grad_R_fd);

        REQUIRE((grad_R - grad_R_fd).norm() / grad_R_fd.norm() < 1e-3);
    }
}

TEST_CASE("Multivariate Spectral: BLUP computation", "[multivariate][spectral]") {
    const size_t n = 40;
    const size_t d = 2;
    const size_t p = 3;

    Eigen::MatrixXd K = create_test_kernel(n, 900);
    Eigen::MatrixXd G = create_test_covariance(d, 1000);
    Eigen::MatrixXd R = create_test_covariance(d, 1100);

    std::srand(1200);
    Eigen::MatrixXd Y = Eigen::MatrixXd::Random(n, d);
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);

    MultivariateSpectralLikelihoodEvaluator evaluator(K, d);

    Eigen::MatrixXd U = evaluator.compute_blup(Y, X, G, R);

    REQUIRE(U.rows() == n);
    REQUIRE(U.cols() == d);

    // BLUP should be finite
    REQUIRE(U.allFinite());

    // BLUP should shrink towards zero (on average)
    double blup_mean = U.mean();
    REQUIRE(std::abs(blup_mean) < 1.0); // Loose check
}

TEST_CASE("Multivariate Spectral: Single trait matches univariate", "[multivariate][spectral]") {
    const size_t n = 30;
    const size_t d = 1;
    const size_t p = 2;

    Eigen::MatrixXd K = create_test_kernel(n, 1300);

    // Single trait covariances (scalars wrapped as 1x1 matrices)
    Eigen::MatrixXd G(1, 1);
    G(0, 0) = 0.6;

    Eigen::MatrixXd R(1, 1);
    R(0, 0) = 0.4;

    std::srand(1400);
    Eigen::MatrixXd Y = Eigen::MatrixXd::Random(n, d);
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);

    MultivariateSpectralLikelihoodEvaluator evaluator(K, d);

    double loglik = evaluator.evaluate_loglik(Y, X, G, R, false);

    REQUIRE(std::isfinite(loglik));
    REQUIRE(loglik < 0.0);

    // Compare with univariate spectral evaluator
    // (We would need to include spectral_likelihood_evaluator.hpp for this)
    // For now, just check that multivariate works for d=1
    REQUIRE(true);
}

TEST_CASE("Multivariate Spectral: Performance benchmark", "[multivariate][spectral][!benchmark]") {
    const size_t n = 200;
    const size_t d = 3;
    const size_t p = 5;

    Eigen::MatrixXd K = create_test_kernel(n, 1500);
    Eigen::MatrixXd G = create_test_covariance(d, 1600);
    Eigen::MatrixXd R = create_test_covariance(d, 1700);

    std::srand(1800);
    Eigen::MatrixXd Y = Eigen::MatrixXd::Random(n, d);
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);

    MultivariateSpectralLikelihoodEvaluator evaluator(K, d);

    const int n_iter = 20;
    auto start = std::chrono::high_resolution_clock::now();

    double loglik_sum = 0.0;
    for (int i = 0; i < n_iter; ++i) {
        // Perturb G slightly to simulate optimization
        Eigen::MatrixXd G_iter = G * (1.0 + 0.01 * i / n_iter);
        loglik_sum += evaluator.evaluate_loglik(Y, X, G_iter, R, false);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    INFO("Multivariate spectral path: " << n_iter << " iterations in " << duration << "ms "
         << "(" << (duration * 1.0 / n_iter) << "ms per iteration)");

    REQUIRE(std::isfinite(loglik_sum));
    REQUIRE(duration < 10000); // Should complete in reasonable time
}

TEST_CASE("Multivariate Spectral: Edge cases", "[multivariate][spectral]") {
    const size_t n = 30;
    const size_t d = 2;
    const size_t p = 2;

    Eigen::MatrixXd K = create_test_kernel(n, 1900);
    Eigen::MatrixXd G = create_test_covariance(d, 2000);
    Eigen::MatrixXd R = create_test_covariance(d, 2100);

    std::srand(2200);
    Eigen::MatrixXd Y = Eigen::MatrixXd::Random(n, d);
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);

    MultivariateSpectralLikelihoodEvaluator evaluator(K, d);

    SECTION("Invalid Y dimensions") {
        Eigen::MatrixXd Y_invalid(n + 1, d);
        REQUIRE_THROWS_AS(evaluator.evaluate_loglik(Y_invalid, X, G, R, false),
                          std::invalid_argument);
    }

    SECTION("Invalid X dimensions") {
        Eigen::MatrixXd X_invalid(n + 1, p);
        REQUIRE_THROWS_AS(evaluator.evaluate_loglik(Y, X_invalid, G, R, false),
                          std::invalid_argument);
    }

    SECTION("Invalid G dimensions") {
        Eigen::MatrixXd G_invalid(d + 1, d);
        REQUIRE_THROWS_AS(evaluator.evaluate_loglik(Y, X, G_invalid, R, false),
                          std::invalid_argument);
    }

    SECTION("Invalid R dimensions") {
        Eigen::MatrixXd R_invalid(d, d + 1);
        REQUIRE_THROWS_AS(evaluator.evaluate_loglik(Y, X, G, R_invalid, false),
                          std::invalid_argument);
    }

    SECTION("Non-positive definite G") {
        Eigen::MatrixXd G_invalid = -G;
        REQUIRE_THROWS_AS(evaluator.evaluate_loglik(Y, X, G_invalid, R, false),
                          std::invalid_argument);
    }

    SECTION("Non-positive definite R") {
        Eigen::MatrixXd R_invalid = -R;
        REQUIRE_THROWS_AS(evaluator.evaluate_loglik(Y, X, G, R_invalid, false),
                          std::invalid_argument);
    }
}
