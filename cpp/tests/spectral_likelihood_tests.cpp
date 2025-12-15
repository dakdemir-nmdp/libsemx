#include "libsemx/spectral_likelihood_evaluator.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <chrono>

using namespace libsemx;

// Helper function to create a test GRM (Genomic Relationship Matrix)
Eigen::MatrixXd create_test_grm(size_t n, unsigned seed = 42) {
    std::srand(seed);

    // Create random symmetric positive definite matrix
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
    Eigen::MatrixXd K = (A * A.transpose()) / n;

    // Ensure positive definite by adding ridge
    K.diagonal().array() += 0.1;

    return K;
}

// Helper function to compute likelihood using direct inversion (baseline)
double compute_loglik_direct(const Eigen::VectorXd& y,
                               const Eigen::MatrixXd& X,
                               const Eigen::MatrixXd& K,
                               double sigma_g_sq,
                               double sigma_e_sq,
                               bool use_reml = false) {
    const size_t n = y.size();
    const size_t p = X.cols();

    // Construct V = σ²_g K + σ²_e I
    Eigen::MatrixXd V = sigma_g_sq * K + sigma_e_sq * Eigen::MatrixXd::Identity(n, n);

    // Compute V^{-1}
    Eigen::LLT<Eigen::MatrixXd> llt(V);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("V is not positive definite");
    }

    // Compute β = (X^T V^{-1} X)^{-1} X^T V^{-1} y
    Eigen::MatrixXd V_inv = V.inverse();
    Eigen::MatrixXd XtVinvX = X.transpose() * V_inv * X;
    Eigen::VectorXd XtVinvy = X.transpose() * V_inv * y;
    Eigen::VectorXd beta = XtVinvX.inverse() * XtVinvy;

    // Compute residuals r = y - Xβ
    Eigen::VectorXd r = y - X * beta;

    // Compute log|V|
    double logdet_V = 2.0 * llt.matrixL().toDenseMatrix().diagonal().array().log().sum();

    // Compute quadratic form r^T V^{-1} r
    double quad_form = r.transpose() * V_inv * r;

    // ML log-likelihood
    double loglik = -0.5 * (logdet_V + quad_form + n * std::log(2.0 * M_PI));

    // REML correction
    if (use_reml && p > 0) {
        Eigen::LLT<Eigen::MatrixXd> llt_XtVinvX(XtVinvX);
        double logdet_XtVinvX = 2.0 * llt_XtVinvX.matrixL().toDenseMatrix().diagonal().array().log().sum();
        loglik -= 0.5 * logdet_XtVinvX;
    }

    return loglik;
}

TEST_CASE("SpectralLikelihoodEvaluator: Construction and eigendecomposition", "[spectral][likelihood]") {
    const size_t n = 50;
    Eigen::MatrixXd K = create_test_grm(n);

    SECTION("Valid kernel matrix") {
        REQUIRE_NOTHROW(SpectralLikelihoodEvaluator(K));

        SpectralLikelihoodEvaluator evaluator(K);

        // Check dimensions
        REQUIRE(evaluator.size() == n);
        REQUIRE(evaluator.eigenvalues().size() == static_cast<long>(n));
        REQUIRE(evaluator.eigenvectors().rows() == static_cast<long>(n));
        REQUIRE(evaluator.eigenvectors().cols() == static_cast<long>(n));

        // Check eigenvalues are positive (with threshold)
        for (size_t i = 0; i < n; ++i) {
            REQUIRE(evaluator.eigenvalues()(i) >= 1e-10);
        }

        // Check orthogonality of eigenvectors
        Eigen::MatrixXd I = evaluator.eigenvectors().transpose() * evaluator.eigenvectors();
        REQUIRE_THAT((I - Eigen::MatrixXd::Identity(n, n)).norm(),
                     Catch::Matchers::WithinAbs(0.0, 1e-10));
    }

    SECTION("Non-square kernel matrix") {
        Eigen::MatrixXd K_bad = Eigen::MatrixXd::Random(n, n + 1);
        REQUIRE_THROWS_AS(SpectralLikelihoodEvaluator(K_bad), std::invalid_argument);
    }
}

TEST_CASE("SpectralLikelihoodEvaluator: ML likelihood matches direct computation", "[spectral][likelihood]") {
    const size_t n = 100;
    const size_t p = 5;

    // Create test data
    Eigen::MatrixXd K = create_test_grm(n);
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    X.col(0).setOnes(); // Intercept

    Eigen::VectorXd y = Eigen::VectorXd::Random(n);

    // Variance components
    double sigma_g_sq = 0.6;
    double sigma_e_sq = 0.4;

    SpectralLikelihoodEvaluator evaluator(K);

    // Compute likelihood using spectral method
    double loglik_spectral = evaluator.evaluate_loglik(y, X, sigma_g_sq, sigma_e_sq, false);

    // Compute likelihood using direct method
    double loglik_direct = compute_loglik_direct(y, X, K, sigma_g_sq, sigma_e_sq, false);

    // Should match within numerical tolerance
    REQUIRE_THAT(loglik_spectral, Catch::Matchers::WithinAbs(loglik_direct, 1e-8));
}

TEST_CASE("SpectralLikelihoodEvaluator: REML likelihood matches direct computation", "[spectral][likelihood]") {
    const size_t n = 100;
    const size_t p = 5;

    // Create test data
    Eigen::MatrixXd K = create_test_grm(n);
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    X.col(0).setOnes(); // Intercept

    Eigen::VectorXd y = Eigen::VectorXd::Random(n);

    // Variance components
    double sigma_g_sq = 0.6;
    double sigma_e_sq = 0.4;

    SpectralLikelihoodEvaluator evaluator(K);

    // Compute likelihood using spectral method
    double loglik_spectral = evaluator.evaluate_loglik(y, X, sigma_g_sq, sigma_e_sq, true);

    // Compute likelihood using direct method
    double loglik_direct = compute_loglik_direct(y, X, K, sigma_g_sq, sigma_e_sq, true);

    // Should match within numerical tolerance
    REQUIRE_THAT(loglik_spectral, Catch::Matchers::WithinAbs(loglik_direct, 1e-8));
}

TEST_CASE("SpectralLikelihoodEvaluator: Gradient computation via finite differences", "[spectral][likelihood]") {
    const size_t n = 50;
    const size_t p = 3;

    // Create test data
    Eigen::MatrixXd K = create_test_grm(n);
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    X.col(0).setOnes();

    Eigen::VectorXd y = Eigen::VectorXd::Random(n);

    // Variance components
    double sigma_g_sq = 0.6;
    double sigma_e_sq = 0.4;

    SpectralLikelihoodEvaluator evaluator(K);

    SECTION("ML gradient check") {
        // Compute analytic gradient
        Eigen::Vector2d grad_analytic = evaluator.evaluate_gradient(y, X, sigma_g_sq, sigma_e_sq, false);

        // Compute numerical gradient via finite differences
        const double h = 1e-6;

        double f0 = evaluator.evaluate_loglik(y, X, sigma_g_sq, sigma_e_sq, false);
        double f_g = evaluator.evaluate_loglik(y, X, sigma_g_sq + h, sigma_e_sq, false);
        double f_e = evaluator.evaluate_loglik(y, X, sigma_g_sq, sigma_e_sq + h, false);

        double grad_g_numerical = (f_g - f0) / h;
        double grad_e_numerical = (f_e - f0) / h;

        // Check gradient accuracy
        REQUIRE_THAT(grad_analytic(0), Catch::Matchers::WithinAbs(grad_g_numerical, 1e-4));
        REQUIRE_THAT(grad_analytic(1), Catch::Matchers::WithinAbs(grad_e_numerical, 1e-4));
    }

    SECTION("REML gradient check") {
        // Compute analytic gradient
        Eigen::Vector2d grad_analytic = evaluator.evaluate_gradient(y, X, sigma_g_sq, sigma_e_sq, true);

        // Compute numerical gradient via finite differences
        const double h = 1e-6;

        double f0 = evaluator.evaluate_loglik(y, X, sigma_g_sq, sigma_e_sq, true);
        double f_g = evaluator.evaluate_loglik(y, X, sigma_g_sq + h, sigma_e_sq, true);
        double f_e = evaluator.evaluate_loglik(y, X, sigma_g_sq, sigma_e_sq + h, true);

        double grad_g_numerical = (f_g - f0) / h;
        double grad_e_numerical = (f_e - f0) / h;

        // Check gradient accuracy
        REQUIRE_THAT(grad_analytic(0), Catch::Matchers::WithinAbs(grad_g_numerical, 1e-4));
        REQUIRE_THAT(grad_analytic(1), Catch::Matchers::WithinAbs(grad_e_numerical, 1e-4));
    }
}

TEST_CASE("SpectralLikelihoodEvaluator: BLUP computation", "[spectral][blup]") {
    const size_t n = 100;
    const size_t p = 5;

    // Create test data
    Eigen::MatrixXd K = create_test_grm(n);
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    X.col(0).setOnes();

    Eigen::VectorXd y = Eigen::VectorXd::Random(n);

    // Variance components
    double sigma_g_sq = 0.6;
    double sigma_e_sq = 0.4;

    SpectralLikelihoodEvaluator evaluator(K);

    // Compute BLUP using spectral method
    Eigen::VectorXd blup_spectral = evaluator.compute_blup(y, X, sigma_g_sq, sigma_e_sq);

    // Compute BLUP using direct method: u = σ²_g K V^{-1} (y - Xβ)
    Eigen::MatrixXd V = sigma_g_sq * K + sigma_e_sq * Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd V_inv = V.inverse();
    Eigen::MatrixXd XtVinvX = X.transpose() * V_inv * X;
    Eigen::VectorXd XtVinvy = X.transpose() * V_inv * y;
    Eigen::VectorXd beta = XtVinvX.inverse() * XtVinvy;
    Eigen::VectorXd r = y - X * beta;
    Eigen::VectorXd blup_direct = sigma_g_sq * K * V_inv * r;

    // Should match within numerical tolerance
    for (size_t i = 0; i < n; ++i) {
        REQUIRE_THAT(blup_spectral(i), Catch::Matchers::WithinAbs(blup_direct(i), 1e-8));
    }
}

TEST_CASE("SpectralLikelihoodEvaluator: Performance benchmark vs direct inversion", "[spectral][benchmark][!benchmark]") {
    const size_t n = 500;
    const size_t p = 10;

    // Create test data
    Eigen::MatrixXd K = create_test_grm(n);
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    X.col(0).setOnes();

    Eigen::VectorXd y = Eigen::VectorXd::Random(n);

    double sigma_g_sq = 0.6;
    double sigma_e_sq = 0.4;

    SECTION("Spectral method timing") {
        // One-time eigendecomposition
        auto start_setup = std::chrono::high_resolution_clock::now();
        SpectralLikelihoodEvaluator evaluator(K);
        auto end_setup = std::chrono::high_resolution_clock::now();

        auto setup_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_setup - start_setup).count();

        // Multiple likelihood evaluations (simulating optimization iterations)
        const int n_iter = 100;
        auto start_eval = std::chrono::high_resolution_clock::now();

        double loglik_sum = 0.0;
        for (int iter = 0; iter < n_iter; ++iter) {
            double sigma_g_test = 0.1 + 0.005 * iter;
            double sigma_e_test = 0.5 + 0.005 * iter;
            loglik_sum += evaluator.evaluate_loglik(y, X, sigma_g_test, sigma_e_test, false);
        }

        auto end_eval = std::chrono::high_resolution_clock::now();
        auto eval_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_eval - start_eval).count();

        INFO("Spectral method: Setup = " << setup_time << "ms, " <<
             n_iter << " iterations = " << eval_time << "ms " <<
             "(" << (eval_time * 1.0 / n_iter) << "ms per iteration)");

        // Ensure computation is not optimized away
        REQUIRE(std::isfinite(loglik_sum));
    }

    SECTION("Direct method timing") {
        // Multiple likelihood evaluations using direct inversion
        const int n_iter = 100;
        auto start = std::chrono::high_resolution_clock::now();

        double loglik_sum = 0.0;
        for (int iter = 0; iter < n_iter; ++iter) {
            double sigma_g_test = 0.1 + 0.005 * iter;
            double sigma_e_test = 0.5 + 0.005 * iter;
            loglik_sum += compute_loglik_direct(y, X, K, sigma_g_test, sigma_e_test, false);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        INFO("Direct method: " << n_iter << " iterations = " << total_time << "ms " <<
             "(" << (total_time * 1.0 / n_iter) << "ms per iteration)");

        // Ensure computation is not optimized away
        REQUIRE(std::isfinite(loglik_sum));
    }
}

TEST_CASE("SpectralLikelihoodEvaluator: Edge cases", "[spectral][likelihood]") {
    const size_t n = 50;
    Eigen::MatrixXd K = create_test_grm(n);
    SpectralLikelihoodEvaluator evaluator(K);

    SECTION("Zero genetic variance") {
        Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, 1);
        Eigen::VectorXd y = Eigen::VectorXd::Random(n);

        double loglik = evaluator.evaluate_loglik(y, X, 0.0, 1.0, false);
        REQUIRE(std::isfinite(loglik));
    }

    SECTION("Large genetic variance") {
        Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, 1);
        Eigen::VectorXd y = Eigen::VectorXd::Random(n);

        double loglik = evaluator.evaluate_loglik(y, X, 100.0, 1.0, false);
        REQUIRE(std::isfinite(loglik));
    }

    SECTION("No fixed effects") {
        Eigen::MatrixXd X(n, 0);
        Eigen::VectorXd y = Eigen::VectorXd::Random(n);

        double loglik = evaluator.evaluate_loglik(y, X, 0.6, 0.4, false);
        REQUIRE(std::isfinite(loglik));
    }

    SECTION("Invalid variance components") {
        Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, 1);
        Eigen::VectorXd y = Eigen::VectorXd::Random(n);

        REQUIRE_THROWS_AS(evaluator.evaluate_loglik(y, X, -0.1, 1.0, false), std::invalid_argument);
        REQUIRE_THROWS_AS(evaluator.evaluate_loglik(y, X, 0.5, 0.0, false), std::invalid_argument);
        REQUIRE_THROWS_AS(evaluator.evaluate_loglik(y, X, 0.5, -0.1, false), std::invalid_argument);
    }

    SECTION("Dimension mismatch") {
        Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, 1);
        Eigen::VectorXd y_bad = Eigen::VectorXd::Random(n + 10);

        REQUIRE_THROWS_AS(evaluator.evaluate_loglik(y_bad, X, 0.6, 0.4, false), std::invalid_argument);
    }
}
