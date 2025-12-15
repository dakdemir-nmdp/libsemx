#include "libsemx/average_information_optimizer.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Dense>
#include <iostream>

using namespace libsemx;

// Helper to create a simple random effects design matrix Z
static Eigen::MatrixXd create_random_effects_design(size_t n, size_t q, unsigned seed = 42) {
    std::srand(seed);
    Eigen::MatrixXd Z = Eigen::MatrixXd::Random(n, q);
    return Z;
}

// Helper to generate synthetic LMM data: y = Xβ + Zu + e
static std::tuple<Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, double, double>
generate_lmm_data(size_t n, size_t p, size_t q, double sigma_u_sq, double sigma_e_sq, unsigned seed = 123) {
    std::srand(seed);

    // Fixed effects design
    Eigen::MatrixXd X(n, p);
    X.col(0) = Eigen::VectorXd::Ones(n); // Intercept
    for (size_t j = 1; j < p; ++j) {
        X.col(j) = Eigen::VectorXd::Random(n);
    }

    // Random effects design
    Eigen::MatrixXd Z = create_random_effects_design(n, q, seed + 1);

    // True fixed effects
    Eigen::VectorXd beta_true(p);
    beta_true << 5.0, 1.5; // Example values

    // Generate random effects u ~ N(0, σ²_u I)
    Eigen::VectorXd u = std::sqrt(sigma_u_sq) * Eigen::VectorXd::Random(q);

    // Generate residuals e ~ N(0, σ²_e I)
    Eigen::VectorXd e = std::sqrt(sigma_e_sq) * Eigen::VectorXd::Random(n);

    // Generate response: y = Xβ + Zu + e
    Eigen::VectorXd y = X * beta_true + Z * u + e;

    return std::make_tuple(y, X, Z, sigma_u_sq, sigma_e_sq);
}

TEST_CASE("AI Optimizer: Simple LMM convergence", "[ai][optimizer]") {
    const size_t n = 100;
    const size_t p = 2;
    const size_t q = 10;

    const double true_sigma_u_sq = 0.6;
    const double true_sigma_e_sq = 0.4;

    auto [y, X, Z, sigma_u_true, sigma_e_true] = generate_lmm_data(n, p, q, true_sigma_u_sq, true_sigma_e_sq);

    // Build variance component matrices
    std::vector<Eigen::MatrixXd> V_matrices(2);
    V_matrices[0] = Z * Z.transpose(); // Random effect covariance
    V_matrices[1] = Eigen::MatrixXd::Identity(n, n); // Residual

    // Initial parameters
    Eigen::VectorXd initial_params(2);
    initial_params << 0.5, 0.5;

    // Optimize using AI algorithm
    AverageInformationOptimizer::Options options;
    options.max_iterations = 100;
    options.tolerance = 1e-6;
    options.use_reml = true;
    options.verbose = false;

    auto result = AverageInformationOptimizer::optimize(y, X, V_matrices, initial_params, {}, options);

    REQUIRE(result.converged);
    REQUIRE(result.iterations < 100);

    // Check that estimates are reasonably close to true values
    // (Note: estimates won't be exact due to sampling variability)
    INFO("Estimated σ²_u: " << result.parameters(0) << " (true: " << sigma_u_true << ")");
    INFO("Estimated σ²_e: " << result.parameters(1) << " (true: " << sigma_e_true << ")");

    REQUIRE(result.parameters(0) > 0);
    REQUIRE(result.parameters(1) > 0);

    // Rough check (within reasonable bounds given sampling variability)
    REQUIRE(std::abs(result.parameters(0) - sigma_u_true) < 0.5);
    REQUIRE(std::abs(result.parameters(1) - sigma_e_true) < 0.5);
}

TEST_CASE("AI Optimizer: REML vs ML", "[ai][optimizer]") {
    const size_t n = 80;
    const size_t p = 2;
    const size_t q = 8;

    auto [y, X, Z, sigma_u_true, sigma_e_true] = generate_lmm_data(n, p, q, 0.7, 0.3);

    std::vector<Eigen::MatrixXd> V_matrices(2);
    V_matrices[0] = Z * Z.transpose();
    V_matrices[1] = Eigen::MatrixXd::Identity(n, n);

    Eigen::VectorXd initial_params(2);
    initial_params << 0.5, 0.5;

    // REML estimation
    AverageInformationOptimizer::Options reml_options;
    reml_options.use_reml = true;
    reml_options.verbose = false;

    auto reml_result = AverageInformationOptimizer::optimize(y, X, V_matrices, initial_params, {}, reml_options);

    // ML estimation
    AverageInformationOptimizer::Options ml_options;
    ml_options.use_reml = false;
    ml_options.verbose = false;

    auto ml_result = AverageInformationOptimizer::optimize(y, X, V_matrices, initial_params, {}, ml_options);

    REQUIRE(reml_result.converged);
    REQUIRE(ml_result.converged);

    // REML and ML should give different estimates (REML corrects for bias)
    REQUIRE(std::abs(reml_result.parameters(0) - ml_result.parameters(0)) > 1e-4);

    // Both should be positive
    REQUIRE(reml_result.parameters(0) > 0);
    REQUIRE(reml_result.parameters(1) > 0);
    REQUIRE(ml_result.parameters(0) > 0);
    REQUIRE(ml_result.parameters(1) > 0);

    INFO("REML: σ²_u=" << reml_result.parameters(0) << " σ²_e=" << reml_result.parameters(1));
    INFO("ML: σ²_u=" << ml_result.parameters(0) << " σ²_e=" << ml_result.parameters(1));
}

TEST_CASE("AI Optimizer: Fixed variance components", "[ai][optimizer]") {
    const size_t n = 60;
    const size_t p = 2;
    const size_t q = 6;

    auto [y, X, Z, sigma_u_true, sigma_e_true] = generate_lmm_data(n, p, q, 0.5, 0.5);

    std::vector<Eigen::MatrixXd> V_matrices(2);
    V_matrices[0] = Z * Z.transpose();
    V_matrices[1] = Eigen::MatrixXd::Identity(n, n);

    Eigen::VectorXd initial_params(2);
    initial_params << 0.5, 1.0; // Fix residual variance at 1.0

    // Fix the residual variance component
    std::vector<bool> fixed_components = {false, true};

    AverageInformationOptimizer::Options options;
    options.verbose = false;

    auto result = AverageInformationOptimizer::optimize(y, X, V_matrices, initial_params, fixed_components, options);

    REQUIRE(result.converged);

    // Residual variance should remain fixed
    REQUIRE(result.parameters(1) == 1.0);

    // Random effect variance should be estimated
    REQUIRE(result.parameters(0) != 0.5);
    REQUIRE(result.parameters(0) > 0);

    INFO("Estimated σ²_u: " << result.parameters(0) << " (σ²_e fixed at 1.0)");
}

TEST_CASE("AI Optimizer: All components fixed", "[ai][optimizer]") {
    const size_t n = 50;
    const size_t p = 2;
    const size_t q = 5;

    auto [y, X, Z, sigma_u_true, sigma_e_true] = generate_lmm_data(n, p, q, 0.6, 0.4);

    std::vector<Eigen::MatrixXd> V_matrices(2);
    V_matrices[0] = Z * Z.transpose();
    V_matrices[1] = Eigen::MatrixXd::Identity(n, n);

    Eigen::VectorXd initial_params(2);
    initial_params << 0.6, 0.4;

    // Fix all components
    std::vector<bool> fixed_components = {true, true};

    AverageInformationOptimizer::Options options;
    options.verbose = false;

    auto result = AverageInformationOptimizer::optimize(y, X, V_matrices, initial_params, fixed_components, options);

    REQUIRE(result.converged);
    REQUIRE(result.iterations == 0);
    REQUIRE(result.message == "All variance components are fixed");

    // Parameters should remain unchanged
    REQUIRE(result.parameters(0) == 0.6);
    REQUIRE(result.parameters(1) == 0.4);

    // Log-likelihood should be computed
    REQUIRE(std::isfinite(result.log_likelihood));
}

TEST_CASE("AI Optimizer: Convergence monitoring", "[ai][optimizer]") {
    const size_t n = 70;
    const size_t p = 2;
    const size_t q = 7;

    auto [y, X, Z, sigma_u_true, sigma_e_true] = generate_lmm_data(n, p, q, 0.8, 0.2);

    std::vector<Eigen::MatrixXd> V_matrices(2);
    V_matrices[0] = Z * Z.transpose();
    V_matrices[1] = Eigen::MatrixXd::Identity(n, n);

    // Start far from truth
    Eigen::VectorXd initial_params(2);
    initial_params << 0.1, 0.9;

    AverageInformationOptimizer::Options options;
    options.max_iterations = 50;
    options.tolerance = 1e-5;
    options.verbose = false;

    auto result = AverageInformationOptimizer::optimize(y, X, V_matrices, initial_params, {}, options);

    // Should converge even from far starting point
    REQUIRE(result.converged);

    INFO("Converged in " << result.iterations << " iterations");
    INFO("Final σ²_u: " << result.parameters(0) << ", σ²_e: " << result.parameters(1));

    // Check final score is near zero (optimum condition)
    REQUIRE(result.final_score.norm() < 0.1);

    // Check AI matrix is positive definite (at optimum)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(result.final_ai_matrix);
    REQUIRE(es.eigenvalues().minCoeff() > 0);
}

TEST_CASE("AI Optimizer: Multiple variance components", "[ai][optimizer]") {
    const size_t n = 90;
    const size_t p = 2;
    const size_t q1 = 5;
    const size_t q2 = 4;

    std::srand(200);
    Eigen::VectorXd y = Eigen::VectorXd::Random(n);
    Eigen::MatrixXd X(n, p);
    X.col(0) = Eigen::VectorXd::Ones(n);
    X.col(1) = Eigen::VectorXd::Random(n);

    Eigen::MatrixXd Z1 = create_random_effects_design(n, q1, 48);
    Eigen::MatrixXd Z2 = create_random_effects_design(n, q2, 49);

    // Three variance components: two random effects + residual
    std::vector<Eigen::MatrixXd> V_matrices(3);
    V_matrices[0] = Z1 * Z1.transpose();
    V_matrices[1] = Z2 * Z2.transpose();
    V_matrices[2] = Eigen::MatrixXd::Identity(n, n);

    Eigen::VectorXd initial_params(3);
    initial_params << 0.4, 0.3, 0.3;

    AverageInformationOptimizer::Options options;
    options.verbose = false;

    auto result = AverageInformationOptimizer::optimize(y, X, V_matrices, initial_params, {}, options);

    REQUIRE(result.converged);

    // All variance components should be positive
    REQUIRE(result.parameters(0) > 0);
    REQUIRE(result.parameters(1) > 0);
    REQUIRE(result.parameters(2) > 0);

    INFO("σ²_u1: " << result.parameters(0) << ", σ²_u2: " << result.parameters(1)
         << ", σ²_e: " << result.parameters(2));
}

TEST_CASE("AI Optimizer: Variance floor enforcement", "[ai][optimizer]") {
    const size_t n = 50;
    const size_t p = 2;
    const size_t q = 5;

    auto [y, X, Z, sigma_u_true, sigma_e_true] = generate_lmm_data(n, p, q, 0.01, 0.99);

    std::vector<Eigen::MatrixXd> V_matrices(2);
    V_matrices[0] = Z * Z.transpose();
    V_matrices[1] = Eigen::MatrixXd::Identity(n, n);

    Eigen::VectorXd initial_params(2);
    initial_params << 0.5, 0.5;

    AverageInformationOptimizer::Options options;
    options.min_variance = 1e-10;
    options.verbose = false;

    auto result = AverageInformationOptimizer::optimize(y, X, V_matrices, initial_params, {}, options);

    REQUIRE(result.converged);

    // All variances should be at least min_variance
    REQUIRE(result.parameters(0) >= options.min_variance);
    REQUIRE(result.parameters(1) >= options.min_variance);
}

TEST_CASE("AI Optimizer: Performance benchmark", "[ai][optimizer][!benchmark]") {
    const size_t n = 200;
    const size_t p = 3;
    const size_t q = 15;

    auto [y, X, Z, sigma_u_true, sigma_e_true] = generate_lmm_data(n, p, q, 0.6, 0.4, 999);

    std::vector<Eigen::MatrixXd> V_matrices(2);
    V_matrices[0] = Z * Z.transpose();
    V_matrices[1] = Eigen::MatrixXd::Identity(n, n);

    Eigen::VectorXd initial_params(2);
    initial_params << 0.5, 0.5;

    AverageInformationOptimizer::Options options;
    options.verbose = true;

    auto start = std::chrono::high_resolution_clock::now();
    auto result = AverageInformationOptimizer::optimize(y, X, V_matrices, initial_params, {}, options);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    INFO("AI optimizer: n=" << n << ", " << result.iterations << " iterations in " << duration << "ms");

    REQUIRE(result.converged);
    REQUIRE(duration < 10000); // Should complete in reasonable time
}
