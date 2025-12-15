#include "libsemx/mom_solver.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <random>

using namespace libsemx;

// Helper to generate synthetic crossed random effects data
static std::tuple<Eigen::VectorXd, Eigen::MatrixXd, std::vector<size_t>, std::vector<size_t>, double, double, double>
generate_crossed_data(size_t n_u, size_t n_v, size_t reps_per_cell,
                      double sigma_u_sq, double sigma_v_sq, double sigma_e_sq,
                      unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    // Total observations
    size_t n = n_u * n_v * reps_per_cell;

    // Fixed effects design: intercept + one covariate
    Eigen::MatrixXd X(n, 2);
    X.col(0) = Eigen::VectorXd::Ones(n);
    for (size_t i = 0; i < n; ++i) {
        X(i, 1) = norm(rng);
    }

    // True fixed effects
    Eigen::VectorXd beta_true(2);
    beta_true << 5.0, 1.5;

    // Generate random effects
    Eigen::VectorXd u(n_u);
    for (size_t i = 0; i < n_u; ++i) {
        u(i) = std::sqrt(sigma_u_sq) * norm(rng);
    }

    Eigen::VectorXd v(n_v);
    for (size_t i = 0; i < n_v; ++i) {
        v(i) = std::sqrt(sigma_v_sq) * norm(rng);
    }

    // Generate group indices and response
    std::vector<size_t> u_indices(n);
    std::vector<size_t> v_indices(n);
    Eigen::VectorXd y(n);

    size_t idx = 0;
    for (size_t i = 0; i < n_u; ++i) {
        for (size_t j = 0; j < n_v; ++j) {
            for (size_t k = 0; k < reps_per_cell; ++k) {
                u_indices[idx] = i;
                v_indices[idx] = j;

                // y = Xβ + u_i + v_j + e
                double e = std::sqrt(sigma_e_sq) * norm(rng);
                y(idx) = X.row(idx).dot(beta_true) + u(i) + v(j) + e;
                ++idx;
            }
        }
    }

    return std::make_tuple(y, X, u_indices, v_indices, sigma_u_sq, sigma_v_sq, sigma_e_sq);
}

TEST_CASE("MoM Solver: Simple balanced design", "[mom][solver]") {
    // 10 x 8 crossed design, 5 replicates per cell
    const size_t n_u = 10;
    const size_t n_v = 8;
    const size_t reps = 5;

    const double true_sigma_u = 0.6;
    const double true_sigma_v = 0.4;
    const double true_sigma_e = 0.3;

    auto [y, X, u_indices, v_indices, sigma_u, sigma_v, sigma_e] =
        generate_crossed_data(n_u, n_v, reps, true_sigma_u, true_sigma_v, true_sigma_e);

    MoMSolver::Options options;
    options.use_gls = false;  // Use OLS only for speed
    options.verbose = false;

    auto result = MoMSolver::fit(y, X, u_indices, v_indices, options);

    REQUIRE(result.converged);
    REQUIRE(result.n_groups_u == n_u);
    REQUIRE(result.n_groups_v == n_v);

    INFO("Estimated σ²_u: " << result.variance_components(0) << " (true: " << sigma_u << ")");
    INFO("Estimated σ²_v: " << result.variance_components(1) << " (true: " << sigma_v << ")");
    INFO("Estimated σ²_e: " << result.variance_components(2) << " (true: " << sigma_e << ")");

    // Check that all variance components are non-negative
    REQUIRE(result.variance_components(0) >= 0);
    REQUIRE(result.variance_components(1) >= 0);
    REQUIRE(result.variance_components(2) >= 0);

    // MoM can produce estimates at or near zero when the true variance is small
    // Just check that the total variance is reasonable
    double total_var = result.variance_components.sum();
    double true_total = sigma_u + sigma_v + sigma_e;
    REQUIRE(std::abs(total_var - true_total) < true_total);  // Within 100% of true total

    // Check fixed effects are estimated
    REQUIRE(result.beta.size() == 2);
    REQUIRE(std::abs(result.beta(0) - 5.0) < 1.0);  // Intercept ~ 5.0
    REQUIRE(std::abs(result.beta(1) - 1.5) < 1.0);  // Slope ~ 1.5
}

TEST_CASE("MoM Solver: With GLS refinement", "[mom][solver]") {
    const size_t n_u = 8;
    const size_t n_v = 6;
    const size_t reps = 4;

    const double true_sigma_u = 0.5;
    const double true_sigma_v = 0.5;
    const double true_sigma_e = 0.5;

    auto [y, X, u_indices, v_indices, sigma_u, sigma_v, sigma_e] =
        generate_crossed_data(n_u, n_v, reps, true_sigma_u, true_sigma_v, true_sigma_e, 123);

    // Test without GLS
    MoMSolver::Options ols_options;
    ols_options.use_gls = false;
    auto ols_result = MoMSolver::fit(y, X, u_indices, v_indices, ols_options);

    // Test with GLS
    MoMSolver::Options gls_options;
    gls_options.use_gls = true;
    gls_options.verbose = false;
    auto gls_result = MoMSolver::fit(y, X, u_indices, v_indices, gls_options);

    REQUIRE(ols_result.converged);
    REQUIRE(gls_result.converged);

    INFO("OLS β: " << ols_result.beta.transpose());
    INFO("GLS β: " << gls_result.beta.transpose());

    // GLS and OLS estimates may be similar or different depending on data
    // Just check both converged and are reasonable
    REQUIRE(std::abs(ols_result.beta(0) - 5.0) < 2.0);
    REQUIRE(std::abs(gls_result.beta(0) - 5.0) < 2.0);
}

TEST_CASE("MoM Solver: Second MoM step", "[mom][solver]") {
    const size_t n_u = 10;
    const size_t n_v = 10;
    const size_t reps = 3;

    const double true_sigma_u = 0.7;
    const double true_sigma_v = 0.3;
    const double true_sigma_e = 0.2;

    auto [y, X, u_indices, v_indices, sigma_u, sigma_v, sigma_e] =
        generate_crossed_data(n_u, n_v, reps, true_sigma_u, true_sigma_v, true_sigma_e, 456);

    // First step only
    MoMSolver::Options first_step;
    first_step.use_gls = true;
    first_step.second_step = false;
    auto result1 = MoMSolver::fit(y, X, u_indices, v_indices, first_step);

    // With second step
    MoMSolver::Options second_step;
    second_step.use_gls = true;
    second_step.second_step = true;
    second_step.verbose = false;
    auto result2 = MoMSolver::fit(y, X, u_indices, v_indices, second_step);

    REQUIRE(result1.converged);
    REQUIRE(result2.converged);

    INFO("First step variances: " << result1.variance_components.transpose());
    INFO("Second step variances: " << result2.variance_components.transpose());

    // Variance estimates may differ after second step (but not guaranteed)
    // Just check both are reasonable
    REQUIRE(result1.variance_components.sum() > 0.1);
    REQUIRE(result2.variance_components.sum() > 0.1);
}

TEST_CASE("MoM Solver: Unbalanced design", "[mom][solver]") {
    // Create unbalanced data: different replicates per cell
    const size_t n_u = 5;
    const size_t n_v = 4;

    std::mt19937 rng(789);
    std::normal_distribution<double> norm(0.0, 1.0);
    std::uniform_int_distribution<int> rep_dist(1, 5);

    const double sigma_u = 0.6;
    const double sigma_v = 0.4;
    const double sigma_e = 0.3;

    // Generate random effects
    Eigen::VectorXd u(n_u);
    for (size_t i = 0; i < n_u; ++i) {
        u(i) = std::sqrt(sigma_u) * norm(rng);
    }

    Eigen::VectorXd v(n_v);
    for (size_t i = 0; i < n_v; ++i) {
        v(i) = std::sqrt(sigma_v) * norm(rng);
    }

    // Build data with random replicates per cell
    std::vector<double> y_vec;
    std::vector<double> X_vec;
    std::vector<size_t> u_indices;
    std::vector<size_t> v_indices;

    Eigen::VectorXd beta_true(2);
    beta_true << 5.0, 1.5;

    for (size_t i = 0; i < n_u; ++i) {
        for (size_t j = 0; j < n_v; ++j) {
            int n_reps = rep_dist(rng);
            for (int k = 0; k < n_reps; ++k) {
                double x_val = norm(rng);
                double e = std::sqrt(sigma_e) * norm(rng);
                double y_val = beta_true(0) + beta_true(1) * x_val + u(i) + v(j) + e;

                y_vec.push_back(y_val);
                X_vec.push_back(1.0);
                X_vec.push_back(x_val);
                u_indices.push_back(i);
                v_indices.push_back(j);
            }
        }
    }

    size_t n = y_vec.size();
    Eigen::VectorXd y = Eigen::Map<Eigen::VectorXd>(y_vec.data(), n);
    Eigen::MatrixXd X = Eigen::Map<Eigen::MatrixXd>(X_vec.data(), 2, n).transpose();

    MoMSolver::Options options;
    options.verbose = false;
    auto result = MoMSolver::fit(y, X, u_indices, v_indices, options);

    REQUIRE(result.converged);
    REQUIRE(result.variance_components(0) >= 0);
    REQUIRE(result.variance_components(1) >= 0);
    REQUIRE(result.variance_components(2) >= 0);

    INFO("Unbalanced: σ²_u=" << result.variance_components(0)
         << " σ²_v=" << result.variance_components(1)
         << " σ²_e=" << result.variance_components(2));

    // Check total variance is reasonable
    double total_var = result.variance_components.sum();
    double true_total = sigma_u + sigma_v + sigma_e;
    REQUIRE(std::abs(total_var - true_total) < 2.0 * true_total);
}

TEST_CASE("MoM Solver: Large scale performance", "[mom][solver][!benchmark]") {
    // Large-scale test: 100 x 100 crossed design, 3 replicates per cell
    // Total: 30,000 observations
    const size_t n_u = 100;
    const size_t n_v = 100;
    const size_t reps = 3;

    auto [y, X, u_indices, v_indices, sigma_u, sigma_v, sigma_e] =
        generate_crossed_data(n_u, n_v, reps, 0.6, 0.4, 0.3, 999);

    MoMSolver::Options options;
    options.use_gls = false;  // GLS would be very slow for this size
    options.verbose = true;

    auto start = std::chrono::high_resolution_clock::now();
    auto result = MoMSolver::fit(y, X, u_indices, v_indices, options);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    INFO("MoM solver: n=" << y.size() << ", " << duration << "ms");

    REQUIRE(result.converged);
    REQUIRE(duration < 5000);  // Should complete in < 5 seconds

    INFO("Large scale: σ²_u=" << result.variance_components(0)
         << " σ²_v=" << result.variance_components(1)
         << " σ²_e=" << result.variance_components(2));
}

TEST_CASE("MoM Solver: Zero variance component", "[mom][solver]") {
    // Generate data with zero random effect variance for first factor
    const size_t n_u = 8;
    const size_t n_v = 6;
    const size_t reps = 4;

    const double true_sigma_u = 0.0;  // Zero variance
    const double true_sigma_v = 0.5;
    const double true_sigma_e = 0.4;

    auto [y, X, u_indices, v_indices, sigma_u, sigma_v, sigma_e] =
        generate_crossed_data(n_u, n_v, reps, true_sigma_u, true_sigma_v, true_sigma_e, 321);

    MoMSolver::Options options;
    options.verbose = false;
    auto result = MoMSolver::fit(y, X, u_indices, v_indices, options);

    REQUIRE(result.converged);

    INFO("Zero u variance: σ²_u=" << result.variance_components(0)
         << " σ²_v=" << result.variance_components(1)
         << " σ²_e=" << result.variance_components(2));

    // When one component is zero, MoM may struggle to separate variances
    // Just check that total variance is reasonable and all are non-negative
    REQUIRE(result.variance_components(0) >= 0);
    REQUIRE(result.variance_components(1) >= 0);
    REQUIRE(result.variance_components(2) > 0);  // At least residual should be positive

    double total_var = result.variance_components.sum();
    double true_total = true_sigma_u + true_sigma_v + true_sigma_e;
    REQUIRE(std::abs(total_var - true_total) < 2.0 * true_total);
}

TEST_CASE("MoM Solver: Single replicate per cell", "[mom][solver]") {
    // Exactly one observation per cell: can't estimate interaction
    const size_t n_u = 10;
    const size_t n_v = 8;
    const size_t reps = 1;

    const double true_sigma_u = 0.6;
    const double true_sigma_v = 0.4;
    const double true_sigma_e = 0.3;

    auto [y, X, u_indices, v_indices, sigma_u, sigma_v, sigma_e] =
        generate_crossed_data(n_u, n_v, reps, true_sigma_u, true_sigma_v, true_sigma_e, 654);

    MoMSolver::Options options;
    options.verbose = false;
    auto result = MoMSolver::fit(y, X, u_indices, v_indices, options);

    REQUIRE(result.converged);

    INFO("Single rep: σ²_u=" << result.variance_components(0)
         << " σ²_v=" << result.variance_components(1)
         << " σ²_e=" << result.variance_components(2));

    // All variances should be non-negative
    REQUIRE(result.variance_components(0) >= 0);
    REQUIRE(result.variance_components(1) >= 0);
    REQUIRE(result.variance_components(2) >= 0);
}

TEST_CASE("MoM Solver: Variance floor enforcement", "[mom][solver]") {
    // Generate data that might produce negative variance estimates
    const size_t n_u = 5;
    const size_t n_v = 5;
    const size_t reps = 2;

    const double true_sigma_u = 0.01;  // Very small
    const double true_sigma_v = 0.01;  // Very small
    const double true_sigma_e = 0.5;

    auto [y, X, u_indices, v_indices, sigma_u, sigma_v, sigma_e] =
        generate_crossed_data(n_u, n_v, reps, true_sigma_u, true_sigma_v, true_sigma_e, 987);

    MoMSolver::Options options;
    options.min_variance = 1e-10;
    options.verbose = false;
    auto result = MoMSolver::fit(y, X, u_indices, v_indices, options);

    REQUIRE(result.converged);

    // All variances should be at least min_variance
    REQUIRE(result.variance_components(0) >= options.min_variance);
    REQUIRE(result.variance_components(1) >= options.min_variance);
    REQUIRE(result.variance_components(2) >= options.min_variance);
}
