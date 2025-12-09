#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Dense>

#include "libsemx/likelihood_driver.hpp"
#include "libsemx/model_ir.hpp"

#include <cmath>
#include <vector>
#include <unordered_map>

TEST_CASE("LikelihoodDriver evaluates Gaussian Random Intercept Model", "[mixed][gaussian]") {
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("cluster", libsemx::VariableKind::Grouping);
    builder.add_variable("u_cluster", libsemx::VariableKind::Latent);
    
    // Random intercept: u ~ N(0, tau^2)
    builder.add_covariance("tau_sq", "diagonal", 1);
    builder.add_random_effect("u_cluster", {"cluster"}, "tau_sq");
    builder.add_edge(libsemx::EdgeKind::Regression, "u_cluster", "y", "1");

    auto model = builder.build();

    // Data: 2 clusters, 2 obs each
    // Cluster 1: y = [1, 2]
    // Cluster 2: y = [3, 4]
    std::vector<double> y = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> cluster = {1.0, 1.0, 2.0, 2.0};
    
    // Fixed effects: intercept = 0
    std::vector<double> preds = {0.0, 0.0, 0.0, 0.0};
    
    // Residual variance: sigma^2 = 1
    std::vector<double> disps = {1.0, 1.0, 1.0, 1.0};

    // Random effect variance: tau^2 = 1
    std::vector<double> tau_sq_params = {1.0};

    std::unordered_map<std::string, std::vector<double>> data = {
        {"y", y},
        {"cluster", cluster}
    };
    std::unordered_map<std::string, std::vector<double>> linear_predictors = {
        {"y", preds}
    };
    std::unordered_map<std::string, std::vector<double>> dispersions = {
        {"y", disps}
    };
    std::unordered_map<std::string, std::vector<double>> cov_params = {
        {"tau_sq", tau_sq_params}
    };

    libsemx::LikelihoodDriver driver;
    double loglik = driver.evaluate_model_loglik(model, data, linear_predictors, dispersions, cov_params);

    // Expected calculation:
    // V_i = J_2 * 1 + I_2 * 1 = [[2, 1], [1, 2]]
    // |V_i| = 3
    // V_i^-1 = 1/3 * [[2, -1], [-1, 2]]
    // Group 1: r = [1, 2]. r^T V^-1 r = 1/3 * (2*1 - 2 + 2*2 - 1*1) ? No.
    // r^T V^-1 r = 1/3 * [1, 2] * [0, 3]^T = 2.
    // Group 2: r = [3, 4]. r^T V^-1 r = 1/3 * [3, 4] * [2, 5]^T = 1/3 * (6 + 20) = 26/3.
    
    // LL_i = -0.5 * (2*log(2pi) + log(3) + quad_form)
    // LL_1 = -0.5 * (2*log(2pi) + log(3) + 2)
    // LL_2 = -0.5 * (2*log(2pi) + log(3) + 26/3)
    
    double log_2pi = std::log(2.0 * 3.14159265358979323846);
    double expected_ll1 = -0.5 * (2 * log_2pi + std::log(3.0) + 2.0);
    double expected_ll2 = -0.5 * (2 * log_2pi + std::log(3.0) + 26.0/3.0);
    double expected_total = expected_ll1 + expected_ll2;

    REQUIRE_THAT(loglik, Catch::Matchers::WithinRel(expected_total, 1e-5));
}

TEST_CASE("LikelihoodDriver evaluates Gaussian Random Slope Model", "[mixed][gaussian]") {
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("cluster", libsemx::VariableKind::Grouping);
    builder.add_variable("x", libsemx::VariableKind::Latent); // Predictor for random slope (Latent to skip likelihood eval)
    builder.add_variable("u_slope", libsemx::VariableKind::Latent);
    
    // Random slope: u ~ N(0, tau^2)
    builder.add_covariance("tau_sq", "diagonal", 1);
    builder.add_random_effect("u_slope", {"cluster", "x"}, "tau_sq");
    builder.add_edge(libsemx::EdgeKind::Regression, "u_slope", "y", "1");

    auto model = builder.build();

    // Data: 1 cluster, 2 obs
    // Cluster 1: y = [1, 2], x = [1, 2]
    std::vector<double> y = {1.0, 2.0};
    std::vector<double> cluster = {1.0, 1.0};
    std::vector<double> x = {1.0, 2.0};
    
    // Fixed effects: intercept = 0
    std::vector<double> preds = {0.0, 0.0};
    
    // Residual variance: sigma^2 = 1
    std::vector<double> disps = {1.0, 1.0};

    // Random effect variance: tau^2 = 0.5
    std::vector<double> tau_sq_params = {0.5};

    std::unordered_map<std::string, std::vector<double>> data = {
        {"y", y},
        {"cluster", cluster},
        {"x", x}
    };
    std::unordered_map<std::string, std::vector<double>> linear_predictors = {
        {"y", preds}
    };
    std::unordered_map<std::string, std::vector<double>> dispersions = {
        {"y", disps}
    };
    std::unordered_map<std::string, std::vector<double>> cov_params = {
        {"tau_sq", tau_sq_params}
    };

    libsemx::LikelihoodDriver driver;
    double loglik = driver.evaluate_model_loglik(model, data, linear_predictors, dispersions, cov_params);

    // Expected calculation:
    // Z_i = [[1], [2]] (from x)
    // G = [0.5]
    // Z G Z^T = [[1], [2]] * [0.5] * [[1, 2]] = [[0.5, 1.0], [1.0, 2.0]]
    // R = I_2
    // V = [[1.5, 1.0], [1.0, 3.0]]
    // |V| = 4.5 - 1 = 3.5
    // V^-1 = 1/3.5 * [[3.0, -1.0], [-1.0, 1.5]]
    
    // r = [1, 2]
    // r^T V^-1 r = 1/3.5 * [1, 2] * [[3, -1], [-1, 1.5]] * [1, 2]^T
    // = 1/3.5 * [1, 2] * [1, 2]^T
    // = 1/3.5 * (1 + 4) = 5/3.5 = 10/7
    
    // LL = -0.5 * (2*log(2pi) + log(3.5) + 10/7)
    
    double log_2pi = std::log(2.0 * 3.14159265358979323846);
    double expected = -0.5 * (2 * log_2pi + std::log(3.5) + 10.0/7.0);

    REQUIRE_THAT(loglik, Catch::Matchers::WithinRel(expected, 1e-5));
}

TEST_CASE("LikelihoodDriver log-likelihood handles crossed grouping factors", "[mixed][gaussian][loglik]") {
    using Catch::Matchers::WithinRel;
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("x", libsemx::VariableKind::Latent);
    builder.add_variable("cluster_a", libsemx::VariableKind::Grouping);
    builder.add_variable("cluster_b", libsemx::VariableKind::Grouping);
    builder.add_variable("u_a", libsemx::VariableKind::Latent);
    builder.add_variable("u_b", libsemx::VariableKind::Latent);

    builder.add_covariance("G_a", "diagonal", 1);
    builder.add_covariance("G_b", "diagonal", 1);
    builder.add_random_effect("u_a", {"cluster_a"}, "G_a");
    builder.add_random_effect("u_b", {"cluster_b"}, "G_b");
    builder.add_edge(libsemx::EdgeKind::Regression, "u_a", "y", "1");
    builder.add_edge(libsemx::EdgeKind::Regression, "u_b", "y", "1");

    auto model = builder.build();

    std::vector<double> y = {2.0, 2.5, 3.5, 4.0};
    std::vector<double> x = {0.2, 0.4, 1.2, 1.4};
    std::vector<double> cluster_a = {1.0, 1.0, 2.0, 2.0};
    std::vector<double> cluster_b = {10.0, 20.0, 10.0, 20.0};

    std::unordered_map<std::string, std::vector<double>> data = {
        {"y", y},
        {"x", x},
        {"cluster_a", cluster_a},
        {"cluster_b", cluster_b}
    };

    double beta = 0.9;
    double sigma_a = 0.6;
    double sigma_b = 0.3;

    std::unordered_map<std::string, std::vector<double>> linear_predictors = {
        {"y", {beta * 0.2, beta * 0.4, beta * 1.2, beta * 1.4}}
    };

    std::unordered_map<std::string, std::vector<double>> dispersions = {
        {"y", {1.0, 1.0, 1.0, 1.0}}
    };

    std::unordered_map<std::string, std::vector<double>> covariance_parameters = {
        {"G_a", {sigma_a}},
        {"G_b", {sigma_b}}
    };

    libsemx::LikelihoodDriver driver;
    double loglik = driver.evaluate_model_loglik(
        model,
        data,
        linear_predictors,
        dispersions,
        covariance_parameters,
        {},
        {},
        {});

    Eigen::Matrix4d V = Eigen::Matrix4d::Zero();
    for (int i = 0; i < 4; ++i) {
        V(i, i) = 1.0; // residual variance
    }

    auto add_group = [&](const std::vector<int>& indices, double variance) {
        for (int r : indices) {
            for (int c : indices) {
                V(r, c) += variance;
            }
        }
    };

    add_group({0, 1}, sigma_a);
    add_group({2, 3}, sigma_a);
    add_group({0, 2}, sigma_b);
    add_group({1, 3}, sigma_b);

    Eigen::LLT<Eigen::Matrix4d> llt(V);
    REQUIRE(llt.info() == Eigen::Success);

    Eigen::Vector4d resid;
    for (int i = 0; i < 4; ++i) {
        resid(i) = y[i] - linear_predictors["y"][i];
    }
    Eigen::Vector4d alpha = llt.solve(resid);
    double quad = resid.dot(alpha);

    Eigen::Matrix4d L = llt.matrixL();
    double log_det = 0.0;
    for (int i = 0; i < 4; ++i) {
        log_det += std::log(L(i, i));
    }
    log_det *= 2.0;

    double log_2pi = std::log(2.0 * 3.14159265358979323846);
    double expected = -0.5 * (4 * log_2pi + log_det + quad);

    REQUIRE_THAT(loglik, WithinRel(expected, 1e-6));
}
