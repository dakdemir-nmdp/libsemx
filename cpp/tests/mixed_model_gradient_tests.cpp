#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "libsemx/likelihood_driver.hpp"
#include "libsemx/model_ir.hpp"

#include <vector>
#include <unordered_map>
#include <cmath>
#include <iostream>

using namespace libsemx;

TEST_CASE("LikelihoodDriver evaluates analytic gradients for Gaussian mixed model", "[gradient][mixed]") {
    ModelIRBuilder builder;
    builder.add_variable("y", VariableKind::Observed, "gaussian");
    builder.add_variable("x", VariableKind::Latent);
    builder.add_variable("cluster", VariableKind::Grouping);
    
    // y = beta * x + u + e
    builder.add_variable("u", VariableKind::Latent);
    builder.add_edge(EdgeKind::Regression, "x", "y", "beta");
    
    // Random intercept u ~ N(0, sigma_u^2)
    builder.add_covariance("G", "diagonal", 1);
    builder.add_random_effect("u", {"cluster"}, "G");
    builder.add_edge(EdgeKind::Regression, "u", "y", "1");
    
    auto model = builder.build();
    
    // Data: 2 clusters, 2 obs each
    std::vector<double> y = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> x = {0.5, 0.5, 1.0, 1.0};
    std::vector<double> cluster = {1.0, 1.0, 2.0, 2.0};
    
    std::unordered_map<std::string, std::vector<double>> data = {
        {"y", y},
        {"x", x},
        {"cluster", cluster}
    };
    
    // Parameters
    double beta = 1.5;
    double sigma_u_sq = 0.5;
    
    std::unordered_map<std::string, std::vector<double>> linear_predictors;
    linear_predictors["y"] = {beta * 0.5, beta * 0.5, beta * 1.0, beta * 1.0};
    
    std::unordered_map<std::string, std::vector<double>> dispersions;
    dispersions["y"] = {1.0, 1.0, 1.0, 1.0}; // Fixed residual variance = 1
    
    std::unordered_map<std::string, std::vector<double>> covariance_parameters;
    covariance_parameters["G"] = {sigma_u_sq};
    
    LikelihoodDriver driver;
    
    // Analytic gradient
    auto gradients = driver.evaluate_model_gradient(model, data, linear_predictors, dispersions, covariance_parameters, {}, {}, {}, EstimationMethod::ML);
    
    // Finite differences
    double epsilon = 1e-6;
    
    // Check beta
    {
        double beta_plus = beta + epsilon;
        std::unordered_map<std::string, std::vector<double>> lp_plus = linear_predictors;
        for(size_t i=0; i<4; ++i) lp_plus["y"][i] = beta_plus * x[i];
        
        double l_plus = driver.evaluate_model_loglik(model, data, lp_plus, dispersions, covariance_parameters, {}, {}, {}, EstimationMethod::ML);
        
        double beta_minus = beta - epsilon;
        std::unordered_map<std::string, std::vector<double>> lp_minus = linear_predictors;
        for(size_t i=0; i<4; ++i) lp_minus["y"][i] = beta_minus * x[i];
        
        double l_minus = driver.evaluate_model_loglik(model, data, lp_minus, dispersions, covariance_parameters, {}, {}, {}, EstimationMethod::ML);
        
        double fd_grad = (l_plus - l_minus) / (2 * epsilon);
        
        REQUIRE_THAT(gradients["beta"], Catch::Matchers::WithinRel(fd_grad, 1e-4));
    }
    
    // Check sigma_u_sq
    {
        double sigma_plus = sigma_u_sq + epsilon;
        std::unordered_map<std::string, std::vector<double>> cp_plus = covariance_parameters;
        cp_plus["G"] = {sigma_plus};
        
        double l_plus = driver.evaluate_model_loglik(model, data, linear_predictors, dispersions, cp_plus, {}, {}, {}, EstimationMethod::ML);
        
        double sigma_minus = sigma_u_sq - epsilon;
        std::unordered_map<std::string, std::vector<double>> cp_minus = covariance_parameters;
        cp_minus["G"] = {sigma_minus};
        
        double l_minus = driver.evaluate_model_loglik(model, data, linear_predictors, dispersions, cp_minus, {}, {}, {}, EstimationMethod::ML);
        
        double fd_grad = (l_plus - l_minus) / (2 * epsilon);
        
        // Parameter name for diagonal covariance is "G_0" in ModelObjective, but evaluate_model_gradient returns map by param_id?
        // Wait, evaluate_model_gradient returns map with keys from edge.parameter_id.
        // But for covariance parameters, what keys does it use?
        // It currently doesn't return covariance gradients in the GLM case (empty).
        // I need to decide on keys.
        // ModelObjective maps "G_0" to index.
        // evaluate_model_gradient should probably return "G_0", "G_1", etc. or a nested map?
        // The signature returns flat map <string, double>.
        // So I should use "G_0" etc.
        
        REQUIRE_THAT(gradients["G_0"], Catch::Matchers::WithinRel(fd_grad, 1e-4));
    }
}

TEST_CASE("LikelihoodDriver analytic gradients handle multiple random effects", "[gradient][mixed]") {
    ModelIRBuilder builder;
    builder.add_variable("y", VariableKind::Observed, "gaussian");
    builder.add_variable("x", VariableKind::Latent);
    builder.add_variable("cluster", VariableKind::Grouping);
    builder.add_variable("u_intercept", VariableKind::Latent);
    builder.add_variable("u_slope", VariableKind::Latent);

    builder.add_edge(EdgeKind::Regression, "x", "y", "beta");

    builder.add_covariance("G_intercept", "diagonal", 1);
    builder.add_covariance("G_slope", "diagonal", 1);

    builder.add_random_effect("u_intercept", {"cluster"}, "G_intercept");
    builder.add_random_effect("u_slope", {"cluster", "x"}, "G_slope");
    builder.add_edge(EdgeKind::Regression, "u_intercept", "y", "1");
    builder.add_edge(EdgeKind::Regression, "u_slope", "y", "1");

    auto model = builder.build();

    std::vector<double> y = {1.1, 1.9, 3.2, 4.5};
    std::vector<double> x = {0.5, 0.6, 1.0, 1.5};
    std::vector<double> cluster = {1.0, 1.0, 2.0, 2.0};

    std::unordered_map<std::string, std::vector<double>> data = {
        {"y", y},
        {"x", x},
        {"cluster", cluster}
    };

    double beta = 1.3;
    double sigma_intercept = 0.4;
    double sigma_slope = 0.2;

    std::unordered_map<std::string, std::vector<double>> linear_predictors;
    linear_predictors["y"] = {beta * 0.5, beta * 0.6, beta * 1.0, beta * 1.5};

    std::unordered_map<std::string, std::vector<double>> dispersions;
    dispersions["y"] = {1.0, 1.0, 1.0, 1.0};

    std::unordered_map<std::string, std::vector<double>> covariance_parameters;
    covariance_parameters["G_intercept"] = {sigma_intercept};
    covariance_parameters["G_slope"] = {sigma_slope};

    LikelihoodDriver driver;
    auto gradients = driver.evaluate_model_gradient(
        model,
        data,
        linear_predictors,
        dispersions,
        covariance_parameters,
        {},
        {},
        {});

    double epsilon = 1e-6;

    auto loglik = [&](double b, double si, double ss) {
        std::unordered_map<std::string, std::vector<double>> lp = linear_predictors;
        for (size_t i = 0; i < x.size(); ++i) {
            lp["y"][i] = b * x[i];
        }
        std::unordered_map<std::string, std::vector<double>> covs = covariance_parameters;
        covs["G_intercept"][0] = si;
        covs["G_slope"][0] = ss;
        return driver.evaluate_model_loglik(model, data, lp, dispersions, covs, {}, {}, {});
    };

    {
        double fd = (loglik(beta + epsilon, sigma_intercept, sigma_slope) -
                     loglik(beta - epsilon, sigma_intercept, sigma_slope)) / (2 * epsilon);
        REQUIRE_THAT(gradients["beta"], Catch::Matchers::WithinRel(fd, 1e-4));
    }

    {
        double fd = (loglik(beta, sigma_intercept + epsilon, sigma_slope) -
                     loglik(beta, sigma_intercept - epsilon, sigma_slope)) / (2 * epsilon);
        REQUIRE_THAT(gradients["G_intercept_0"], Catch::Matchers::WithinRel(fd, 1e-4));
    }

    {
        double fd = (loglik(beta, sigma_intercept, sigma_slope + epsilon) -
                     loglik(beta, sigma_intercept, sigma_slope - epsilon)) / (2 * epsilon);
        REQUIRE_THAT(gradients["G_slope_0"], Catch::Matchers::WithinRel(fd, 1e-4));
    }
}

TEST_CASE("LikelihoodDriver analytic gradients handle crossed grouping factors", "[gradient][mixed]") {
    ModelIRBuilder builder;
    builder.add_variable("y", VariableKind::Observed, "gaussian");
    builder.add_variable("x", VariableKind::Latent);
    builder.add_variable("cluster_a", VariableKind::Grouping);
    builder.add_variable("cluster_b", VariableKind::Grouping);
    builder.add_variable("u_a", VariableKind::Latent);
    builder.add_variable("u_b", VariableKind::Latent);

    builder.add_edge(EdgeKind::Regression, "x", "y", "beta");

    builder.add_covariance("G_a", "diagonal", 1);
    builder.add_covariance("G_b", "diagonal", 1);

    builder.add_random_effect("u_a", {"cluster_a"}, "G_a");
    builder.add_random_effect("u_b", {"cluster_b"}, "G_b");
    builder.add_edge(EdgeKind::Regression, "u_a", "y", "1");
    builder.add_edge(EdgeKind::Regression, "u_b", "y", "1");

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

    std::unordered_map<std::string, std::vector<double>> linear_predictors;
    linear_predictors["y"] = {beta * 0.2, beta * 0.4, beta * 1.2, beta * 1.4};

    std::unordered_map<std::string, std::vector<double>> dispersions;
    dispersions["y"] = {1.0, 1.0, 1.0, 1.0};

    std::unordered_map<std::string, std::vector<double>> covariance_parameters;
    covariance_parameters["G_a"] = {sigma_a};
    covariance_parameters["G_b"] = {sigma_b};

    LikelihoodDriver driver;
    auto gradients = driver.evaluate_model_gradient(
        model,
        data,
        linear_predictors,
        dispersions,
        covariance_parameters,
        {},
        {},
        {});

    auto loglik = [&](double b, double sa, double sb) {
        std::unordered_map<std::string, std::vector<double>> lp = linear_predictors;
        for (size_t i = 0; i < x.size(); ++i) {
            lp["y"][i] = b * x[i];
        }
        std::unordered_map<std::string, std::vector<double>> covs = covariance_parameters;
        covs["G_a"][0] = sa;
        covs["G_b"][0] = sb;
        return driver.evaluate_model_loglik(model, data, lp, dispersions, covs, {}, {}, {});
    };

    double eps = 1e-6;

    {
        double fd = (loglik(beta + eps, sigma_a, sigma_b) - loglik(beta - eps, sigma_a, sigma_b)) / (2 * eps);
        REQUIRE_THAT(gradients["beta"], Catch::Matchers::WithinRel(fd, 1e-4));
    }

    {
        double fd = (loglik(beta, sigma_a + eps, sigma_b) - loglik(beta, sigma_a - eps, sigma_b)) / (2 * eps);
        REQUIRE_THAT(gradients["G_a_0"], Catch::Matchers::WithinRel(fd, 1e-4));
    }

    {
        double fd = (loglik(beta, sigma_a, sigma_b + eps) - loglik(beta, sigma_a, sigma_b - eps)) / (2 * eps);
        REQUIRE_THAT(gradients["G_b_0"], Catch::Matchers::WithinRel(fd, 1e-4));
    }
}

TEST_CASE("LikelihoodDriver Laplace gradients match finite differences", "[gradient][laplace]") {
    ModelIRBuilder builder;
    builder.add_variable("y", VariableKind::Observed, "binomial");
    builder.add_variable("x", VariableKind::Exogenous);
    builder.add_variable("cluster", VariableKind::Grouping);
    builder.add_variable("u", VariableKind::Latent);

    builder.add_edge(EdgeKind::Regression, "x", "y", "beta");

    builder.add_covariance("G", "diagonal", 1);
    builder.add_random_effect("u", {"cluster"}, "G");
    builder.add_edge(EdgeKind::Regression, "u", "y", "1");

    auto model = builder.build();

    std::vector<double> y = {0, 1, 0, 1, 0, 1};
    std::vector<double> x = {-1.0, -0.5, 0.2, 0.7, 1.0, 1.5};
    std::vector<double> cluster = {1.0, 1.0, 2.0, 2.0, 3.0, 3.0};

    std::unordered_map<std::string, std::vector<double>> data = {
        {"y", y},
        {"x", x},
        {"cluster", cluster}
    };

    double beta = 0.8;
    double sigma = 0.6;

    std::unordered_map<std::string, std::vector<double>> linear_predictors;
    linear_predictors["y"].resize(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        linear_predictors["y"][i] = beta * x[i];
    }

    std::unordered_map<std::string, std::vector<double>> dispersions;
    dispersions["y"] = std::vector<double>(y.size(), 1.0);

    std::unordered_map<std::string, std::vector<double>> covariance_parameters;
    covariance_parameters["G"] = {sigma};

    LikelihoodDriver driver;
    auto gradients = driver.evaluate_model_gradient(
        model,
        data,
        linear_predictors,
        dispersions,
        covariance_parameters,
        {},
        {},
        {},
        EstimationMethod::ML);

    auto loglik = [&](double b, double s) {
        std::unordered_map<std::string, std::vector<double>> lp = linear_predictors;
        for (size_t i = 0; i < x.size(); ++i) {
            lp["y"][i] = b * x[i];
        }
        std::unordered_map<std::string, std::vector<double>> covs = covariance_parameters;
        covs["G"][0] = s;
        return driver.evaluate_model_loglik(model, data, lp, dispersions, covs, {}, {}, {}, EstimationMethod::ML);
    };

    const double eps = 1e-5;
    double fd_beta = (loglik(beta + eps, sigma) - loglik(beta - eps, sigma)) / (2 * eps);
    double fd_sigma = (loglik(beta, sigma + eps) - loglik(beta, sigma - eps)) / (2 * eps);

    REQUIRE_THAT(gradients.at("beta"), Catch::Matchers::WithinRel(fd_beta, 1e-4));
    REQUIRE_THAT(gradients.at("G_0"), Catch::Matchers::WithinRel(fd_sigma, 1e-4));
}
