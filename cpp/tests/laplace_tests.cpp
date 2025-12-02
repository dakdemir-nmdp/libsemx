#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "libsemx/likelihood_driver.hpp"
#include "libsemx/model_ir.hpp"

#include <cmath>
#include <vector>
#include <unordered_map>

TEST_CASE("LikelihoodDriver evaluates Laplace for Binomial GLMM", "[laplace][mixed]") {
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "binomial");
    builder.add_variable("cluster", libsemx::VariableKind::Grouping);
    
    // Random intercept: u ~ N(0, tau^2)
    builder.add_covariance("tau_sq", "diagonal", 1);
    builder.add_random_effect("u_cluster", {"cluster"}, "tau_sq");

    auto model = builder.build();

    // Data: 1 cluster, 2 obs
    // y = [1, 0]
    std::vector<double> y = {1.0, 0.0};
    std::vector<double> cluster = {1.0, 1.0};
    
    // Fixed effects: mu=0 => preds=0
    std::vector<double> preds = {0.0, 0.0};
    
    // Dispersion: 1 (ignored for binomial usually, but passed)
    std::vector<double> disps = {1.0, 1.0};

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

    // Expected: -2 log(2) - 0.5 log(1.5)
    double expected = -2.0 * std::log(2.0) - 0.5 * std::log(1.5);
    
    REQUIRE_THAT(loglik, Catch::Matchers::WithinRel(expected, 1e-4));
}

TEST_CASE("Laplace fit mixes new covariance structures", "[laplace][covariance]") {
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "binomial");
    builder.add_variable("cluster", libsemx::VariableKind::Grouping);
    builder.add_variable("intercept", libsemx::VariableKind::Latent);
    builder.add_variable("x", libsemx::VariableKind::Latent);
    builder.add_variable("z0", libsemx::VariableKind::Latent);
    builder.add_variable("z1", libsemx::VariableKind::Latent);

    builder.add_covariance("G_cs", "compound_symmetry", 2);
    builder.add_covariance("G_diag", "diagonal", 1);
    builder.add_covariance("G_toep", "toeplitz", 2);

    builder.add_random_effect("u_cs", {"cluster", "intercept", "x"}, "G_cs");
    builder.add_random_effect("u_diag", {"cluster"}, "G_diag");
    builder.add_random_effect("u_toep", {"cluster", "z0", "z1"}, "G_toep");

    auto model = builder.build();

    const std::vector<double> y = {0.0, 1.0, 0.0, 1.0};
    const std::vector<double> cluster = {1.0, 1.0, 2.0, 2.0};
    const std::vector<double> intercept = {1.0, 1.0, 1.0, 1.0};
    const std::vector<double> x = {-1.0, -0.2, 0.4, 1.1};
    const std::vector<double> z0 = {1.0, 0.0, 1.0, 0.0};
    const std::vector<double> z1 = {0.0, 1.0, 0.0, 1.0};

    std::unordered_map<std::string, std::vector<double>> data = {
        {"y", y},
        {"cluster", cluster},
        {"intercept", intercept},
        {"x", x},
        {"z0", z0},
        {"z1", z1}
    };

    std::unordered_map<std::string, std::vector<double>> linear_predictors = {
        {"y", {0.0, 0.0, 0.0, 0.0}}
    };

    std::unordered_map<std::string, std::vector<double>> dispersions = {
        {"y", {1.0, 1.0, 1.0, 1.0}}
    };

    std::unordered_map<std::string, std::vector<double>> covariance_parameters = {
        {"G_cs", {0.4, 0.15}},
        {"G_diag", {0.3}},
        {"G_toep", {0.6, 2.2}}
    };

    libsemx::LikelihoodDriver driver;
    double loglik = driver.evaluate_model_loglik(model, data, linear_predictors, dispersions, covariance_parameters);

    REQUIRE(std::isfinite(loglik));
    REQUIRE_THAT(loglik, Catch::Matchers::WithinAbs(-3.224501, 1e-3));
}

TEST_CASE("LikelihoodDriver evaluates Laplace for Poisson GLMM", "[laplace][poisson]") {
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "poisson");
    builder.add_variable("cluster", libsemx::VariableKind::Grouping);
    
    // Random intercept: u ~ N(0, tau^2)
    builder.add_covariance("tau_sq", "diagonal", 1);
    builder.add_random_effect("u_cluster", {"cluster"}, "tau_sq");

    auto model = builder.build();

    // Data: 1 cluster, 2 obs
    // y = [1, 2]
    std::vector<double> y = {1.0, 2.0};
    std::vector<double> cluster = {1.0, 1.0};
    
    // Fixed effects: mu=exp(0)=1 => preds=0
    std::vector<double> preds = {0.0, 0.0};
    
    // Dispersion: 1 (ignored)
    std::vector<double> disps = {1.0, 1.0};

    // Random effect variance: tau^2 = 0.5
    std::vector<double> tau_sq_params = {0.5};

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

    REQUIRE(std::isfinite(loglik));
}

TEST_CASE("LikelihoodDriver evaluates Laplace for Negative Binomial GLMM", "[laplace][nbinom]") {
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "negative_binomial");
    builder.add_variable("cluster", libsemx::VariableKind::Grouping);
    
    // Random intercept: u ~ N(0, tau^2)
    builder.add_covariance("tau_sq", "diagonal", 1);
    builder.add_random_effect("u_cluster", {"cluster"}, "tau_sq");

    auto model = builder.build();

    // Data: 1 cluster, 2 obs
    // y = [1, 2]
    std::vector<double> y = {1.0, 2.0};
    std::vector<double> cluster = {1.0, 1.0};
    
    // Fixed effects: mu=exp(0)=1 => preds=0
    std::vector<double> preds = {0.0, 0.0};
    
    // Dispersion: k=2
    std::vector<double> disps = {2.0, 2.0};

    // Random effect variance: tau^2 = 0.5
    std::vector<double> tau_sq_params = {0.5};

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

    REQUIRE(std::isfinite(loglik));
}
