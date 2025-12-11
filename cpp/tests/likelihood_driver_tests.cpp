#include "libsemx/likelihood_driver.hpp"
#include "libsemx/gaussian_outcome.hpp"
#include "libsemx/model_ir.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <memory>
#include <unordered_map>
#include <vector>

TEST_CASE("LikelihoodDriver evaluates total log-likelihood for Gaussian outcomes", "[likelihood_driver]") {
    libsemx::LikelihoodDriver driver;
    libsemx::GaussianOutcome gaussian;

    SECTION("Single observation") {
        std::vector<double> observed = {1.0};
        std::vector<double> linear_predictors = {0.0};
        std::vector<double> dispersions = {1.0};

        const double total_loglik = driver.evaluate_total_loglik(observed, linear_predictors, dispersions, gaussian);

        // For Gaussian, loglik = -0.5 * (log(2*pi*variance) + residual^2 / variance)
        // residual = 1.0 - 0.0 = 1.0, variance = 1.0
        // loglik = -0.5 * (log(2*pi*1) + 1^2 / 1) = -0.5 * (log(2*pi) + 1)
        const double expected = -0.5 * (std::log(2 * 3.141592653589793) + 1.0);
        REQUIRE_THAT(total_loglik, Catch::Matchers::WithinRel(expected));
    }

    SECTION("Multiple observations") {
        std::vector<double> observed = {1.0, 2.0, 3.0};
        std::vector<double> linear_predictors = {0.5, 1.5, 2.5};
        std::vector<double> dispersions = {1.0, 1.0, 1.0};

        const double total_loglik = driver.evaluate_total_loglik(observed, linear_predictors, dispersions, gaussian);

        // Sum of individual logliks
        double expected = 0.0;
        for (size_t i = 0; i < observed.size(); ++i) {
            const double residual = observed[i] - linear_predictors[i];
            expected += -0.5 * (std::log(2 * 3.141592653589793 * 1.0) + residual * residual / 1.0);
        }
        REQUIRE_THAT(total_loglik, Catch::Matchers::WithinRel(expected));
    }

    SECTION("Empty vectors") {
        std::vector<double> observed;
        std::vector<double> linear_predictors;
        std::vector<double> dispersions;

        const double total_loglik = driver.evaluate_total_loglik(observed, linear_predictors, dispersions, gaussian);
        REQUIRE(total_loglik == 0.0);
    }
}

TEST_CASE("LikelihoodDriver throws on mismatched vector sizes", "[likelihood_driver]") {
    libsemx::LikelihoodDriver driver;
    libsemx::GaussianOutcome gaussian;

    std::vector<double> observed = {1.0, 2.0};
    std::vector<double> linear_predictors = {0.0};
    std::vector<double> dispersions = {1.0, 1.0};

    REQUIRE_THROWS_AS(driver.evaluate_total_loglik(observed, linear_predictors, dispersions, gaussian), std::invalid_argument);
}

TEST_CASE("LikelihoodDriver evaluates total log-likelihood for mixed outcomes", "[likelihood_driver]") {
    libsemx::LikelihoodDriver driver;
    libsemx::GaussianOutcome gaussian;

    SECTION("Mixed families") {
        std::vector<double> observed = {1.0, 2.0};
        std::vector<double> linear_predictors = {0.0, 1.0};
        std::vector<double> dispersions = {1.0, 1.0};
        std::vector<const libsemx::OutcomeFamily*> families = {&gaussian, &gaussian};

        const double total_loglik = driver.evaluate_total_loglik_mixed(observed, linear_predictors, dispersions, families);

        // Should be same as single family case
        const double single_loglik = driver.evaluate_total_loglik(observed, linear_predictors, dispersions, gaussian);
        REQUIRE_THAT(total_loglik, Catch::Matchers::WithinRel(single_loglik));
    }

    SECTION("Throws on mismatched family vector size") {
        std::vector<double> observed = {1.0};
        std::vector<double> linear_predictors = {0.0};
        std::vector<double> dispersions = {1.0};
        std::vector<const libsemx::OutcomeFamily*> families = {&gaussian, &gaussian};

        REQUIRE_THROWS_AS(driver.evaluate_total_loglik_mixed(observed, linear_predictors, dispersions, families), std::invalid_argument);
    }
}

TEST_CASE("LikelihoodDriver evaluates model log-likelihood with ModelIR", "[likelihood_driver]") {
    libsemx::LikelihoodDriver driver;
    libsemx::ModelIRBuilder builder;

    builder.add_variable("y1", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("y2", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("latent", libsemx::VariableKind::Latent, "");

    const auto model = builder.build();

    std::unordered_map<std::string, std::vector<double>> data = {
        {"y1", {1.0, 2.0}},
        {"y2", {0.5, 1.5}}
    };
    std::unordered_map<std::string, std::vector<double>> linear_predictors = {
        {"y1", {0.0, 1.0}},
        {"y2", {0.0, 1.0}}
    };
    std::unordered_map<std::string, std::vector<double>> dispersions = {
        {"y1", {1.0, 1.0}},
        {"y2", {1.0, 1.0}}
    };

    const double total_loglik = driver.evaluate_model_loglik(model, data, linear_predictors, dispersions);

    // Should sum logliks for y1 and y2
    libsemx::GaussianOutcome gaussian;
    const double y1_loglik = driver.evaluate_total_loglik(data["y1"], linear_predictors["y1"], dispersions["y1"], gaussian);
    const double y2_loglik = driver.evaluate_total_loglik(data["y2"], linear_predictors["y2"], dispersions["y2"], gaussian);
    const double expected = y1_loglik + y2_loglik;

    REQUIRE_THAT(total_loglik, Catch::Matchers::WithinRel(expected));
}

TEST_CASE("LikelihoodDriver throws on missing data for ModelIR", "[likelihood_driver]") {
    libsemx::LikelihoodDriver driver;
    libsemx::ModelIRBuilder builder;

    builder.add_variable("y1", libsemx::VariableKind::Observed, "gaussian");
    const auto model = builder.build();

    std::unordered_map<std::string, std::vector<double>> data = {{"y1", {1.0}}};
    std::unordered_map<std::string, std::vector<double>> linear_predictors = {{"y1", {0.0}}};
    std::unordered_map<std::string, std::vector<double>> dispersions;  // Missing dispersions

    REQUIRE_THROWS_AS(driver.evaluate_model_loglik(model, data, linear_predictors, dispersions), std::invalid_argument);
}

TEST_CASE("LikelihoodDriver fits simple regression model", "[likelihood_driver]") {
    libsemx::LikelihoodDriver driver;
    
    libsemx::ModelIR model;
    model.variables.push_back({"y", libsemx::VariableKind::Observed, "gaussian"});
    // x is a predictor, not modeled, so we don't add it to variables list for this simple regression
    
    // y ~ x (beta)
    model.edges.push_back({libsemx::EdgeKind::Regression, "x", "y", "beta"});
    
    std::unordered_map<std::string, std::vector<double>> data;
    data["y"] = {1.0, 2.0, 3.0};
    data["x"] = {1.0, 2.0, 3.0};
    
    libsemx::OptimizationOptions options;
    options.max_iterations = 100;
    options.tolerance = 1e-6;
    
    auto result = driver.fit(model, data, options, "lbfgs");
    
    REQUIRE(result.optimization_result.converged);
    REQUIRE(result.optimization_result.parameters.size() == 1);
    REQUIRE_THAT(result.optimization_result.parameters[0], Catch::Matchers::WithinAbs(1.0, 1e-3));
}

TEST_CASE("Model parameter ordering stays aligned with gradient map", "[likelihood_driver][gradient][parameter_catalog]") {
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("intercept", libsemx::VariableKind::Exogenous, "gaussian");
    builder.add_variable("x1", libsemx::VariableKind::Exogenous, "gaussian");
    builder.add_variable("x2", libsemx::VariableKind::Exogenous, "gaussian");
    builder.add_variable("cluster", libsemx::VariableKind::Grouping);
    builder.add_edge(libsemx::EdgeKind::Regression, "intercept", "y", "beta_intercept");
    builder.add_edge(libsemx::EdgeKind::Regression, "x1", "y", "beta_x1");
    builder.add_edge(libsemx::EdgeKind::Regression, "x2", "y", "beta_x2");
    builder.add_covariance("G_cluster", "diagonal", 1);
    builder.add_random_effect("u_cluster", {"cluster"}, "G_cluster");

    auto model = builder.build();
    libsemx::LikelihoodDriver driver;

    std::unordered_map<std::string, std::vector<double>> data = {
        {"y", {-0.4, 0.5, 0.2, 1.2, 1.8, 2.4}},
        {"intercept", {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}},
        {"x1", {-1.5, -0.5, 0.0, 0.5, 1.0, 1.5}},
        {"x2", {0.3, -0.2, 0.4, -0.1, 0.7, -0.4}},
        {"cluster", {1.0, 1.0, 2.0, 2.0, 3.0, 3.0}}
    };

    libsemx::OptimizationOptions options;
    options.max_iterations = 400;
    options.tolerance = 1e-6;

    auto result = driver.fit(model, data, options, "lbfgs");
    REQUIRE(result.optimization_result.converged);

    const auto structural_count = model.parameters.size();
    REQUIRE(structural_count == 3);
    REQUIRE(result.optimization_result.parameters.size() == structural_count + 1);

    std::unordered_map<std::string, double> beta_lookup;
    for (std::size_t i = 0; i < structural_count; ++i) {
        beta_lookup.emplace(model.parameters[i].id, result.optimization_result.parameters[i]);
    }

    std::unordered_map<std::string, std::vector<double>> linear_predictors;
    const auto n = data.at("y").size();
    linear_predictors["y"] = std::vector<double>(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        linear_predictors["y"][i] = beta_lookup.at("beta_intercept") * data.at("intercept")[i]
                                     + beta_lookup.at("beta_x1") * data.at("x1")[i]
                                     + beta_lookup.at("beta_x2") * data.at("x2")[i];
    }

    std::unordered_map<std::string, std::vector<double>> dispersions = {
        {"y", std::vector<double>(n, 1.0)}
    };

    const double sigma = result.optimization_result.parameters[structural_count];

    std::unordered_map<std::string, std::vector<double>> covariance_parameters = {
        {"G_cluster", {sigma}}
    };

    auto gradients = driver.evaluate_model_gradient(
        model,
        data,
        linear_predictors,
        dispersions,
        covariance_parameters,
        {},
        {},
        {});

    REQUIRE(gradients.count("beta_intercept"));
    REQUIRE(gradients.count("beta_x1"));
    REQUIRE(gradients.count("beta_x2"));
    REQUIRE(gradients.count("G_cluster_0"));

    std::vector<std::string> beta_ids;
    beta_ids.reserve(structural_count);
    for (const auto& spec : model.parameters) {
        beta_ids.push_back(spec.id);
    }

    const auto loglik = [&](const std::unordered_map<std::string, double>& betas,
                            double cov_param) {
        std::unordered_map<std::string, std::vector<double>> lp;
        lp["y"] = std::vector<double>(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            lp["y"][i] = betas.at("beta_intercept") * data.at("intercept")[i]
                           + betas.at("beta_x1") * data.at("x1")[i]
                           + betas.at("beta_x2") * data.at("x2")[i];
        }
        std::unordered_map<std::string, std::vector<double>> disp = {
            {"y", std::vector<double>(n, 1.0)}
        };
        std::unordered_map<std::string, std::vector<double>> cov = {
            {"G_cluster", {cov_param}}
        };
        return driver.evaluate_model_loglik(model, data, lp, disp, cov, {}, {}, {});
    };

    const double beta_step = 5e-4;
    for (const auto& id : beta_ids) {
        auto beta_plus = beta_lookup;
        auto beta_minus = beta_lookup;
        beta_plus[id] += beta_step;
        beta_minus[id] -= beta_step;
        const double fd = (loglik(beta_plus, sigma) - loglik(beta_minus, sigma)) / (2.0 * beta_step);
        REQUIRE_THAT(gradients.at(id), Catch::Matchers::WithinAbs(fd, 1e-3));
    }

    const double theta_step = 1e-3;
    const double sigma_plus = sigma * std::exp(theta_step);
    const double sigma_minus = sigma * std::exp(-theta_step);
    const double fd_sigma = (loglik(beta_lookup, sigma_plus) - loglik(beta_lookup, sigma_minus)) /
                            (sigma_plus - sigma_minus);
    REQUIRE_THAT(gradients.at("G_cluster_0"), Catch::Matchers::WithinAbs(fd_sigma, 5e-3));
}

TEST_CASE("LikelihoodDriver fits binomial mixed model with Laplace gradients", "[likelihood_driver][laplace][fit]") {
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "binomial");
    builder.add_variable("x", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("cluster", libsemx::VariableKind::Grouping);
    builder.add_variable("u", libsemx::VariableKind::Latent);
    builder.add_edge(libsemx::EdgeKind::Regression, "x", "y", "beta");
    builder.add_covariance("G", "diagonal", 1);
    builder.add_random_effect("u", {"cluster"}, "G");
    builder.add_edge(libsemx::EdgeKind::Regression, "u", "y", "1");

    auto model = builder.build();
    libsemx::LikelihoodDriver driver;

    std::unordered_map<std::string, std::vector<double>> data = {
        {"y", {0, 1, 0, 1, 0, 1}},
        {"x", {-1.0, -0.5, 0.2, 0.7, 1.0, 1.5}},
        {"cluster", {1.0, 1.0, 2.0, 2.0, 3.0, 3.0}}
    };

    libsemx::OptimizationOptions options;
    options.max_iterations = 200;
    options.tolerance = 1e-4;

    auto result = driver.fit(model, data, options, "lbfgs");
    REQUIRE(result.optimization_result.parameters.size() == 2);
    REQUIRE(result.optimization_result.converged);
    double beta = result.optimization_result.parameters[0];
    double sigma = result.optimization_result.parameters[1];
    REQUIRE(sigma > 0.0);

    std::unordered_map<std::string, std::vector<double>> linear_predictors = {
        {"y", {}}
    };
    for (double x_val : data["x"]) {
        linear_predictors["y"].push_back(beta * x_val);
    }
    std::unordered_map<std::string, std::vector<double>> dispersions = {
        {"y", std::vector<double>(data["y"].size(), 1.0)}
    };
    std::unordered_map<std::string, std::vector<double>> covariance_parameters = {
        {"G", {sigma}}
    };

    auto gradients = driver.evaluate_model_gradient(
        model,
        data,
        linear_predictors,
        dispersions,
        covariance_parameters,
        {},
        {},
        {});

    REQUIRE_THAT(gradients.at("beta"), Catch::Matchers::WithinAbs(0.0, 1e-3));
    REQUIRE(sigma < 1e-3);
    REQUIRE(gradients.at("G_0") <= 0.0);
}

TEST_CASE("LikelihoodDriver fits multi-effect binomial Laplace model", "[likelihood_driver][laplace][fit][multi_effect]") {
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "binomial");
    builder.add_variable("x", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("cluster", libsemx::VariableKind::Grouping);
    builder.add_variable("batch", libsemx::VariableKind::Grouping);
    builder.add_variable("u_cluster", libsemx::VariableKind::Latent);
    builder.add_variable("u_batch", libsemx::VariableKind::Latent);
    builder.add_edge(libsemx::EdgeKind::Regression, "x", "y", "beta");
    builder.add_covariance("G_cluster", "diagonal", 1);
    builder.add_covariance("G_batch", "diagonal", 1);
    builder.add_random_effect("u_cluster", {"cluster"}, "G_cluster");
    builder.add_random_effect("u_batch", {"batch"}, "G_batch");
    builder.add_edge(libsemx::EdgeKind::Regression, "u_cluster", "y", "1");
    builder.add_edge(libsemx::EdgeKind::Regression, "u_batch", "y", "1");

    auto model = builder.build();
    libsemx::LikelihoodDriver driver;

    std::unordered_map<std::string, std::vector<double>> data = {
        {"y", {0, 1, 0, 1, 0, 1, 0, 1}},
        {"x", {-1.5, -0.8, -0.2, 0.4, 0.9, 1.2, 1.5, 1.8}},
        {"cluster", {1, 1, 2, 2, 3, 3, 4, 4}},
        {"batch", {1, 1, 1, 1, 2, 2, 2, 2}}
    };

    libsemx::OptimizationOptions options;
    options.max_iterations = 250;
    options.tolerance = 5e-4;

    auto result = driver.fit(model, data, options, "lbfgs");
    INFO("multi-effect iterations: " << result.optimization_result.iterations);
    INFO("multi-effect grad norm: " << result.optimization_result.gradient_norm);
    REQUIRE(result.optimization_result.converged);
    REQUIRE(result.optimization_result.parameters.size() == 3);

    double beta = result.optimization_result.parameters[0];
    double sigma_cluster = result.optimization_result.parameters[1];
    double sigma_batch = result.optimization_result.parameters[2];
    REQUIRE(sigma_cluster > 0.0);
    REQUIRE(sigma_batch > 0.0);

    std::unordered_map<std::string, std::vector<double>> linear_predictors = {
        {"y", {}}
    };
    for (double x_val : data["x"]) {
        linear_predictors["y"].push_back(beta * x_val);
    }
    std::unordered_map<std::string, std::vector<double>> dispersions = {
        {"y", std::vector<double>(data["y"].size(), 1.0)}
    };
    std::unordered_map<std::string, std::vector<double>> covariance_parameters = {
        {"G_cluster", {sigma_cluster}},
        {"G_batch", {sigma_batch}}
    };

    auto gradients = driver.evaluate_model_gradient(
        model,
        data,
        linear_predictors,
        dispersions,
        covariance_parameters,
        {},
        {},
        {});

    REQUIRE_THAT(gradients.at("beta"), Catch::Matchers::WithinAbs(0.0, 1e-3));
    REQUIRE(gradients.at("G_cluster_0") <= 0.0);
    REQUIRE(gradients.at("G_batch_0") <= 0.0);
}

TEST_CASE("LikelihoodDriver fits mixed covariance Laplace model", "[likelihood_driver][laplace][fit][fixed_covariance]") {
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "binomial");
    builder.add_variable("x", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("cluster", libsemx::VariableKind::Grouping);
    builder.add_variable("batch", libsemx::VariableKind::Grouping);
    builder.add_variable("u_cluster", libsemx::VariableKind::Latent);
    builder.add_variable("u_batch", libsemx::VariableKind::Latent);
    builder.add_edge(libsemx::EdgeKind::Regression, "x", "y", "beta");
    builder.add_covariance("G_cluster", "diagonal", 1);
    builder.add_covariance("G_batch_fixed", "scaled_fixed", 1);
    builder.add_random_effect("u_cluster", {"cluster"}, "G_cluster");
    builder.add_random_effect("u_batch", {"batch"}, "G_batch_fixed");
    builder.add_edge(libsemx::EdgeKind::Regression, "u_cluster", "y", "1");
    builder.add_edge(libsemx::EdgeKind::Regression, "u_batch", "y", "1");

    auto model = builder.build();
    libsemx::LikelihoodDriver driver;

    std::unordered_map<std::string, std::vector<double>> data = {
        {"y", {0, 1, 0, 1, 0, 1, 0, 1}},
        {"x", {-1.5, -0.8, -0.2, 0.4, 0.9, 1.2, 1.5, 1.8}},
        {"cluster", {1, 1, 2, 2, 3, 3, 4, 4}},
        {"batch", {1, 1, 1, 1, 2, 2, 2, 2}}
    };

    std::unordered_map<std::string, std::vector<std::vector<double>>> fixed_covariance_data = {
        {"G_batch_fixed", {{1.0}}}
    };

    libsemx::OptimizationOptions options;
    options.max_iterations = 250;
    options.tolerance = 5e-4;

    auto result = driver.fit(model, data, options, "lbfgs", fixed_covariance_data);
    INFO("mixed iterations: " << result.optimization_result.iterations);
    INFO("mixed grad norm: " << result.optimization_result.gradient_norm);
    REQUIRE(result.optimization_result.converged);
    REQUIRE(result.optimization_result.parameters.size() == 3);

    double beta = result.optimization_result.parameters[0];
    double sigma_cluster = result.optimization_result.parameters[1];
    double sigma_batch_scale = result.optimization_result.parameters[2];
    REQUIRE(sigma_cluster > 0.0);
    REQUIRE(sigma_batch_scale > 0.0);

    std::unordered_map<std::string, std::vector<double>> linear_predictors = {
        {"y", {}}
    };
    for (double x_val : data["x"]) {
        linear_predictors["y"].push_back(beta * x_val);
    }
    std::unordered_map<std::string, std::vector<double>> dispersions = {
        {"y", std::vector<double>(data["y"].size(), 1.0)}
    };
    std::unordered_map<std::string, std::vector<double>> covariance_parameters = {
        {"G_cluster", {sigma_cluster}},
        {"G_batch_fixed", {sigma_batch_scale}}
    };

    auto gradients = driver.evaluate_model_gradient(
        model,
        data,
        linear_predictors,
        dispersions,
        covariance_parameters,
        {},
        {},
        fixed_covariance_data);

    REQUIRE_THAT(gradients.at("beta"), Catch::Matchers::WithinAbs(0.0, 1e-3));
    REQUIRE(gradients.at("G_cluster_0") <= 0.0);
    REQUIRE(gradients.at("G_batch_fixed_0") <= 0.0);
}

TEST_CASE("LikelihoodDriver fits random-slope Laplace model", "[likelihood_driver][laplace][fit][random_slope]") {
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "binomial");
    builder.add_variable("x", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("cluster", libsemx::VariableKind::Grouping);
    builder.add_variable("intercept_col", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("z", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("u_cluster2", libsemx::VariableKind::Latent);
    builder.add_edge(libsemx::EdgeKind::Regression, "x", "y", "beta");
    builder.add_covariance("G_cluster2", "unstructured", 2);
    builder.add_random_effect("u_cluster2", {"cluster", "intercept_col", "z"}, "G_cluster2");
    builder.add_edge(libsemx::EdgeKind::Regression, "u_cluster2", "y", "1");

    auto model = builder.build();
    libsemx::LikelihoodDriver driver;

    std::unordered_map<std::string, std::vector<double>> data = {
        {"y", {0, 1, 0, 1, 0, 1, 0, 1}},
        {"x", {-1.4, -0.9, -0.3, 0.2, 0.7, 1.1, 1.5, 1.9}},
        {"cluster", {1, 1, 2, 2, 3, 3, 4, 4}},
        {"intercept_col", {1, 1, 1, 1, 1, 1, 1, 1}},
        {"z", {-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 0.9, 1.3}}
    };

    libsemx::OptimizationOptions options;
    options.max_iterations = 300;
    options.tolerance = 5e-4;

    auto result = driver.fit(model, data, options, "lbfgs");
    INFO("random slope iterations: " << result.optimization_result.iterations);
    INFO("random slope grad norm: " << result.optimization_result.gradient_norm);
    REQUIRE(result.optimization_result.converged);
    REQUIRE(result.optimization_result.parameters.size() == 4);

    double beta = result.optimization_result.parameters[0];
    double sigma_intercept = result.optimization_result.parameters[1];
    double cov_intercept_slope = result.optimization_result.parameters[2];
    double sigma_slope = result.optimization_result.parameters[3];
    REQUIRE(sigma_intercept > 0.0);
    REQUIRE(sigma_slope > 0.0);
    REQUIRE(std::abs(cov_intercept_slope) < 1.0);

    std::unordered_map<std::string, std::vector<double>> linear_predictors = {
        {"y", {}}
    };
    for (double x_val : data["x"]) {
        linear_predictors["y"].push_back(beta * x_val);
    }
    std::unordered_map<std::string, std::vector<double>> dispersions = {
        {"y", std::vector<double>(data["y"].size(), 1.0)}
    };
    std::unordered_map<std::string, std::vector<double>> covariance_parameters = {
        {"G_cluster2", {sigma_intercept, cov_intercept_slope, sigma_slope}}
    };

    auto gradients = driver.evaluate_model_gradient(
        model,
        data,
        linear_predictors,
        dispersions,
        covariance_parameters);

    REQUIRE_THAT(gradients.at("beta"), Catch::Matchers::WithinAbs(0.0, 1e-3));
    REQUIRE(gradients.at("G_cluster2_0") <= 0.0);
    REQUIRE_THAT(gradients.at("G_cluster2_1"), Catch::Matchers::WithinAbs(0.0, 5e-3));
    REQUIRE(gradients.at("G_cluster2_2") <= 0.0);
}

TEST_CASE("LikelihoodDriver fits Kronecker Laplace model with multi-kernel covariance", "[likelihood_driver][laplace][fit][kronecker]") {
    auto kronecker = [](const std::vector<double>& A, std::size_t a_dim,
                        const std::vector<double>& B, std::size_t b_dim) {
        std::vector<double> out(a_dim * b_dim * a_dim * b_dim, 0.0);
        for (std::size_t i = 0; i < a_dim; ++i) {
            for (std::size_t j = 0; j < a_dim; ++j) {
                for (std::size_t p = 0; p < b_dim; ++p) {
                    for (std::size_t q = 0; q < b_dim; ++q) {
                        const std::size_t row = i * b_dim + p;
                        const std::size_t col = j * b_dim + q;
                        out[row * (a_dim * b_dim) + col] = A[i * a_dim + j] * B[p * b_dim + q];
                    }
                }
            }
        }
        return out;
    };

    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "binomial");
    builder.add_variable("x", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("cluster", libsemx::VariableKind::Grouping);
    builder.add_variable("t1e1", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("t1e2", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("t2e1", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("t2e2", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("u_kron", libsemx::VariableKind::Latent);
    builder.add_variable("u_diag", libsemx::VariableKind::Latent);
    builder.add_edge(libsemx::EdgeKind::Regression, "x", "y", "beta");
    builder.add_covariance("G_kron", "multi_kernel", 4);
    builder.add_covariance("G_diag", "diagonal", 1);
    builder.add_random_effect("u_kron", {"cluster", "t1e1", "t1e2", "t2e1", "t2e2"}, "G_kron");
    builder.add_random_effect("u_diag", {"cluster"}, "G_diag");
    builder.add_edge(libsemx::EdgeKind::Regression, "u_kron", "y", "1");
    builder.add_edge(libsemx::EdgeKind::Regression, "u_diag", "y", "1");

    auto model = builder.build();
    libsemx::LikelihoodDriver driver;

    std::unordered_map<std::string, std::vector<double>> data = {
        {"y", {0, 1, 0, 1, 0, 1, 0, 1}},
        {"x", {-1.6, -1.1, -0.6, -0.1, 0.4, 0.9, 1.4, 1.9}},
        {"cluster", {1, 2, 1, 2, 1, 2, 1, 2}},
        {"t1e1", {1, 0, 0, 0, 1, 0, 0, 0}},
        {"t1e2", {0, 1, 0, 0, 0, 1, 0, 0}},
        {"t2e1", {0, 0, 1, 0, 0, 0, 1, 0}},
        {"t2e2", {0, 0, 0, 1, 0, 0, 0, 1}},
    };

    std::vector<double> trait_cov = {1.0, 0.35, 0.35, 1.0};
    std::vector<double> env_cov = {1.0, 0.2, 0.2, 1.0};
    std::vector<double> identity2 = {1.0, 0.0, 0.0, 1.0};

    std::unordered_map<std::string, std::vector<std::vector<double>>> fixed_covariance_data;
    fixed_covariance_data["G_kron"] = {
        kronecker(trait_cov, 2, identity2, 2),
        kronecker(identity2, 2, env_cov, 2)
    };

    libsemx::OptimizationOptions options;
    options.max_iterations = 600;
    options.tolerance = 5e-4;
    options.learning_rate = 0.2;

    auto result = driver.fit(model, data, options, "lbfgs", fixed_covariance_data);
    INFO("kronecker iterations: " << result.optimization_result.iterations);
    INFO("kronecker grad norm: " << result.optimization_result.gradient_norm);
    REQUIRE(result.optimization_result.parameters.size() == 5);

    double beta = result.optimization_result.parameters[0];
    double sigma_kron = result.optimization_result.parameters[1];
    double weight_trait = result.optimization_result.parameters[2];
    double weight_env = result.optimization_result.parameters[3];
    double sigma_diag = result.optimization_result.parameters[4];
    INFO("beta estimate: " << beta);
    INFO("sigma_kron: " << sigma_kron);
    INFO("weight_trait: " << weight_trait);
    INFO("weight_env: " << weight_env);
    INFO("sigma_diag: " << sigma_diag);
    REQUIRE(sigma_kron > 0.0);
    REQUIRE(sigma_diag > 0.0);

    std::unordered_map<std::string, std::vector<double>> linear_predictors = {
        {"y", {}}
    };
    for (double x_val : data.at("x")) {
        linear_predictors["y"].push_back(beta * x_val);
    }

    std::unordered_map<std::string, std::vector<double>> dispersions = {
        {"y", std::vector<double>(data.at("y").size(), 1.0)}
    };
    std::unordered_map<std::string, std::vector<double>> covariance_parameters = {
        {"G_kron", {sigma_kron, weight_trait, weight_env}},
        {"G_diag", {sigma_diag}}
    };

    auto gradients = driver.evaluate_model_gradient(
        model,
        data,
        linear_predictors,
        dispersions,
        covariance_parameters,
        {},
        {},
        fixed_covariance_data);
    INFO("grad beta: " << gradients.at("beta"));
    INFO("grad G_kron_0: " << gradients.at("G_kron_0"));
    INFO("grad G_kron_1: " << gradients.at("G_kron_1"));
    INFO("grad G_kron_2: " << gradients.at("G_kron_2"));
    INFO("grad G_diag_0: " << gradients.at("G_diag_0"));

    REQUIRE_THAT(gradients.at("beta"), Catch::Matchers::WithinAbs(0.0, 2e-3));
    REQUIRE(gradients.at("G_kron_0") <= 0.0);
    REQUIRE_THAT(gradients.at("G_kron_1"), Catch::Matchers::WithinAbs(0.0, 5e-3));
    REQUIRE_THAT(gradients.at("G_kron_2"), Catch::Matchers::WithinAbs(0.0, 5e-3));
    REQUIRE(gradients.at("G_diag_0") <= 0.0);
    REQUIRE(result.optimization_result.converged);
}

TEST_CASE("LikelihoodDriver throws on random effects in ModelIR", "[likelihood_driver]") {
    libsemx::LikelihoodDriver driver;
    libsemx::ModelIRBuilder builder;

    builder.add_variable("y1", libsemx::VariableKind::Observed, "gaussian");
    builder.add_covariance("G", "unstructured", 2);
    builder.add_random_effect("u", {"y1"}, "G");
    const auto model = builder.build();

    std::unordered_map<std::string, std::vector<double>> data = {{"y1", {1.0}}};
    std::unordered_map<std::string, std::vector<double>> linear_predictors = {{"y1", {0.0}}};
    std::unordered_map<std::string, std::vector<double>> dispersions = {{"y1", {1.0}}};

    REQUIRE_THROWS_AS(driver.evaluate_model_loglik(model, data, linear_predictors, dispersions), std::runtime_error);
}