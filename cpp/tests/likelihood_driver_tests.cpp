#include "libsemx/likelihood_driver.hpp"
#include "libsemx/gaussian_outcome.hpp"
#include "libsemx/model_ir.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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

    REQUIRE_THROWS_AS(driver.evaluate_model_loglik(model, data, linear_predictors, dispersions), std::runtime_error);
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
    
    REQUIRE(result.converged);
    REQUIRE(result.parameters.size() == 1);
    REQUIRE_THAT(result.parameters[0], Catch::Matchers::WithinAbs(1.0, 1e-3));
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