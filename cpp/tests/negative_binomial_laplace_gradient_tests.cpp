#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "libsemx/likelihood_driver.hpp"
#include "libsemx/model_ir.hpp"

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace {

libsemx::ModelIR build_negative_binomial_model() {
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "negative_binomial");
    builder.add_variable("x", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("cluster", libsemx::VariableKind::Grouping);
    builder.add_edge(libsemx::EdgeKind::Regression, "x", "y", "beta");
    builder.add_covariance("G_nb", "diagonal", 1);
    builder.add_random_effect("u_cluster", {"cluster"}, "G_nb");
    return builder.build();
}

std::unordered_map<std::string, std::vector<double>> build_data() {
    return {
        {"y", {0, 1, 1, 2, 3, 2, 1, 4}},
        {"x", {-1.4, -0.9, -0.3, 0.2, 0.7, 1.1, 1.5, 1.9}},
        {"cluster", {1, 1, 2, 2, 3, 3, 4, 4}}
    };
}

std::unordered_map<std::string, std::vector<double>> build_dispersions(std::size_t n) {
    return {{"y", std::vector<double>(n, 1.0)}};
}

} // namespace

TEST_CASE("Negative binomial Laplace gradients match finite differences", "[laplace][negative_binomial]") {
    auto model = build_negative_binomial_model();
    auto data = build_data();
    auto dispersions = build_dispersions(data.at("y").size());

    libsemx::LikelihoodDriver driver;

    libsemx::OptimizationOptions options;
    options.max_iterations = 400;
    options.tolerance = 5e-4;
    options.learning_rate = 0.1;

    auto result = driver.fit(model, data, options, "lbfgs");
    REQUIRE(result.optimization_result.converged);
    REQUIRE(result.optimization_result.parameters.size() == 2);

    const double beta = result.optimization_result.parameters[0];
    const double sigma = result.optimization_result.parameters[1];

    auto linear_predictors = std::unordered_map<std::string, std::vector<double>>{{"y", {}}};
    for (double value : data.at("x")) {
        linear_predictors.at("y").push_back(beta * value);
    }
    std::unordered_map<std::string, std::vector<double>> covariance_parameters = {
        {"G_nb", {sigma}}
    };

    auto gradients = driver.evaluate_model_gradient(
        model,
        data,
        linear_predictors,
        dispersions,
        covariance_parameters);

    auto evaluate_loglik = [&](double beta_val, double sigma_val) {
        std::unordered_map<std::string, std::vector<double>> lp = {{"y", {}}};
        for (double value : data.at("x")) {
            lp.at("y").push_back(beta_val * value);
        }
        std::unordered_map<std::string, std::vector<double>> cov = {{"G_nb", {sigma_val}}};
        return driver.evaluate_model_loglik(
            model,
            data,
            lp,
            dispersions,
            cov);
    };

    const double step_beta = 5e-4;
    const double step_sigma = std::max(1e-5, 0.1 * sigma);

    const double fd_beta = (evaluate_loglik(beta + step_beta, sigma) -
                            evaluate_loglik(beta - step_beta, sigma)) /
                           (2.0 * step_beta);
    const double fd_sigma = (evaluate_loglik(beta, sigma + step_sigma) -
                             evaluate_loglik(beta, sigma - step_sigma)) /
                            (2.0 * step_sigma);

    REQUIRE_THAT(gradients.at("beta"), Catch::Matchers::WithinAbs(fd_beta, 2e-3));
    REQUIRE_THAT(gradients.at("G_nb_0"), Catch::Matchers::WithinAbs(fd_sigma, 2e-3));
}
