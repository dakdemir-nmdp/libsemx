#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "libsemx/likelihood_driver.hpp"
#include "libsemx/model_ir.hpp"

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace {

libsemx::ModelIR build_ordinal_model() {
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "ordinal");
    builder.add_variable("x", libsemx::VariableKind::Exogenous);
    builder.add_variable("cluster", libsemx::VariableKind::Grouping);
    builder.add_edge(libsemx::EdgeKind::Regression, "x", "y", "beta");
    builder.add_covariance("G_ord", "diagonal", 1);
    builder.add_random_effect("u_cluster", {"cluster"}, "G_ord");
    return builder.build();
}

std::unordered_map<std::string, std::vector<double>> build_data() {
    return {
        {"y", {0, 1, 1, 2, 1, 2, 2, 0}},
        {"x", {-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0}},
        {"cluster", {1, 1, 2, 2, 3, 3, 4, 4}}
    };
}

std::unordered_map<std::string, std::vector<double>> build_dispersions(std::size_t n) {
    return {{"y", std::vector<double>(n, 1.0)}};
}

}  // namespace

TEST_CASE("Ordinal Laplace gradients match finite differences", "[laplace][ordinal]") {
    auto model = build_ordinal_model();
    auto data = build_data();
    const std::size_t n = data.at("y").size();
    auto dispersions = build_dispersions(n);

    libsemx::LikelihoodDriver driver;

    const double beta = 0.7;
    const double sigma = 0.8;

    std::unordered_map<std::string, std::vector<double>> linear_predictors = {{"y", {}}};
    linear_predictors.at("y").reserve(n);
    for (double value : data.at("x")) {
        linear_predictors.at("y").push_back(beta * value);
    }

    std::unordered_map<std::string, std::vector<double>> covariance_parameters = {
        {"G_ord", {sigma}}
    };

    const std::vector<double> thresholds = {-0.4, 0.6};
    std::unordered_map<std::string, std::vector<double>> extra_params = {
        {"y", thresholds}
    };

    auto gradients = driver.evaluate_model_gradient(
        model,
        data,
        linear_predictors,
        dispersions,
        covariance_parameters,
        {},
        extra_params);

    auto loglik = [&](double beta_val, double sigma_val) {
        std::unordered_map<std::string, std::vector<double>> lp = {{"y", {}}};
        lp.at("y").reserve(n);
        for (double value : data.at("x")) {
            lp.at("y").push_back(beta_val * value);
        }
        std::unordered_map<std::string, std::vector<double>> cov = {{"G_ord", {sigma_val}}};
        return driver.evaluate_model_loglik(
            model,
            data,
            lp,
            dispersions,
            cov,
            {},
            extra_params);
    };

    const double step_beta = 5e-4;
    const double step_sigma = std::max(1e-5, 0.1 * sigma);

    const double fd_beta = (loglik(beta + step_beta, sigma) -
                            loglik(beta - step_beta, sigma)) /
                           (2.0 * step_beta);
    const double fd_sigma = (loglik(beta, sigma + step_sigma) -
                             loglik(beta, sigma - step_sigma)) /
                            (2.0 * step_sigma);

    REQUIRE_THAT(gradients.at("beta"), Catch::Matchers::WithinAbs(fd_beta, 2e-3));
    REQUIRE_THAT(gradients.at("G_ord_0"), Catch::Matchers::WithinAbs(fd_sigma, 2e-3));
}
