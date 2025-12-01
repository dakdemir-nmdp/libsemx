#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "libsemx/likelihood_driver.hpp"
#include "libsemx/model_ir.hpp"

#include <unordered_map>
#include <vector>

namespace {

std::vector<double> kronecker_product(const std::vector<double>& A,
                                      std::size_t a_dim,
                                      const std::vector<double>& B,
                                      std::size_t b_dim) {
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
}

libsemx::ModelIR build_model() {
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "binomial");
    builder.add_variable("x", libsemx::VariableKind::Latent);
    builder.add_variable("cluster", libsemx::VariableKind::Grouping);
    builder.add_variable("t1e1", libsemx::VariableKind::Latent);
    builder.add_variable("t1e2", libsemx::VariableKind::Latent);
    builder.add_variable("t2e1", libsemx::VariableKind::Latent);
    builder.add_variable("t2e2", libsemx::VariableKind::Latent);
    builder.add_edge(libsemx::EdgeKind::Regression, "x", "y", "beta");
    builder.add_covariance("G_kron", "multi_kernel", 4);
    builder.add_covariance("G_diag", "diagonal", 1);
    builder.add_random_effect("u_kron", {"cluster", "t1e1", "t1e2", "t2e1", "t2e2"}, "G_kron");
    builder.add_random_effect("u_diag", {"cluster"}, "G_diag");
    return builder.build();
}

std::unordered_map<std::string, std::vector<double>> build_data() {
    return {
        {"y", {0, 1, 0, 1, 0, 1, 0, 1}},
        {"x", {-1.6, -1.1, -0.6, -0.1, 0.4, 0.9, 1.4, 1.9}},
        {"cluster", {1, 2, 1, 2, 1, 2, 1, 2}},
        {"t1e1", {1, 0, 0, 0, 1, 0, 0, 0}},
        {"t1e2", {0, 1, 0, 0, 0, 1, 0, 0}},
        {"t2e1", {0, 0, 1, 0, 0, 0, 1, 0}},
        {"t2e2", {0, 0, 0, 1, 0, 0, 0, 1}},
    };
}

std::unordered_map<std::string, std::vector<std::vector<double>>>
build_fixed_covariance_data() {
    const std::vector<double> trait_cov = {1.0, 0.35, 0.35, 1.0};
    const std::vector<double> env_cov = {1.0, 0.2, 0.2, 1.0};
    const std::vector<double> identity2 = {1.0, 0.0, 0.0, 1.0};

    std::unordered_map<std::string, std::vector<std::vector<double>>> fixed_covariance_data;
    fixed_covariance_data["G_kron"] = {
        kronecker_product(trait_cov, 2, identity2, 2),
        kronecker_product(identity2, 2, env_cov, 2)
    };
    return fixed_covariance_data;
}

std::unordered_map<std::string, std::vector<double>>
build_dispersions(std::size_t n_obs) {
    return {{"y", std::vector<double>(n_obs, 1.0)}};
}

} // namespace

TEST_CASE("Kronecker Laplace gradients match finite differences", "[kronecker][laplace][gradient]") {
    auto model = build_model();
    auto data = build_data();
    auto fixed_covariance_data = build_fixed_covariance_data();
    auto dispersions = build_dispersions(data.at("y").size());

    libsemx::LikelihoodDriver driver;

    libsemx::OptimizationOptions options;
    options.max_iterations = 600;
    options.tolerance = 5e-4;
    options.learning_rate = 0.2;

    auto result = driver.fit(model, data, options, "lbfgs", fixed_covariance_data);
    REQUIRE(result.converged);
    REQUIRE(result.parameters.size() == 5);

    const double beta = result.parameters[0];
    const double sigma_kron = result.parameters[1];
    const double weight_trait = result.parameters[2];
    const double weight_env = result.parameters[3];
    const double sigma_diag = result.parameters[4];

    auto build_linear_predictors = [&](double beta_val) {
        std::unordered_map<std::string, std::vector<double>> linear_predictors = { {"y", {}} };
        for (double x_val : data.at("x")) {
            linear_predictors["y"].push_back(beta_val * x_val);
        }
        return linear_predictors;
    };

    auto build_covariance_parameters = [&](double sigma_kron_val,
                                           double weight_trait_val,
                                           double weight_env_val,
                                           double sigma_diag_val) {
        return std::unordered_map<std::string, std::vector<double>>{
            {"G_kron", {sigma_kron_val, weight_trait_val, weight_env_val}},
            {"G_diag", {sigma_diag_val}}
        };
    };

    auto evaluate_loglik = [&](double beta_val,
                               double sigma_kron_val,
                               double weight_trait_val,
                               double weight_env_val,
                               double sigma_diag_val) {
        auto lp = build_linear_predictors(beta_val);
        auto cov_params = build_covariance_parameters(
            sigma_kron_val, weight_trait_val, weight_env_val, sigma_diag_val);
        return driver.evaluate_model_loglik(
            model,
            data,
            lp,
            dispersions,
            cov_params,
            {},
            {},
            fixed_covariance_data);
    };

    auto linear_predictors = build_linear_predictors(beta);
    auto covariance_parameters = build_covariance_parameters(
        sigma_kron, weight_trait, weight_env, sigma_diag);

    auto analytic_gradients = driver.evaluate_model_gradient(
        model,
        data,
        linear_predictors,
        dispersions,
        covariance_parameters,
        {},
        {},
        fixed_covariance_data);

    const double step_beta = 5e-4;
    const double step_weight = 5e-4;
    const double step_sigma = 1e-4;

    auto central_difference = [&](int which, double step) {
        auto eval_shifted = [&](double delta) {
            double beta_shift = beta;
            double sigma_kron_shift = sigma_kron;
            double weight_trait_shift = weight_trait;
            double weight_env_shift = weight_env;
            double sigma_diag_shift = sigma_diag;
            switch (which) {
                case 0: beta_shift += delta; break;
                case 1: sigma_kron_shift += delta; break;
                case 2: weight_trait_shift += delta; break;
                case 3: weight_env_shift += delta; break;
                case 4: sigma_diag_shift += delta; break;
                default: break;
            }
            return evaluate_loglik(beta_shift,
                                   sigma_kron_shift,
                                   weight_trait_shift,
                                   weight_env_shift,
                                   sigma_diag_shift);
        };
        const double ll_plus = eval_shifted(step);
        const double ll_minus = eval_shifted(-step);
        return (ll_plus - ll_minus) / (2.0 * step);
    };

    const double grad_beta_fd = central_difference(0, step_beta);
    REQUIRE_THAT(analytic_gradients.at("beta"), Catch::Matchers::WithinAbs(grad_beta_fd, 2e-3));

    const double grad_trait_fd = central_difference(2, step_weight);
    REQUIRE_THAT(analytic_gradients.at("G_kron_1"), Catch::Matchers::WithinAbs(grad_trait_fd, 3e-3));

    const double grad_env_fd = central_difference(3, step_weight);
    REQUIRE_THAT(analytic_gradients.at("G_kron_2"), Catch::Matchers::WithinAbs(grad_env_fd, 3e-3));

    const double grad_sigma_fd = central_difference(1, step_sigma);
    REQUIRE_THAT(analytic_gradients.at("G_kron_0"), Catch::Matchers::WithinAbs(grad_sigma_fd, 3e-3));

    const double grad_diag_fd = central_difference(4, step_sigma);
    REQUIRE_THAT(analytic_gradients.at("G_diag_0"), Catch::Matchers::WithinAbs(grad_diag_fd, 3e-3));
}
