#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "libsemx/likelihood_driver.hpp"
#include "libsemx/model_ir.hpp"
#include "libsemx/outcome_family_factory.hpp"

#include <vector>
#include <unordered_map>
#include <cmath>

TEST_CASE("LikelihoodDriver evaluates dispersion gradients for Gaussian", "[gradient][dispersion]") {
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "gaussian");
    
    // Dispersion parameter "sigma2"
    // We model it as a self-covariance edge
    builder.add_edge(libsemx::EdgeKind::Covariance, "y", "y", "sigma2");
    
    auto model = builder.build();

    std::vector<double> y = {1.0, 2.0};
    std::vector<double> preds = {0.0, 0.0}; // mu = 0
    double sigma2 = 2.0;
    std::vector<double> disps = {sigma2, sigma2};

    std::unordered_map<std::string, std::vector<double>> data = {{"y", y}};
    std::unordered_map<std::string, std::vector<double>> linear_predictors = {{"y", preds}};
    std::unordered_map<std::string, std::vector<double>> dispersions = {{"y", disps}};

    libsemx::LikelihoodDriver driver;
    auto gradients = driver.evaluate_model_gradient(model, data, linear_predictors, dispersions);

    // Expected gradient:
    // i=0: y=1, mu=0, sigma2=2
    // dL/dsigma2 = 0.5 * ((1^2)/2 - 1) / 2 = 0.5 * (0.5 - 1) / 2 = 0.5 * (-0.5) / 2 = -0.125
    // i=1: y=2, mu=0, sigma2=2
    // dL/dsigma2 = 0.5 * ((2^2)/2 - 1) / 2 = 0.5 * (2 - 1) / 2 = 0.5 * 1 / 2 = 0.25
    // Total = -0.125 + 0.25 = 0.125

    REQUIRE(gradients.count("sigma2"));
    REQUIRE_THAT(gradients.at("sigma2"), Catch::Matchers::WithinRel(0.125, 1e-5));
}

TEST_CASE("LikelihoodDriver evaluates dispersion gradients for Negative Binomial", "[gradient][dispersion]") {
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "negative_binomial");
    
    // Dispersion parameter "k"
    builder.add_edge(libsemx::EdgeKind::Covariance, "y", "y", "k");
    
    auto model = builder.build();

    std::vector<double> y = {2.0};
    std::vector<double> preds = {std::log(4.0)}; // mu = 4 (log link)
    double k = 2.0;
    std::vector<double> disps = {k};

    std::unordered_map<std::string, std::vector<double>> data = {{"y", y}};
    std::unordered_map<std::string, std::vector<double>> linear_predictors = {{"y", preds}};
    std::unordered_map<std::string, std::vector<double>> dispersions = {{"y", disps}};

    libsemx::LikelihoodDriver driver;
    auto gradients = driver.evaluate_model_gradient(model, data, linear_predictors, dispersions);

    // Expected gradient:
    // y=2, mu=4, k=2
    // dL/dk = psi(y+k) - psi(k) + log(k/(mu+k)) + (mu-y)/(mu+k)
    //       = psi(4) - psi(2) + log(2/6) + (4-2)/6
    //       = psi(4) - psi(2) + log(1/3) + 1/3
    // psi(n) for integer n: -gamma + sum_{k=1}^{n-1} 1/k
    // psi(1) = -gamma
    // psi(2) = -gamma + 1
    // psi(3) = -gamma + 1 + 1/2
    // psi(4) = -gamma + 1 + 1/2 + 1/3 = -gamma + 11/6
    // psi(4) - psi(2) = (11/6) - 1 = 5/6
    // dL/dk = 5/6 + log(1/3) + 1/3 = 7/6 - log(3)
    // 7/6 approx 1.166666
    // log(3) approx 1.098612
    // Result approx 0.068054

    double expected = 7.0/6.0 - std::log(3.0);

    REQUIRE(gradients.count("k"));
    REQUIRE_THAT(gradients.at("k"), Catch::Matchers::WithinRel(expected, 1e-5));
}
