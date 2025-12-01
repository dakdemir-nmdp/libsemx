#include "libsemx/likelihood_driver.hpp"
#include "libsemx/model_ir.hpp"
#include "libsemx/lognormal_outcome.hpp"
#include "libsemx/loglogistic_outcome.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <unordered_map>
#include <vector>

namespace {

template <typename Outcome>
double manual_loglik(const Outcome& outcome,
                     const std::vector<double>& observed,
                     const std::vector<double>& predictors,
                     const std::vector<double>& dispersions,
                     const std::vector<double>& status) {
    double total = 0.0;
    for (std::size_t i = 0; i < observed.size(); ++i) {
        total += outcome.evaluate(observed[i], predictors[i], dispersions[i], status[i]).log_likelihood;
    }
    return total;
}

}  // namespace

TEST_CASE("LikelihoodDriver evaluates lognormal survival outcomes", "[survival][likelihood_driver]") {
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "lognormal");
    auto model = builder.build();

    std::unordered_map<std::string, std::vector<double>> data{
        {"y", {1.5, 2.2, 0.9}}
    };
    std::unordered_map<std::string, std::vector<double>> predictors{
        {"y", {0.3, -0.1, 0.2}}
    };
    std::unordered_map<std::string, std::vector<double>> dispersions{
        {"y", {0.8, 0.9, 0.7}}
    };
    std::unordered_map<std::string, std::vector<double>> status{
        {"y", {1.0, 0.0, 1.0}}
    };

    libsemx::LikelihoodDriver driver;
    const double loglik = driver.evaluate_model_loglik(model, data, predictors, dispersions, {}, status);

    libsemx::LognormalOutcome outcome;
    const double expected = manual_loglik(outcome, data["y"], predictors["y"], dispersions["y"], status["y"]);

    REQUIRE_THAT(loglik, Catch::Matchers::WithinRel(expected, 1e-10));
}

TEST_CASE("LikelihoodDriver evaluates loglogistic survival outcomes", "[survival][likelihood_driver]") {
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y", libsemx::VariableKind::Observed, "loglogistic_aft");
    auto model = builder.build();

    std::unordered_map<std::string, std::vector<double>> data{
        {"y", {1.2, 2.5, 3.0}}
    };
    std::unordered_map<std::string, std::vector<double>> predictors{
        {"y", {0.1, 0.2, -0.4}}
    };
    std::unordered_map<std::string, std::vector<double>> dispersions{
        {"y", {1.1, 0.9, 1.3}}
    };
    std::unordered_map<std::string, std::vector<double>> status{
        {"y", {0.0, 1.0, 1.0}}
    };

    libsemx::LikelihoodDriver driver;
    const double loglik = driver.evaluate_model_loglik(model, data, predictors, dispersions, {}, status);

    libsemx::LogLogisticOutcome outcome;
    const double expected = manual_loglik(outcome, data["y"], predictors["y"], dispersions["y"], status["y"]);

    REQUIRE_THAT(loglik, Catch::Matchers::WithinRel(expected, 1e-10));
}
