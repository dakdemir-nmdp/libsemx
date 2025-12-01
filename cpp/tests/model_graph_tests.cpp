#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <stdexcept>

#include "libsemx/model_graph.hpp"
#include "libsemx/parameter.hpp"
#include "libsemx/parameter_transform.hpp"

TEST_CASE("ModelGraph registers observed and latent variables", "[model_graph]") {
    libsemx::ModelGraph graph;

    graph.add_variable("y1", libsemx::VariableType::Observed);
    graph.add_variable("eta1", libsemx::VariableType::Latent);

    REQUIRE(graph.contains("y1"));
    REQUIRE(graph.contains("eta1"));
    REQUIRE(graph.get("y1").type == libsemx::VariableType::Observed);
    REQUIRE(graph.get("eta1").type == libsemx::VariableType::Latent);
}

TEST_CASE("ModelGraph rejects duplicate variable names", "[model_graph]") {
    libsemx::ModelGraph graph;
    graph.add_variable("y1", libsemx::VariableType::Observed);

    REQUIRE_THROWS_AS(graph.add_variable("y1", libsemx::VariableType::Latent), std::invalid_argument);
}

TEST_CASE("Parameter enforces transform constraints", "[parameter]") {
    libsemx::Parameter tau("tau", 1.0, libsemx::make_log_transform());
    REQUIRE(tau.value() == Catch::Approx(1.0));
    tau.set_value(2.0);
    REQUIRE(tau.value() == Catch::Approx(2.0));
    REQUIRE(tau.unconstrained_value() == Catch::Approx(std::log(2.0)));
    REQUIRE_THROWS_AS(tau.set_value(0.0), std::out_of_range);

    auto bounded = libsemx::make_logistic_transform(0.0, 1.0);
    REQUIRE_THROWS_AS(libsemx::Parameter("rho_bad", -0.1, bounded), std::out_of_range);

    libsemx::Parameter rho("rho", 0.5, bounded);
    rho.set_unconstrained_value(0.0);
    REQUIRE(rho.value() == Catch::Approx(0.5));
    rho.set_value(0.25);
    REQUIRE(rho.value() == Catch::Approx(0.25));
    REQUIRE_THROWS_AS(rho.set_value(-0.05), std::out_of_range);
    REQUIRE_THROWS_AS(rho.set_value(1.0), std::out_of_range);

    auto identity = libsemx::make_identity_transform();
    rho.set_transform(identity);
    REQUIRE(rho.value() == Catch::Approx(0.25));
}
