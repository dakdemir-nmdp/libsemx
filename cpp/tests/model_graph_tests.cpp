#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <stdexcept>

#include "libsemx/model_graph.hpp"
#include "libsemx/model_ir.hpp"
#include "libsemx/parameter.hpp"
#include "libsemx/parameter_transform.hpp"

TEST_CASE("ModelGraph registers variable metadata", "[model_graph]") {
    libsemx::ModelGraph graph;

    graph.add_variable("y1", libsemx::VariableKind::Observed, "gaussian");
    graph.add_variable("eta1", libsemx::VariableKind::Latent);
    graph.add_variable("cluster", libsemx::VariableKind::Grouping);

    const auto& vars = graph.variables();
    REQUIRE(vars.size() == 3);
    REQUIRE(vars[0].name == "y1");
    REQUIRE(vars[0].kind == libsemx::VariableKind::Observed);
    REQUIRE(vars[0].family == "gaussian");
    REQUIRE(vars[1].kind == libsemx::VariableKind::Latent);
    REQUIRE(vars[1].family.empty());
    REQUIRE(vars[2].kind == libsemx::VariableKind::Grouping);
}

TEST_CASE("ModelGraph validates observed outcomes", "[model_graph]") {
    libsemx::ModelGraph graph;

    REQUIRE_THROWS_AS(graph.add_variable("y", libsemx::VariableKind::Observed, ""), std::invalid_argument);

    graph.add_variable("y", libsemx::VariableKind::Observed, "gaussian");
    REQUIRE_THROWS_AS(graph.add_variable("y", libsemx::VariableKind::Observed, "gaussian"), std::invalid_argument);
}

TEST_CASE("ModelGraph serializes edges, covariances, and random effects", "[model_graph]") {
    libsemx::ModelGraph graph;
    graph.add_variable("y", libsemx::VariableKind::Observed, "gaussian");
    graph.add_variable("x", libsemx::VariableKind::Latent);
    graph.add_variable("cluster", libsemx::VariableKind::Grouping);

    graph.add_edge(libsemx::EdgeKind::Regression, "x", "y", "beta_xy");
    graph.add_covariance("G", "diagonal", 1);
    graph.add_random_effect("u", {"cluster"}, "G");

    auto ir = graph.to_model_ir();
    REQUIRE(ir.variables.size() == 3);
    REQUIRE(ir.edges.size() == 1);
    REQUIRE(ir.covariances.size() == 1);
    REQUIRE(ir.random_effects.size() == 1);
    REQUIRE(ir.edges.front().parameter_id == "beta_xy");
    REQUIRE(ir.random_effects.front().covariance_id == "G");
}

TEST_CASE("ModelGraph rejects empty serialization", "[model_graph]") {
    libsemx::ModelGraph graph;
    REQUIRE_THROWS_AS(graph.to_model_ir(), std::invalid_argument);
}

TEST_CASE("ModelGraph validates covariance specifications", "[model_graph]") {
    libsemx::ModelGraph graph;
    graph.add_variable("cluster", libsemx::VariableKind::Grouping);

    REQUIRE_THROWS_AS(graph.add_covariance("", "diagonal", 1), std::invalid_argument);
    REQUIRE_THROWS_AS(graph.add_covariance("G", "", 1), std::invalid_argument);
    REQUIRE_THROWS_AS(graph.add_covariance("G", "diagonal", 0), std::invalid_argument);

    graph.add_covariance("G", "diagonal", 1);
    REQUIRE_THROWS_AS(graph.add_covariance("G", "diagonal", 1), std::invalid_argument);
}

TEST_CASE("ModelGraph rejects invalid edges and random effects", "[model_graph]") {
    libsemx::ModelGraph graph;
    graph.add_variable("y", libsemx::VariableKind::Observed, "gaussian");
    graph.add_variable("cluster", libsemx::VariableKind::Grouping);
    graph.add_covariance("G", "diagonal", 1);

    REQUIRE_THROWS_AS(graph.add_edge(libsemx::EdgeKind::Regression, "x", "y", "beta"), std::invalid_argument);
    REQUIRE_THROWS_AS(graph.add_edge(libsemx::EdgeKind::Regression, "cluster", "y", ""), std::invalid_argument);
    REQUIRE_THROWS_AS(graph.add_edge(libsemx::EdgeKind::Regression, "cluster", "missing", "beta"), std::invalid_argument);

    REQUIRE_THROWS_AS(graph.add_random_effect("u", {}, "G"), std::invalid_argument);
    REQUIRE_THROWS_AS(graph.add_random_effect("u", {"missing"}, "G"), std::invalid_argument);
    REQUIRE_THROWS_AS(graph.add_random_effect("u", {"cluster", "cluster"}, "G"), std::invalid_argument);
    REQUIRE_THROWS_AS(graph.add_random_effect("u", {"cluster"}, "missing"), std::invalid_argument);
}

TEST_CASE("ModelGraph records parameter specs for structural edges", "[model_graph]") {
    libsemx::ModelGraph graph;
    graph.add_variable("y", libsemx::VariableKind::Observed, "gaussian");
    graph.add_variable("x1", libsemx::VariableKind::Latent);
    graph.add_variable("x2", libsemx::VariableKind::Latent);

    graph.add_edge(libsemx::EdgeKind::Regression, "x1", "y", "beta_x1");
    graph.add_edge(libsemx::EdgeKind::Regression, "x2", "y", "beta_x2");
    graph.add_edge(libsemx::EdgeKind::Regression, "x1", "y", "beta_x1");
    graph.add_edge(libsemx::EdgeKind::Regression, "x2", "y", "0.25");

    const auto& params = graph.parameters();
    REQUIRE(params.size() == 2);
    REQUIRE(params[0].id == "beta_x1");
    REQUIRE(params[1].id == "beta_x2");
    for (const auto& param : params) {
        REQUIRE(param.constraint == libsemx::ParameterConstraint::Free);
        REQUIRE(param.initial_value == Catch::Approx(0.0));
    }
}

TEST_CASE("ModelGraph reject conflicting parameter constraints", "[model_graph]") {
    libsemx::ModelGraph graph;
    graph.add_variable("y", libsemx::VariableKind::Observed, "gaussian");
    graph.add_variable("x", libsemx::VariableKind::Latent);

    graph.register_parameter("theta", libsemx::ParameterConstraint::Free, 0.1);
    REQUIRE_NOTHROW(graph.register_parameter("theta", libsemx::ParameterConstraint::Free, 0.0));
    REQUIRE_THROWS_AS(graph.register_parameter("theta", libsemx::ParameterConstraint::Positive, 0.2), std::invalid_argument);
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
