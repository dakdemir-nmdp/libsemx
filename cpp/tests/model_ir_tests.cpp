#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

#include "libsemx/model_ir.hpp"

TEST_CASE("ModelIRBuilder assembles valid model", "[ir]") {
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y1", libsemx::VariableKind::Observed, "gaussian");
    builder.add_variable("eta1", libsemx::VariableKind::Latent);
    builder.add_variable("cluster", libsemx::VariableKind::Grouping);

    builder.add_edge(libsemx::EdgeKind::Loading, "eta1", "y1", "lambda_y1");
    builder.add_edge(libsemx::EdgeKind::Regression, "eta1", "cluster", "beta_cluster");

    builder.add_covariance("resid", "diagonal", 1);
    builder.add_random_effect("u_cluster", {"cluster"}, "resid");

    const auto ir = builder.build();
    REQUIRE(ir.variables.size() == 3);
    REQUIRE(ir.edges.size() == 2);
    REQUIRE(ir.covariances.size() == 1);
    REQUIRE(ir.random_effects.size() == 1);
    REQUIRE(ir.variables.front().family == "gaussian");
}

TEST_CASE("ModelIRBuilder enforces invariants", "[ir]") {
    libsemx::ModelIRBuilder builder;
    builder.add_variable("y1", libsemx::VariableKind::Observed, "gaussian");

    REQUIRE_THROWS_AS(builder.add_variable("y1", libsemx::VariableKind::Observed, "gaussian"),
                      std::invalid_argument);

    REQUIRE_THROWS(builder.add_edge(libsemx::EdgeKind::Loading, "eta1", "y1", "lambda"));
    builder.add_variable("eta1", libsemx::VariableKind::Latent);
    REQUIRE_THROWS(builder.add_edge(libsemx::EdgeKind::Loading, "eta1", "y1", ""));

    builder.add_covariance("resid", "diagonal", 1);
    REQUIRE_THROWS(builder.add_covariance("resid", "diagonal", 1));
    REQUIRE_THROWS(builder.add_random_effect("u1", {"cluster"}, "resid"));
    builder.add_variable("cluster", libsemx::VariableKind::Grouping);
    REQUIRE_THROWS(builder.add_random_effect("u1", {"cluster", "cluster"}, "resid"));
    REQUIRE_THROWS(builder.add_random_effect("u1", {"cluster"}, "unknown_cov"));

    builder.add_random_effect("u1", {"cluster"}, "resid");
    REQUIRE_NOTHROW(builder.build());
}
