#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "libsemx/parameter_catalog.hpp"
#include "libsemx/parameter_transform.hpp"

TEST_CASE("ParameterCatalog registers parameters once", "[parameter_catalog]") {
    libsemx::ParameterCatalog catalog;
    auto idx0 = catalog.register_parameter("beta", 0.5, libsemx::make_identity_transform());
    auto idx1 = catalog.register_parameter("sigma", 1.0, libsemx::make_log_transform());
    auto idx0_dup = catalog.register_parameter("beta", 0.2, libsemx::make_identity_transform());

    REQUIRE(idx0 == 0);
    REQUIRE(idx1 == 1);
    REQUIRE(idx0_dup == idx0);
    REQUIRE(catalog.size() == 2);
    REQUIRE(catalog.contains("beta"));
    REQUIRE_FALSE(catalog.contains("theta"));
}

TEST_CASE("ParameterCatalog constrains and differentiates", "[parameter_catalog]") {
    libsemx::ParameterCatalog catalog;
    catalog.register_parameter("beta", 0.0, libsemx::make_identity_transform());
    catalog.register_parameter("sigma", 0.5, libsemx::make_log_transform());

    auto initial = catalog.initial_unconstrained();
    REQUIRE(initial.size() == 2);

    std::vector<double> unconstrained = {0.25, 0.0};
    auto constrained = catalog.constrain(unconstrained);
    REQUIRE(constrained.size() == 2);
    REQUIRE(constrained[0] == Catch::Approx(0.25));
    REQUIRE(constrained[1] == Catch::Approx(1.0));

    auto derivatives = catalog.constrained_derivatives(unconstrained);
    REQUIRE(derivatives.size() == 2);
    REQUIRE(derivatives[0] == Catch::Approx(1.0));
    REQUIRE(derivatives[1] == Catch::Approx(1.0));
}
