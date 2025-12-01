#include "libsemx/outcome_family_factory.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("OutcomeFamilyFactory creates correct families", "[outcome_family_factory]") {
    SECTION("Gaussian") {
        auto family = libsemx::OutcomeFamilyFactory::create("gaussian");
        REQUIRE(family != nullptr);
        REQUIRE(family->has_dispersion());
    }

    SECTION("Binomial") {
        auto family = libsemx::OutcomeFamilyFactory::create("binomial");
        REQUIRE(family != nullptr);
        REQUIRE_FALSE(family->has_dispersion());
    }

    SECTION("Negative Binomial") {
        auto family = libsemx::OutcomeFamilyFactory::create("negative_binomial");
        REQUIRE(family != nullptr);
        REQUIRE(family->has_dispersion());
    }

    SECTION("NB alias") {
        auto family = libsemx::OutcomeFamilyFactory::create("nbinom");
        REQUIRE(family != nullptr);
    }

    SECTION("Unknown family") {
        REQUIRE_THROWS_AS(libsemx::OutcomeFamilyFactory::create("unknown"), std::invalid_argument);
    }
}