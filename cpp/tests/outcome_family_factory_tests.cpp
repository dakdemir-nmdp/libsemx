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

    SECTION("Weibull") {
        auto family = libsemx::OutcomeFamilyFactory::create("weibull");
        REQUIRE(family != nullptr);
        REQUIRE(family->has_dispersion());
    }

    SECTION("Exponential") {
        auto family = libsemx::OutcomeFamilyFactory::create("exponential");
        REQUIRE(family != nullptr);
        REQUIRE_FALSE(family->has_dispersion());
    }

    SECTION("Lognormal") {
        auto family = libsemx::OutcomeFamilyFactory::create("lognormal_aft");
        REQUIRE(family != nullptr);
        REQUIRE(family->has_dispersion());
    }

    SECTION("Loglogistic") {
        auto family = libsemx::OutcomeFamilyFactory::create("loglogistic");
        REQUIRE(family != nullptr);
        REQUIRE(family->has_dispersion());
    }

    SECTION("Unknown family") {
        REQUIRE_THROWS_AS(libsemx::OutcomeFamilyFactory::create("unknown"), std::invalid_argument);
    }
}