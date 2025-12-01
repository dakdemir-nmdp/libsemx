#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "libsemx/kronecker_covariance.hpp"
#include "libsemx/covariance_structure.hpp"
#include "libsemx/fixed_covariance.hpp"
#include <vector>
#include <memory>

using namespace libsemx;

TEST_CASE("KroneckerCovariance: Computes product correctly", "[covariance][kronecker]") {
    std::vector<std::unique_ptr<CovarianceStructure>> components;
    components.push_back(std::make_unique<DiagonalCovariance>(2));
    components.push_back(std::make_unique<DiagonalCovariance>(2));
    
    KroneckerCovariance kron(std::move(components));
    
    REQUIRE(kron.dimension() == 4);
    REQUIRE(kron.parameter_count() == 4); // 2 + 2
    
    std::vector<double> params = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> matrix = kron.materialize(params);
    
    REQUIRE_THAT(matrix[0], Catch::Matchers::WithinRel(3.0));
    REQUIRE_THAT(matrix[5], Catch::Matchers::WithinRel(4.0));
    REQUIRE_THAT(matrix[10], Catch::Matchers::WithinRel(6.0));
    REQUIRE_THAT(matrix[15], Catch::Matchers::WithinRel(8.0));
}

TEST_CASE("KroneckerCovariance: 3 components", "[covariance][kronecker]") {
    std::vector<std::unique_ptr<CovarianceStructure>> components;
    // A: 1x1 fixed [2]
    components.push_back(std::make_unique<FixedCovariance>(std::vector<double>{2.0}, 1));
    // B: 2x2 diagonal
    components.push_back(std::make_unique<DiagonalCovariance>(2));
    // C: 1x1 fixed [3]
    components.push_back(std::make_unique<FixedCovariance>(std::vector<double>{3.0}, 1));

    KroneckerCovariance kron(std::move(components));
    
    REQUIRE(kron.dimension() == 2); // 1*2*1
    REQUIRE(kron.parameter_count() == 2); // 0 + 2 + 0
    
    std::vector<double> params = {1.0, 4.0};
    std::vector<double> matrix = kron.materialize(params);
    
    // 2 * diag(1, 4) * 3 = diag(6, 24)
    REQUIRE_THAT(matrix[0], Catch::Matchers::WithinRel(6.0));
    REQUIRE_THAT(matrix[3], Catch::Matchers::WithinRel(24.0));
}

TEST_CASE("KroneckerCovariance: Learn Scale with Normalization", "[covariance][kronecker]") {
    std::vector<std::unique_ptr<CovarianceStructure>> components;
    // A: 2x2 Diagonal. Params [10, 20]. Trace=30. Mean=15.
    components.push_back(std::make_unique<DiagonalCovariance>(2));
    // B: 2x2 Diagonal. Params [2, 4]. Trace=6. Mean=3.
    components.push_back(std::make_unique<DiagonalCovariance>(2));

    // learn_scale = true
    KroneckerCovariance kron(std::move(components), true);

    REQUIRE(kron.parameter_count() == 5); // 1 (global) + 2 + 2

    // Global scale = 2.0
    // A params = [10, 20] -> diag(10, 20). Normalized by 15 -> diag(2/3, 4/3)
    // B params = [2, 4] -> diag(2, 4). Normalized by 3 -> diag(2/3, 4/3)
    // Product = diag(4/9, 8/9, 8/9, 16/9)
    // Scaled by 2.0 -> diag(8/9, 16/9, 16/9, 32/9)
    
    std::vector<double> params = {2.0, 10.0, 20.0, 2.0, 4.0};
    std::vector<double> matrix = kron.materialize(params);

    REQUIRE_THAT(matrix[0], Catch::Matchers::WithinRel(8.0/9.0));
    REQUIRE_THAT(matrix[5], Catch::Matchers::WithinRel(16.0/9.0));
    REQUIRE_THAT(matrix[15], Catch::Matchers::WithinRel(32.0/9.0));
}
