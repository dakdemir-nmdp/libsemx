#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "libsemx/optimizer.hpp"

namespace {
class QuadraticObjective final : public libsemx::ObjectiveFunction {
public:
    QuadraticObjective(std::vector<double> center, double scale)
        : center_(std::move(center)), scale_(scale) {}

    [[nodiscard]] double value(const std::vector<double>& parameters) const override {
        double accum = 0.0;
        for (std::size_t i = 0; i < parameters.size(); ++i) {
            const double diff = parameters[i] - center_[i];
            accum += diff * diff;
        }
        return 0.5 * scale_ * accum;
    }

    [[nodiscard]] std::vector<double> gradient(const std::vector<double>& parameters) const override {
        std::vector<double> grad(parameters.size(), 0.0);
        for (std::size_t i = 0; i < parameters.size(); ++i) {
            grad[i] = scale_ * (parameters[i] - center_[i]);
        }
        return grad;
    }

private:
    std::vector<double> center_;
    double scale_;
};
}  // namespace

TEST_CASE("GradientDescentOptimizer converges on quadratic", "[optimizer]") {
    auto optimizer = libsemx::make_gradient_descent_optimizer();
    QuadraticObjective objective({1.0, -2.0}, 2.0);

    std::vector<double> initial{5.0, 5.0};
    libsemx::OptimizationOptions options;
    options.max_iterations = 500;
    options.tolerance = 1e-8;
    options.learning_rate = 0.1;

    const auto result = optimizer->optimize(objective, initial, options);
    REQUIRE(result.converged);
    REQUIRE(result.iterations > 0);
    REQUIRE(result.gradient_norm <= options.tolerance * 10.0);
    REQUIRE(result.parameters[0] == Catch::Approx(1.0).margin(1e-4));
    REQUIRE(result.parameters[1] == Catch::Approx(-2.0).margin(1e-4));
}

TEST_CASE("LBFGSOptimizer converges on quadratic", "[optimizer]") {
    auto optimizer = libsemx::make_lbfgs_optimizer();
    QuadraticObjective objective({1.0, -2.0}, 2.0);

    std::vector<double> initial{5.0, 5.0};
    libsemx::OptimizationOptions options;
    options.max_iterations = 100;
    options.tolerance = 1e-6;
    // learning_rate is ignored by LBFGS usually, or used for line search guess

    const auto result = optimizer->optimize(objective, initial, options);
    REQUIRE(result.converged);
    REQUIRE(result.iterations > 0);
    REQUIRE(result.parameters[0] == Catch::Approx(1.0).margin(1e-4));
    REQUIRE(result.parameters[1] == Catch::Approx(-2.0).margin(1e-4));
    REQUIRE(result.objective_value == Catch::Approx(0.0).margin(1e-6));
}

TEST_CASE("GradientDescentOptimizer validates options", "[optimizer]") {
    auto optimizer = libsemx::make_gradient_descent_optimizer();
    QuadraticObjective objective({0.0}, 1.0);
    std::vector<double> initial{0.5};

    libsemx::OptimizationOptions options;
    options.max_iterations = 0;
    REQUIRE_THROWS_AS(optimizer->optimize(objective, initial, options), std::invalid_argument);

    options.max_iterations = 10;
    options.tolerance = -1.0;
    REQUIRE_THROWS_AS(optimizer->optimize(objective, initial, options), std::invalid_argument);

    options.tolerance = 1e-6;
    options.learning_rate = 0.0;
    REQUIRE_THROWS_AS(optimizer->optimize(objective, initial, options), std::invalid_argument);
}
