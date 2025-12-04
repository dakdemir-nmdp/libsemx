cmake_minimum_required(VERSION 3.20)

project(libsemx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Disable Eigen testing" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "Disable Eigen documentation" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "Disable Eigen pkgconfig" FORCE)

include(FetchContent)

add_library(libsemx STATIC
    src/libsemx/model_graph.cpp
    src/libsemx/parameter.cpp
    src/libsemx/parameter_transform.cpp
    src/libsemx/parameter_catalog.cpp
    src/libsemx/covariance_structure.cpp
    src/libsemx/kronecker_covariance.cpp
    src/libsemx/genomic_kernel.cpp
    src/libsemx/fixed_covariance.cpp
    src/libsemx/scaled_fixed_covariance.cpp
    src/libsemx/multi_kernel_covariance.cpp
    src/libsemx/gaussian_outcome.cpp
    src/libsemx/binomial_outcome.cpp
    src/libsemx/poisson_outcome.cpp
    src/libsemx/negative_binomial_outcome.cpp
    src/libsemx/weibull_outcome.cpp
    src/libsemx/exponential_outcome.cpp
    src/libsemx/lognormal_outcome.cpp
    src/libsemx/loglogistic_outcome.cpp
    src/libsemx/ordinal_outcome.cpp
    src/libsemx/model_ir.cpp
    src/libsemx/optimizer.cpp
    src/libsemx/likelihood_driver.cpp
    src/libsemx/model_objective.cpp
    src/libsemx/outcome_family_factory.cpp
    src/libsemx/post_estimation.cpp
)

target_include_directories(libsemx
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(libsemx PUBLIC cxx_std_20)

target_compile_options(libsemx PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.4
    FIND_PACKAGE_ARGS NAMES Catch2
)

FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)

FetchContent_Declare(
    LBFGSpp
    GIT_REPOSITORY https://github.com/yixuan/LBFGSpp.git
    GIT_TAG v0.3.0
)

# Disable testing for dependencies
set(BUILD_TESTING_SAVED ${BUILD_TESTING})
set(BUILD_TESTING OFF)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Disable Eigen testing" FORCE)

FetchContent_MakeAvailable(Catch2 Eigen3 LBFGSpp)

# Restore testing for our project
set(BUILD_TESTING ${BUILD_TESTING_SAVED})

target_link_libraries(libsemx PUBLIC Eigen3::Eigen lbfgspp)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_libsemx src/bindings/python_bindings.cpp)
target_link_libraries(_libsemx PRIVATE libsemx)

add_executable(libsemx_tests
    tests/model_graph_tests.cpp
    tests/gaussian_outcome_tests.cpp
    tests/binomial_outcome_tests.cpp
    tests/poisson_outcome_tests.cpp
    tests/mixed_model_tests.cpp
    tests/weibull_outcome_tests.cpp
    tests/exponential_outcome_tests.cpp
    tests/lognormal_outcome_tests.cpp
    tests/loglogistic_outcome_tests.cpp
    tests/ordinal_outcome_tests.cpp
    tests/survival_likelihood_tests.cpp
    tests/competing_risks_tests.cpp
    tests/negative_binomial_outcome_tests.cpp
    tests/outcome_family_factory_tests.cpp
    tests/parameter_catalog_tests.cpp
    tests/model_ir_tests.cpp
    tests/optimizer_tests.cpp
    tests/likelihood_driver_tests.cpp
    tests/reml_tests.cpp
    tests/laplace_tests.cpp
    tests/kronecker_covariance_tests.cpp
    tests/fit_statistics_tests.cpp
    tests/fiml_tests.cpp
    tests/kronecker_laplace_gradient_tests.cpp
    tests/negative_binomial_laplace_gradient_tests.cpp
    tests/ordinal_laplace_gradient_tests.cpp
    tests/grm_tests.cpp
    tests/multi_kernel_tests.cpp
    tests/analytic_gradient_tests.cpp
    tests/dispersion_gradient_tests.cpp
    tests/reml_plumbing_tests.cpp
    tests/covariance_optimization_tests.cpp
    tests/fixed_covariance_fit_tests.cpp
    tests/mixed_model_gradient_tests.cpp
    tests/post_estimation_tests.cpp
    tests/sem_gradient_tests.cpp
    tests/lme4_alignment_tests.cpp
    tests/sommer_comparison_tests_v2.cpp
    tests/lavaan_comparison_tests.cpp
)

target_link_libraries(libsemx_tests
    PRIVATE
        libsemx
        Catch2::Catch2WithMain
)

include(CTest)
include(Catch)

catch_discover_tests(libsemx_tests)
