#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "libsemx/model_ir.hpp"
#include "libsemx/likelihood_driver.hpp"
#include "libsemx/optimizer.hpp"

namespace py = pybind11;
using namespace libsemx;

PYBIND11_MODULE(_libsemx, m) {
    m.doc() = "libsemx python bindings";

    py::enum_<VariableKind>(m, "VariableKind")
        .value("Observed", VariableKind::Observed)
        .value("Latent", VariableKind::Latent)
        .value("Grouping", VariableKind::Grouping)
        .export_values();

    py::enum_<EdgeKind>(m, "EdgeKind")
        .value("Loading", EdgeKind::Loading)
        .value("Regression", EdgeKind::Regression)
        .value("Covariance", EdgeKind::Covariance)
        .export_values();

    py::enum_<EstimationMethod>(m, "EstimationMethod")
        .value("ML", EstimationMethod::ML)
        .value("REML", EstimationMethod::REML)
        .export_values();

    py::class_<OptimizationOptions>(m, "OptimizationOptions")
        .def(py::init<>())
        .def_readwrite("max_iterations", &OptimizationOptions::max_iterations)
        .def_readwrite("tolerance", &OptimizationOptions::tolerance)
        .def_readwrite("learning_rate", &OptimizationOptions::learning_rate);

    py::class_<OptimizationResult>(m, "OptimizationResult")
        .def_readwrite("parameters", &OptimizationResult::parameters)
        .def_readwrite("objective_value", &OptimizationResult::objective_value)
        .def_readwrite("gradient_norm", &OptimizationResult::gradient_norm)
        .def_readwrite("iterations", &OptimizationResult::iterations)
        .def_readwrite("converged", &OptimizationResult::converged);

    py::class_<VariableSpec>(m, "VariableSpec")
        .def(py::init<>())
        .def(py::init<std::string, VariableKind, std::string>(), 
             py::arg("name"), py::arg("kind"), py::arg("family") = "")
        .def_readwrite("name", &VariableSpec::name)
        .def_readwrite("kind", &VariableSpec::kind)
        .def_readwrite("family", &VariableSpec::family);

    py::class_<EdgeSpec>(m, "EdgeSpec")
        .def(py::init<>())
        .def(py::init<EdgeKind, std::string, std::string, std::string>(),
             py::arg("kind"), py::arg("source"), py::arg("target"), py::arg("parameter_id") = "")
        .def_readwrite("kind", &EdgeSpec::kind)
        .def_readwrite("source", &EdgeSpec::source)
        .def_readwrite("target", &EdgeSpec::target)
        .def_readwrite("parameter_id", &EdgeSpec::parameter_id);

    py::class_<CovarianceSpec>(m, "CovarianceSpec")
        .def(py::init<>())
        .def_readwrite("id", &CovarianceSpec::id)
        .def_readwrite("structure", &CovarianceSpec::structure)
        .def_readwrite("dimension", &CovarianceSpec::dimension);

    py::class_<RandomEffectSpec>(m, "RandomEffectSpec")
        .def(py::init<>())
        .def_readwrite("id", &RandomEffectSpec::id)
        .def_readwrite("variables", &RandomEffectSpec::variables)
        .def_readwrite("covariance_id", &RandomEffectSpec::covariance_id);

    py::class_<ModelIR>(m, "ModelIR")
        .def(py::init<>())
        .def_readwrite("variables", &ModelIR::variables)
        .def_readwrite("edges", &ModelIR::edges)
        .def_readwrite("covariances", &ModelIR::covariances)
        .def_readwrite("random_effects", &ModelIR::random_effects);

    py::class_<ModelIRBuilder>(m, "ModelIRBuilder")
        .def(py::init<>())
        .def("add_variable", &ModelIRBuilder::add_variable,
             py::arg("name"), py::arg("kind"), py::arg("family") = std::string())
        .def("add_edge", &ModelIRBuilder::add_edge)
        .def("add_covariance", &ModelIRBuilder::add_covariance)
        .def("add_random_effect", &ModelIRBuilder::add_random_effect)
        .def("build", &ModelIRBuilder::build);

    py::class_<LikelihoodDriver>(m, "LikelihoodDriver")
        .def(py::init<>())
        .def("evaluate_model_loglik", &LikelihoodDriver::evaluate_model_loglik,
             py::arg("model"),
             py::arg("data"),
             py::arg("linear_predictors"),
             py::arg("dispersions"),
             py::arg("covariance_parameters") = std::unordered_map<std::string, std::vector<double>>{},
             py::arg("status") = std::unordered_map<std::string, std::vector<double>>{},
             py::arg("extra_params") = std::unordered_map<std::string, std::vector<double>>{},
             py::arg("fixed_covariance_data") = std::unordered_map<std::string, std::vector<std::vector<double>>>{},
             py::arg("method") = EstimationMethod::ML)
        .def("fit", &LikelihoodDriver::fit,
             py::arg("model"),
             py::arg("data"),
             py::arg("options") = OptimizationOptions(),
             py::arg("optimizer_name") = "lbfgs");
}
