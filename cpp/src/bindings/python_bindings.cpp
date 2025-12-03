#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "libsemx/genomic_kernel.hpp"
#include "libsemx/model_ir.hpp"
#include "libsemx/likelihood_driver.hpp"
#include "libsemx/optimizer.hpp"
#include "libsemx/post_estimation.hpp"

namespace py = pybind11;
using namespace libsemx;

PYBIND11_MODULE(_libsemx, m) {
    m.doc() = "libsemx python bindings";

    // ... (existing enums) ...

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

    py::enum_<ParameterConstraint>(m, "ParameterConstraint")
        .value("Free", ParameterConstraint::Free)
        .value("Positive", ParameterConstraint::Positive)
        .export_values();

    py::enum_<EstimationMethod>(m, "EstimationMethod")
        .value("ML", EstimationMethod::ML)
        .value("REML", EstimationMethod::REML)
        .export_values();

    // ... (existing classes) ...

    py::class_<OptimizationOptions>(m, "OptimizationOptions")
        .def(py::init<>())
        .def_readwrite("max_iterations", &OptimizationOptions::max_iterations)
        .def_readwrite("tolerance", &OptimizationOptions::tolerance)
        .def_readwrite("learning_rate", &OptimizationOptions::learning_rate)
        .def_readwrite("m", &OptimizationOptions::m)
        .def_readwrite("past", &OptimizationOptions::past)
        .def_readwrite("delta", &OptimizationOptions::delta)
        .def_readwrite("max_linesearch", &OptimizationOptions::max_linesearch)
        .def_readwrite("linesearch_type", &OptimizationOptions::linesearch_type);

    py::class_<OptimizationResult>(m, "OptimizationResult")
        .def(py::init<>())
        .def_readwrite("parameters", &OptimizationResult::parameters)
        .def_readwrite("objective_value", &OptimizationResult::objective_value)
        .def_readwrite("gradient_norm", &OptimizationResult::gradient_norm)
        .def_readwrite("iterations", &OptimizationResult::iterations)
        .def_readwrite("converged", &OptimizationResult::converged);

    py::class_<FitResult>(m, "FitResult")
        .def(py::init<>())
        .def_readwrite("optimization_result", &FitResult::optimization_result)
        .def_readwrite("standard_errors", &FitResult::standard_errors)
        .def_readwrite("vcov", &FitResult::vcov)
        .def_readwrite("parameter_names", &FitResult::parameter_names)
        .def_readwrite("aic", &FitResult::aic)
        .def_readwrite("bic", &FitResult::bic)
        .def_readwrite("chi_square", &FitResult::chi_square)
        .def_readwrite("df", &FitResult::df)
        .def_readwrite("p_value", &FitResult::p_value)
        .def_readwrite("cfi", &FitResult::cfi)
        .def_readwrite("tli", &FitResult::tli)
        .def_readwrite("rmsea", &FitResult::rmsea)
        .def_readwrite("srmr", &FitResult::srmr)
        .def_readwrite("covariance_matrices", &FitResult::covariance_matrices);

    // Post-estimation structs
    py::class_<StandardizedEdgeResult>(m, "StandardizedEdgeResult")
        .def_readwrite("std_lv", &StandardizedEdgeResult::std_lv)
        .def_readwrite("std_all", &StandardizedEdgeResult::std_all);

    py::class_<StandardizedSolution>(m, "StandardizedSolution")
        .def_readwrite("edges", &StandardizedSolution::edges);

    py::class_<ModelDiagnostics>(m, "ModelDiagnostics")
        .def_readwrite("implied_means", &ModelDiagnostics::implied_means)
        .def_readwrite("implied_covariance", &ModelDiagnostics::implied_covariance)
        .def_readwrite("mean_residuals", &ModelDiagnostics::mean_residuals)
        .def_readwrite("covariance_residuals", &ModelDiagnostics::covariance_residuals)
        .def_readwrite("correlation_residuals", &ModelDiagnostics::correlation_residuals)
        .def_readwrite("srmr", &ModelDiagnostics::srmr);

    py::class_<ModificationIndex>(m, "ModificationIndex")
        .def_readwrite("source", &ModificationIndex::source)
        .def_readwrite("target", &ModificationIndex::target)
        .def_readwrite("kind", &ModificationIndex::kind)
        .def_readwrite("mi", &ModificationIndex::mi)
        .def_readwrite("epc", &ModificationIndex::epc)
        .def_readwrite("gradient", &ModificationIndex::gradient);

    // Post-estimation functions
    m.def("compute_standardized_estimates", &compute_standardized_estimates,
          py::arg("model"), py::arg("parameter_names"), py::arg("parameter_values"),
          py::arg("fixed_covariance_data") = std::unordered_map<std::string, std::vector<std::vector<double>>>());

    m.def("compute_model_diagnostics", &compute_model_diagnostics,
          py::arg("model"), py::arg("parameter_names"), py::arg("parameter_values"),
          py::arg("sample_means"), py::arg("sample_covariance"),
          py::arg("fixed_covariance_data") = std::unordered_map<std::string, std::vector<std::vector<double>>>());

    m.def("compute_modification_indices", &compute_modification_indices,
          py::arg("model"), py::arg("parameter_names"), py::arg("parameter_values"),
          py::arg("sample_covariance"), py::arg("sample_size"),
          py::arg("fixed_covariance_data") = std::unordered_map<std::string, std::vector<std::vector<double>>>());

    py::class_<VariableSpec>(m, "VariableSpec")
        .def(py::init<>())
        .def(py::init<std::string, VariableKind, std::string, std::string, std::string>(), 
             py::arg("name"), py::arg("kind"), py::arg("family") = "", py::arg("label") = "", py::arg("measurement_level") = "")
        .def_readwrite("name", &VariableSpec::name)
        .def_readwrite("kind", &VariableSpec::kind)
        .def_readwrite("family", &VariableSpec::family)
        .def_readwrite("label", &VariableSpec::label)
        .def_readwrite("measurement_level", &VariableSpec::measurement_level);

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

    py::class_<ParameterSpec>(m, "ParameterSpec")
        .def(py::init<>())
        .def_readwrite("id", &ParameterSpec::id)
        .def_readwrite("constraint", &ParameterSpec::constraint)
        .def_readwrite("initial_value", &ParameterSpec::initial_value);

    py::class_<ModelIR>(m, "ModelIR")
        .def(py::init<>())
        .def_readwrite("variables", &ModelIR::variables)
        .def_readwrite("edges", &ModelIR::edges)
        .def_readwrite("covariances", &ModelIR::covariances)
        .def_readwrite("random_effects", &ModelIR::random_effects)
        .def_readwrite("parameters", &ModelIR::parameters);

    py::class_<ModelIRBuilder>(m, "ModelIRBuilder")
        .def(py::init<>())
        .def("add_variable", &ModelIRBuilder::add_variable,
             py::arg("name"), py::arg("kind"), py::arg("family") = std::string(), py::arg("label") = std::string(), py::arg("measurement_level") = std::string())
        .def("add_edge", &ModelIRBuilder::add_edge)
        .def("add_covariance", &ModelIRBuilder::add_covariance)
        .def("add_random_effect", &ModelIRBuilder::add_random_effect)
        .def("build", &ModelIRBuilder::build);

    py::class_<LikelihoodDriver::DataParamMapping>(m, "DataParamMapping")
        .def(py::init<>())
        .def_readwrite("pattern", &LikelihoodDriver::DataParamMapping::pattern)
        .def_readwrite("stride", &LikelihoodDriver::DataParamMapping::stride);

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
            .def("evaluate_model_gradient", &LikelihoodDriver::evaluate_model_gradient,
                 py::arg("model"),
                 py::arg("data"),
                 py::arg("linear_predictors"),
                 py::arg("dispersions"),
                 py::arg("covariance_parameters") = std::unordered_map<std::string, std::vector<double>>{},
                 py::arg("status") = std::unordered_map<std::string, std::vector<double>>{},
                 py::arg("extra_params") = std::unordered_map<std::string, std::vector<double>>{},
                 py::arg("fixed_covariance_data") = std::unordered_map<std::string, std::vector<std::vector<double>>>{},
             py::arg("method") = EstimationMethod::ML,
             py::arg("data_param_mappings") = std::unordered_map<std::string, LikelihoodDriver::DataParamMapping>{},
             py::arg("dispersion_param_mappings") = std::unordered_map<std::string, LikelihoodDriver::DataParamMapping>{})
            .def("fit", &LikelihoodDriver::fit,
         py::arg("model"),
         py::arg("data"),
         py::arg("options"),
         py::arg("optimizer_name") = "lbfgs",
         py::arg("fixed_covariance_data") = std::unordered_map<std::string, std::vector<std::vector<double>>>{},
         py::arg("status") = std::unordered_map<std::string, std::vector<double>>{},
         py::arg("method") = EstimationMethod::ML);

    py::class_<GenomicRelationshipMatrix>(m, "GenomicRelationshipMatrix")
        .def_static("vanraden",
                    [](const std::vector<double>& markers,
                       std::size_t n_individuals,
                       std::size_t n_markers,
                       bool center,
                       bool normalize) {
                        GenomicKernelOptions opts{center, normalize};
                        return GenomicRelationshipMatrix::vanraden(markers, n_individuals, n_markers, opts);
                    },
                    py::arg("markers"),
                    py::arg("n_individuals"),
                    py::arg("n_markers"),
                    py::arg("center") = true,
                    py::arg("normalize") = true)
        .def_static("kronecker",
                    &GenomicRelationshipMatrix::kronecker,
                    py::arg("left"),
                    py::arg("left_dim"),
                    py::arg("right"),
                    py::arg("right_dim"));
}
