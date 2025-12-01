#include <Rcpp.h>
#include "libsemx/model_ir.hpp"
#include "libsemx/likelihood_driver.hpp"
#include "libsemx/optimizer.hpp"

using namespace Rcpp;

RCPP_EXPOSED_CLASS_NODECL(libsemx::ModelIR)
RCPP_EXPOSED_CLASS_NODECL(libsemx::ModelIRBuilder)
RCPP_EXPOSED_CLASS_NODECL(libsemx::LikelihoodDriver)
RCPP_EXPOSED_CLASS_NODECL(libsemx::OptimizationOptions)
RCPP_EXPOSED_CLASS_NODECL(libsemx::OptimizationResult)

using namespace libsemx;

namespace {

std::unordered_map<std::string, std::vector<double>> list_to_map_impl(const Rcpp::List& l) {
    std::unordered_map<std::string, std::vector<double>> map;
    if (l.size() == 0) {
        return map;
    }

    Rcpp::CharacterVector names = l.names();
    for (int i = 0; i < l.size(); ++i) {
        std::string key;
        if (names.size() > i && names[i] != NA_STRING) {
            key = Rcpp::as<std::string>(names[i]);
        } else {
            key = std::to_string(i + 1);
        }
        map[key] = Rcpp::as<std::vector<double>>(l[i]);
    }
    return map;
}

std::unordered_map<std::string, std::vector<double>> list_to_map(const Rcpp::List& l) {
    return list_to_map_impl(l);
}

std::unordered_map<std::string, std::vector<double>> list_to_map(const Rcpp::Nullable<Rcpp::List>& l) {
    if (l.isNull()) {
        return {};
    }
    return list_to_map_impl(Rcpp::List(l));
}

std::vector<double> flatten_numeric(SEXP obj) {
    if (Rf_isNull(obj)) {
        return {};
    }
    if (Rf_isMatrix(obj)) {
        Rcpp::NumericMatrix mat(obj);
        return std::vector<double>(mat.begin(), mat.end());
    }
    Rcpp::NumericVector vec(obj);
    return std::vector<double>(vec.begin(), vec.end());
}

std::unordered_map<std::string, std::vector<std::vector<double>>> list_to_matrix_map(const Rcpp::Nullable<Rcpp::List>& l) {
    std::unordered_map<std::string, std::vector<std::vector<double>>> map;
    if (l.isNull()) {
        return map;
    }

    Rcpp::List list(l);
    if (list.size() == 0) {
        return map;
    }

    Rcpp::CharacterVector names = list.names();
    for (int i = 0; i < list.size(); ++i) {
        std::string key;
        if (names.size() > i && names[i] != NA_STRING) {
            key = Rcpp::as<std::string>(names[i]);
        } else {
            key = std::to_string(i + 1);
        }

        std::vector<std::vector<double>> entries;
        SEXP obj = list[i];
        if (Rf_isNull(obj)) {
            map[key] = entries;
            continue;
        }

        if (Rf_isNewList(obj)) {
            Rcpp::List nested(obj);
            for (int j = 0; j < nested.size(); ++j) {
                entries.push_back(flatten_numeric(nested[j]));
            }
        } else {
            entries.push_back(flatten_numeric(obj));
        }
        map[key] = std::move(entries);
    }
    return map;
}

Rcpp::List gradient_map_to_list(const std::unordered_map<std::string, double>& gradients) {
    Rcpp::List result;
    for (const auto& entry : gradients) {
        result[entry.first] = entry.second;
    }
    return result;
}

Rcpp::CharacterVector ModelIR_parameter_ids(ModelIR* model) {
    if (model == nullptr) {
        return Rcpp::CharacterVector();
    }
    Rcpp::CharacterVector ids(model->parameters.size());
    for (std::size_t i = 0; i < model->parameters.size(); ++i) {
        ids[i] = model->parameters[i].id;
    }
    return ids;
}

} // namespace

// Wrapper for ModelIRBuilder::add_variable to handle enum
void ModelIRBuilder_add_variable(ModelIRBuilder* builder, std::string name, int kind, std::string family) {
    builder->add_variable(name, static_cast<VariableKind>(kind), family);
}

// Wrapper for ModelIRBuilder::add_edge
void ModelIRBuilder_add_edge(ModelIRBuilder* builder, int kind, std::string source, std::string target, std::string parameter_id) {
    builder->add_edge(static_cast<EdgeKind>(kind), source, target, parameter_id);
}

// Basic wrapper preserves the original four-argument API
double LikelihoodDriver_evaluate_model_loglik(LikelihoodDriver* driver, 
                                              const ModelIR& model, 
                                              Rcpp::List data, 
                                              Rcpp::List linear_predictors, 
                                              Rcpp::List dispersions) {
    return driver->evaluate_model_loglik(model, 
                                         list_to_map(data), 
                                         list_to_map(linear_predictors), 
                                         list_to_map(dispersions));
}

double LikelihoodDriver_evaluate_model_loglik_full(LikelihoodDriver* driver,
                                                   const ModelIR& model,
                                                   Rcpp::List data,
                                                   Rcpp::List linear_predictors,
                                                   Rcpp::List dispersions,
                                                   Rcpp::List covariance_parameters,
                                                   Rcpp::List status,
                                                   Rcpp::List extra_params,
                                                   Rcpp::Nullable<Rcpp::List> fixed_covariance_data,
                                                   int method) {
    return driver->evaluate_model_loglik(model,
                                         list_to_map(data),
                                         list_to_map(linear_predictors),
                                         list_to_map(dispersions),
                                         list_to_map(covariance_parameters),
                                         list_to_map(status),
                                         list_to_map(extra_params),
                                         list_to_matrix_map(fixed_covariance_data),
                                         static_cast<EstimationMethod>(method));
}

Rcpp::List LikelihoodDriver_evaluate_model_gradient(LikelihoodDriver* driver,
                                                    const ModelIR& model,
                                                    Rcpp::List data,
                                                    Rcpp::List linear_predictors,
                                                    Rcpp::List dispersions,
                                                    Rcpp::List covariance_parameters,
                                                    Rcpp::List status,
                                                    Rcpp::List extra_params,
                                                    Rcpp::Nullable<Rcpp::List> fixed_covariance_data,
                                                    int method) {
    auto gradients = driver->evaluate_model_gradient(
        model,
        list_to_map(data),
        list_to_map(linear_predictors),
        list_to_map(dispersions),
        list_to_map(covariance_parameters),
        list_to_map(status),
        list_to_map(extra_params),
        list_to_matrix_map(fixed_covariance_data),
        static_cast<EstimationMethod>(method));
    return gradient_map_to_list(gradients);
}

// Wrapper for LikelihoodDriver::fit
OptimizationResult LikelihoodDriver_fit(LikelihoodDriver* driver,
                                        const ModelIR& model,
                                        Rcpp::List data,
                                        OptimizationOptions options,
                                        std::string optimizer_name) {
    return driver->fit(model, list_to_map(data), options, optimizer_name);
}

OptimizationResult LikelihoodDriver_fit_with_fixed(LikelihoodDriver* driver,
                                                   const ModelIR& model,
                                                   Rcpp::List data,
                                                   OptimizationOptions options,
                                                   std::string optimizer_name,
                                                   Rcpp::Nullable<Rcpp::List> fixed_covariance_data) {
    return driver->fit(model,
                       list_to_map(data),
                       options,
                       optimizer_name,
                       list_to_matrix_map(fixed_covariance_data));
}

RCPP_MODULE(semx) {
    class_<ModelIR>("ModelIR")
        .method("parameter_ids", &ModelIR_parameter_ids);

    class_<ModelIRBuilder>("ModelIRBuilder")
        .constructor()
        .method("add_variable", &ModelIRBuilder_add_variable)
        .method("add_edge", &ModelIRBuilder_add_edge)
        .method("add_covariance", &ModelIRBuilder::add_covariance)
        .method("add_random_effect", &ModelIRBuilder::add_random_effect)
        .method("build", &ModelIRBuilder::build)
    ;

    class_<OptimizationOptions>("OptimizationOptions")
        .constructor()
        .field("max_iterations", &OptimizationOptions::max_iterations)
        .field("tolerance", &OptimizationOptions::tolerance)
        .field("learning_rate", &OptimizationOptions::learning_rate)
    ;

    class_<OptimizationResult>("OptimizationResult")
        .field("parameters", &OptimizationResult::parameters)
        .field("objective_value", &OptimizationResult::objective_value)
        .field("gradient_norm", &OptimizationResult::gradient_norm)
        .field("iterations", &OptimizationResult::iterations)
        .field("converged", &OptimizationResult::converged)
    ;

    class_<LikelihoodDriver>("LikelihoodDriver")
        .constructor()
        .method("evaluate_model_loglik", &LikelihoodDriver_evaluate_model_loglik)
        .method("evaluate_model_loglik_full", &LikelihoodDriver_evaluate_model_loglik_full)
        .method("evaluate_model_gradient", &LikelihoodDriver_evaluate_model_gradient)
        .method("fit", &LikelihoodDriver_fit)
        .method("fit_with_fixed", &LikelihoodDriver_fit_with_fixed)
    ;
}


