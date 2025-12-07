#include <Rcpp.h>
#include "libsemx/genomic_kernel.hpp"
#include "libsemx/model_ir.hpp"
#include "libsemx/likelihood_driver.hpp"
#include "libsemx/optimizer.hpp"
#include "libsemx/post_estimation.hpp"

using namespace Rcpp;

RCPP_EXPOSED_CLASS_NODECL(libsemx::ModelIR)
RCPP_EXPOSED_CLASS_NODECL(libsemx::ModelIRBuilder)
RCPP_EXPOSED_CLASS_NODECL(libsemx::LikelihoodDriver)
RCPP_EXPOSED_CLASS_NODECL(libsemx::OptimizationOptions)
RCPP_EXPOSED_CLASS_NODECL(libsemx::OptimizationResult)
RCPP_EXPOSED_CLASS_NODECL(libsemx::FitResult)

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

std::unordered_map<std::string, std::vector<std::string>> list_to_string_vector_map(const Rcpp::Nullable<Rcpp::List>& l) {
    std::unordered_map<std::string, std::vector<std::string>> map;
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
        
        if (Rf_isNull(list[i])) {
            continue;
        }
        
        Rcpp::CharacterVector vec = Rcpp::as<Rcpp::CharacterVector>(list[i]);
        std::vector<std::string> val;
        for(int j=0; j<vec.size(); ++j) {
            val.push_back(Rcpp::as<std::string>(vec[j]));
        }
        map[key] = val;
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

Rcpp::List ModelIR_variables(ModelIR* model) {
    Rcpp::List list;
    for (const auto& var : model->variables) {
        list.push_back(Rcpp::List::create(
            Rcpp::Named("name") = var.name,
            Rcpp::Named("kind") = static_cast<int>(var.kind),
            Rcpp::Named("family") = var.family,
            Rcpp::Named("label") = var.label,
            Rcpp::Named("measurement_level") = var.measurement_level
        ));
    }
    return list;
}

Rcpp::List ModelIR_covariances(ModelIR* model) {
    Rcpp::List list;
    for (const auto& cov : model->covariances) {
        list.push_back(Rcpp::List::create(
            Rcpp::Named("id") = cov.id,
            Rcpp::Named("structure") = cov.structure,
            Rcpp::Named("dimension") = cov.dimension
        ));
    }
    return list;
}

Rcpp::List ModelIR_random_effects(ModelIR* model) {
    Rcpp::List list;
    for (const auto& re : model->random_effects) {
        list.push_back(Rcpp::List::create(
            Rcpp::Named("id") = re.id,
            Rcpp::Named("variables") = re.variables,
            Rcpp::Named("covariance_id") = re.covariance_id
        ));
    }
    return list;
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

std::vector<double> to_row_major(const Rcpp::NumericMatrix& mat) {
    std::size_t n_rows = static_cast<std::size_t>(mat.nrow());
    std::size_t n_cols = static_cast<std::size_t>(mat.ncol());
    std::vector<double> out(n_rows * n_cols);
    for (std::size_t i = 0; i < n_rows; ++i) {
        for (std::size_t j = 0; j < n_cols; ++j) {
            out[i * n_cols + j] = mat(i, j);
        }
    }
    return out;
}

std::vector<double> grm_vanraden_cpp(const Rcpp::NumericMatrix& markers,
                                     bool center,
                                     bool normalize) {
    std::size_t n_individuals = static_cast<std::size_t>(markers.nrow());
    std::size_t n_markers = static_cast<std::size_t>(markers.ncol());
    auto flattened = to_row_major(markers);
    GenomicKernelOptions opts{center, normalize};
    return GenomicRelationshipMatrix::vanraden(flattened, n_individuals, n_markers, opts);
}

std::vector<double> grm_kronecker_cpp(const Rcpp::NumericMatrix& left,
                                      const Rcpp::NumericMatrix& right) {
    std::size_t left_dim = static_cast<std::size_t>(left.nrow());
    std::size_t right_dim = static_cast<std::size_t>(right.nrow());
    auto left_rm = to_row_major(left);
    auto right_rm = to_row_major(right);
    return GenomicRelationshipMatrix::kronecker(left_rm, left_dim, right_rm, right_dim);
}

} // namespace

// Wrapper for ModelIRBuilder::add_variable
void ModelIRBuilder_add_variable(ModelIRBuilder* builder, std::string name, int kind, std::string family, std::string label, std::string measurement_level) {
    builder->add_variable(name, static_cast<VariableKind>(kind), family, label, measurement_level);
}

// Wrapper for ModelIRBuilder::add_edge
void ModelIRBuilder_add_edge(ModelIRBuilder* builder, int kind, std::string source, std::string target, std::string parameter_id) {
    builder->add_edge(static_cast<EdgeKind>(kind), source, target, parameter_id);
}

// Basic wrapper preserves the original four-argument API
double LikelihoodDriver_evaluate_model_loglik(LikelihoodDriver* driver, 
                                              ModelIR* model, 
                                              Rcpp::List data, 
                                              Rcpp::List linear_predictors, 
                                              Rcpp::List dispersions) {
    return driver->evaluate_model_loglik(*model, 
                                         list_to_map(data), 
                                         list_to_map(linear_predictors), 
                                         list_to_map(dispersions));
}

double LikelihoodDriver_evaluate_model_loglik_full(LikelihoodDriver* driver,
                                                   ModelIR* model,
                                                   Rcpp::List data,
                                                   Rcpp::List linear_predictors,
                                                   Rcpp::List dispersions,
                                                   Rcpp::List covariance_parameters,
                                                   Rcpp::List status,
                                                   Rcpp::List extra_params,
                                                   Rcpp::Nullable<Rcpp::List> fixed_covariance_data,
                                                   int method) {
    return driver->evaluate_model_loglik(*model,
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
                                                    ModelIR* model,
                                                    Rcpp::List data,
                                                    Rcpp::List linear_predictors,
                                                    Rcpp::List dispersions,
                                                    Rcpp::List covariance_parameters,
                                                    Rcpp::List status,
                                                    Rcpp::List extra_params,
                                                    Rcpp::Nullable<Rcpp::List> fixed_covariance_data,
                                                    int method,
                                                    Rcpp::Nullable<Rcpp::List> extra_param_mappings = R_NilValue) {
    auto gradients = driver->evaluate_model_gradient(
        *model,
        list_to_map(data),
        list_to_map(linear_predictors),
        list_to_map(dispersions),
        list_to_map(covariance_parameters),
        list_to_map(status),
        list_to_map(extra_params),
        list_to_matrix_map(fixed_covariance_data),
        static_cast<EstimationMethod>(method),
        {}, // data_param_mappings
        {}, // dispersion_param_mappings
        list_to_string_vector_map(extra_param_mappings));
    return gradient_map_to_list(gradients);
}

// Wrapper for LikelihoodDriver::fit
FitResult LikelihoodDriver_fit(LikelihoodDriver* driver,
                                        ModelIR* model,
                                        Rcpp::List data,
                                        OptimizationOptions* options,
                                        std::string optimizer_name,
                                        Rcpp::Nullable<Rcpp::List> extra_param_mappings,
                                        int method) {
    Rcpp::Rcerr << "LikelihoodDriver_fit wrapper called" << std::endl;
    auto data_map = list_to_map(data);
    Rcpp::Rcerr << "Data map created, size: " << data_map.size() << std::endl;
    auto extra_map = list_to_string_vector_map(extra_param_mappings);
    Rcpp::Rcerr << "Extra map created" << std::endl;
    return driver->fit(*model, data_map, *options, optimizer_name, {}, {}, static_cast<EstimationMethod>(method), extra_map);
}

FitResult LikelihoodDriver_fit_with_fixed(LikelihoodDriver* driver,
                                                   ModelIR* model,
                                                   Rcpp::List data,
                                                   OptimizationOptions* options,
                                                   std::string optimizer_name,
                                                   Rcpp::Nullable<Rcpp::List> fixed_covariance_data,
                                                   Rcpp::Nullable<Rcpp::List> extra_param_mappings,
                                                   int method) {
    return driver->fit(*model,
                       list_to_map(data),
                       *options,
                       optimizer_name,
                       list_to_matrix_map(fixed_covariance_data),
                       {},
                       static_cast<EstimationMethod>(method),
                       list_to_string_vector_map(extra_param_mappings));
}

FitResult LikelihoodDriver_fit_with_status(LikelihoodDriver* driver,
                                        ModelIR* model,
                                        Rcpp::List data,
                                        OptimizationOptions* options,
                                        std::string optimizer_name,
                                        Rcpp::List status,
                                        Rcpp::Nullable<Rcpp::List> extra_param_mappings,
                                        int method) {
    return driver->fit(*model, list_to_map(data), *options, optimizer_name, {}, list_to_map(status), static_cast<EstimationMethod>(method), list_to_string_vector_map(extra_param_mappings));
}

FitResult LikelihoodDriver_fit_with_fixed_and_status(LikelihoodDriver* driver,
                                                   ModelIR* model,
                                                   Rcpp::List data,
                                                   OptimizationOptions* options,
                                                   std::string optimizer_name,
                                                   Rcpp::Nullable<Rcpp::List> fixed_covariance_data,
                                                   Rcpp::List status,
                                                   Rcpp::Nullable<Rcpp::List> extra_param_mappings,
                                                   int method) {
    return driver->fit(*model,
                       list_to_map(data),
                       *options,
                       optimizer_name,
                       list_to_matrix_map(fixed_covariance_data),
                       list_to_map(status),
                       static_cast<EstimationMethod>(method),
                       list_to_string_vector_map(extra_param_mappings));
}

std::string edge_kind_to_string(EdgeKind k) {
    switch(k) {
        case EdgeKind::Loading: return "Loading";
        case EdgeKind::Regression: return "Regression";
        case EdgeKind::Covariance: return "Covariance";
        default: return "Unknown";
    }
}

struct SampleStats {
    std::vector<double> means;
    std::vector<double> covariance;
    size_t n_obs;
};

SampleStats compute_sample_stats(const ModelIR& model, const std::unordered_map<std::string, std::vector<double>>& data) {
    std::vector<std::string> observed_vars;
    for(const auto& v : model.variables) {
        if (v.kind == VariableKind::Observed) {
            observed_vars.push_back(v.name);
        }
    }
    
    size_t n_vars = observed_vars.size();
    size_t n_obs = 0;
    if (n_vars > 0 && data.count(observed_vars[0])) {
        n_obs = data.at(observed_vars[0]).size();
    }
    
    std::vector<double> means(n_vars, 0.0);
    std::vector<double> covariance(n_vars * n_vars, 0.0);
    
    if (n_obs == 0) return {std::vector<double>(model.variables.size(), 0.0), std::vector<double>(model.variables.size() * model.variables.size(), 0.0), 0};
    
    // Compute means
    for(size_t i=0; i<n_vars; ++i) {
        const auto& vec = data.at(observed_vars[i]);
        double sum = 0.0;
        for(double val : vec) sum += val;
        means[i] = sum / n_obs;
    }
    
    // Compute covariance
    for(size_t i=0; i<n_vars; ++i) {
        for(size_t j=0; j<n_vars; ++j) {
            const auto& vec_i = data.at(observed_vars[i]);
            const auto& vec_j = data.at(observed_vars[j]);
            double sum_sq = 0.0;
            for(size_t k=0; k<n_obs; ++k) {
                sum_sq += (vec_i[k] - means[i]) * (vec_j[k] - means[j]);
            }
            covariance[i*n_vars + j] = sum_sq / (n_obs - 1);
        }
    }
    
    std::vector<double> full_means(model.variables.size(), 0.0);
    std::vector<double> full_cov(model.variables.size() * model.variables.size(), 0.0);
    
    std::unordered_map<std::string, size_t> var_idx;
    for(size_t i=0; i<model.variables.size(); ++i) {
        var_idx[model.variables[i].name] = i;
    }
    
    for(size_t i=0; i<n_vars; ++i) {
        size_t idx_i = var_idx[observed_vars[i]];
        full_means[idx_i] = means[i];
        
        for(size_t j=0; j<n_vars; ++j) {
            size_t idx_j = var_idx[observed_vars[j]];
            full_cov[idx_i * model.variables.size() + idx_j] = covariance[i*n_vars + j];
        }
    }
    
    return {full_means, full_cov, n_obs};
}

Rcpp::List compute_standardized_estimates_wrapper(
    ModelIR model,
    std::vector<std::string> parameter_names,
    std::vector<double> parameter_values,
    Rcpp::Nullable<Rcpp::List> data) {
    
    // Standardized estimates don't need observed data, only fixed covariance data if any.
    // For now we assume no fixed covariance data passed here (it should be passed if needed).
    // But the signature in R takes 'data'.
    // We pass empty map for fixed_covariance_data.
    
    auto std_sol = compute_standardized_estimates(model, parameter_names, parameter_values, {});
    
    std::vector<double> std_lv;
    std::vector<double> std_all;
    for(const auto& e : std_sol.edges) {
        std_lv.push_back(e.std_lv);
        std_all.push_back(e.std_all);
    }
    
    return Rcpp::List::create(
        Rcpp::Named("edges") = Rcpp::DataFrame::create(
            Rcpp::Named("std_lv") = std_lv,
            Rcpp::Named("std_all") = std_all
        )
    );
}

Rcpp::List compute_model_diagnostics_wrapper(
    ModelIR model,
    std::vector<std::string> parameter_names,
    std::vector<double> parameter_values,
    Rcpp::Nullable<Rcpp::List> data) {
    
    auto stats = compute_sample_stats(model, list_to_map(data));
    auto diag = compute_model_diagnostics(model, parameter_names, parameter_values, stats.means, stats.covariance, {});
    
    return Rcpp::List::create(
        Rcpp::Named("implied_means") = diag.implied_means,
        Rcpp::Named("implied_covariance") = diag.implied_covariance,
        Rcpp::Named("mean_residuals") = diag.mean_residuals,
        Rcpp::Named("covariance_residuals") = diag.covariance_residuals,
        Rcpp::Named("correlation_residuals") = diag.correlation_residuals,
        Rcpp::Named("srmr") = diag.srmr
    );
}

Rcpp::DataFrame compute_modification_indices_wrapper(
    ModelIR model,
    std::vector<std::string> parameter_names,
    std::vector<double> parameter_values,
    Rcpp::Nullable<Rcpp::List> data) {
    
    auto stats = compute_sample_stats(model, list_to_map(data));
    auto mis = compute_modification_indices(model, parameter_names, parameter_values, stats.covariance, stats.n_obs, {});
    
    std::vector<std::string> source;
    std::vector<std::string> target;
    std::vector<std::string> kind;
    std::vector<double> mi_val;
    std::vector<double> epc;
    std::vector<double> gradient;
    
    for(const auto& m : mis) {
        source.push_back(m.source);
        target.push_back(m.target);
        kind.push_back(edge_kind_to_string(m.kind));
        mi_val.push_back(m.mi);
        epc.push_back(m.epc);
        gradient.push_back(m.gradient);
    }
    
    return Rcpp::DataFrame::create(
        Rcpp::Named("source") = source,
        Rcpp::Named("target") = target,
        Rcpp::Named("kind") = kind,
        Rcpp::Named("mi") = mi_val,
        Rcpp::Named("epc") = epc,
        Rcpp::Named("gradient") = gradient
    );
}

std::unordered_map<std::string, std::vector<double>> get_fit_covariance_matrices(FitResult* obj) {
    return obj->covariance_matrices;
}

// Wrapper for ModelIRBuilder::build
ModelIR* ModelIRBuilder_build(ModelIRBuilder* builder) {
    return new ModelIR(builder->build());
}

RCPP_MODULE(semx) {
    class_<ModelIR>("ModelIR")
        .method("parameter_ids", &ModelIR_parameter_ids)
        .property("variables", &ModelIR_variables, "Get variables")
        .property("covariances", &ModelIR_covariances, "Get covariances")
        .property("random_effects", &ModelIR_random_effects, "Get random effects");

    class_<ModelIRBuilder>("ModelIRBuilder")
        .constructor()
        .method("add_variable", &ModelIRBuilder_add_variable)
        .method("add_edge", &ModelIRBuilder_add_edge)
        .method("add_covariance", &ModelIRBuilder::add_covariance)
        .method("add_random_effect", &ModelIRBuilder::add_random_effect)
        .method("register_parameter", &ModelIRBuilder::register_parameter)
        .method("build", &ModelIRBuilder_build)
    ;

    class_<OptimizationOptions>("OptimizationOptions")
        .constructor()
        .field("max_iterations", &OptimizationOptions::max_iterations)
        .field("tolerance", &OptimizationOptions::tolerance)
        .field("learning_rate", &OptimizationOptions::learning_rate)
        .field("m", &OptimizationOptions::m)
        .field("past", &OptimizationOptions::past)
        .field("delta", &OptimizationOptions::delta)
        .field("max_linesearch", &OptimizationOptions::max_linesearch)
        .field("linesearch_type", &OptimizationOptions::linesearch_type)
    ;

    class_<OptimizationResult>("OptimizationResult")
        .field("parameters", &OptimizationResult::parameters)
        .field("objective_value", &OptimizationResult::objective_value)
        .field("gradient_norm", &OptimizationResult::gradient_norm)
        .field("iterations", &OptimizationResult::iterations)
        .field("converged", &OptimizationResult::converged)
    ;

    class_<FitResult>("FitResult")
        .field("optimization_result", &FitResult::optimization_result)
        .field("standard_errors", &FitResult::standard_errors)
        .field("vcov", &FitResult::vcov)
        .field("parameter_names", &FitResult::parameter_names)
        .field("aic", &FitResult::aic)
        .field("bic", &FitResult::bic)
        .field("chi_square", &FitResult::chi_square)
        .field("df", &FitResult::df)
        .field("p_value", &FitResult::p_value)
        .field("cfi", &FitResult::cfi)
        .field("tli", &FitResult::tli)
        .field("rmsea", &FitResult::rmsea)
        .field("srmr", &FitResult::srmr)
        .property("covariance_matrices", &get_fit_covariance_matrices, "Get covariance matrices")
    ;

    class_<LikelihoodDriver>("LikelihoodDriver")
        .constructor()
        .method("evaluate_model_loglik", &LikelihoodDriver_evaluate_model_loglik)
        .method("evaluate_model_loglik_full", &LikelihoodDriver_evaluate_model_loglik_full)
        .method("evaluate_model_gradient", &LikelihoodDriver_evaluate_model_gradient)
        .method("fit", &LikelihoodDriver_fit)
        .method("fit_with_fixed", &LikelihoodDriver_fit_with_fixed)
        .method("fit_with_status", &LikelihoodDriver_fit_with_status)
        .method("fit_with_fixed_and_status", &LikelihoodDriver_fit_with_fixed_and_status)
    ;

    function("grm_vanraden_cpp", &grm_vanraden_cpp);
    function("grm_kronecker_cpp", &grm_kronecker_cpp);
    function("compute_standardized_estimates_wrapper", &compute_standardized_estimates_wrapper);
    function("compute_model_diagnostics_wrapper", &compute_model_diagnostics_wrapper);
    function("compute_modification_indices_wrapper", &compute_modification_indices_wrapper);
}
