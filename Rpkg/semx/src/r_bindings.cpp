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

// Helper to convert R list to unordered_map
std::unordered_map<std::string, std::vector<double>> list_to_map(Rcpp::List l) {
    std::unordered_map<std::string, std::vector<double>> map;
    if (Rf_isNull(l)) return map;
    
    std::vector<std::string> names = l.names();
    for (int i = 0; i < l.size(); ++i) {
        map[names[i]] = Rcpp::as<std::vector<double>>(l[i]);
    }
    return map;
}

// Wrapper for ModelIRBuilder::add_variable to handle enum
void ModelIRBuilder_add_variable(ModelIRBuilder* builder, std::string name, int kind, std::string family) {
    builder->add_variable(name, static_cast<VariableKind>(kind), family);
}

// Wrapper for ModelIRBuilder::add_edge
void ModelIRBuilder_add_edge(ModelIRBuilder* builder, int kind, std::string source, std::string target, std::string parameter_id) {
    builder->add_edge(static_cast<EdgeKind>(kind), source, target, parameter_id);
}

// Wrapper for LikelihoodDriver::evaluate_model_loglik
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

// Wrapper for LikelihoodDriver::fit
OptimizationResult LikelihoodDriver_fit(LikelihoodDriver* driver,
                                        const ModelIR& model,
                                        Rcpp::List data,
                                        OptimizationOptions options,
                                        std::string optimizer_name) {
    return driver->fit(model, list_to_map(data), options, optimizer_name);
}

RCPP_MODULE(semx) {
    class_<ModelIR>("ModelIR");

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
        .method("fit", &LikelihoodDriver_fit)
    ;
}


