#include "libsemx/model_objective.hpp"
#include "libsemx/parameter_transform.hpp"
#include "libsemx/covariance_structure.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <iostream>
#include <fstream>
#include <unordered_set>

namespace libsemx {

namespace {
constexpr double kDefaultCoefficientInit = 0.0;
constexpr double kDefaultVarianceInit = 0.5;
}

ModelObjective::ModelObjective(const LikelihoodDriver& driver,
                               const ModelIR& model,
                               const std::unordered_map<std::string, std::vector<double>>& data,
                               const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data,
                               const std::unordered_map<std::string, std::vector<double>>& status,
                               EstimationMethod method,
                               const std::unordered_map<std::string, std::vector<std::string>>& extra_param_mappings)
    : driver_(driver), model_(model), data_(data), fixed_covariance_data_(fixed_covariance_data), status_(status), method_(method), extra_param_mappings_(extra_param_mappings) {
    
    // Pre-calculate stats for observed variables
    std::unordered_map<std::string, double> var_means;
    std::unordered_map<std::string, double> var_variances;
    
    for (const auto& [name, vec] : data_) {
        if (vec.empty()) continue;
        double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
        double mean = sum / vec.size();
        var_means[name] = mean;
        
        double sq_sum = 0.0;
        for(double v : vec) sq_sum += (v - mean) * (v - mean);
        var_variances[name] = sq_sum / vec.size(); // ML estimate
    }

    // Map parameter IDs to edges for heuristic initialization
    std::unordered_map<std::string, const EdgeSpec*> param_to_edge;
    for (const auto& edge : model_.edges) {
        if (!edge.parameter_id.empty()) {
            param_to_edge[edge.parameter_id] = &edge;
        }
    }

    if (!model_.parameters.empty()) {
        for (const auto& param : model_.parameters) {
            std::shared_ptr<const ParameterTransform> transform;
            switch (param.constraint) {
                case ParameterConstraint::Positive:
                    transform = make_log_transform();
                    break;
                case ParameterConstraint::Free:
                default:
                    transform = make_identity_transform();
                    break;
            }
            double init_val = param.initial_value;
            
            // Apply heuristic if parameter is linked to an edge
            if (param_to_edge.count(param.id)) {
                const auto* edge = param_to_edge.at(param.id);
                if (edge->kind == EdgeKind::Covariance && edge->source == edge->target) {
                    // Check family to decide heuristic
                    std::string fam = "";
                    for (const auto& v : model_.variables) {
                        if (v.name == edge->source) {
                            fam = v.family;
                            break;
                        }
                    }

                    if (fam == "gaussian" || fam == "") {
                        if (var_variances.count(edge->source)) {
                            double heuristic = var_variances.at(edge->source) * 0.5;
                            if (heuristic > 1e-4) {
                                init_val = heuristic;
                                // std::cout << "Override variance for " << param.id << " = " << init_val << std::endl;
                            }
                        }
                    } else if (fam == "weibull" || fam == "weibull_aft" || fam == "gamma" || fam == "exponential") {
                        init_val = 1.0;
                        // std::cout << "Override shape/dispersion for " << param.id << " = " << init_val << std::endl;
                    } else {
                        // Default for others (nbinom, etc)
                        init_val = 1.0;
                    }
                } else if (edge->kind == EdgeKind::Regression && (edge->source == "_intercept" || edge->source == "1")) {
                    if (var_means.count(edge->target)) {
                        init_val = var_means.at(edge->target);
                    }
                } else if (edge->kind == EdgeKind::Loading) {
                    if (std::abs(init_val) < 1e-6) { // Only override if 0
                        init_val = 0.8;
                    }
                }
            }

            if (status_.count(param.id) && !status_.at(param.id).empty()) {
                init_val = status_.at(param.id)[0];
            }
            catalog_.register_parameter(param.id, init_val, std::move(transform));
        }
    } else {
        // std::cout << "Inferring parameters from edges" << std::endl;
        for (const auto& edge : model_.edges) {
            if (edge.parameter_id.empty()) {
                continue;
            }
            char* end;
            std::strtod(edge.parameter_id.c_str(), &end);
            if (end != edge.parameter_id.c_str() && *end == '\0') {
                continue;  // numeric literal
            }
            if (catalog_.find_index(edge.parameter_id) != ParameterCatalog::npos) {
                continue;
            }
            if (edge.kind == EdgeKind::Covariance && edge.source == edge.target) {
                double init_val = kDefaultVarianceInit;
                if (var_variances.count(edge.source)) {
                    init_val = var_variances.at(edge.source) * 0.5;
                    if (init_val < 1e-4) init_val = kDefaultVarianceInit;
                    // std::cout << "Init variance for " << edge.parameter_id << " (" << edge.source << ") = " << init_val << std::endl;
                } else {
                    // std::cout << "No variance stats for " << edge.source << ", using default " << kDefaultVarianceInit << std::endl;
                }

                if (status_.count(edge.parameter_id) && !status_.at(edge.parameter_id).empty()) {
                    init_val = status_.at(edge.parameter_id)[0];
                }
                catalog_.register_parameter(edge.parameter_id, init_val, make_log_transform());
            } else {
                double init_val = kDefaultCoefficientInit;
                
                if (edge.kind == EdgeKind::Regression && (edge.source == "_intercept" || edge.source == "1")) {
                    if (var_means.count(edge.target)) {
                        init_val = var_means.at(edge.target);
                        // std::cout << "Init intercept for " << edge.parameter_id << " (" << edge.target << ") = " << init_val << std::endl;
                    }
                } else if (edge.kind == EdgeKind::Loading) {
                    init_val = 0.8;
                    // std::cout << "Init loading for " << edge.parameter_id << " = " << init_val << std::endl;
                }

                if (status_.count(edge.parameter_id) && !status_.at(edge.parameter_id).empty()) {
                    init_val = status_.at(edge.parameter_id)[0];
                }
                catalog_.register_parameter(edge.parameter_id, init_val, make_identity_transform());
            }
        }
    }

    // Add covariance parameters
    for (const auto& cov : model_.covariances) {
        auto structure = create_covariance_structure(cov, fixed_covariance_data_);
        size_t count = structure->parameter_count();
        if (count == 0) {
            continue;
        }
        size_t start_idx = catalog_.size();
        auto mask = build_covariance_positive_mask(cov, *structure);
        if (mask.size() < count) {
            mask.resize(count, false);
        }
        for (size_t i = 0; i < count; ++i) {
            std::string param_name = cov.id + "_" + std::to_string(i);
            bool positive = mask[i];
            auto transform = positive ? make_log_transform() : make_identity_transform();
            double init_val = positive ? kDefaultVarianceInit : kDefaultCoefficientInit;
            
            if (status_.count(param_name) && !status_.at(param_name).empty()) {
                init_val = status_.at(param_name)[0];
            }
            
            catalog_.register_parameter(param_name, init_val, transform);
        }
        covariance_param_ranges_[cov.id] = {start_idx, count};
    }

    prepare_sem_structures();
}

double ModelObjective::value(const std::vector<double>& parameters) const {
    const auto constrained = to_constrained(parameters);
    
    if (sem_mode_) {
        update_sem_data(constrained);
        std::unordered_map<std::string, std::vector<double>> linear_predictors;
        std::unordered_map<std::string, std::vector<double>> dispersions;
        std::unordered_map<std::string, std::vector<double>> covariance_parameters;
        
        // 1. Linear Predictors (Fixed Effects)
        auto& lp = linear_predictors["_stacked_y"];
        lp.assign(sem_data_.at("_stacked_y").size(), 0.0);
        
        size_t n_obs = data_.at(sem_outcomes_[0]).size();
        size_t n_outcomes = sem_outcomes_.size();
        size_t total_rows = n_obs * n_outcomes;

        for (const auto& edge : model_.edges) {
            if (edge.kind == EdgeKind::Regression) {
                auto it = std::find(sem_outcomes_.begin(), sem_outcomes_.end(), edge.target);
                if (it != sem_outcomes_.end()) {
                    size_t outcome_idx = std::distance(sem_outcomes_.begin(), it);
                    if (data_.count(edge.source)) {
                        double weight = 0.0;
                        if (!edge.parameter_id.empty()) {
                            size_t idx = catalog_.find_index(edge.parameter_id);
                            if (idx != ParameterCatalog::npos) {
                                weight = constrained[idx];
                            } else {
                                try { weight = std::stod(edge.parameter_id); } catch(...) {}
                            }
                        } else {
                            try { weight = std::stod(edge.parameter_id); } catch(...) {}
                        }
                        
                        const auto& src_vec = data_.at(edge.source);
                        for (size_t i = 0; i < n_obs; ++i) {
                            lp[i * n_outcomes + outcome_idx] += src_vec[i] * weight;
                        }
                    }
                }
            }
        }
        
        // 2. Dispersions
        std::vector<double> disp(total_rows, 1.0);
        for (const auto& info : residual_infos_) {
            double val = info.fixed_value;
            if (info.param_index != ParameterCatalog::npos) {
                val = constrained[info.param_index];
            }
            for (size_t i = 0; i < n_obs; ++i) {
                disp[i * n_outcomes + info.outcome_index] = val;
            }
        }
        dispersions["_stacked_y"] = std::move(disp);
        
        // 3. Covariance Parameters
        // Standard ranges (if any)
        for (const auto& [id, range] : covariance_param_ranges_) {
            std::vector<double> params;
            params.reserve(range.second);
            for (size_t i = 0; i < range.second; ++i) {
                params.push_back(constrained[range.first + i]);
            }
            covariance_parameters[id] = std::move(params);
        }
        // SEM mappings
        for (const auto& mapping : sem_covariance_mappings_) {
            std::vector<double> params;
            params.reserve(mapping.elements.size());
            for (const auto& elem : mapping.elements) {
                if (elem.param_index != ParameterCatalog::npos) {
                    params.push_back(constrained[elem.param_index]);
                } else {
                    params.push_back(elem.fixed_value);
                }
            }
            covariance_parameters[mapping.id] = std::move(params);
        }
        
        // Prepare status and extra_params for _stacked_y
        auto local_status = status_;
        if (sem_data_.count("_stacked_item_idx")) {
            local_status["_stacked_y"] = sem_data_.at("_stacked_item_idx");
        }

        auto local_extra_params = build_extra_params(constrained);
        if (extra_param_mappings_.count("_stacked_y")) {
            const auto& param_ids = extra_param_mappings_.at("_stacked_y");
            std::vector<double> stacked_ep;
            stacked_ep.reserve(param_ids.size());
            for (const auto& pid : param_ids) {
                size_t idx = catalog_.find_index(pid);
                if (idx != ParameterCatalog::npos) {
                    stacked_ep.push_back(constrained[idx]);
                } else {
                    try { stacked_ep.push_back(std::stod(pid)); } catch(...) { stacked_ep.push_back(0.0); }
                }
            }
            local_extra_params["_stacked_y"] = std::move(stacked_ep);
        }

        try {
            double ll = -driver_.evaluate_model_loglik(sem_model_, sem_data_, linear_predictors, dispersions, covariance_parameters, local_status, local_extra_params, fixed_covariance_data_, method_);
            return ll;
        } catch (const std::runtime_error& e) {
            // std::cerr << "ModelObjective::value (SEM) exception: " << e.what() << std::endl;
            return std::numeric_limits<double>::infinity();
        }
    }

    std::unordered_map<std::string, std::vector<double>> linear_predictors;
    std::unordered_map<std::string, std::vector<double>> dispersions;
    std::unordered_map<std::string, std::vector<double>> covariance_parameters;

    build_prediction_workspaces(constrained, linear_predictors, dispersions, covariance_parameters);

    try {
        double ll = -driver_.evaluate_model_loglik(model_, data_, linear_predictors, dispersions, covariance_parameters, status_, build_extra_params(constrained), fixed_covariance_data_, method_);
        return ll;
    } catch (const std::runtime_error& e) {
        std::cerr << "ModelObjective::value (Non-SEM) exception: " << e.what() << std::endl;
        return std::numeric_limits<double>::infinity();
    }
}

std::vector<double> ModelObjective::gradient(const std::vector<double>& parameters) const {
    std::vector<double> grad;
    value_and_gradient(parameters, grad);
    return grad;
}

double ModelObjective::value_and_gradient(const std::vector<double>& parameters, std::vector<double>& grad) const {
    const auto constrained = to_constrained(parameters);
    
    if (sem_mode_) {
        update_sem_data(constrained);
        std::unordered_map<std::string, std::vector<double>> linear_predictors;
        std::unordered_map<std::string, std::vector<double>> dispersions;
        std::unordered_map<std::string, std::vector<double>> covariance_parameters;
        
        // 1. Linear Predictors (Fixed Effects)
        auto& lp = linear_predictors["_stacked_y"];
        lp.assign(sem_data_.at("_stacked_y").size(), 0.0);
        
        size_t n_obs = data_.at(sem_outcomes_[0]).size();
        size_t n_outcomes = sem_outcomes_.size();
        size_t total_rows = n_obs * n_outcomes;

        for (const auto& edge : model_.edges) {
            if (edge.kind == EdgeKind::Regression) {
                auto it = std::find(sem_outcomes_.begin(), sem_outcomes_.end(), edge.target);
                if (it != sem_outcomes_.end()) {
                    size_t outcome_idx = std::distance(sem_outcomes_.begin(), it);
                    if (data_.count(edge.source)) {
                        double weight = 0.0;
                        if (!edge.parameter_id.empty()) {
                            size_t idx = catalog_.find_index(edge.parameter_id);
                            if (idx != ParameterCatalog::npos) {
                                weight = constrained[idx];
                            } else {
                                try { weight = std::stod(edge.parameter_id); } catch(...) {}
                            }
                        } else {
                            try { weight = std::stod(edge.parameter_id); } catch(...) {}
                        }
                        
                        const auto& src_vec = data_.at(edge.source);
                        for (size_t i = 0; i < n_obs; ++i) {
                            lp[i * n_outcomes + outcome_idx] += src_vec[i] * weight;
                        }
                    }
                }
            }
        }

        // Handle SEM-specific regression edges (e.g. latent means)
        for (const auto& edge : sem_model_.edges) {
            if (edge.kind == EdgeKind::Regression && edge.target == "_stacked_y") {
                double weight = 0.0;
                if (!edge.parameter_id.empty()) {
                    size_t idx = catalog_.find_index(edge.parameter_id);
                    if (idx != ParameterCatalog::npos) {
                        weight = constrained[idx];
                    } else {
                        try { weight = std::stod(edge.parameter_id); } catch(...) {}
                    }
                } else {
                    try { weight = std::stod(edge.parameter_id); } catch(...) {}
                }

                if (sem_data_.count(edge.source)) {
                    const auto& src_vec = sem_data_.at(edge.source);
                    if (src_vec.size() == lp.size()) {
                        for(size_t i=0; i<src_vec.size(); ++i) {
                            lp[i] += src_vec[i] * weight;
                        }
                    }
                }
            }
        }
        
        // 2. Dispersions
        std::vector<double> disp(total_rows, 1.0);
        for (const auto& info : residual_infos_) {
            double val = info.fixed_value;
            if (info.param_index != ParameterCatalog::npos) {
                val = constrained[info.param_index];
            }
            for (size_t i = 0; i < n_obs; ++i) {
                disp[i * n_outcomes + info.outcome_index] = val;
            }
        }
        dispersions["_stacked_y"] = std::move(disp);
        
        // 3. Covariance Parameters
        for (const auto& [id, range] : covariance_param_ranges_) {
            std::vector<double> params;
            params.reserve(range.second);
            for (size_t i = 0; i < range.second; ++i) {
                params.push_back(constrained[range.first + i]);
            }
            covariance_parameters[id] = std::move(params);
        }
        for (const auto& mapping : sem_covariance_mappings_) {
            std::vector<double> params;
            params.reserve(mapping.elements.size());
            for (const auto& elem : mapping.elements) {
                if (elem.param_index != ParameterCatalog::npos) {
                    params.push_back(constrained[elem.param_index]);
                } else {
                    params.push_back(elem.fixed_value);
                }
            }
            covariance_parameters[mapping.id] = std::move(params);
        }
        
        // Construct data_param_mappings for loadings
        std::unordered_map<std::string, LikelihoodDriver::DataParamMapping> data_param_mappings;
        
        // Initialize mappings for all loading variables
        for(const auto& latent : sem_latents_) {
            std::string loading_var = "_loading_" + latent;
            LikelihoodDriver::DataParamMapping mapping;
            mapping.stride = n_outcomes;
            mapping.pattern.resize(n_outcomes);
            data_param_mappings[loading_var] = mapping;
        }
        
        // Fill patterns
        for (const auto& info : loading_infos_) {
            if (info.param_index != ParameterCatalog::npos) {
                std::string loading_var = "_loading_" + sem_latents_[info.latent_index];
                std::string param_id = catalog_.names()[info.param_index];
                data_param_mappings[loading_var].pattern[info.outcome_index] = param_id;
            }
        }
        
        // Build dispersion mappings for analytic gradients
        std::unordered_map<std::string, LikelihoodDriver::DataParamMapping> dispersion_param_mappings;
        
        std::vector<std::string> pattern(sem_outcomes_.size());
        for (const auto& info : residual_infos_) {
            if (info.param_index != ParameterCatalog::npos) {
                pattern[info.outcome_index] = catalog_.names()[info.param_index];
            }
        }
        dispersion_param_mappings["_stacked_y"] = LikelihoodDriver::DataParamMapping{pattern, sem_outcomes_.size()};

        std::pair<double, std::unordered_map<std::string, double>> value_and_grad;
        try {
            auto mappings = build_extra_param_mappings();
            for(const auto& [k, v] : extra_param_mappings_) {
                mappings[k] = v;
            }

            std::unordered_map<std::string, std::vector<double>> sem_status = status_;
            if (sem_mode_ && sem_data_.count("_stacked_item_idx")) {
                sem_status["_stacked_y"] = sem_data_.at("_stacked_item_idx");
            }

            auto local_extra_params = build_extra_params(constrained);
            if (extra_param_mappings_.count("_stacked_y")) {
                const auto& param_ids = extra_param_mappings_.at("_stacked_y");
                std::vector<double> stacked_ep;
                stacked_ep.reserve(param_ids.size());
                for (const auto& pid : param_ids) {
                    size_t idx = catalog_.find_index(pid);
                    if (idx != ParameterCatalog::npos) {
                        stacked_ep.push_back(constrained[idx]);
                    } else {
                        try { stacked_ep.push_back(std::stod(pid)); } catch(...) { stacked_ep.push_back(0.0); }
                    }
                }
                local_extra_params["_stacked_y"] = std::move(stacked_ep);
            }

            value_and_grad = driver_.evaluate_model_loglik_and_gradient(
                sem_model_,
                sem_data_,
                linear_predictors,
                dispersions,
                covariance_parameters,
                sem_status,
                local_extra_params,
                fixed_covariance_data_,
                method_,
                data_param_mappings,
                dispersion_param_mappings,
                mappings
            );
        } catch (const std::exception& e) {
            // std::cerr << "ModelObjective::value_and_gradient (SEM) exception: " << e.what() << std::endl;
            grad.assign(parameters.size(), 0.0);
            return std::numeric_limits<double>::infinity();
        }

        grad.assign(parameters.size(), 0.0);
        auto chain = catalog_.constrained_derivatives(parameters);
        
        // Map gradients back to parameters
        for (const auto& [param_id, g] : value_and_grad.second) {
            // Check if it's a covariance parameter gradient (e.g. _cov_latents_0)
            bool handled = false;
            for (const auto& mapping : sem_covariance_mappings_) {
                if (param_id.rfind(mapping.id + "_", 0) == 0) {
                    // Extract index
                    try {
                        size_t idx = std::stoul(param_id.substr(mapping.id.length() + 1));
                        if (idx < mapping.elements.size()) {
                            const auto& elem = mapping.elements[idx];
                            if (elem.param_index != ParameterCatalog::npos) {
                                grad[elem.param_index] -= g * chain[elem.param_index];
                            }
                            handled = true;
                        }
                    } catch(...) {}
                }
            }
            
            if (!handled) {
                auto idx = catalog_.find_index(param_id);
                if (idx != ParameterCatalog::npos) {
                    grad[idx] -= g * chain[idx];
                }
            }
        }
        
        return -value_and_grad.first;
    }


    std::unordered_map<std::string, std::vector<double>> linear_predictors;
    std::unordered_map<std::string, std::vector<double>> dispersions;
    std::unordered_map<std::string, std::vector<double>> covariance_parameters;

    build_prediction_workspaces(constrained, linear_predictors, dispersions, covariance_parameters);

    std::unordered_map<std::string, LikelihoodDriver::DataParamMapping> dispersion_param_mappings;
    std::unordered_map<std::string, LikelihoodDriver::DataParamMapping> data_param_mappings;

    // Build data parameter mappings (for factor loadings in Z)
    for (const auto& re : model_.random_effects) {
        for (const auto& var_name : re.variables) {
            if (catalog_.find_index(var_name) != ParameterCatalog::npos) {
                LikelihoodDriver::DataParamMapping mapping;
                mapping.stride = 1;
                mapping.pattern = {var_name};
                data_param_mappings[var_name] = mapping;
            }
        }
    }
    
    if (!sem_mode_) {
        for (const auto& var : model_.variables) {
            if (var.kind == VariableKind::Observed) {
                std::string param_id;
                for (const auto& edge : model_.edges) {
                    if (edge.kind == EdgeKind::Covariance && edge.source == var.name && edge.target == var.name) {
                        param_id = edge.parameter_id;
                        break;
                    }
                }
                
                if (!param_id.empty()) {
                    if (catalog_.find_index(param_id) != ParameterCatalog::npos) {
                        LikelihoodDriver::DataParamMapping mapping;
                        mapping.stride = 1;
                        mapping.pattern = {param_id};
                        dispersion_param_mappings[var.name] = mapping;
                    }
                }
            }
        }
    } else {
        std::vector<std::string> outcomes;
        for (const auto& var : model_.variables) {
            if (var.kind == VariableKind::Observed) {
                outcomes.push_back(var.name);
            }
        }
        for (const auto& info : residual_infos_) {
            if (info.param_index != ParameterCatalog::npos && info.outcome_index < outcomes.size()) {
                std::string var_name = outcomes[info.outcome_index];
                std::string param_id = catalog_.names()[info.param_index];
                LikelihoodDriver::DataParamMapping mapping;
                mapping.stride = 1;
                mapping.pattern = {param_id};
                dispersion_param_mappings[var_name] = mapping;
            }
        }
    }

    std::pair<double, std::unordered_map<std::string, double>> value_and_grad;
    static int iteration = 0;
    iteration++;
    try {
        auto mappings = build_extra_param_mappings();
        for(const auto& [k, v] : extra_param_mappings_) {
            mappings[k] = v;
        }

        value_and_grad = driver_.evaluate_model_loglik_and_gradient(
            model_,
            data_,
            linear_predictors,
            dispersions,
            covariance_parameters,
            status_,
            build_extra_params(constrained),
            fixed_covariance_data_,
            method_,
            data_param_mappings,
            dispersion_param_mappings,
            mappings);

        if (iteration % 10 == 0 || iteration == 1) {
            std::cout << "Iter (V&G) " << iteration << " NLL=" << -value_and_grad.first << " Params: ";
            for (size_t i = 0; i < parameters.size(); ++i) {
                 std::cout << catalog_.names()[i] << "=" << parameters[i] << " ";
            }
            std::cout << std::endl;
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "ModelObjective::value_and_gradient (Non-SEM) exception: " << e.what() << std::endl;
        grad.assign(parameters.size(), 0.0);
        return std::numeric_limits<double>::infinity();
    }
        
    grad.assign(parameters.size(), 0.0);
    auto chain = catalog_.constrained_derivatives(parameters);
    for (const auto& [param_id, g] : value_and_grad.second) {
        auto idx = catalog_.find_index(param_id);
        if (idx != ParameterCatalog::npos) {
            grad[idx] -= g * chain[idx];
        }
    }
    
    return -value_and_grad.first;
}

void ModelObjective::prepare_sem_structures() {
    // 1. Identify Latents and Outcomes
    for (const auto& var : model_.variables) {
        if (var.kind == VariableKind::Latent) {
            bool is_re = false;
            for(const auto& re : model_.random_effects) {
                if (re.id == var.name) {
                    is_re = true;
                    break;
                }
            }
            if (!is_re) {
                sem_latents_.push_back(var.name);
            }
        } else if (var.kind == VariableKind::Observed) {
            bool is_target = false;
            for (const auto& edge : model_.edges) {
                if (edge.target == var.name) {
                    is_target = true;
                    break;
                }
            }
            if (is_target) {
                sem_outcomes_.push_back(var.name);
            }
        }
    }

    if (sem_latents_.empty()) {
        sem_mode_ = false;
        return;
    }
    sem_mode_ = true;

    // 2. Stack Data
    size_t n_obs = 0;
    if (!sem_outcomes_.empty() && data_.count(sem_outcomes_[0])) {
        n_obs = data_.at(sem_outcomes_[0]).size();
    }
    size_t n_outcomes = sem_outcomes_.size();
    size_t total_rows = n_obs * n_outcomes;

    std::vector<double> stacked_y(total_rows);
    std::vector<double> stacked_obs_idx(total_rows);
    std::vector<double> stacked_item_idx(total_rows);

    std::string stacked_family = "gaussian";
    std::vector<std::string> stacked_extra_params;
    bool mixed_families = false;
    std::string first_family;
    std::stringstream mixed_config;
    mixed_config << "mixed";

    for (size_t k = 0; k < n_outcomes; ++k) {
        const std::string& outcome = sem_outcomes_[k];
        auto it = std::find_if(model_.variables.begin(), model_.variables.end(), [&](const auto& v){ return v.name == outcome; });
        std::string family = (it != model_.variables.end()) ? it->family : "gaussian";
        
        if (k == 0) first_family = family;
        else if (family != first_family) mixed_families = true;

        size_t param_count = 0;
        if (extra_param_mappings_.count(outcome)) {
            const auto& params = extra_param_mappings_.at(outcome);
            param_count = params.size();
            stacked_extra_params.insert(stacked_extra_params.end(), params.begin(), params.end());
        }
        mixed_config << ";" << family << "," << param_count;
    }

    if (n_outcomes > 1 && (mixed_families || !stacked_extra_params.empty())) {
         stacked_family = mixed_config.str();
         extra_param_mappings_["_stacked_y"] = stacked_extra_params;
    } else if (n_outcomes == 1) {
         stacked_family = first_family;
         if (!stacked_extra_params.empty()) {
             extra_param_mappings_["_stacked_y"] = stacked_extra_params;
         }
    }
    
    for (size_t k = 0; k < n_outcomes; ++k) {
        const auto& y_vec = data_.at(sem_outcomes_[k]);
        for (size_t i = 0; i < n_obs; ++i) {
            size_t row = i * n_outcomes + k;
            stacked_y[row] = y_vec[i];
            stacked_obs_idx[row] = static_cast<double>(i);
            stacked_item_idx[row] = static_cast<double>(k);
        }
    }
    
    sem_data_["_stacked_y"] = std::move(stacked_y);
    sem_data_["_stacked_obs_idx"] = std::move(stacked_obs_idx);
    sem_data_["_stacked_item_idx"] = std::move(stacked_item_idx);

    // 3. Build SEM Model IR
    sem_model_ = model_; 
    sem_model_.variables.clear();
    sem_model_.edges.clear(); 
    sem_model_.random_effects.clear(); 

    sem_model_.variables.push_back({ "_stacked_y", VariableKind::Observed, stacked_family });
    sem_model_.variables.push_back({ "_stacked_obs_idx", VariableKind::Grouping, "" });

    // 4. Handle Latents (Random Effects) - Grouped
    if (!sem_latents_.empty()) {
        std::vector<std::string> re_vars;
        re_vars.push_back("_stacked_obs_idx");
        
        for (size_t l = 0; l < sem_latents_.size(); ++l) {
            const std::string& latent = sem_latents_[l];
            std::string loading_var = "_loading_" + latent;
            sem_data_[loading_var] = std::vector<double>(total_rows, 0.0);
            re_vars.push_back(loading_var);

            // Handle Latent Means (Intercepts)
            for (const auto& edge : model_.edges) {
                if (edge.kind == EdgeKind::Regression && edge.target == latent && 
                   (edge.source == "_intercept" || edge.source == "1")) {
                    
                    sem_model_.edges.push_back({
                        EdgeKind::Regression,
                        loading_var,
                        "_stacked_y",
                        edge.parameter_id
                    });
                    
                    bool exists = false;
                    for(const auto& v : sem_model_.variables) {
                        if (v.name == loading_var) { exists = true; break; }
                    }
                    if (!exists) {
                        sem_model_.variables.push_back({ loading_var, VariableKind::Observed, "" });
                    }
                    break;
                }
            }
            
            // Collect loading infos
            for (size_t k = 0; k < n_outcomes; ++k) {
                std::string outcome = sem_outcomes_[k];
                for(const auto& edge : model_.edges) {
                    if ((edge.kind == EdgeKind::Loading || edge.kind == EdgeKind::Regression) &&
                        edge.source == latent && edge.target == outcome) {
                        
                        LoadingInfo info;
                        info.latent_index = l;
                        info.outcome_index = k;
                        
                        if (!edge.parameter_id.empty()) {
                            size_t idx = catalog_.find_index(edge.parameter_id);
                            if (idx != ParameterCatalog::npos) {
                                info.param_index = idx;
                            } else {
                                try {
                                    info.fixed_value = std::stod(edge.parameter_id);
                                    info.param_index = ParameterCatalog::npos;
                                } catch(...) {}
                            }
                        } else {
                             info.param_index = ParameterCatalog::npos;
                             info.fixed_value = 0.0;
                        }

                        if (info.param_index == ParameterCatalog::npos) {
                             for(size_t i=0; i < n_obs; ++i) {
                                 sem_data_[loading_var][i * n_outcomes + k] = info.fixed_value;
                             }
                        }

                        loading_infos_.push_back(info);
                    }
                }
            }
        }

        std::string cov_id = "_cov_latents";
        CovarianceSpec cov_spec;
        cov_spec.id = cov_id;
        cov_spec.structure = "unstructured";
        cov_spec.dimension = sem_latents_.size();
        sem_model_.covariances.push_back(cov_spec);

        RandomEffectSpec re;
        re.id = "_re_latents";
        re.variables = re_vars;
        re.covariance_id = cov_id;
        sem_model_.random_effects.push_back(re);

        // Add regression edge from random effect to stacked outcome
        sem_model_.edges.push_back({
            EdgeKind::Regression,
            "_re_latents",
            "_stacked_y",
            ""
        });

        // Build Covariance Mapping
        SemCovarianceMapping mapping;
        mapping.id = cov_id;
        size_t dim = sem_latents_.size();
        
        for (size_t i = 0; i < dim; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                std::string p_id;
                for (const auto& edge : model_.edges) {
                    if (edge.kind == EdgeKind::Covariance) {
                        if ((edge.source == sem_latents_[i] && edge.target == sem_latents_[j]) ||
                            (edge.source == sem_latents_[j] && edge.target == sem_latents_[i])) {
                            p_id = edge.parameter_id;
                            break;
                        }
                    }
                }

                SemCovarianceMapping::Element elem;
                if (!p_id.empty()) {
                    size_t idx = catalog_.find_index(p_id);
                    if (idx != ParameterCatalog::npos) {
                        elem.param_index = idx;
                    } else {
                        try {
                            elem.fixed_value = std::stod(p_id);
                            elem.param_index = ParameterCatalog::npos;
                        } catch(...) {
                            elem.fixed_value = 0.0;
                            elem.param_index = ParameterCatalog::npos;
                        }
                    }
                } else {
                    elem.fixed_value = (i == j) ? 1.0 : 0.0;
                    elem.param_index = ParameterCatalog::npos;
                }
                mapping.elements.push_back(elem);
            }
        }
        sem_covariance_mappings_.push_back(mapping);
    }
    
    // 5. Handle Residuals
    for (size_t k = 0; k < n_outcomes; ++k) {
        std::string outcome = sem_outcomes_[k];
        std::string param_id;
        for(const auto& edge : model_.edges) {
            if (edge.kind == EdgeKind::Covariance && edge.source == outcome && edge.target == outcome) {
                param_id = edge.parameter_id;
                break;
            }
        }
        
        ResidualInfo info;
        info.outcome_index = k;
        if (!param_id.empty()) {
            size_t idx = catalog_.find_index(param_id);
            if (idx != ParameterCatalog::npos) {
                info.param_index = idx;
            } else {
                try {
                    info.fixed_value = std::stod(param_id);
                    info.param_index = ParameterCatalog::npos;
                } catch(...) {}
            }
        } else {
            info.param_index = ParameterCatalog::npos;
            info.fixed_value = 1.0;
        }
        residual_infos_.push_back(info);
    }

    // 6. Handle Fixed Effects (Regressions)
    for (const auto& edge : model_.edges) {
        if (edge.kind == EdgeKind::Regression) {
            auto it = std::find(sem_outcomes_.begin(), sem_outcomes_.end(), edge.target);
            if (it != sem_outcomes_.end()) {
                size_t outcome_idx = std::distance(sem_outcomes_.begin(), it);
                if (data_.count(edge.source)) {
                    std::string fe_var_name = "_fe_" + edge.source + "_on_" + edge.target;
                    std::vector<double> fe_data(total_rows, 0.0);
                    const auto& src_vec = data_.at(edge.source);
                    for (size_t i = 0; i < n_obs; ++i) {
                        fe_data[i * n_outcomes + outcome_idx] = src_vec[i];
                    }
                    sem_data_[fe_var_name] = std::move(fe_data);
                    sem_model_.variables.push_back({ fe_var_name, VariableKind::Observed, "" });
                    sem_model_.edges.push_back({
                        EdgeKind::Regression,
                        fe_var_name,
                        "_stacked_y",
                        edge.parameter_id
                    });
                }
            }
        }
    }
}

void ModelObjective::update_sem_data(const std::vector<double>& constrained_parameters) const {
    size_t n_obs = data_.at(sem_outcomes_[0]).size();
    size_t n_outcomes = sem_outcomes_.size();
    
    // Update Loadings
    for (const auto& info : loading_infos_) {
        double val = info.fixed_value;
        if (info.param_index != ParameterCatalog::npos) {
            val = constrained_parameters[info.param_index];
        }
        
        std::string loading_var = "_loading_" + sem_latents_[info.latent_index];
        auto& vec = sem_data_.at(loading_var);
        
        for (size_t i = 0; i < n_obs; ++i) {
            vec[i * n_outcomes + info.outcome_index] = val;
        }
    }
}

std::vector<double> ModelObjective::to_constrained(const std::vector<double>& unconstrained) const {
    return catalog_.constrain(unconstrained);
}

std::vector<double> ModelObjective::constrained_derivatives(const std::vector<double>& unconstrained) const {
    return catalog_.constrained_derivatives(unconstrained);
}

std::unordered_map<std::string, std::vector<double>> ModelObjective::get_covariance_matrices(const std::vector<double>& constrained_parameters) const {
    std::unordered_map<std::string, std::vector<double>> matrices;
    for (const auto& cov : model_.covariances) {
        auto structure = create_covariance_structure(cov, fixed_covariance_data_);
        auto it = covariance_param_ranges_.find(cov.id);
        if (it != covariance_param_ranges_.end()) {
            std::vector<double> params;
            params.reserve(it->second.second);
            for (size_t i = 0; i < it->second.second; ++i) {
                params.push_back(constrained_parameters[it->second.first + i]);
            }
            matrices[cov.id] = structure->materialize(params);
        }
    }
    return matrices;
}

void ModelObjective::build_prediction_workspaces(const std::vector<double>& constrained_parameters,
                                                 std::unordered_map<std::string, std::vector<double>>& linear_predictors,
                                                 std::unordered_map<std::string, std::vector<double>>& dispersions,
                                                 std::unordered_map<std::string, std::vector<double>>& covariance_parameters) const {
    linear_predictors.clear();
    dispersions.clear();
    covariance_parameters.clear();

    std::unordered_set<std::string> response_vars;
    for (const auto& edge : model_.edges) {
        if (edge.kind == EdgeKind::Regression) {
            response_vars.insert(edge.target);
        }
    }

    auto register_response = [&](const std::string& name) {
        if (!data_.contains(name)) return;
        linear_predictors[name] = std::vector<double>(data_.at(name).size(), 0.0);
        dispersions[name] = std::vector<double>(data_.at(name).size(), 1.0);
    };

    for (const auto& var : model_.variables) {
        if (var.kind == VariableKind::Observed) {
            register_response(var.name);
        }
    }

    for (const auto& edge : model_.edges) {
        if (edge.kind != EdgeKind::Regression) continue;

        double weight = 0.0;
        if (!edge.parameter_id.empty()) {
            auto idx = catalog_.find_index(edge.parameter_id);
            if (idx != ParameterCatalog::npos) {
                weight = constrained_parameters[idx];
            } else {
                try {
                    weight = std::stod(edge.parameter_id);
                } catch (...) {
                    weight = 0.0;
                }
            }
        }

        if (data_.count(edge.source) && linear_predictors.count(edge.target)) {
            const auto& src_data = data_.at(edge.source);
            auto& tgt_lp = linear_predictors.at(edge.target);
            for (size_t i = 0; i < tgt_lp.size(); ++i) {
                tgt_lp[i] += src_data[i] * weight;
            }
        }
    }

    for (const auto& edge : model_.edges) {
        if (edge.kind == EdgeKind::Covariance && edge.source == edge.target) {
            if (dispersions.count(edge.source) && !edge.parameter_id.empty()) {
                auto idx = catalog_.find_index(edge.parameter_id);
                if (idx != ParameterCatalog::npos) {
                    double val = constrained_parameters[idx];
                    std::fill(dispersions[edge.source].begin(), dispersions[edge.source].end(), val);
                } else {
                    try {
                        double val = std::stod(edge.parameter_id);
                        std::fill(dispersions[edge.source].begin(), dispersions[edge.source].end(), val);
                    } catch (...) {}
                }
            }
        }
    }

    for (const auto& [id, range] : covariance_param_ranges_) {
        std::vector<double> params;
        params.reserve(range.second);
        for (size_t i = 0; i < range.second; ++i) {
            params.push_back(constrained_parameters[range.first + i]);
        }
        covariance_parameters[id] = std::move(params);
    }
}

const std::vector<std::string>& ModelObjective::parameter_names() const { return catalog_.names(); }

std::vector<double> ModelObjective::initial_parameters() const {
    return catalog_.initial_unconstrained();
}

std::vector<double> ModelObjective::convert_to_model_parameters(const std::vector<double>& optimizer_parameters) const {
    std::vector<double> model_params = optimizer_parameters;

    for (const auto& mapping : sem_covariance_mappings_) {
        // Find the covariance spec
        const CovarianceSpec* spec = nullptr;
        for (const auto& s : sem_model_.covariances) {
            if (s.id == mapping.id) {
                spec = &s;
                break;
            }
        }
        
        if (spec && spec->structure == "unstructured") {
            size_t dim = spec->dimension;
            Eigen::MatrixXd L = Eigen::MatrixXd::Zero(dim, dim);
            
            // Reconstruct L
            size_t elem_idx = 0;
            for (size_t i = 0; i < dim; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    if (elem_idx < mapping.elements.size()) {
                        const auto& elem = mapping.elements[elem_idx];
                        double val = elem.fixed_value;
                        if (elem.param_index != ParameterCatalog::npos) {
                            val = optimizer_parameters[elem.param_index];
                        }
                        L(i, j) = val;
                    }
                    elem_idx++;
                }
            }
            
            // Compute Psi = L * L^T
            Eigen::MatrixXd Psi = L * L.transpose();
            
            // Update model_params
            elem_idx = 0;
            for (size_t i = 0; i < dim; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    if (elem_idx < mapping.elements.size()) {
                        const auto& elem = mapping.elements[elem_idx];
                        if (elem.param_index != ParameterCatalog::npos) {
                            model_params[elem.param_index] = Psi(i, j);
                        }
                    }
                    elem_idx++;
                }
            }
        }
    }
    
    return model_params;
}

std::unordered_map<std::string, std::vector<double>> ModelObjective::build_extra_params(const std::vector<double>& constrained_parameters) const {
    std::unordered_map<std::string, std::vector<double>> extra_params;
    
    for (const auto& var : model_.variables) {
        if (var.family == "ordinal") {
            std::vector<double> thresholds;
            int k = 1;
            while (true) {
                std::string param_id = var.name + "_threshold_" + std::to_string(k);
                auto idx = catalog_.find_index(param_id);
                if (idx != ParameterCatalog::npos) {
                    thresholds.push_back(constrained_parameters[idx]);
                    k++;
                } else {
                    break;
                }
            }
            if (!thresholds.empty()) {
                extra_params[var.name] = thresholds;
            }
        }
    }
    return extra_params;
}

std::unordered_map<std::string, std::vector<std::string>> ModelObjective::build_extra_param_mappings() const {
    std::unordered_map<std::string, std::vector<std::string>> mappings;
    
    std::ofstream debug_file("data/debug_semx_mappings.log", std::ios::app);
    debug_file << "Building extra param mappings..." << std::endl;
    
    for (const auto& var : model_.variables) {
        debug_file << "Checking variable " << var.name << " family: " << var.family << std::endl;
        
        if (var.family == "ordinal") {
            std::vector<std::string> param_ids;
            int k = 1;
            while (true) {
                std::string param_id = var.name + "_threshold_" + std::to_string(k);
                if (catalog_.find_index(param_id) != ParameterCatalog::npos) {
                    param_ids.push_back(param_id);
                    k++;
                } else {
                    debug_file << "  Did not find " << param_id << std::endl;
                    break;
                }
            }
            if (!param_ids.empty()) {
                mappings[var.name] = param_ids;
                debug_file << "  Added " << param_ids.size() << " thresholds for " << var.name << std::endl;
            }
        }
    }
    return mappings;
}

}  // namespace libsemx
