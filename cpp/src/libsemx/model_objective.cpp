#include "libsemx/model_objective.hpp"
#include "libsemx/parameter_transform.hpp"
#include "libsemx/covariance_structure.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <algorithm>
#include <cstdlib>
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
                               EstimationMethod method)
    : driver_(driver), model_(model), data_(data), fixed_covariance_data_(fixed_covariance_data), status_(status), method_(method) {
    
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
            if (status_.count(param.id) && !status_.at(param.id).empty()) {
                init_val = status_.at(param.id)[0];
            }
            catalog_.register_parameter(param.id, init_val, std::move(transform));
        }
    } else {
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
                if (status_.count(edge.parameter_id) && !status_.at(edge.parameter_id).empty()) {
                    init_val = status_.at(edge.parameter_id)[0];
                }
                catalog_.register_parameter(edge.parameter_id, init_val, make_log_transform());
            } else {
                double init_val = kDefaultCoefficientInit;
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
        
        try {
            return -driver_.evaluate_model_loglik(sem_model_, sem_data_, linear_predictors, dispersions, covariance_parameters, status_, {}, fixed_covariance_data_, method_);
        } catch (const std::runtime_error&) {
            return std::numeric_limits<double>::infinity();
        }
    }

    std::unordered_map<std::string, std::vector<double>> linear_predictors;
    std::unordered_map<std::string, std::vector<double>> dispersions;
    std::unordered_map<std::string, std::vector<double>> covariance_parameters;

    build_prediction_workspaces(constrained, linear_predictors, dispersions, covariance_parameters);

    try {
        return -driver_.evaluate_model_loglik(model_, data_, linear_predictors, dispersions, covariance_parameters, status_, {}, fixed_covariance_data_, method_);
    } catch (const std::runtime_error&) {
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
            value_and_grad = driver_.evaluate_model_loglik_and_gradient(
                sem_model_,
                sem_data_,
                linear_predictors,
                dispersions,
                covariance_parameters,
                status_,
                {},
                fixed_covariance_data_,
                method_,
                data_param_mappings,
                dispersion_param_mappings
            );
        } catch (const std::runtime_error&) {
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
    try {
        value_and_grad = driver_.evaluate_model_loglik_and_gradient(
            model_,
            data_,
            linear_predictors,
            dispersions,
            covariance_parameters,
            status_,
            {},
            fixed_covariance_data_,
            method_,
            {},
            dispersion_param_mappings);
    } catch (const std::runtime_error&) {
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
            sem_latents_.push_back(var.name);
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
    
    for (size_t k = 0; k < n_outcomes; ++k) {
        const auto& y_vec = data_.at(sem_outcomes_[k]);
        for (size_t i = 0; i < n_obs; ++i) {
            size_t row = i * n_outcomes + k;
            stacked_y[row] = y_vec[i];
            stacked_obs_idx[row] = static_cast<double>(i);
        }
    }
    
    sem_data_["_stacked_y"] = std::move(stacked_y);
    sem_data_["_stacked_obs_idx"] = std::move(stacked_obs_idx);

    // 3. Build SEM Model IR
    sem_model_ = model_; 
    sem_model_.variables.clear();
    sem_model_.edges.clear(); 
    sem_model_.random_effects.clear(); 

    sem_model_.variables.push_back({ "_stacked_y", VariableKind::Observed, "gaussian" });
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
                    elem.fixed_value = 0.0;
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

}  // namespace libsemx
