#include "libsemx/likelihood_driver.hpp"
#include "libsemx/outcome_family_factory.hpp"
#include "libsemx/covariance_structure.hpp"

#include "libsemx/scaled_fixed_covariance.hpp"
#include "libsemx/multi_kernel_covariance.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace libsemx {

namespace {

std::unique_ptr<CovarianceStructure> create_covariance_structure(
    const CovarianceSpec& spec,
    const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data) {
    
    if (spec.structure == "unstructured") {
        return std::make_unique<UnstructuredCovariance>(spec.dimension);
    } else if (spec.structure == "diagonal") {
        return std::make_unique<DiagonalCovariance>(spec.dimension);
    } else if (spec.structure == "scaled_fixed") {
        auto fixed_it = fixed_covariance_data.find(spec.id);
        if (fixed_it == fixed_covariance_data.end()) {
             throw std::runtime_error("Missing fixed covariance data for: " + spec.id);
        }
        if (fixed_it->second.empty()) {
             throw std::runtime_error("Fixed covariance data is empty for: " + spec.id);
        }
        return std::make_unique<ScaledFixedCovariance>(fixed_it->second[0], spec.dimension);
    } else if (spec.structure == "multi_kernel") {
        auto fixed_it = fixed_covariance_data.find(spec.id);
        if (fixed_it == fixed_covariance_data.end()) {
             throw std::runtime_error("Missing fixed covariance data for: " + spec.id);
        }
        return std::make_unique<MultiKernelCovariance>(fixed_it->second, spec.dimension);
    } else {
        throw std::runtime_error("Unknown covariance structure: " + spec.structure);
    }
}

// Simple dense matrix helpers (row-major)
// A is n x n
void cholesky(std::size_t n, const std::vector<double>& A, std::vector<double>& L) {
    L.assign(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < j; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            if (i == j) {
                double val = A[i * n + i] - sum;
                if (val <= 0.0) throw std::runtime_error("Matrix is not positive definite");
                L[i * n + j] = std::sqrt(val);
            } else {
                L[i * n + j] = (A[i * n + j] - sum) / L[j * n + j];
            }
        }
    }
}

// Solve L * y = b, then L^T * x = y. Returns x = A^-1 * b
std::vector<double> solve_cholesky(std::size_t n, const std::vector<double>& L, const std::vector<double>& b) {
    std::vector<double> y(n);
    // Forward substitution
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (std::size_t k = 0; k < i; ++k) {
            sum += L[i * n + k] * y[k];
        }
        y[i] = (b[i] - sum) / L[i * n + i];
    }
    // Backward substitution
    std::vector<double> x(n);
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (std::size_t k = i + 1; k < n; ++k) {
            sum += L[k * n + i] * x[k];
        }
        x[i] = (y[i] - sum) / L[i * n + i];
    }
    return x;
}

double log_det_cholesky(std::size_t n, const std::vector<double>& L) {
    double log_det = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        log_det += 2.0 * std::log(L[i * n + i]);
    }
    return log_det;
}

} // namespace

double LikelihoodDriver::evaluate_total_loglik(const std::vector<double>& observed,
                                                const std::vector<double>& linear_predictors,
                                                const std::vector<double>& dispersions,
                                                const OutcomeFamily& family,
                                                const std::vector<double>& status,
                                                const std::vector<double>& extra_params) const {
    if (observed.size() != linear_predictors.size() || observed.size() != dispersions.size()) {
        throw std::invalid_argument("observed, linear_predictors, and dispersions must have the same size");
    }
    if (!status.empty() && status.size() != observed.size()) {
        throw std::invalid_argument("status vector must match observed size if provided");
    }
    double total = 0.0;
    for (std::size_t i = 0; i < observed.size(); ++i) {
        double s = status.empty() ? 1.0 : status[i];
        const auto eval = family.evaluate(observed[i], linear_predictors[i], dispersions[i], s, extra_params);
        total += eval.log_likelihood;
    }
    return total;
}

double LikelihoodDriver::evaluate_total_loglik_mixed(const std::vector<double>& observed,
                                                      const std::vector<double>& linear_predictors,
                                                      const std::vector<double>& dispersions,
                                                      const std::vector<const OutcomeFamily*>& families,
                                                      const std::vector<double>& status,
                                                      const std::vector<std::vector<double>>& extra_params) const {
    if (observed.size() != linear_predictors.size() || observed.size() != dispersions.size() ||
        observed.size() != families.size()) {
        throw std::invalid_argument("all input vectors must have the same size");
    }
    if (!status.empty() && status.size() != observed.size()) {
        throw std::invalid_argument("status vector must match observed size if provided");
    }
    if (!extra_params.empty() && extra_params.size() != observed.size()) {
        throw std::invalid_argument("extra_params vector must match observed size if provided");
    }
    double total = 0.0;
    for (std::size_t i = 0; i < observed.size(); ++i) {
        double s = status.empty() ? 1.0 : status[i];
        const std::vector<double>& ep = extra_params.empty() ? std::vector<double>{} : extra_params[i];
        const auto eval = families[i]->evaluate(observed[i], linear_predictors[i], dispersions[i], s, ep);
        total += eval.log_likelihood;
    }
    return total;
}

double LikelihoodDriver::evaluate_model_loglik(const ModelIR& model,
                                                const std::unordered_map<std::string, std::vector<double>>& data,
                                                const std::unordered_map<std::string, std::vector<double>>& linear_predictors,
                                                const std::unordered_map<std::string, std::vector<double>>& dispersions,
                                                const std::unordered_map<std::string, std::vector<double>>& covariance_parameters,
                                                const std::unordered_map<std::string, std::vector<double>>& status,
                                                const std::unordered_map<std::string, std::vector<double>>& extra_params,
                                                const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data,
                                                EstimationMethod method) const {
    
    if (model.random_effects.empty()) {
        double total = 0.0;
        for (const auto& var : model.variables) {
            if (var.kind != VariableKind::Observed) {
                continue;  // Skip latent and grouping variables for now
            }
            auto data_it = data.find(var.name);
            auto pred_it = linear_predictors.find(var.name);
            auto disp_it = dispersions.find(var.name);
            if (data_it == data.end() || pred_it == linear_predictors.end() || disp_it == dispersions.end()) {
                throw std::invalid_argument("Missing data for variable: " + var.name);
            }
            const auto& obs = data_it->second;
            const auto& preds = pred_it->second;
            const auto& disps = disp_it->second;
            if (obs.size() != preds.size() || obs.size() != disps.size()) {
                throw std::invalid_argument("Data vectors for variable " + var.name + " have mismatched sizes");
            }
            
            const std::vector<double>* status_vec = nullptr;
            auto status_it = status.find(var.name);
            if (status_it != status.end()) {
                status_vec = &status_it->second;
                if (status_vec->size() != obs.size()) {
                    throw std::invalid_argument("Status vector for variable " + var.name + " has mismatched size");
                }
            }

            const std::vector<double>* extra_vec = nullptr;
            auto extra_it = extra_params.find(var.name);
            if (extra_it != extra_params.end()) {
                extra_vec = &extra_it->second;
            }
            const std::vector<double>& ep = extra_vec ? *extra_vec : std::vector<double>{};

            auto family = OutcomeFamilyFactory::create(var.family);
            for (std::size_t i = 0; i < obs.size(); ++i) {
                double s = status_vec ? (*status_vec)[i] : 1.0;
                const auto eval = family->evaluate(obs[i], preds[i], disps[i], s, ep);
                total += eval.log_likelihood;
            }
        }
        return total;
    }

    // Mixed model logic
    // 1. Identify outcome variable
    std::string obs_var_name;
    std::string obs_family_name;
    // We need to find the LAST observed variable that is NOT used as a predictor in random effects?
    // Or just the last one defined?
    // In grm_tests, we defined "id1", "id2", "y". "y" is the outcome.
    // In multi_kernel_tests, we defined "y", "group", "id1", "id2".
    // "id1" and "id2" are Observed variables (dummy).
    // The loop below picks the LAST one.
    // So it picks "id2".
    // But "id2" is a predictor, not the outcome.
    // We need a better way to identify the outcome.
    // Maybe the outcome is the one that is NOT in random effect variables?
    // Or we rely on the order.
    // Let's change the test to define "y" LAST.
    for (const auto& var : model.variables) {
        if (var.kind == VariableKind::Observed) {
            obs_var_name = var.name; 
            obs_family_name = var.family;
        }
    }
    if (obs_var_name.empty()) throw std::runtime_error("No observed variable found");

    // 2. Identify grouping variable and random effect structure
    const auto& re_spec = model.random_effects.front();
    
    std::string grouping_var_name;
    for (const auto& v_name : re_spec.variables) {
        for (const auto& mv : model.variables) {
            if (mv.name == v_name && mv.kind == VariableKind::Grouping) {
                grouping_var_name = v_name;
                break;
            }
        }
        if (!grouping_var_name.empty()) break;
    }

    if (grouping_var_name.empty()) {
        throw std::runtime_error("No grouping variable found in random effect spec");
    }

    // 3. Get Covariance Structure
    const CovarianceSpec* cov_spec = nullptr;
    for (const auto& cs : model.covariances) {
        if (cs.id == re_spec.covariance_id) {
            cov_spec = &cs;
            break;
        }
    }
    if (!cov_spec) throw std::runtime_error("Covariance spec not found: " + re_spec.covariance_id);

    std::unique_ptr<CovarianceStructure> G_struct = create_covariance_structure(*cov_spec, fixed_covariance_data);

    auto cov_params_it = covariance_parameters.find(re_spec.covariance_id);
    if (cov_params_it == covariance_parameters.end()) {
        throw std::runtime_error("Missing covariance parameters for: " + re_spec.covariance_id);
    }
    std::vector<double> G_mat = G_struct->materialize(cov_params_it->second);

    // Identify predictors
    std::vector<std::string> predictor_vars;
    for (const auto& v_name : re_spec.variables) {
        if (v_name != grouping_var_name) {
            predictor_vars.push_back(v_name);
        }
    }

    // Validate dimension
    if (predictor_vars.empty()) {
        if (cov_spec->dimension != 1 && cov_spec->structure != "scaled_fixed" && cov_spec->structure != "multi_kernel") {
             throw std::runtime_error("No predictor variables specified for random effect, but dimension > 1");
        }
        // Implicit intercept or identity
    } else {
        if (cov_spec->dimension != predictor_vars.size()) {
             throw std::runtime_error("Mismatch between random effect dimension and number of predictors");
        }
    }

    // 4. Process by group
    auto group_it = data.find(grouping_var_name);
    if (group_it == data.end()) {
        throw std::runtime_error("Grouping variable not found in data: " + grouping_var_name);
    }
    const auto& group_data = group_it->second;
    std::map<double, std::vector<std::size_t>> groups;
    for (std::size_t i = 0; i < group_data.size(); ++i) {
        groups[group_data[i]].push_back(i);
    }

    auto obs_it = data.find(obs_var_name);
    if (obs_it == data.end()) {
        throw std::runtime_error("Observed variable not found in data: " + obs_var_name);
    }
    const auto& obs_data = obs_it->second;

    auto pred_it = linear_predictors.find(obs_var_name);
    if (pred_it == linear_predictors.end()) {
        throw std::runtime_error("Linear predictors not found for: " + obs_var_name);
    }
    const auto& pred_data = pred_it->second;

    auto disp_it = dispersions.find(obs_var_name);
    if (disp_it == dispersions.end()) {
        throw std::runtime_error("Dispersions not found for: " + obs_var_name);
    }
    const auto& disp_data = disp_it->second;

    // Identify fixed effect predictors for REML
    std::vector<std::string> fixed_effect_vars;
    if (method == EstimationMethod::REML) {
        for (const auto& edge : model.edges) {
            if (edge.kind == EdgeKind::Regression && edge.target == obs_var_name) {
                fixed_effect_vars.push_back(edge.source);
            }
        }
        std::sort(fixed_effect_vars.begin(), fixed_effect_vars.end());
    }
    std::size_t p = fixed_effect_vars.size();
    std::vector<double> Xt_Vinv_X(p * p, 0.0);

    auto outcome_family = OutcomeFamilyFactory::create(obs_family_name);
    std::vector<double> G_inv;
    double log_det_G = 0.0;
    if (obs_family_name != "gaussian") {
        std::size_t q = cov_spec->dimension;
        std::vector<double> L_G(q * q);
        try {
            cholesky(q, G_mat, L_G);
            log_det_G = log_det_cholesky(q, L_G);
            std::vector<double> I(q * q, 0.0);
            for(size_t i=0; i<q; ++i) I[i*q+i] = 1.0;
            G_inv.resize(q * q);
            for(size_t j=0; j<q; ++j) {
                std::vector<double> col(q);
                for(size_t i=0; i<q; ++i) col[i] = I[i*q+j];
                std::vector<double> res = solve_cholesky(q, L_G, col);
                for(size_t i=0; i<q; ++i) G_inv[i*q+j] = res[i];
            }
        } catch (...) {
            return -std::numeric_limits<double>::infinity();
        }
    }

    double total_loglik = 0.0;
    const double log_2pi = std::log(2.0 * 3.14159265358979323846);

    for (const auto& [group_id, indices] : groups) {
        std::size_t n_i = indices.size();
        std::size_t q = cov_spec->dimension;

        // Construct Z_i (n_i x q)
        std::vector<double> Z_i(n_i * q, 0.0);
        if (predictor_vars.empty()) {
            if ((cov_spec->structure == "scaled_fixed" || cov_spec->structure == "multi_kernel") && q == n_i) {
                // Identity matrix
                for (std::size_t k = 0; k < n_i; ++k) {
                    Z_i[k * q + k] = 1.0;
                }
            } else {
                std::fill(Z_i.begin(), Z_i.end(), 1.0);
            }
        } else {
            for (std::size_t k = 0; k < n_i; ++k) {
                std::size_t row_idx = indices[k];
                for (std::size_t j = 0; j < q; ++j) {
                    // Check if predictor is in data or linear_predictors
                    auto it = data.find(predictor_vars[j]);
                    if (it != data.end()) {
                        Z_i[k * q + j] = it->second[row_idx];
                    } else {
                        // Try linear_predictors? Unlikely for random effect design matrix.
                        // But let's check.
                        auto it2 = linear_predictors.find(predictor_vars[j]);
                        if (it2 != linear_predictors.end()) {
                            Z_i[k * q + j] = it2->second[row_idx];
                        } else {
                             throw std::runtime_error("Predictor variable not found in data: " + predictor_vars[j]);
                        }
                    }
                }
            }
        }

        if (obs_family_name == "gaussian") {
            // Compute ZG = Z_i * G (n_i x q)
            std::vector<double> ZG(n_i * q, 0.0);
            for (std::size_t r = 0; r < n_i; ++r) {
                for (std::size_t c = 0; c < q; ++c) {
                    double sum = 0.0;
                    for (std::size_t k = 0; k < q; ++k) {
                        sum += Z_i[r * q + k] * G_mat[k * q + c];
                    }
                    ZG[r * q + c] = sum;
                }
            }

            // Compute V_i = ZG * Z_i^T + R_i (n_i x n_i)
            std::vector<double> V_i(n_i * n_i, 0.0);
            for (std::size_t r = 0; r < n_i; ++r) {
                for (std::size_t c = 0; c < n_i; ++c) {
                    double sum = 0.0;
                    for (std::size_t k = 0; k < q; ++k) {
                        sum += ZG[r * q + k] * Z_i[c * q + k];
                    }
                    V_i[r * n_i + c] = sum;
                }
            }

            // Add R_i
            for (std::size_t k = 0; k < n_i; ++k) {
                V_i[k * n_i + k] += disp_data[indices[k]];
            }

            std::vector<double> L(n_i * n_i);
            cholesky(n_i, V_i, L);
            double log_det = log_det_cholesky(n_i, L);

            std::vector<double> resid(n_i);
            for (std::size_t k = 0; k < n_i; ++k) {
                resid[k] = obs_data[indices[k]] - pred_data[indices[k]];
            }

            std::vector<double> alpha = solve_cholesky(n_i, L, resid);

            double quad_form = 0.0;
            for (std::size_t k = 0; k < n_i; ++k) {
                quad_form += resid[k] * alpha[k];
            }

            total_loglik += -0.5 * (n_i * log_2pi + log_det + quad_form);

            if (method == EstimationMethod::REML && p > 0) {
                // Construct X_i (n_i x p)
                std::vector<double> X_i(n_i * p);
                for (std::size_t k = 0; k < n_i; ++k) {
                    std::size_t row_idx = indices[k];
                    for (std::size_t j = 0; j < p; ++j) {
                        auto it = data.find(fixed_effect_vars[j]);
                        if (it != data.end()) {
                            X_i[k * p + j] = it->second[row_idx];
                        } else {
                            throw std::runtime_error("Missing data for fixed effect: " + fixed_effect_vars[j]);
                        }
                    }
                }

                // Compute V_i^-1 * X_i
                std::vector<double> Vinv_X_i(n_i * p);
                for (std::size_t j = 0; j < p; ++j) {
                    std::vector<double> x_col(n_i);
                    for (std::size_t k = 0; k < n_i; ++k) {
                        x_col[k] = X_i[k * p + j];
                    }
                    std::vector<double> m_col = solve_cholesky(n_i, L, x_col);
                    for (std::size_t k = 0; k < n_i; ++k) {
                        Vinv_X_i[k * p + j] = m_col[k];
                    }
                }

                // Accumulate X_i^T * (V_i^-1 * X_i)
                for (std::size_t r = 0; r < p; ++r) {
                    for (std::size_t c = 0; c < p; ++c) {
                        double sum = 0.0;
                        for (std::size_t k = 0; k < n_i; ++k) {
                            sum += X_i[k * p + r] * Vinv_X_i[k * p + c];
                        }
                        Xt_Vinv_X[r * p + c] += sum;
                    }
                }
            }
        } else {
            // Laplace approximation
            std::vector<double> u_i(q, 0.0);
            
            // Newton-Raphson
            for (int iter = 0; iter < 20; ++iter) {
                std::vector<double> grad(q, 0.0);
                std::vector<double> hess(q * q, 0.0);
                
                // Prior term
                for (std::size_t j = 0; j < q; ++j) {
                    double sum = 0.0;
                    for (std::size_t k = 0; k < q; ++k) {
                        sum += G_inv[j * q + k] * u_i[k];
                    }
                    grad[j] -= sum;
                }
                for (std::size_t j = 0; j < q * q; ++j) {
                    hess[j] -= G_inv[j];
                }
                
                // Likelihood term
                for (std::size_t k = 0; k < n_i; ++k) {
                    std::size_t idx = indices[k];
                    double eta = pred_data[idx];
                    for (std::size_t j = 0; j < q; ++j) {
                        eta += Z_i[k * q + j] * u_i[j];
                    }
                    
                    auto status_it = status.find(obs_var_name);
                    double s = (status_it != status.end()) ? status_it->second[idx] : 1.0;
                    
                    auto extra_it = extra_params.find(obs_var_name);
                    const std::vector<double>& ep = (extra_it != extra_params.end()) ? extra_it->second : std::vector<double>{};
                    
                    auto eval = outcome_family->evaluate(obs_data[idx], eta, disp_data[idx], s, ep);
                    
                    for (std::size_t j = 0; j < q; ++j) {
                        grad[j] += Z_i[k * q + j] * eval.first_derivative;
                    }
                    
                    for (std::size_t r = 0; r < q; ++r) {
                        for (std::size_t c = 0; c < q; ++c) {
                            hess[r * q + c] += Z_i[k * q + r] * Z_i[k * q + c] * eval.second_derivative;
                        }
                    }
                }
                
                std::vector<double> neg_hess(q * q);
                for (std::size_t j = 0; j < q * q; ++j) neg_hess[j] = -hess[j];
                
                std::vector<double> L_H(q * q);
                try {
                    cholesky(q, neg_hess, L_H);
                } catch (...) {
                    break; 
                }
                
                std::vector<double> delta = solve_cholesky(q, L_H, grad);
                
                double max_delta = 0.0;
                for (std::size_t j = 0; j < q; ++j) {
                    u_i[j] += delta[j];
                    max_delta = std::max(max_delta, std::abs(delta[j]));
                }
                
                if (max_delta < 1e-6) break;
            }
            
            // Compute Laplace approximation at mode u_i
            double u_Ginv_u = 0.0;
            for (std::size_t r = 0; r < q; ++r) {
                for (std::size_t c = 0; c < q; ++c) {
                    u_Ginv_u += u_i[r] * G_inv[r * q + c] * u_i[c];
                }
            }
            double log_prior = -0.5 * (q * log_2pi + log_det_G + u_Ginv_u);
            
            double log_lik_data = 0.0;
            std::vector<double> hess(q * q, 0.0);
            for (std::size_t j = 0; j < q * q; ++j) hess[j] -= G_inv[j];
            
            for (std::size_t k = 0; k < n_i; ++k) {
                std::size_t idx = indices[k];
                double eta = pred_data[idx];
                for (std::size_t j = 0; j < q; ++j) eta += Z_i[k * q + j] * u_i[j];
                
                auto status_it = status.find(obs_var_name);
                double s = (status_it != status.end()) ? status_it->second[idx] : 1.0;
                
                auto extra_it = extra_params.find(obs_var_name);
                const std::vector<double>& ep = (extra_it != extra_params.end()) ? extra_it->second : std::vector<double>{};

                auto eval = outcome_family->evaluate(obs_data[idx], eta, disp_data[idx], s, ep);
                log_lik_data += eval.log_likelihood;
                
                for (std::size_t r = 0; r < q; ++r) {
                    for (std::size_t c = 0; c < q; ++c) {
                        hess[r * q + c] += Z_i[k * q + r] * Z_i[k * q + c] * eval.second_derivative;
                    }
                }
            }
            
            std::vector<double> neg_hess(q * q);
            for (std::size_t j = 0; j < q * q; ++j) neg_hess[j] = -hess[j];
            std::vector<double> L_H(q * q);
            try {
                cholesky(q, neg_hess, L_H);
                double log_det_neg_hess = log_det_cholesky(q, L_H);
                total_loglik += log_prior + log_lik_data + 0.5 * q * log_2pi - 0.5 * log_det_neg_hess;
            } catch (...) {
                total_loglik += -std::numeric_limits<double>::infinity();
            }
        }
    }

    if (method == EstimationMethod::REML && p > 0) {
        std::vector<double> L_rem(p * p);
        cholesky(p, Xt_Vinv_X, L_rem);
        double log_det_rem = log_det_cholesky(p, L_rem);
        total_loglik -= 0.5 * log_det_rem;
        total_loglik += 0.5 * p * log_2pi;
    }

    return total_loglik;
}

std::unordered_map<std::string, double> LikelihoodDriver::evaluate_model_gradient(const ModelIR& model,
                                                const std::unordered_map<std::string, std::vector<double>>& data,
                                                const std::unordered_map<std::string, std::vector<double>>& linear_predictors,
                                                const std::unordered_map<std::string, std::vector<double>>& dispersions,
                                                const std::unordered_map<std::string, std::vector<double>>& covariance_parameters,
                                                const std::unordered_map<std::string, std::vector<double>>& status,
                                                const std::unordered_map<std::string, std::vector<double>>& extra_params,
                                                const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data,
                                                EstimationMethod method) const {
    std::unordered_map<std::string, double> gradients;

    if (model.random_effects.empty()) {
        // Pre-process edges for efficient lookup: target -> [(source, param_id)]
        std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> incoming_edges;
        for (const auto& edge : model.edges) {
            if (edge.kind == EdgeKind::Regression && !edge.parameter_id.empty()) {
                incoming_edges[edge.target].emplace_back(edge.source, edge.parameter_id);
            }
        }

        for (const auto& var : model.variables) {
            if (var.kind != VariableKind::Observed) continue;

            auto data_it = data.find(var.name);
            auto pred_it = linear_predictors.find(var.name);
            auto disp_it = dispersions.find(var.name);
            
            if (data_it == data.end() || pred_it == linear_predictors.end() || disp_it == dispersions.end()) {
                 continue; 
            }

            const auto& obs = data_it->second;
            const auto& preds = pred_it->second;
            const auto& disps = disp_it->second;
            
            const std::vector<double>* status_vec = nullptr;
            if (status.count(var.name)) status_vec = &status.at(var.name);

            const std::vector<double>* extra_vec = nullptr;
            if (extra_params.count(var.name)) extra_vec = &extra_params.at(var.name);
            const std::vector<double>& ep = extra_vec ? *extra_vec : std::vector<double>{};

            auto family = OutcomeFamilyFactory::create(var.family);
            
            // Get incoming edges for this variable
            const auto& edges = incoming_edges[var.name];

            for (size_t i = 0; i < obs.size(); ++i) {
                double s = status_vec ? (*status_vec)[i] : 1.0;
                auto eval = family->evaluate(obs[i], preds[i], disps[i], s, ep);
                double d_lp = eval.first_derivative; // d(loglik)/d(eta)

                for (const auto& edge : edges) {
                    const std::string& source = edge.first;
                    const std::string& param_id = edge.second;
                    
                    double source_val = 0.0;
                    if (data.count(source)) {
                        source_val = data.at(source)[i];
                    }
                    
                    gradients[param_id] += d_lp * source_val;
                }
            }
        }
        return gradients;
    } else {
        // Mixed model gradient
        // 1. Validate single random effect (limitation for now)
        if (model.random_effects.size() > 1) {
            throw std::runtime_error("Multiple random effects not yet supported for analytic gradients");
        }
        const auto& re_spec = model.random_effects[0];

        // 2. Identify grouping variable
        std::string grouping_var_name;
        for (const auto& v_name : re_spec.variables) {
            for (const auto& mv : model.variables) {
                if (mv.name == v_name && mv.kind == VariableKind::Grouping) {
                    grouping_var_name = v_name;
                    break;
                }
            }
            if (!grouping_var_name.empty()) break;
        }
        if (grouping_var_name.empty()) {
            throw std::runtime_error("No grouping variable found in random effect spec");
        }

        // 3. Get Covariance Structure
        const CovarianceSpec* cov_spec = nullptr;
        for (const auto& cs : model.covariances) {
            if (cs.id == re_spec.covariance_id) {
                cov_spec = &cs;
                break;
            }
        }
        if (!cov_spec) throw std::runtime_error("Covariance spec not found: " + re_spec.covariance_id);

        std::unique_ptr<CovarianceStructure> G_struct = create_covariance_structure(*cov_spec, fixed_covariance_data);

        auto cov_params_it = covariance_parameters.find(re_spec.covariance_id);
        if (cov_params_it == covariance_parameters.end()) {
            throw std::runtime_error("Missing covariance parameters for: " + re_spec.covariance_id);
        }
        const std::vector<double>& theta = cov_params_it->second;
        std::vector<double> G_mat = G_struct->materialize(theta);
        std::vector<std::vector<double>> G_grads = G_struct->parameter_gradients(theta);

        // Identify predictors
        std::vector<std::string> predictor_vars;
        for (const auto& v_name : re_spec.variables) {
            if (v_name != grouping_var_name) {
                predictor_vars.push_back(v_name);
            }
        }

        // 4. Process by group
        auto group_it = data.find(grouping_var_name);
        if (group_it == data.end()) {
            throw std::runtime_error("Grouping variable not found in data: " + grouping_var_name);
        }
        const auto& group_data = group_it->second;
        std::map<double, std::vector<std::size_t>> groups;
        for (std::size_t i = 0; i < group_data.size(); ++i) {
            groups[group_data[i]].push_back(i);
        }

        if (linear_predictors.size() != 1) {
            throw std::runtime_error("Analytic gradients for mixed models currently require exactly one outcome variable");
        }
        const std::string& obs_var_name = linear_predictors.begin()->first;

        std::string obs_family_name;
        for (const auto& var : model.variables) {
            if (var.name == obs_var_name && var.kind == VariableKind::Observed) {
                obs_family_name = var.family;
                break;
            }
        }
        if (obs_family_name.empty()) {
            throw std::runtime_error("Outcome variable metadata not found: " + obs_var_name);
        }
        if (obs_family_name != "gaussian") {
            throw std::runtime_error("Analytic gradients only implemented for Gaussian mixed models");
        }

        auto obs_it = data.find(obs_var_name);
        const auto& obs_data = obs_it->second;
        auto pred_it = linear_predictors.find(obs_var_name);
        const auto& pred_data = pred_it->second;
        auto disp_it = dispersions.find(obs_var_name);
        const auto& disp_data = disp_it->second;

        // Identify fixed effect predictors (edges pointing to obs_var_name)
        std::vector<std::pair<std::string, std::string>> fixed_effects; // source, param_id
        for (const auto& edge : model.edges) {
            if (edge.kind == EdgeKind::Regression && edge.target == obs_var_name && !edge.parameter_id.empty()) {
                fixed_effects.emplace_back(edge.source, edge.parameter_id);
            }
        }

        for (const auto& [group_id, indices] : groups) {
            std::size_t n_i = indices.size();
            std::size_t q = cov_spec->dimension;

            // Construct Z_i (n_i x q)
            std::vector<double> Z_i(n_i * q, 0.0);
            if (predictor_vars.empty()) {
                if ((cov_spec->structure == "scaled_fixed" || cov_spec->structure == "multi_kernel") && q == n_i) {
                    for (std::size_t k = 0; k < n_i; ++k) Z_i[k * q + k] = 1.0;
                } else {
                    std::fill(Z_i.begin(), Z_i.end(), 1.0);
                }
            } else {
                for (std::size_t k = 0; k < n_i; ++k) {
                    std::size_t row_idx = indices[k];
                    for (std::size_t j = 0; j < q; ++j) {
                        auto it = data.find(predictor_vars[j]);
                        if (it != data.end()) {
                            Z_i[k * q + j] = it->second[row_idx];
                        } else {
                             // Try linear_predictors?
                             auto it2 = linear_predictors.find(predictor_vars[j]);
                             if (it2 != linear_predictors.end()) {
                                 Z_i[k * q + j] = it2->second[row_idx];
                             } else {
                                 throw std::runtime_error("Predictor variable not found: " + predictor_vars[j]);
                             }
                        }
                    }
                }
            }

            // Compute ZG = Z_i * G
            std::vector<double> ZG(n_i * q, 0.0);
            for (std::size_t r = 0; r < n_i; ++r) {
                for (std::size_t c = 0; c < q; ++c) {
                    double sum = 0.0;
                    for (std::size_t k = 0; k < q; ++k) sum += Z_i[r * q + k] * G_mat[k * q + c];
                    ZG[r * q + c] = sum;
                }
            }

            // Compute V_i = ZG * Z_i^T + R_i
            std::vector<double> V_i(n_i * n_i, 0.0);
            for (std::size_t r = 0; r < n_i; ++r) {
                for (std::size_t c = 0; c < n_i; ++c) {
                    double sum = 0.0;
                    for (std::size_t k = 0; k < q; ++k) sum += ZG[r * q + k] * Z_i[c * q + k];
                    V_i[r * n_i + c] = sum;
                }
            }
            for (std::size_t k = 0; k < n_i; ++k) {
                V_i[k * n_i + k] += disp_data[indices[k]];
            }

            // Invert V_i
            std::vector<double> L(n_i * n_i);
            cholesky(n_i, V_i, L);
            
            std::vector<double> V_inv(n_i * n_i);
            std::vector<double> I(n_i * n_i, 0.0);
            for(size_t k=0; k<n_i; ++k) I[k*n_i+k] = 1.0;
            
            for(size_t j=0; j<n_i; ++j) {
                std::vector<double> col(n_i);
                for(size_t i=0; i<n_i; ++i) col[i] = I[i*n_i+j];
                std::vector<double> res = solve_cholesky(n_i, L, col);
                for(size_t i=0; i<n_i; ++i) V_inv[i*n_i+j] = res[i];
            }

            // Residuals e_i
            std::vector<double> e_i(n_i);
            for (std::size_t k = 0; k < n_i; ++k) {
                e_i[k] = obs_data[indices[k]] - pred_data[indices[k]];
            }

            // alpha_i = V_inv * e_i
            std::vector<double> alpha_i(n_i, 0.0);
            for (std::size_t r = 0; r < n_i; ++r) {
                for (std::size_t c = 0; c < n_i; ++c) {
                    alpha_i[r] += V_inv[r * n_i + c] * e_i[c];
                }
            }

            // Fixed effects gradients: X_i^T * alpha_i
            for (const auto& fe : fixed_effects) {
                const std::string& source = fe.first;
                const std::string& param_id = fe.second;
                
                double dot = 0.0;
                if (data.count(source)) {
                    const auto& src_vec = data.at(source);
                    for (std::size_t k = 0; k < n_i; ++k) {
                        dot += src_vec[indices[k]] * alpha_i[k];
                    }
                }
                gradients[param_id] += dot;
            }

            // Covariance gradients
            // dL/dtheta = 0.5 * tr( (alpha * alpha^T - V_inv) * dV/dtheta )
            // dV/dtheta = Z * dG/dtheta * Z^T
            // tr( (alpha * alpha^T - V_inv) * Z * dG * Z^T )
            // = tr( Z^T * (alpha * alpha^T - V_inv) * Z * dG )
            
            // Compute M = Z^T * (alpha * alpha^T - V_inv) * Z
            // First compute temp = (alpha * alpha^T - V_inv) * Z
            // (n x n) * (n x q) -> (n x q)
            
            std::vector<double> temp(n_i * q, 0.0);
            for (std::size_t r = 0; r < n_i; ++r) {
                for (std::size_t c = 0; c < q; ++c) {
                    double sum = 0.0;
                    for (std::size_t k = 0; k < n_i; ++k) {
                        double term = alpha_i[r] * alpha_i[k] - V_inv[r * n_i + k];
                        sum += term * Z_i[k * q + c];
                    }
                    temp[r * q + c] = sum;
                }
            }
            
            // M = Z^T * temp (q x q)
            std::vector<double> M(q * q, 0.0);
            for (std::size_t r = 0; r < q; ++r) {
                for (std::size_t c = 0; c < q; ++c) {
                    double sum = 0.0;
                    for (std::size_t k = 0; k < n_i; ++k) {
                        sum += Z_i[k * q + r] * temp[k * q + c];
                    }
                    M[r * q + c] = sum;
                }
            }
            
            // For each parameter k
            for (std::size_t k = 0; k < G_grads.size(); ++k) {
                const auto& dG = G_grads[k];
                double trace = 0.0;
                for (std::size_t r = 0; r < q; ++r) {
                    for (std::size_t c = 0; c < q; ++c) {
                        trace += M[r * q + c] * dG[c * q + r]; // Note: dG is symmetric usually, but trace(AB) = sum A_ij B_ji
                    }
                }
                
                // Parameter name construction needs to match ModelObjective
                // ModelObjective uses "cov_id_index"
                std::string param_name = re_spec.covariance_id + "_" + std::to_string(k);
                gradients[param_name] += 0.5 * trace;
            }
        }
        return gradients;
    }
}

namespace {

class ModelObjective : public ObjectiveFunction {
public:
    ModelObjective(const LikelihoodDriver& driver,
                   const ModelIR& model,
                   const std::unordered_map<std::string, std::vector<double>>& data,
                   const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data = {})
        : driver_(driver), model_(model), data_(data), fixed_covariance_data_(fixed_covariance_data) {
        
        for (const auto& edge : model_.edges) {
            if (!edge.parameter_id.empty()) {
                char* end;
                std::strtod(edge.parameter_id.c_str(), &end);
                if (end == edge.parameter_id.c_str() || *end != '\0') {
                    if (param_map_.find(edge.parameter_id) == param_map_.end()) {
                        param_map_[edge.parameter_id] = param_names_.size();
                        param_names_.push_back(edge.parameter_id);
                    }
                }
            }
        }

        // Add covariance parameters
        for (const auto& cov : model_.covariances) {
            auto structure = create_covariance_structure(cov, fixed_covariance_data_);
            size_t count = structure->parameter_count();
            if (count > 0) {
                size_t start_idx = param_names_.size();
                for (size_t i = 0; i < count; ++i) {
                    std::string param_name = cov.id + "_" + std::to_string(i);
                    param_map_[param_name] = param_names_.size();
                    param_names_.push_back(param_name);
                }
                covariance_param_ranges_[cov.id] = {start_idx, count};
            }
        }
    }

    [[nodiscard]] double value(const std::vector<double>& parameters) const override {
        std::unordered_map<std::string, std::vector<double>> linear_predictors;
        std::unordered_map<std::string, std::vector<double>> dispersions;
        std::unordered_map<std::string, std::vector<double>> covariance_parameters;

        build_prediction_workspaces(parameters, linear_predictors, dispersions, covariance_parameters);

        return -driver_.evaluate_model_loglik(model_, data_, linear_predictors, dispersions, covariance_parameters, {}, {}, fixed_covariance_data_);
    }

    [[nodiscard]] std::vector<double> gradient(const std::vector<double>& parameters) const override {
        std::unordered_map<std::string, std::vector<double>> linear_predictors;
        std::unordered_map<std::string, std::vector<double>> dispersions;
        std::unordered_map<std::string, std::vector<double>> covariance_parameters;

        build_prediction_workspaces(parameters, linear_predictors, dispersions, covariance_parameters);

        auto grad_map = driver_.evaluate_model_gradient(
            model_,
            data_,
            linear_predictors,
            dispersions,
            covariance_parameters,
            {},
            {},
            fixed_covariance_data_);
        
        std::vector<double> grad(parameters.size(), 0.0);
        for (const auto& [param_id, g] : grad_map) {
            auto it = param_map_.find(param_id);
            if (it != param_map_.end()) {
                grad[it->second] -= g; // Negative gradient for minimization
            }
        }
        return grad;
    }
    
    const std::vector<std::string>& parameter_names() const { return param_names_; }

private:
    const LikelihoodDriver& driver_;
    const ModelIR& model_;
    const std::unordered_map<std::string, std::vector<double>>& data_;
    const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data_;
    std::vector<std::string> param_names_;
    std::unordered_map<std::string, size_t> param_map_;
    std::unordered_map<std::string, std::pair<size_t, size_t>> covariance_param_ranges_;

    void build_prediction_workspaces(const std::vector<double>& parameters,
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

        if (!response_vars.empty()) {
            for (const auto& name : response_vars) {
                register_response(name);
            }
        } else {
            for (const auto& var : model_.variables) {
                if (var.kind == VariableKind::Observed) {
                    register_response(var.name);
                }
            }
        }

        for (const auto& edge : model_.edges) {
            if (edge.kind != EdgeKind::Regression) continue;

            double weight = 0.0;
            if (!edge.parameter_id.empty()) {
                auto it = param_map_.find(edge.parameter_id);
                if (it != param_map_.end()) {
                    weight = parameters[it->second];
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

        for (const auto& [id, range] : covariance_param_ranges_) {
            std::vector<double> params;
            params.reserve(range.second);
            for (size_t i = 0; i < range.second; ++i) {
                params.push_back(parameters[range.first + i]);
            }
            covariance_parameters[id] = std::move(params);
        }
    }
};

} // namespace

OptimizationResult LikelihoodDriver::fit(const ModelIR& model,
                                         const std::unordered_map<std::string, std::vector<double>>& data,
                                         const OptimizationOptions& options,
                                         const std::string& optimizer_name,
                                         const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data) const {
    
    ModelObjective objective(*this, model, data, fixed_covariance_data);
    std::vector<double> initial_params(objective.parameter_names().size(), 0.1);
    
    std::unique_ptr<Optimizer> optimizer;
    if (optimizer_name == "lbfgs") {
        optimizer = make_lbfgs_optimizer();
    } else {
        optimizer = make_gradient_descent_optimizer();
    }
    
    return optimizer->optimize(objective, initial_params, options);
}

}  // namespace libsemx
