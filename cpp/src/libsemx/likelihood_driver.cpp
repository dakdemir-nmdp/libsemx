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
#include <vector>

namespace libsemx {

namespace {

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

    std::unique_ptr<CovarianceStructure> G_struct;
    if (cov_spec->structure == "unstructured") {
        G_struct = std::make_unique<UnstructuredCovariance>(cov_spec->dimension);
    } else if (cov_spec->structure == "diagonal") {
        G_struct = std::make_unique<DiagonalCovariance>(cov_spec->dimension);
    } else if (cov_spec->structure == "scaled_fixed") {
        auto fixed_it = fixed_covariance_data.find(cov_spec->id);
        if (fixed_it == fixed_covariance_data.end()) {
             throw std::runtime_error("Missing fixed covariance data for: " + cov_spec->id);
        }
        if (fixed_it->second.empty()) {
             throw std::runtime_error("Fixed covariance data is empty for: " + cov_spec->id);
        }
        G_struct = std::make_unique<ScaledFixedCovariance>(fixed_it->second[0], cov_spec->dimension);
    } else if (cov_spec->structure == "multi_kernel") {
        auto fixed_it = fixed_covariance_data.find(cov_spec->id);
        if (fixed_it == fixed_covariance_data.end()) {
             throw std::runtime_error("Missing fixed covariance data for: " + cov_spec->id);
        }
        G_struct = std::make_unique<MultiKernelCovariance>(fixed_it->second, cov_spec->dimension);
    } else {
        throw std::runtime_error("Unknown covariance structure: " + cov_spec->structure);
    }

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
        if (cov_spec->dimension != 1) {
             throw std::runtime_error("No predictor variables specified for random effect, but dimension > 1");
        }
        // Implicit intercept
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
        std::vector<double> Z_i(n_i * q);
        if (predictor_vars.empty()) {
            std::fill(Z_i.begin(), Z_i.end(), 1.0);
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

namespace {

class ModelObjective : public ObjectiveFunction {
public:
    ModelObjective(const LikelihoodDriver& driver,
                   const ModelIR& model,
                   const std::unordered_map<std::string, std::vector<double>>& data)
        : driver_(driver), model_(model), data_(data) {
        
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
    }

    [[nodiscard]] double value(const std::vector<double>& parameters) const override {
        std::unordered_map<std::string, std::vector<double>> linear_predictors;
        std::unordered_map<std::string, std::vector<double>> dispersions;
        
        for (const auto& var : model_.variables) {
            if (var.kind == VariableKind::Observed) {
                if (data_.find(var.name) != data_.end()) {
                     linear_predictors[var.name] = std::vector<double>(data_.at(var.name).size(), 0.0);
                     dispersions[var.name] = std::vector<double>(data_.at(var.name).size(), 1.0);
                }
            }
        }

        for (const auto& edge : model_.edges) {
            if (edge.kind == EdgeKind::Regression) {
                double weight = 0.0;
                if (!edge.parameter_id.empty()) {
                    auto it = param_map_.find(edge.parameter_id);
                    if (it != param_map_.end()) {
                        weight = parameters[it->second];
                    } else {
                        weight = std::stod(edge.parameter_id);
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
        }

        return -driver_.evaluate_model_loglik(model_, data_, linear_predictors, dispersions);
    }

    [[nodiscard]] std::vector<double> gradient(const std::vector<double>& parameters) const override {
        std::vector<double> grad(parameters.size());
        double base_val = value(parameters);
        double epsilon = 1e-6;
        
        std::vector<double> p = parameters;
        for (size_t i = 0; i < parameters.size(); ++i) {
            p[i] += epsilon;
            double val_plus = value(p);
            grad[i] = (val_plus - base_val) / epsilon;
            p[i] -= epsilon;
        }
        return grad;
    }
    
    const std::vector<std::string>& parameter_names() const { return param_names_; }

private:
    const LikelihoodDriver& driver_;
    const ModelIR& model_;
    const std::unordered_map<std::string, std::vector<double>>& data_;
    std::vector<std::string> param_names_;
    std::unordered_map<std::string, size_t> param_map_;
};

} // namespace

OptimizationResult LikelihoodDriver::fit(const ModelIR& model,
                                         const std::unordered_map<std::string, std::vector<double>>& data,
                                         const OptimizationOptions& options,
                                         const std::string& optimizer_name) const {
    ModelObjective objective(*this, model, data);
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
