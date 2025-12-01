#include "libsemx/likelihood_driver.hpp"
#include "libsemx/outcome_family_factory.hpp"
#include "libsemx/covariance_structure.hpp"

#include "libsemx/scaled_fixed_covariance.hpp"
#include "libsemx/multi_kernel_covariance.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace libsemx {

namespace {

using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct RandomEffectInfo {
    std::string id;
    std::string grouping_var;
    std::vector<std::string> design_vars;
    std::string covariance_id;
    const CovarianceSpec* cov_spec;
    std::vector<double> G_matrix;
    std::vector<std::vector<double>> G_gradients;
};

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
        throw std::runtime_error("Unsupported covariance structure: " + spec.structure);
    }
}

void cholesky(std::size_t n, const std::vector<double>& matrix, std::vector<double>& lower) {
    if (matrix.size() != n * n) {
        throw std::invalid_argument("Matrix size mismatch for Cholesky factorization");
    }
    Eigen::Map<const RowMajorMatrix> mat(matrix.data(), n, n);
    Eigen::LLT<RowMajorMatrix> llt(mat);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("Cholesky factorization failed");
    }
    RowMajorMatrix L = llt.matrixL();
    lower.resize(n * n);
    Eigen::Map<RowMajorMatrix>(lower.data(), n, n) = L;
}

double log_det_cholesky(std::size_t n, const std::vector<double>& lower) {
    if (lower.size() != n * n) {
        throw std::invalid_argument("Matrix size mismatch for log-det computation");
    }
    Eigen::Map<const RowMajorMatrix> L(lower.data(), n, n);
    double log_det = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double val = L(i, i);
        if (!(val > 0.0)) {
            throw std::runtime_error("Non-positive diagonal encountered in Cholesky factor");
        }
        log_det += std::log(val);
    }
    return 2.0 * log_det;
}

std::vector<double> solve_cholesky(std::size_t n, const std::vector<double>& lower, const std::vector<double>& rhs) {
    if (lower.size() != n * n || rhs.size() != n) {
        throw std::invalid_argument("Dimension mismatch in solve_cholesky");
    }
    Eigen::Map<const RowMajorMatrix> L(lower.data(), n, n);
    Eigen::Map<const Eigen::VectorXd> b(rhs.data(), n);
    Eigen::VectorXd y = L.triangularView<Eigen::Lower>().solve(b);
    Eigen::VectorXd x = L.transpose().triangularView<Eigen::Upper>().solve(y);
    std::vector<double> result(n);
    Eigen::Map<Eigen::VectorXd>(result.data(), n) = x;
    return result;
}

std::vector<double> invert_from_cholesky(std::size_t n, const std::vector<double>& lower) {
    if (lower.size() != n * n) {
        throw std::invalid_argument("Matrix size mismatch when forming inverse");
    }
    Eigen::Map<const RowMajorMatrix> L(lower.data(), n, n);
    RowMajorMatrix identity = RowMajorMatrix::Identity(n, n);
    RowMajorMatrix Linv = L.triangularView<Eigen::Lower>().solve(identity);
    RowMajorMatrix inv = Linv.transpose() * Linv;
    std::vector<double> result(n * n);
    Eigen::Map<RowMajorMatrix>(result.data(), n, n) = inv;
    return result;
}

std::vector<RandomEffectInfo> build_random_effect_infos(
    const ModelIR& model,
    const std::unordered_map<std::string, std::vector<double>>& covariance_parameters,
    const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data,
    bool need_gradients) {
    std::unordered_map<std::string, const CovarianceSpec*> cov_lookup;
    for (const auto& cov : model.covariances) {
        cov_lookup.emplace(cov.id, &cov);
    }

    std::vector<RandomEffectInfo> infos;
    infos.reserve(model.random_effects.size());

    for (const auto& re : model.random_effects) {
        if (re.variables.empty()) {
            throw std::runtime_error("Random effect " + re.id + " is missing grouping/design variables");
        }
        auto cov_it = cov_lookup.find(re.covariance_id);
        if (cov_it == cov_lookup.end()) {
            throw std::runtime_error("Covariance spec not found for random effect: " + re.covariance_id);
        }

        auto structure = create_covariance_structure(*cov_it->second, fixed_covariance_data);
        std::vector<double> params;
        if (structure->parameter_count() > 0) {
            auto param_it = covariance_parameters.find(re.covariance_id);
            if (param_it == covariance_parameters.end()) {
                throw std::runtime_error("Missing covariance parameters for: " + re.covariance_id);
            }
            params = param_it->second;
        }

        RandomEffectInfo info;
        info.id = re.id;
        info.grouping_var = re.variables.front();
        info.design_vars.assign(re.variables.begin() + 1, re.variables.end());
        info.covariance_id = re.covariance_id;
        info.cov_spec = cov_it->second;
        info.G_matrix = structure->materialize(params);
        if (need_gradients && structure->parameter_count() > 0) {
            info.G_gradients = structure->parameter_gradients(params);
        }
        infos.push_back(std::move(info));
    }

    return infos;
}

std::map<double, std::vector<std::size_t>> build_group_map(
    const RandomEffectInfo& info,
    const std::unordered_map<std::string, std::vector<double>>& data,
    std::size_t n) {
    auto it = data.find(info.grouping_var);
    if (it == data.end()) {
        throw std::runtime_error("Grouping data missing for variable: " + info.grouping_var);
    }
    if (it->second.size() != n) {
        throw std::runtime_error("Grouping variable " + info.grouping_var + " size mismatch");
    }
    std::map<double, std::vector<std::size_t>> groups;
    for (std::size_t i = 0; i < n; ++i) {
        groups[it->second[i]].push_back(i);
    }
    return groups;
}

const std::vector<double>& resolve_design_values(
    const std::string& name,
    const std::unordered_map<std::string, std::vector<double>>& data,
    const std::unordered_map<std::string, std::vector<double>>& linear_predictors) {
    if (auto d_it = data.find(name); d_it != data.end()) {
        return d_it->second;
    }
    if (auto lp_it = linear_predictors.find(name); lp_it != linear_predictors.end()) {
        return lp_it->second;
    }
    throw std::runtime_error("Design variable not found: " + name);
}

std::vector<double> build_design_matrix(
    const RandomEffectInfo& info,
    const std::vector<std::size_t>& indices,
    const std::unordered_map<std::string, std::vector<double>>& data,
    const std::unordered_map<std::string, std::vector<double>>& linear_predictors) {
    std::size_t n_i = indices.size();
    std::size_t q = info.cov_spec->dimension;
    std::vector<double> Z(n_i * q, 0.0);

    if (info.design_vars.empty()) {
        if (q != 1) {
            throw std::runtime_error("Random effect " + info.id + " expects " + std::to_string(q) +
                                     " design columns but none were specified");
        }
        for (std::size_t r = 0; r < n_i; ++r) {
            Z[r * q] = 1.0;
        }
        return Z;
    }

    if (info.design_vars.size() != q) {
        throw std::runtime_error("Design variable count does not match covariance dimension for " + info.id);
    }

    for (std::size_t col = 0; col < q; ++col) {
        const auto& values = resolve_design_values(info.design_vars[col], data, linear_predictors);
        for (std::size_t row = 0; row < n_i; ++row) {
            if (indices[row] >= values.size()) {
                throw std::runtime_error("Design variable " + info.design_vars[col] + " has insufficient data length");
            }
            Z[row * q + col] = values[indices[row]];
        }
    }
    return Z;
}

} // namespace

double LikelihoodDriver::evaluate_total_loglik(const std::vector<double>& observed,
                                               const std::vector<double>& linear_predictors,
                                               const std::vector<double>& dispersions,
                                               const OutcomeFamily& family,
                                               const std::vector<double>& status,
                                               const std::vector<double>& extra_params) const {
    if (observed.size() != linear_predictors.size() || observed.size() != dispersions.size()) {
        throw std::invalid_argument("all input vectors must have the same size");
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
    auto random_effect_infos = build_random_effect_infos(model, covariance_parameters, fixed_covariance_data, false);
    if (random_effect_infos.empty()) {
        throw std::runtime_error("Random effect metadata missing");
    }

    std::string obs_var_name;
    std::string obs_family_name;
    for (const auto& var : model.variables) {
        if (var.kind == VariableKind::Observed) {
            obs_var_name = var.name;
            obs_family_name = var.family;
        }
    }
    if (obs_var_name.empty()) {
        throw std::runtime_error("No observed variable found");
    }

    auto pred_it = linear_predictors.find(obs_var_name);
    if (pred_it == linear_predictors.end()) {
        if (linear_predictors.size() == 1) {
            obs_var_name = linear_predictors.begin()->first;
            auto var_it = std::find_if(model.variables.begin(), model.variables.end(), [&](const auto& v) {
                return v.name == obs_var_name;
            });
            if (var_it == model.variables.end()) {
                throw std::runtime_error("Outcome variable metadata not found: " + obs_var_name);
            }
            obs_family_name = var_it->family;
            pred_it = linear_predictors.begin();
        } else {
            throw std::runtime_error("Linear predictors for outcome variable not found");
        }
    }
    const auto& pred_data = pred_it->second;

    auto obs_it = data.find(obs_var_name);
    if (obs_it == data.end()) {
        throw std::runtime_error("Observed variable not found in data: " + obs_var_name);
    }
    const auto& obs_data = obs_it->second;

    if (obs_data.size() != pred_data.size()) {
        throw std::runtime_error("Outcome data and predictors have mismatched sizes");
    }
    std::size_t n = obs_data.size();

    auto disp_it = dispersions.find(obs_var_name);
    if (disp_it == dispersions.end()) {
        throw std::runtime_error("Dispersions not found for: " + obs_var_name);
    }
    const auto& disp_data = disp_it->second;
    if (disp_data.size() != n) {
        throw std::runtime_error("Dispersion vector has mismatched size");
    }

    const double log_2pi = std::log(2.0 * 3.14159265358979323846);

    // Identify fixed effect predictors for REML adjustments
    std::vector<std::string> fixed_effect_vars;
    if (method == EstimationMethod::REML) {
        for (const auto& edge : model.edges) {
            if (edge.kind == EdgeKind::Regression && edge.target == obs_var_name) {
                fixed_effect_vars.push_back(edge.source);
            }
        }
        std::sort(fixed_effect_vars.begin(), fixed_effect_vars.end());
    }

    if (obs_family_name == "gaussian") {
        std::vector<std::map<double, std::vector<std::size_t>>> group_maps;
        group_maps.reserve(random_effect_infos.size());
        for (const auto& info : random_effect_infos) {
            group_maps.push_back(build_group_map(info, data, n));
        }

        std::vector<double> V_full(n * n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            V_full[i * n + i] = disp_data[i];
        }

        for (std::size_t re_idx = 0; re_idx < random_effect_infos.size(); ++re_idx) {
            const auto& info = random_effect_infos[re_idx];
            const auto& groups = group_maps[re_idx];
            for (const auto& [_, indices] : groups) {
                auto Z_i = build_design_matrix(info, indices, data, linear_predictors);
                std::size_t n_i = indices.size();
                std::size_t q = info.cov_spec->dimension;

                std::vector<double> ZG(n_i * q, 0.0);
                for (std::size_t r = 0; r < n_i; ++r) {
                    for (std::size_t c = 0; c < q; ++c) {
                        double sum = 0.0;
                        for (std::size_t k = 0; k < q; ++k) {
                            sum += Z_i[r * q + k] * info.G_matrix[k * q + c];
                        }
                        ZG[r * q + c] = sum;
                    }
                }

                for (std::size_t r = 0; r < n_i; ++r) {
                    for (std::size_t c = 0; c < n_i; ++c) {
                        double sum = 0.0;
                        for (std::size_t k = 0; k < q; ++k) {
                            sum += ZG[r * q + k] * Z_i[c * q + k];
                        }
                        V_full[indices[r] * n + indices[c]] += sum;
                    }
                }
            }
        }

        std::vector<double> L(n * n);
        cholesky(n, V_full, L);
        double log_det = log_det_cholesky(n, L);

        std::vector<double> resid(n);
        for (std::size_t i = 0; i < n; ++i) {
            resid[i] = obs_data[i] - pred_data[i];
        }
        std::vector<double> alpha = solve_cholesky(n, L, resid);

        double quad_form = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            quad_form += resid[i] * alpha[i];
        }

        double total_loglik = -0.5 * (n * log_2pi + log_det + quad_form);

        if (method == EstimationMethod::REML && !fixed_effect_vars.empty()) {
            std::vector<double> V_inv = invert_from_cholesky(n, L);
            std::size_t p = fixed_effect_vars.size();
            std::vector<double> X(n * p);
            for (std::size_t j = 0; j < p; ++j) {
                auto it = data.find(fixed_effect_vars[j]);
                if (it == data.end() || it->second.size() != n) {
                    throw std::runtime_error("Missing data for fixed effect: " + fixed_effect_vars[j]);
                }
                for (std::size_t i = 0; i < n; ++i) {
                    X[i * p + j] = it->second[i];
                }
            }

            std::vector<double> Vinv_X(n * p, 0.0);
            for (std::size_t col = 0; col < p; ++col) {
                for (std::size_t row = 0; row < n; ++row) {
                    double sum = 0.0;
                    for (std::size_t k = 0; k < n; ++k) {
                        sum += V_inv[row * n + k] * X[k * p + col];
                    }
                    Vinv_X[row * p + col] = sum;
                }
            }

            std::vector<double> Xt_Vinv_X(p * p, 0.0);
            for (std::size_t r = 0; r < p; ++r) {
                for (std::size_t c = 0; c < p; ++c) {
                    double sum = 0.0;
                    for (std::size_t row = 0; row < n; ++row) {
                        sum += X[row * p + r] * Vinv_X[row * p + c];
                    }
                    Xt_Vinv_X[r * p + c] = sum;
                }
            }

            std::vector<double> L_reml(p * p);
            cholesky(p, Xt_Vinv_X, L_reml);
            double log_det_reml = log_det_cholesky(p, L_reml);
            total_loglik -= 0.5 * (p * log_2pi + log_det_reml);
        }

        return total_loglik;
    }

    if (random_effect_infos.size() != 1) {
        throw std::runtime_error("Non-Gaussian mixed models currently support only a single random effect");
    }

    auto outcome_family = OutcomeFamilyFactory::create(obs_family_name);
    const auto& info = random_effect_infos.front();
    std::size_t q = info.cov_spec->dimension;
    std::vector<double> L_G(q * q);
    double log_det_G = 0.0;
    std::vector<double> G_inv;
    try {
        cholesky(q, info.G_matrix, L_G);
        log_det_G = log_det_cholesky(q, L_G);
        G_inv = invert_from_cholesky(q, L_G);
    } catch (...) {
        return -std::numeric_limits<double>::infinity();
    }

    auto status_it = status.find(obs_var_name);
    const std::vector<double>* status_vec = nullptr;
    if (status_it != status.end()) {
        if (status_it->second.size() != n) {
            throw std::runtime_error("Status vector has mismatched size");
        }
        status_vec = &status_it->second;
    }

    auto group_map = build_group_map(info, data, n);
    double total_loglik = 0.0;

    for (const auto& [_, indices] : group_map) {
        std::size_t n_i = indices.size();
        std::vector<double> Z_i = build_design_matrix(info, indices, data, linear_predictors);
        std::vector<double> u_i(q, 0.0);

        for (int iter = 0; iter < 20; ++iter) {
            std::vector<double> grad(q, 0.0);
            std::vector<double> hess(q * q, 0.0);

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

            for (std::size_t k = 0; k < n_i; ++k) {
                std::size_t idx = indices[k];
                double eta = pred_data[idx];
                for (std::size_t j = 0; j < q; ++j) {
                    eta += Z_i[k * q + j] * u_i[j];
                }

                double s = status_vec ? (*status_vec)[idx] : 1.0;
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
            if (max_delta < 1e-6) {
                break;
            }
        }

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
            for (std::size_t j = 0; j < q; ++j) {
                eta += Z_i[k * q + j] * u_i[j];
            }

            double s = status_vec ? (*status_vec)[idx] : 1.0;
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
    (void)method;

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
        auto random_effect_infos = build_random_effect_infos(model, covariance_parameters, fixed_covariance_data, true);
        if (random_effect_infos.empty()) {
            throw std::runtime_error("Random effect metadata missing");
        }

        std::string obs_var_name;
        std::string obs_family_name;
        for (const auto& var : model.variables) {
            if (var.kind == VariableKind::Observed) {
                obs_var_name = var.name;
                obs_family_name = var.family;
            }
        }
        if (obs_var_name.empty()) {
            throw std::runtime_error("No observed variable found");
        }

        auto pred_it = linear_predictors.find(obs_var_name);
        if (pred_it == linear_predictors.end()) {
            if (linear_predictors.size() == 1) {
                obs_var_name = linear_predictors.begin()->first;
                auto it = std::find_if(model.variables.begin(), model.variables.end(), [&](const auto& v) {
                    return v.name == obs_var_name;
                });
                if (it == model.variables.end()) {
                    throw std::runtime_error("Outcome variable metadata not found: " + obs_var_name);
                }
                obs_family_name = it->family;
                pred_it = linear_predictors.begin();
            } else {
                throw std::runtime_error("Linear predictors for outcome variable not found");
            }
        }

        if (obs_family_name != "gaussian") {
            throw std::runtime_error("Analytic gradients only implemented for Gaussian mixed models");
        }

        auto obs_it = data.find(obs_var_name);
        if (obs_it == data.end()) {
            throw std::runtime_error("Observed variable not found in data: " + obs_var_name);
        }
        const auto& obs_data = obs_it->second;
        const auto& pred_data = pred_it->second;
        if (obs_data.size() != pred_data.size()) {
            throw std::runtime_error("Outcome data and predictors have mismatched sizes");
        }
        std::size_t n = obs_data.size();

        auto disp_it = dispersions.find(obs_var_name);
        if (disp_it == dispersions.end()) {
            throw std::runtime_error("Dispersions not found for: " + obs_var_name);
        }
        const auto& disp_data = disp_it->second;
        if (disp_data.size() != n) {
            throw std::runtime_error("Dispersion vector has mismatched size");
        }

        std::vector<std::map<double, std::vector<std::size_t>>> group_maps;
        group_maps.reserve(random_effect_infos.size());
        for (const auto& info : random_effect_infos) {
            group_maps.push_back(build_group_map(info, data, n));
        }

        std::vector<double> V_full(n * n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            V_full[i * n + i] = disp_data[i];
        }

        for (std::size_t re_idx = 0; re_idx < random_effect_infos.size(); ++re_idx) {
            const auto& info = random_effect_infos[re_idx];
            const auto& groups = group_maps[re_idx];
            for (const auto& [_, indices] : groups) {
                auto Z_i = build_design_matrix(info, indices, data, linear_predictors);
                std::size_t n_i = indices.size();
                std::size_t q = info.cov_spec->dimension;

                std::vector<double> ZG(n_i * q, 0.0);
                for (std::size_t r = 0; r < n_i; ++r) {
                    for (std::size_t c = 0; c < q; ++c) {
                        double sum = 0.0;
                        for (std::size_t k = 0; k < q; ++k) {
                            sum += Z_i[r * q + k] * info.G_matrix[k * q + c];
                        }
                        ZG[r * q + c] = sum;
                    }
                }

                for (std::size_t r = 0; r < n_i; ++r) {
                    for (std::size_t c = 0; c < n_i; ++c) {
                        double sum = 0.0;
                        for (std::size_t k = 0; k < q; ++k) {
                            sum += ZG[r * q + k] * Z_i[c * q + k];
                        }
                        V_full[indices[r] * n + indices[c]] += sum;
                    }
                }
            }
        }

        std::vector<double> L(n * n);
        cholesky(n, V_full, L);
        std::vector<double> V_inv = invert_from_cholesky(n, L);

        std::vector<double> resid(n);
        for (std::size_t i = 0; i < n; ++i) {
            resid[i] = obs_data[i] - pred_data[i];
        }
        std::vector<double> alpha = solve_cholesky(n, L, resid);

        for (const auto& edge : model.edges) {
            if (edge.kind != EdgeKind::Regression || edge.target != obs_var_name || edge.parameter_id.empty()) {
                continue;
            }
            if (!data.count(edge.source)) {
                continue;
            }
            const auto& src_vec = data.at(edge.source);
            double accum = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                accum += src_vec[i] * alpha[i];
            }
            gradients[edge.parameter_id] += accum;
        }

        for (std::size_t re_idx = 0; re_idx < random_effect_infos.size(); ++re_idx) {
            const auto& info = random_effect_infos[re_idx];
            if (info.G_gradients.empty()) continue;
            const auto& groups = group_maps[re_idx];
            for (const auto& [_, indices] : groups) {
                auto Z_i = build_design_matrix(info, indices, data, linear_predictors);
                std::size_t n_i = indices.size();
                std::size_t q = info.cov_spec->dimension;

                std::vector<double> temp(n_i * q, 0.0);
                for (std::size_t r = 0; r < n_i; ++r) {
                    for (std::size_t c = 0; c < q; ++c) {
                        double sum = 0.0;
                        for (std::size_t k = 0; k < n_i; ++k) {
                            std::size_t row = indices[r];
                            std::size_t col = indices[k];
                            double term = alpha[row] * alpha[col] - V_inv[row * n + col];
                            sum += term * Z_i[k * q + c];
                        }
                        temp[r * q + c] = sum;
                    }
                }

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

                for (std::size_t k = 0; k < info.G_gradients.size(); ++k) {
                    const auto& dG = info.G_gradients[k];
                    double trace = 0.0;
                    for (std::size_t r = 0; r < q; ++r) {
                        for (std::size_t c = 0; c < q; ++c) {
                            trace += M[r * q + c] * dG[c * q + r];
                        }
                    }
                    std::string param_name = info.covariance_id + "_" + std::to_string(k);
                    gradients[param_name] += 0.5 * trace;
                }
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
