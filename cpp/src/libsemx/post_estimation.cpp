#include "libsemx/post_estimation.hpp"
#include "libsemx/covariance_structure.hpp"
#include "libsemx/parameter_transform.hpp"

#include <Eigen/Dense>
#include <unordered_map>
#include <set>
#include <iostream>
#include <cmath>

namespace libsemx {

namespace {

double get_parameter_value(const std::string& param_id,
                           const std::unordered_map<std::string, double>& param_map) {
    if (param_id.empty()) return 0.0;
    
    // Try to look up in map
    auto it = param_map.find(param_id);
    if (it != param_map.end()) {
        return it->second;
    }
    
    // Try to parse as number
    char* end;
    double val = std::strtod(param_id.c_str(), &end);
    if (end != param_id.c_str() && *end == '\0') {
        return val;
    }
    
    return 0.0; // Should not happen if validated
}

struct ImpliedMatrices {
    Eigen::MatrixXd Sigma;
    Eigen::MatrixXd A;
    Eigen::MatrixXd S;
    Eigen::VectorXd Mu;
    std::unordered_map<std::string, int> var_idx;
    std::vector<VariableKind> var_kinds;
    bool valid{false};
};

ImpliedMatrices compute_implied_matrices_internal(
    const ModelIR& model,
    const std::vector<std::string>& parameter_names,
    const std::vector<double>& parameter_values,
    const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data)
{
    ImpliedMatrices result;
    
    // 1. Map parameters
    std::unordered_map<std::string, double> param_map;
    for (size_t i = 0; i < parameter_names.size() && i < parameter_values.size(); ++i) {
        param_map[parameter_names[i]] = parameter_values[i];
    }

    // 2. Map variables to indices
    int n = 0;
    for (const auto& var : model.variables) {
        result.var_idx[var.name] = n++;
        result.var_kinds.push_back(var.kind);
    }

    // 3. Construct matrices
    result.A = Eigen::MatrixXd::Zero(n, n);
    result.S = Eigen::MatrixXd::Zero(n, n);
    result.Mu = Eigen::VectorXd::Zero(n); // Assuming zero means for now if not modeled

    // Edges
    for (const auto& edge : model.edges) {
        if (result.var_idx.find(edge.source) == result.var_idx.end() || result.var_idx.find(edge.target) == result.var_idx.end()) {
            continue;
        }
        int i = result.var_idx[edge.source];
        int j = result.var_idx[edge.target];
        double val = get_parameter_value(edge.parameter_id, param_map);

        if (edge.kind == EdgeKind::Regression || edge.kind == EdgeKind::Loading) {
            // target = val * source + ... => A[target, source] = val
            result.A(j, i) += val;
        } else if (edge.kind == EdgeKind::Covariance) {
            result.S(i, j) += val;
            if (i != j) {
                result.S(j, i) += val;
            }
        }
    }

    // Covariance Structures (Random Effects)
    std::unordered_map<std::string, const CovarianceSpec*> cov_specs;
    for (const auto& cov : model.covariances) {
        cov_specs[cov.id] = &cov;
    }

    for (const auto& re : model.random_effects) {
        if (cov_specs.find(re.covariance_id) == cov_specs.end()) continue;
        const auto& spec = *cov_specs[re.covariance_id];
        
        auto structure = create_covariance_structure(spec, fixed_covariance_data);
        size_t count = structure->parameter_count();
        
        std::vector<double> cov_params;
        cov_params.reserve(count);
        
        for (size_t k = 0; k < count; ++k) {
            std::string pname = spec.id + "_" + std::to_string(k);
            if (param_map.count(pname)) {
                cov_params.push_back(param_map[pname]);
            } else {
                cov_params.push_back(0.0); 
            }
        }
        
        std::vector<int> re_indices;
        for (const auto& vname : re.variables) {
            if (result.var_idx.count(vname)) {
                re_indices.push_back(result.var_idx[vname]);
            }
        }
        
        if (re_indices.empty()) continue;

        std::vector<double> cov_matrix_flat = structure->materialize(cov_params);
        size_t dim = structure->dimension();
        
        if (cov_matrix_flat.size() != dim * dim) continue;
        
        for (size_t r = 0; r < dim; ++r) {
            for (size_t c = 0; c < dim; ++c) {
                if (r < re_indices.size() && c < re_indices.size()) {
                    int global_r = re_indices[r];
                    int global_c = re_indices[c];
                    result.S(global_r, global_c) += cov_matrix_flat[r * dim + c];
                }
            }
        }
    }
    
    // 4. Solve for Sigma
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd I_minus_A = I - result.A;
    
    Eigen::FullPivLU<Eigen::MatrixXd> lu(I_minus_A);
    if (!lu.isInvertible()) {
        result.valid = false;
        return result;
    }
    Eigen::MatrixXd Inv = lu.inverse();
    result.Sigma = Inv * result.S * Inv.transpose();
    
    // Implied Means: Mu = (I-A)^-1 * alpha (intercepts)
    // Currently alpha is zero, so Mu is zero.
    
    result.valid = true;
    return result;
}

} // namespace

StandardizedSolution compute_standardized_estimates(
    const ModelIR& model,
    const std::vector<std::string>& parameter_names,
    const std::vector<double>& parameter_values,
    const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data)
{
    auto matrices = compute_implied_matrices_internal(model, parameter_names, parameter_values, fixed_covariance_data);
    
    StandardizedSolution solution;
    if (!matrices.valid) return solution;
    
    solution.edges.reserve(model.edges.size());
    
    Eigen::VectorXd variances = matrices.Sigma.diagonal();
    Eigen::VectorXd std_devs = variances.cwiseSqrt();
    
    // Need param_map again for values
    std::unordered_map<std::string, double> param_map;
    for (size_t i = 0; i < parameter_names.size() && i < parameter_values.size(); ++i) {
        param_map[parameter_names[i]] = parameter_values[i];
    }

    for (const auto& edge : model.edges) {
        StandardizedEdgeResult res;
        res.std_lv = 0.0;
        res.std_all = 0.0;
        
        if (matrices.var_idx.find(edge.source) == matrices.var_idx.end() || 
            matrices.var_idx.find(edge.target) == matrices.var_idx.end()) {
            solution.edges.push_back(res);
            continue;
        }
        
        int i = matrices.var_idx[edge.source];
        int j = matrices.var_idx[edge.target];
        double val = get_parameter_value(edge.parameter_id, param_map);
        
        double sd_source = std_devs(i);
        double sd_target = std_devs(j);
        
        bool source_latent = (matrices.var_kinds[i] == VariableKind::Latent);
        bool target_latent = (matrices.var_kinds[j] == VariableKind::Latent);
        
        if (edge.kind == EdgeKind::Regression || edge.kind == EdgeKind::Loading) {
            res.std_lv = val * (source_latent ? sd_source : 1.0);
            if (sd_target > 1e-9) {
                res.std_all = val * sd_source / sd_target;
            }
        } else if (edge.kind == EdgeKind::Covariance) {
            if (i == j) {
                double scale_lv = (source_latent ? (sd_source*sd_source) : 1.0);
                if (scale_lv > 1e-9) res.std_lv = val / scale_lv;
                if (variances(i) > 1e-9) res.std_all = val / variances(i);
            } else {
                double scale_src_lv = (source_latent ? sd_source : 1.0);
                double scale_tgt_lv = (target_latent ? sd_target : 1.0);
                if (scale_src_lv > 1e-9 && scale_tgt_lv > 1e-9) {
                    res.std_lv = val / (scale_src_lv * scale_tgt_lv);
                }
                if (sd_source > 1e-9 && sd_target > 1e-9) {
                    res.std_all = val / (sd_source * sd_target);
                }
            }
        }
        solution.edges.push_back(res);
    }
    
    return solution;
}

ModelDiagnostics compute_model_diagnostics(
    const ModelIR& model,
    const std::vector<std::string>& parameter_names,
    const std::vector<double>& parameter_values,
    const std::vector<double>& sample_means,
    const std::vector<double>& sample_covariance,
    const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data)
{
    ModelDiagnostics diag;
    auto matrices = compute_implied_matrices_internal(model, parameter_names, parameter_values, fixed_covariance_data);
    
    if (!matrices.valid) return diag;
    
    int n = matrices.Sigma.rows();
    
    // Convert Eigen types to std::vector
    diag.implied_means.resize(n);
    Eigen::VectorXd::Map(&diag.implied_means[0], n) = matrices.Mu;
    
    diag.implied_covariance.resize(n * n);
    for(int r=0; r<n; ++r) {
        for(int c=0; c<n; ++c) {
            diag.implied_covariance[r*n + c] = matrices.Sigma(r, c);
        }
    }
    
    // Residuals
    diag.mean_residuals.resize(n);
    diag.covariance_residuals.resize(n * n);
    diag.correlation_residuals.resize(n * n);
    
    double sum_sq_resid = 0.0;
    double sum_sq_obs = 0.0;
    
    for(int i=0; i<n; ++i) {
        if (i < (int)sample_means.size()) {
            diag.mean_residuals[i] = sample_means[i] - matrices.Mu(i);
        } else {
            diag.mean_residuals[i] = 0.0; 
        }
    }
    
    for(int r=0; r<n; ++r) {
        for(int c=0; c<n; ++c) {
            double obs = 0.0;
            if (r*n + c < (int)sample_covariance.size()) {
                obs = sample_covariance[r*n + c];
            }
            double imp = matrices.Sigma(r, c);
            double resid = obs - imp;
            
            diag.covariance_residuals[r*n + c] = resid;
            
            double obs_var_r = (r*n + r < (int)sample_covariance.size()) ? sample_covariance[r*n + r] : 1.0;
            double obs_var_c = (c*n + c < (int)sample_covariance.size()) ? sample_covariance[c*n + c] : 1.0;
            
            double denom = std::sqrt(obs_var_r * obs_var_c);
            if (denom > 1e-9) {
                diag.correlation_residuals[r*n + c] = resid / denom;
            } else {
                diag.correlation_residuals[r*n + c] = 0.0;
            }
            
            // SRMR accumulation (lower triangle + diagonal)
            if (r >= c) {
                // Check if both are observed
                if (matrices.var_kinds[r] == VariableKind::Observed && 
                    matrices.var_kinds[c] == VariableKind::Observed) {
                    double corr_resid = diag.correlation_residuals[r*n + c];
                    sum_sq_resid += corr_resid * corr_resid;
                    sum_sq_obs += 1.0;
                }
            }
        }
    }
    
    if (sum_sq_obs > 0) {
        diag.srmr = std::sqrt(sum_sq_resid / sum_sq_obs);
    } else {
        diag.srmr = 0.0;
    }
    
    return diag;
}

std::vector<ModificationIndex> compute_modification_indices(
    const ModelIR& model,
    const std::vector<std::string>& parameter_names,
    const std::vector<double>& parameter_values,
    const std::vector<double>& sample_covariance,
    size_t sample_size,
    const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data)
{
    std::vector<ModificationIndex> indices;
    auto matrices = compute_implied_matrices_internal(model, parameter_names, parameter_values, fixed_covariance_data);
    
    if (!matrices.valid) return indices;
    
    int n = matrices.Sigma.rows();
    
    // 1. Extract Observed Submatrices
    std::vector<int> obs_indices;
    for(int i=0; i<n; ++i) {
        if (matrices.var_kinds[i] == VariableKind::Observed) {
            obs_indices.push_back(i);
        }
    }
    
    int n_obs = obs_indices.size();
    Eigen::MatrixXd Sigma_obs(n_obs, n_obs);
    Eigen::MatrixXd S_obs(n_obs, n_obs);
    
    for(int r=0; r<n_obs; ++r) {
        for(int c=0; c<n_obs; ++c) {
            Sigma_obs(r, c) = matrices.Sigma(obs_indices[r], obs_indices[c]);
            
            // S_obs from sample_covariance
            // sample_covariance is flattened n x n (with latents as 0)
            int orig_r = obs_indices[r];
            int orig_c = obs_indices[c];
            if (orig_r*n + orig_c < (int)sample_covariance.size()) {
                S_obs(r, c) = sample_covariance[orig_r*n + orig_c];
            } else {
                S_obs(r, c) = 0.0;
            }
        }
    }
    
    // 2. Compute Omega_obs
    Eigen::FullPivLU<Eigen::MatrixXd> lu_obs(Sigma_obs);
    if (!lu_obs.isInvertible()) return indices;
    Eigen::MatrixXd SigmaInv_obs = lu_obs.inverse();
    
    Eigen::MatrixXd Omega_obs = SigmaInv_obs * (Sigma_obs - S_obs) * SigmaInv_obs;
    
    // 3. Embed into full Omega (n x n)
    Eigen::MatrixXd Omega = Eigen::MatrixXd::Zero(n, n);
    for(int r=0; r<n_obs; ++r) {
        for(int c=0; c<n_obs; ++c) {
            Omega(obs_indices[r], obs_indices[c]) = Omega_obs(r, c);
        }
    }
    
    // Compute T = (I - A)^-1
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd I_minus_A = I - matrices.A;
    Eigen::FullPivLU<Eigen::MatrixXd> lu_A(I_minus_A);
    Eigen::MatrixXd T = lu_A.inverse();
    
    // Precompute matrices for gradient calculation
    Eigen::MatrixXd M = matrices.Sigma * Omega * T;
    
    // For S_ij (covariance): dSigma/dS_ij = T (Delta_ij + Delta_ji) T^T (if i != j)
    //                                     = T Delta_ii T^T (if i == j)
    // Gradient g_ij = tr( Omega T (Delta_ij + Delta_ji) T^T )
    //               = tr( T^T Omega T (Delta_ij + Delta_ji) )
    // Let Q = T^T Omega T.
    // If i != j: tr( Q Delta_ij ) + tr( Q Delta_ji ) = Q_ji + Q_ij = 2 Q_ij (since Q symmetric)
    // If i == j: tr( Q Delta_ii ) = Q_ii
    
    Eigen::MatrixXd Q = T.transpose() * Omega * T;
    
    // Identify existing edges to skip
    std::set<std::pair<int, int>> existing_A;
    std::set<std::pair<int, int>> existing_S;
    
    for (const auto& edge : model.edges) {
        if (matrices.var_idx.find(edge.source) == matrices.var_idx.end() || 
            matrices.var_idx.find(edge.target) == matrices.var_idx.end()) continue;
            
        int i = matrices.var_idx[edge.source];
        int j = matrices.var_idx[edge.target];
        
        if (edge.kind == EdgeKind::Regression || edge.kind == EdgeKind::Loading) {
            existing_A.insert({j, i}); // A(target, source)
        } else if (edge.kind == EdgeKind::Covariance) {
            if (i > j) std::swap(i, j);
            existing_S.insert({i, j});
        }
    }
    
    // Precompute Z = T^T * SigmaInv * T
    // SigmaInv should be the inverse of the observed part, padded with zeros
    Eigen::MatrixXd SigmaInv_padded = Eigen::MatrixXd::Zero(n, n);
    for(int r=0; r<n_obs; ++r) {
        for(int c=0; c<n_obs; ++c) {
            SigmaInv_padded(obs_indices[r], obs_indices[c]) = SigmaInv_obs(r, c);
        }
    }
    
    Eigen::MatrixXd Z = T.transpose() * SigmaInv_padded * T;
    
    // Iterate over all possible missing edges
    // 1. Regressions / Loadings (A matrix)
    // Target (row) <- Source (col)
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            if (r == c) continue; // No self-loops in A
            if (existing_A.count({r, c})) continue;
            
            double g = 2.0 * M(c, r);
            double h = 2.0 * ( (T(c, r) * T(c, r)) + matrices.Sigma(c, c) * Z(r, r) );
            
            if (h > 1e-9) {
                double mi = (sample_size - 1) * (g * g) / h;
                double epc = -g / h;
                
                if (mi > 3.84) { // Chi-square 1df p=0.05
                    ModificationIndex idx;
                    // Find names
                    for(auto& kv : matrices.var_idx) {
                        if (kv.second == c) idx.source = kv.first;
                        if (kv.second == r) idx.target = kv.first;
                    }
                    idx.kind = EdgeKind::Regression; 
                    
                    if (matrices.var_kinds[c] == VariableKind::Latent && 
                        matrices.var_kinds[r] == VariableKind::Observed) {
                        idx.kind = EdgeKind::Loading;
                    }
                    
                    idx.mi = mi;
                    idx.epc = epc;
                    idx.gradient = g;
                    indices.push_back(idx);
                }
            }
        }
    }
    
    // 2. Covariances (S matrix)
    // Symmetric, so iterate r <= c.
    // We look for missing covariances S_rc.
    
    for (int r = 0; r < n; ++r) {
        for (int c = r; c < n; ++c) {
            if (existing_S.count({r, c})) continue;
            
            // Gradient g_rc = 2 * Q_rc (if r != c), else Q_rr
            double g = (r == c) ? Q(r, c) : 2.0 * Q(r, c);
            
            // Hessian H_rc,rc
            // For r != c: H = 2 * (Z_rc^2 + Z_rr * Z_cc)
            // For r == c:
            // dSigma/dS_rr = T Delta_rr T^T
            // Y = L Delta_rr T^T
            // Y_uv = L_ur T_vr
            // tr(Y^2) = Sum_uv L_ur T_vr L_vr T_ur
            // Sum_u L_ur T_ur = Z_rr
            // Sum_v T_vr L_vr = Z_rr
            // H = Z_rr * Z_rr
            
            double h = 0.0;
            if (r == c) {
                h = Z(r, r) * Z(r, r);
            } else {
                h = 2.0 * (Z(r, c) * Z(r, c) + Z(r, r) * Z(c, c));
            }
            
            if (h > 1e-9) {
                double mi = (sample_size - 1) * (g * g) / h;
                double epc = -g / h;
                
                if (mi > 3.84) {
                    ModificationIndex idx;
                    for(auto& kv : matrices.var_idx) {
                        if (kv.second == r) idx.source = kv.first;
                        if (kv.second == c) idx.target = kv.first;
                    }
                    idx.kind = EdgeKind::Covariance;
                    idx.mi = mi;
                    idx.epc = epc;
                    idx.gradient = g;
                    indices.push_back(idx);
                }
            }
        }
    }
    
    return indices;
}

} // namespace libsemx
