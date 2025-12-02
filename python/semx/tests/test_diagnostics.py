import pytest
import numpy as np
import pandas as pd
from _libsemx import FitResult, OptimizationResult, OptimizationOptions
from semx import Model, SemFit

def test_diagnostics_simple_cfa():
    # Generate data for a simple CFA: F -> y1, y2, y3
    np.random.seed(42)
    n = 200
    F = np.random.normal(0, 1, n)
    y1 = 1.0 * F + np.random.normal(0, 0.6, n)
    y2 = 0.8 * F + np.random.normal(0, 0.6, n)
    y3 = 0.6 * F + np.random.normal(0, 0.6, n)
    
    data = pd.DataFrame({"y1": y1, "y2": y2, "y3": y3})
    
    # Specify model
    model = Model(
        equations=[
            "F =~ y1 + y2 + y3",
            "F ~~ F",
            "y1 ~~ y1",
            "y2 ~~ y2",
            "y3 ~~ y3"
        ],
        families={"y1": "gaussian", "y2": "gaussian", "y3": "gaussian"}
    )
    
    # Manually construct SemFit with "true" parameters to bypass estimation issues
    # We need to know parameter order.
    ir = model.to_ir()
    param_names = [p.id for p in ir.parameters]
    
    # Map names to values
    # y1 loading is fixed to 1.0 (not in parameters)
    # y2 loading: 0.8
    # y3 loading: 0.6
    # F variance: 1.0
    # Residual variances: 0.6^2 = 0.36
    
    true_values = {}
    for name in param_names:
        if "lambda" in name:
            if "y2" in name: true_values[name] = 0.8
            elif "y3" in name: true_values[name] = 0.6
            else: true_values[name] = 0.0 # Should not happen if y1 fixed
        elif "psi" in name:
            if "F_F" in name: true_values[name] = 1.0
            else: true_values[name] = 0.36
            
    param_vec = [true_values.get(name, 0.0) for name in param_names]
    
    # Create FitResult
    opt_res = OptimizationResult()
    opt_res.parameters = param_vec
    fit_res = FitResult()
    fit_res.optimization_result = opt_res
    
    # Compute sample stats manually as Model.fit does
    means = data.mean().values.tolist()
    cov_df = data.cov()
    
    # We need full stats (including latent F as 0)
    all_vars = [v.name for v in ir.variables]
    var_to_idx = {name: i for i, name in enumerate(all_vars)}
    n_total = len(all_vars)
    
    full_cov = np.zeros((n_total, n_total))
    full_means = np.zeros(n_total)
    
    observed_vars = ["y1", "y2", "y3"]
    for i, name in enumerate(observed_vars):
        idx = var_to_idx[name]
        full_means[idx] = means[i]
        
    cov_values = cov_df.values
    for r_i, r_name in enumerate(observed_vars):
        r_idx = var_to_idx[r_name]
        for c_i, c_name in enumerate(observed_vars):
            c_idx = var_to_idx[c_name]
            full_cov[r_idx, c_idx] = cov_values[r_i, c_i]
            
    sample_stats = {
        "means": full_means.tolist(),
        "covariance": full_cov.flatten().tolist(),
        "n_obs": n
    }
    
    fit = SemFit(ir, fit_res, sample_stats)
    
    # Check standardized estimates
    std = fit.standardized_solution()
    assert len(std.edges) > 0
    
    # Check diagnostics
    diag = fit.diagnostics()
    # With true parameters, fit should be good (SRMR small)
    # Sampling error exists, but n=200 is decent.
    assert diag.srmr < 0.2 
    
    # Check modification indices
    mis = fit.modification_indices()
    assert isinstance(mis, list)

def test_modification_indices_misspecified():
    # Generate data with a cross-loading or correlation
    # F -> y1, y2
    # y3 = 0.5 * F + noise
    np.random.seed(42)
    n = 1000 # Increase N to stabilize MI
    F = np.random.normal(0, 1, n)
    y1 = 1.0 * F + np.random.normal(0, 0.5, n)
    y2 = 0.8 * F + np.random.normal(0, 0.5, n)
    y3 = 0.5 * F + np.random.normal(0, 0.5, n) 
    
    data = pd.DataFrame({"y1": y1, "y2": y2, "y3": y3})
    
    # Misspecified model: F -> y1, y2 (y3 is independent)
    model = Model(
        equations=[
            "F =~ y1 + y2",
            "F ~~ F",
            "y1 ~~ y1",
            "y2 ~~ y2",
            "y3 ~~ y3" 
        ],
        families={"y1": "gaussian", "y2": "gaussian", "y3": "gaussian"}
    )
    
    ir = model.to_ir()
    param_names = [p.id for p in ir.parameters]
    
    # Set parameters to "best fit" for misspecified model
    # F -> y1 (fixed 1.0), F -> y2 (0.8)
    # Var(F) = 1.0
    # Var(y1) = 0.25, Var(y2) = 0.25
    # y3 is independent. Var(y3) = Var(0.5*F + e) = 0.25 + 0.25 = 0.5
    
    true_values = {}
    for name in param_names:
        if "lambda" in name:
            if "y2" in name: true_values[name] = 0.8
            else: true_values[name] = 0.0
        elif "psi" in name:
            if "F_F" in name: true_values[name] = 1.0
            elif "y3_y3" in name: true_values[name] = 0.5
            else: true_values[name] = 0.25
            
    param_vec = [true_values.get(name, 0.0) for name in param_names]
    
    opt_res = OptimizationResult()
    opt_res.parameters = param_vec
    fit_res = FitResult()
    fit_res.optimization_result = opt_res
    
    # Compute sample stats
    means = data.mean().values.tolist()
    cov_df = data.cov()
    
    all_vars = [v.name for v in ir.variables]
    var_to_idx = {name: i for i, name in enumerate(all_vars)}
    n_total = len(all_vars)
    
    full_cov = np.zeros((n_total, n_total))
    full_means = np.zeros(n_total)
    
    observed_vars = ["y1", "y2", "y3"]
    for i, name in enumerate(observed_vars):
        idx = var_to_idx[name]
        full_means[idx] = means[i]
        
    cov_values = cov_df.values
    for r_i, r_name in enumerate(observed_vars):
        r_idx = var_to_idx[r_name]
        for c_i, c_name in enumerate(observed_vars):
            c_idx = var_to_idx[c_name]
            full_cov[r_idx, c_idx] = cov_values[r_i, c_i]
            
    sample_stats = {
        "means": full_means.tolist(),
        "covariance": full_cov.flatten().tolist(),
        "n_obs": n
    }
    
    fit = SemFit(ir, fit_res, sample_stats)
    
    # Should have bad fit
    diag = fit.diagnostics()
    assert diag.srmr > 0.05
    
    # MI should suggest F -> y3
    mis = fit.modification_indices()
    
    found = False
    for mi in mis:
        if mi.source == "F" and mi.target == "y3" and mi.kind.name == "Loading":
            found = True
            assert mi.mi > 10.0
            # EPC should be around 0.5
            assert abs(mi.epc - 0.5) < 0.2
            break
            
    assert found
