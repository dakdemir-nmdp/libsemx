
import pytest
import pandas as pd
import numpy as np
import semx

def test_variance_components_random_intercept():
    # Simulate data
    np.random.seed(42)
    n_groups = 20
    n_per_group = 10
    N = n_groups * n_per_group
    
    groups = np.repeat(np.arange(n_groups), n_per_group)
    group_effects = np.random.randn(n_groups) * 2.0 # SD = 2.0, Var = 4.0
    
    x = np.random.randn(N)
    y = 1.0 + 0.5 * x + group_effects[groups] + np.random.randn(N) * 1.0 # Residual SD = 1.0
    
    df = pd.DataFrame({"x": x, "y": y, "group": groups})
    
    model = semx.Model(
        ["y ~ x + (1 | group)"],
        families={"y": "gaussian", "x": "gaussian"}
    )
    
    fit = model.fit(df)
    assert fit.optimization_result.converged
    
    print("FitResult attributes:", dir(fit.fit_result))
    if hasattr(fit.fit_result, "covariance_matrices"):
        print("Covariance Matrices:", fit.fit_result.covariance_matrices)

    vc = fit.variance_components()
    print(vc)
    
    assert not vc.empty
    assert "Group" in vc.columns
    assert "Variance" in vc.columns
    assert "Std.Dev" in vc.columns
    
    # Check random intercept
    # Group should be "group"
    # Name1 should be "(Intercept)" for implicit intercepts
    
    ri_row = vc[(vc["Group"] == "group") & (vc["Name1"] == "(Intercept)")]
    assert len(ri_row) == 1
    
    est_sd = ri_row.iloc[0]["Std.Dev"]
    assert abs(est_sd - 2.0) < 0.5 # Rough check

def test_variance_components_random_slope():
    # Simulate data
    np.random.seed(42)
    n_groups = 50
    n_per_group = 20
    N = n_groups * n_per_group
    
    groups = np.repeat(np.arange(n_groups), n_per_group)
    
    # Random effects: Intercept and Slope
    # Covariance matrix: [[1.0, 0.5], [0.5, 1.0]]
    # SDs = 1.0, Corr = 0.5
    
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    re = np.random.multivariate_normal([0, 0], cov, size=n_groups)
    
    x = np.random.randn(N)
    
    y = np.zeros(N)
    for i in range(N):
        g = groups[i]
        y[i] = (1.0 + re[g, 0]) + (0.5 + re[g, 1]) * x[i] + np.random.randn() * 0.5
        
    df = pd.DataFrame({"x": x, "y": y, "group": groups})
    
    model = semx.Model(
        ["y ~ x + (1 + x | group)"],
        families={"y": "gaussian", "x": "gaussian"}
    )
    
    fit = model.fit(df)
    assert fit.optimization_result.converged
    
    vc = fit.variance_components()
    print(vc)
    
    # Check dimensions
    # 2x2 matrix -> 3 rows (Var1, Var2, Cov12)
    assert len(vc) == 3
    
    # Check correlation
    # Find row with Name1="_intercept" and Name2="x" (or vice versa)
    corr_row = vc[((vc["Name1"] == "_intercept") & (vc["Name2"] == "x")) | 
                  ((vc["Name1"] == "x") & (vc["Name2"] == "_intercept"))]
    
    assert len(corr_row) == 1
    est_corr = corr_row.iloc[0]["Corr"]
    assert abs(est_corr - 0.5) < 0.2
