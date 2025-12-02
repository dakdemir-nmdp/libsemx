import pytest
import pandas as pd
import numpy as np
import semx

def test_fit_indices_cfa():
    # Simulate simple CFA data
    np.random.seed(42)
    N = 500
    # True model: f1 -> x1, x2, x3
    f1 = np.random.randn(N)
    x1 = 0.8 * f1 + np.random.randn(N) * 0.6
    x2 = 0.7 * f1 + np.random.randn(N) * 0.7
    x3 = 0.6 * f1 + np.random.randn(N) * 0.8
    
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    
    model = semx.Model(
        [
            "f1 =~ x1 + x2 + x3", 
            "f1 ~~ f1",
            "x1 ~~ x1", "x2 ~~ x2", "x3 ~~ x3",
            "x1 ~ 1", "x2 ~ 1", "x3 ~ 1"
        ],
        families={"x1": "gaussian", "x2": "gaussian", "x3": "gaussian"}
    )
    fit = model.fit(df)
    
    assert fit.optimization_result.converged
    
    indices = fit.fit_indices
    print(indices)
    
    assert not np.isnan(indices["chisq"])
    assert not np.isnan(indices["df"])
    assert not np.isnan(indices["cfi"])
    assert not np.isnan(indices["tli"])
    assert not np.isnan(indices["rmsea"])
    assert not np.isnan(indices["srmr"])
    
    # Check reasonable values for a true model
    # Note: 3 indicators is saturated for covariance (6 moments vs 3 loadings + 3 residuals = 6 params)
    # But we fix one loading to 1.0.
    # Params: 
    # Loadings: lam2, lam3 (lam1=1) -> 2
    # Residuals: e1, e2, e3 -> 3
    # Latent var: psi1 -> 1
    # Total 6 parameters.
    # Moments: 3*4/2 = 6 covariances.
    # DF = 9 (saturated moments) - 6 (params) = 3.
    # (3 means not estimated, but counted in saturated df)
    assert indices["df"] == 3
    assert indices["chisq"] < 15
    assert indices["cfi"] > 0.95
    assert indices["rmsea"] < 0.1
    assert indices["srmr"] < 0.1

def test_fit_indices_mixed_model():
    # Mixed model should NOT have these indices (except AIC/BIC)
    # because my implementation only enables them for SEM (no random effects)
    
    # Simulate mixed data
    N = 100
    groups = np.repeat(np.arange(10), 10)
    u = np.random.randn(10)
    x = np.random.randn(N)
    y = 1 + x + u[groups] + np.random.randn(N)
    
    df = pd.DataFrame({"y": y, "x": x, "g": groups})
    
    model = semx.Model(
        ["y ~ x + (1 | g)"],
        families={"y": "gaussian", "x": "gaussian"}
    )
    fit = model.fit(df)
    
    indices = fit.fit_indices
    assert np.isnan(indices["chisq"])
    assert np.isnan(indices["cfi"])
    assert not np.isnan(indices["aic"])
