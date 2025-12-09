import pytest
import numpy as np
import pandas as pd
from semx import Model

def test_multi_group_api_simple():
    np.random.seed(123)
    # Simulate data for 2 groups
    # Group 1: y = 1*x + 0 + e, var(e)=1
    # Group 2: y = 2*x + 0 + e, var(e)=1
    
    n = 100
    x1 = np.random.normal(0, 1, n)
    y1 = 1.0 * x1 + np.random.normal(0, 1, n)
    g1 = ["A"] * n
    
    x2 = np.random.normal(0, 1, n)
    y2 = 2.0 * x2 + np.random.normal(0, 1, n)
    g2 = ["B"] * n
    
    df = pd.DataFrame({
        "x": np.concatenate([x1, x2]),
        "y": np.concatenate([y1, y2]),
        "group": np.concatenate([g1, g2])
    })
    
    model = Model(["y ~ x"], families={"y": "gaussian", "x": "gaussian"})
    
    # Fit without constraints (slopes should differ)
    fit = model.fit(df, group="group")
    
    assert fit.fit_result.optimization_result.converged
    params = fit.parameter_estimates
    
    print("Free params:", params.keys())
    
    # Parameter names depend on _to_ir_for_group logic.
    # Default ID for y ~ x is "beta_y_on_x"
    # With suffix: "beta_y_on_x_A", "beta_y_on_x_B".
    
    assert "beta_y_on_x_A" in params
    assert "beta_y_on_x_B" in params
    assert params["beta_y_on_x_A"] == pytest.approx(1.0, abs=0.2)
    assert params["beta_y_on_x_B"] == pytest.approx(2.0, abs=0.2)
    
    # Fit WITH constraints (slopes equal)
    fit_eq = model.fit(df, group="group", group_equal=["regressions"])
    
    assert fit_eq.fit_result.optimization_result.converged
    params_eq = fit_eq.parameter_estimates
    
    print("Constrained params:", params_eq.keys())
    
    # Should have only one beta_y_on_x
    assert "beta_y_on_x" in params_eq
    assert "beta_y_on_x_A" not in params_eq
    
    # Estimate should be between 1 and 2 (around 1.5)
    assert params_eq["beta_y_on_x"] == pytest.approx(1.5, abs=0.2)

def test_multi_group_intercepts():
    np.random.seed(123)
    # Group 1: y = 0 + e
    # Group 2: y = 5 + e
    n = 100
    y1 = np.random.normal(0, 1, n)
    y2 = np.random.normal(5, 1, n)
    df = pd.DataFrame({
        "y": np.concatenate([y1, y2]),
        "group": ["A"]*n + ["B"]*n
    })
    
    model = Model(["y ~ 1"], families={"y": "gaussian"})
    
    # Free intercepts
    fit = model.fit(df, group="group")
    params = fit.parameter_estimates
    
    # Intercept parameter ID: "beta_y_on__intercept"
    # Suffix: "beta_y_on__intercept_A", "beta_y_on__intercept_B"
    
    assert "beta_y_on__intercept_A" in params
    assert params["beta_y_on__intercept_A"] == pytest.approx(0.0, abs=0.2)
    assert params["beta_y_on__intercept_B"] == pytest.approx(5.0, abs=0.2)
    
    # Constrained intercepts
    fit_eq = model.fit(df, group="group", group_equal=["intercepts"])
    params_eq = fit_eq.parameter_estimates
    
    assert "beta_y_on__intercept" in params_eq
    assert params_eq["beta_y_on__intercept"] == pytest.approx(2.5, abs=0.2)
