import pytest
import numpy as np
import pandas as pd
from semx import Model

def test_simple_cfa_estimation():
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
    
    # Fit
    # This is expected to fail or produce 0 estimates for loadings if the bug exists
    try:
        fit = model.fit(data)
        print("Estimates:", fit.parameter_estimates)
    except Exception as e:
        pytest.fail(f"Fit failed with error: {e}")
    
    # Check loadings
    # We expect loadings around 1.0, 0.8, 0.6
    # But currently they might be 0 because they are not used in design matrix
    
    estimates = fit.parameter_estimates
    
    # Check if we have non-zero estimates for loadings
    # Loadings are usually named "lambda_y2_F", "lambda_y3_F" (y1 is fixed)
    
    has_loading = False
    for name, val in estimates.items():
        if "lambda" in name and abs(val) > 0.1:
            has_loading = True
            print(f"Found loading: {name} = {val}")
            
    if not has_loading:
        pytest.fail("No significant loadings estimated! Likely the SEM estimation bug.")
