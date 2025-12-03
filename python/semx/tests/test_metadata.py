
import pytest
import semx
import pandas as pd
import numpy as np

def test_variable_metadata():
    # Define a simple model with metadata
    model = semx.Model(
        ["y ~ x"],
        families={"y": "gaussian", "x": "gaussian"},
        labels={"y": "Outcome Variable", "x": "Predictor Variable"},
        measurement_levels={"y": "interval", "x": "ratio"}
    )
    
    ir = model.to_ir()
    
    # Check metadata in IR
    y_var = next(v for v in ir.variables if v.name == "y")
    x_var = next(v for v in ir.variables if v.name == "x")
    
    assert y_var.label == "Outcome Variable"
    assert y_var.measurement_level == "interval"
    assert x_var.label == "Predictor Variable"
    assert x_var.measurement_level == "ratio"

def test_summary_metadata():
    # Simulate data
    np.random.seed(42)
    N = 100
    x = np.random.randn(N)
    y = 0.5 * x + np.random.randn(N)
    df = pd.DataFrame({"x": x, "y": y})
    
    model = semx.Model(
        ["y ~ x"],
        families={"y": "gaussian", "x": "gaussian"},
        labels={"y": "Outcome Variable", "x": "Predictor Variable"},
        measurement_levels={"y": "interval", "x": "ratio"}
    )
    
    fit = model.fit(df)
    summary = fit.summary()
    
    # Check if metadata is in summary string
    summary_str = str(summary)
    print(summary_str)
    
    assert "Variable Metadata:" in summary_str
    assert "Outcome Variable" in summary_str
    assert "Predictor Variable" in summary_str
    assert "interval" in summary_str
    assert "ratio" in summary_str

if __name__ == "__main__":
    test_variable_metadata()
    test_summary_metadata()
