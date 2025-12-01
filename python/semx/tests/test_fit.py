import pytest
import semx
import math

def test_fit_simple_regression():
    driver = semx.LikelihoodDriver()
    
    model = semx.ModelIR()
    model.variables = [
        semx.VariableSpec("y", semx.VariableKind.Observed, "gaussian")
    ]
    # x is not added as a variable because it's just a predictor
    
    model.edges = [
        semx.EdgeSpec(semx.EdgeKind.Regression, "x", "y", "beta")
    ]
    
    data = {
        "y": [1.0, 2.0, 3.0],
        "x": [1.0, 2.0, 3.0]
    }
    
    options = semx.OptimizationOptions()
    options.max_iterations = 100
    options.tolerance = 1e-6
    
    result = driver.fit(model, data, options, "lbfgs")
    
    assert result.converged
    assert len(result.parameters) == 1
    assert abs(result.parameters[0] - 1.0) < 1e-3
