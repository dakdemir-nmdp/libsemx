import pytest
import numpy as np
from semx import ModelIRBuilder, LikelihoodDriver, OptimizationOptions, VariableKind, EdgeKind

def test_multi_group_pooled_mean():
    np.random.seed(123)
    # Group 1
    b1 = ModelIRBuilder()
    b1.add_variable("y", VariableKind.Observed, "gaussian")
    b1.add_variable("one", VariableKind.Exogenous, "gaussian")
    b1.register_parameter("mu", 0.0)
    # b1.register_parameter("sigma2", 1.0) # Let add_edge register it as Positive
    b1.add_edge(EdgeKind.Regression, "one", "y", "mu")
    b1.add_edge(EdgeKind.Covariance, "y", "y", "sigma2")
    model1 = b1.build()
    
    data1 = {
        "y": np.random.normal(5, 1, 100).tolist(),
        "one": [1.0] * 100
    }
    
    # Group 2
    b2 = ModelIRBuilder()
    b2.add_variable("y", VariableKind.Observed, "gaussian")
    b2.add_variable("one", VariableKind.Exogenous, "gaussian")
    b2.register_parameter("mu", 0.0)
    # b2.register_parameter("sigma2", 1.0)
    b2.add_edge(EdgeKind.Regression, "one", "y", "mu")
    b2.add_edge(EdgeKind.Covariance, "y", "y", "sigma2")
    model2 = b2.build()
    
    data2 = {
        "y": np.random.normal(5, 1, 100).tolist(),
        "one": [1.0] * 100
    }
    
    driver = LikelihoodDriver()
    options = OptimizationOptions()
    options.max_iterations = 100
    
    result = driver.fit_multi_group([model1, model2], [data1, data2], options, "lbfgs")
    
    assert result.optimization_result.converged
    
    params = dict(zip(result.parameter_names, result.optimization_result.parameters))
    assert params["mu"] == pytest.approx(5.0, abs=0.2)
    assert params["sigma2"] == pytest.approx(1.0, abs=0.2)

def test_multi_group_specific_means():
    np.random.seed(123)
    # Group 1
    b1 = ModelIRBuilder()
    b1.add_variable("y", VariableKind.Observed, "gaussian")
    b1.add_variable("one", VariableKind.Exogenous, "gaussian")
    b1.register_parameter("mu1", 0.0)
    b1.add_edge(EdgeKind.Regression, "one", "y", "mu1")
    b1.add_edge(EdgeKind.Covariance, "y", "y", "sigma2")
    model1 = b1.build()
    
    data1 = {
        "y": np.random.normal(2, 1, 100).tolist(),
        "one": [1.0] * 100
    }
    
    # Group 2
    b2 = ModelIRBuilder()
    b2.add_variable("y", VariableKind.Observed, "gaussian")
    b2.add_variable("one", VariableKind.Exogenous, "gaussian")
    b2.register_parameter("mu2", 0.0)
    b2.add_edge(EdgeKind.Regression, "one", "y", "mu2")
    b2.add_edge(EdgeKind.Covariance, "y", "y", "sigma2")
    model2 = b2.build()
    
    data2 = {
        "y": np.random.normal(8, 1, 100).tolist(),
        "one": [1.0] * 100
    }
    
    driver = LikelihoodDriver()
    options = OptimizationOptions()
    
    result = driver.fit_multi_group([model1, model2], [data1, data2], options, "lbfgs")
    
    assert result.optimization_result.converged
    
    params = dict(zip(result.parameter_names, result.optimization_result.parameters))
    assert params["mu1"] == pytest.approx(2.0, abs=0.2)
    assert params["mu2"] == pytest.approx(8.0, abs=0.2)
    assert params["sigma2"] == pytest.approx(1.0, abs=0.2)
