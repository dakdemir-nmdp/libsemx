"""Tests for likelihood driver functionality."""

import pytest
import math

try:
    import _libsemx as semx_cpp
except ImportError:
    pytest.skip("C++ extension not available", allow_module_level=True)

class TestLikelihoodDriver:
    """Test suite for LikelihoodDriver mirroring C++ fixtures."""

    def test_evaluates_model_loglik(self):
        """Mirror C++: LikelihoodDriver evaluates model log-likelihood."""
        builder = semx_cpp.ModelIRBuilder()
        builder.add_variable("y", semx_cpp.VariableKind.Observed, "gaussian")
        model = builder.build()
        
        driver = semx_cpp.LikelihoodDriver()
        
        data = {"y": [1.0, 2.0, 3.0]}
        linear_predictors = {"y": [1.0, 2.0, 3.0]} # Perfect fit
        dispersions = {"y": [1.0, 1.0, 1.0]} # sigma^2 = 1, per observation
        
        # loglik for N(mu, 1) at x=mu is -0.5 * log(2*pi)
        # Total = 3 * (-0.5 * log(2*pi))
        expected = 3 * (-0.5 * math.log(2 * math.pi))
        
        loglik = driver.evaluate_model_loglik(model, data, linear_predictors, dispersions)
        assert loglik == pytest.approx(expected)

    def test_throws_on_missing_data(self):
        """Mirror C++: LikelihoodDriver throws on missing data."""
        builder = semx_cpp.ModelIRBuilder()
        builder.add_variable("y", semx_cpp.VariableKind.Observed, "gaussian")
        model = builder.build()
        
        driver = semx_cpp.LikelihoodDriver()
        
        data = {} # Missing y
        linear_predictors = {"y": [1.0]}
        dispersions = {"y": [1.0]}
        
        with pytest.raises(ValueError): # C++ throws invalid_argument
            driver.evaluate_model_loglik(model, data, linear_predictors, dispersions)
