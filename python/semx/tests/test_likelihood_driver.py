"""Tests for likelihood driver functionality."""

import pytest
import math

try:
    import semx as semx_cpp
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
    
    LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)

    @staticmethod
    def lognormal_loglik(time, eta, sigma, status):
        z = (math.log(time) - eta) / sigma
        if status:
            return -0.5 * z * z - math.log(time) - math.log(sigma) - TestLikelihoodDriver.LOG_SQRT_TWO_PI
        survival = 0.5 * math.erfc(z / math.sqrt(2.0))
        return math.log(survival)

    @staticmethod
    def loglogistic_loglik(time, eta, gamma, status):
        diff = math.log(time) - eta
        u = math.exp(gamma * diff)
        if status:
            return math.log(gamma) + gamma * diff - math.log(time) - 2.0 * math.log1p(u)
        return -math.log1p(u)
    
    def test_survival_lognormal_loglik(self):
        builder = semx_cpp.ModelIRBuilder()
        builder.add_variable("y", semx_cpp.VariableKind.Observed, "lognormal")
        model = builder.build()
    
        data = {"y": [1.4, 2.2, 0.95]}
        linear_predictors = {"y": [0.3, -0.2, 0.1]}
        dispersions = {"y": [0.9, 1.0, 0.85]}
        status = {"y": [1.0, 0.0, 1.0]}
    
        driver = semx_cpp.LikelihoodDriver()
        loglik = driver.evaluate_model_loglik(model, data, linear_predictors, dispersions, status=status)
    
        expected = sum(
            self.lognormal_loglik(t, eta, sigma, stat)
            for t, eta, sigma, stat in zip(data["y"], linear_predictors["y"], dispersions["y"], status["y"])
        )
        assert loglik == pytest.approx(expected, rel=1e-10)
    
    def test_survival_cif_payload(self):
        builder = semx_cpp.ModelIRBuilder()
        builder.add_variable("cause_lognormal", semx_cpp.VariableKind.Observed, "lognormal")
        builder.add_variable("cause_loglogistic", semx_cpp.VariableKind.Observed, "loglogistic")
        model = builder.build()
    
        times = [1.3, 2.0, 3.6]
        data = {
            "cause_lognormal": times,
            "cause_loglogistic": times,
        }
        linear_predictors = {
            "cause_lognormal": [0.2, -0.15, 0.25],
            "cause_loglogistic": [-0.3, 0.35, 0.05],
        }
        dispersions = {
            "cause_lognormal": [0.8, 1.05, 0.9],
            "cause_loglogistic": [1.2, 1.0, 1.1],
        }
        status = {
            "cause_lognormal": [1.0, 0.0, 0.0],
            "cause_loglogistic": [0.0, 1.0, 0.0],
        }
    
        driver = semx_cpp.LikelihoodDriver()
        loglik = driver.evaluate_model_loglik(
            model,
            data,
            linear_predictors,
            dispersions,
            status=status,
        )
    
        expected = 0.0
        expected += sum(
            self.lognormal_loglik(t, eta, sigma, stat)
            for t, eta, sigma, stat in zip(
                data["cause_lognormal"],
                linear_predictors["cause_lognormal"],
                dispersions["cause_lognormal"],
                status["cause_lognormal"],
            )
        )
        expected += sum(
            self.loglogistic_loglik(t, eta, gamma, stat)
            for t, eta, gamma, stat in zip(
                data["cause_loglogistic"],
                linear_predictors["cause_loglogistic"],
                dispersions["cause_loglogistic"],
                status["cause_loglogistic"],
            )
        )
    
        assert loglik == pytest.approx(expected, rel=1e-10)
