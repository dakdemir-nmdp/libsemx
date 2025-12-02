import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from semx import SemFit, ModelIR, EdgeKind, EdgeSpec

class MockOptimizationResult:
    def __init__(self):
        self.converged = True
        self.iterations = 10
        self.objective_value = -100.0
        self.parameters = [0.5, 1.2]

class MockFitResult:
    def __init__(self):
        self.optimization_result = MockOptimizationResult()
        self.standard_errors = [0.1, 0.2]
        self.chi_square = 5.0
        self.df = 2.0
        self.p_value = 0.08
        self.cfi = 0.95
        self.tli = 0.94
        self.rmsea = 0.05
        self.srmr = 0.04
        self.aic = 210.0
        self.bic = 220.0

class MockParameter:
    def __init__(self, id):
        self.id = id

class MockModelIR:
    def __init__(self):
        self.parameters = [MockParameter("beta"), MockParameter("sigma")]
        self.edges = [
            EdgeSpec(EdgeKind.Regression, "x", "y", "beta")
        ]

@pytest.fixture
def mock_sem_fit():
    model_ir = MockModelIR()
    fit_result = MockFitResult()
    return SemFit(model_ir, fit_result)

def test_summary_structure(mock_sem_fit):
    summary = mock_sem_fit.summary()
    
    assert hasattr(summary, "fit_indices")
    assert hasattr(summary, "parameters")
    
    assert summary.fit_indices["chisq"] == 5.0
    assert summary.fit_indices["cfi"] == 0.95
    
    df = summary.parameters
    assert isinstance(df, pd.DataFrame)
    assert "Estimate" in df.columns
    assert "Std.Error" in df.columns
    assert "z-value" in df.columns
    assert "P(>|z|)" in df.columns
    
    assert df.loc["beta", "Estimate"] == 0.5
    assert df.loc["sigma", "Estimate"] == 1.2

def test_summary_repr(mock_sem_fit):
    summary = mock_sem_fit.summary()
    repr_str = repr(summary)
    
    assert "Optimization converged: True" in repr_str
    assert "Log-likelihood: 100.000" in repr_str
    assert "CFI: 0.950" in repr_str
    assert "beta" in repr_str
    assert "sigma" in repr_str

def test_predict(mock_sem_fit):
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 3.0, 4.0]})
    preds = mock_sem_fit.predict(data)
    
    assert isinstance(preds, pd.DataFrame)
    assert "y" in preds.columns
    # beta is 0.5, so y_hat = 0.5 * x
    expected = np.array([0.5, 1.0, 1.5])
    np.testing.assert_allclose(preds["y"], expected)

def test_plot_residuals(mock_sem_fit):
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 3.0, 4.0]})
    
    with patch("matplotlib.pyplot.subplots") as mock_subplots:
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        ax = mock_sem_fit.plot(data, kind="residuals")
        
        assert ax is mock_ax
        mock_ax.scatter.assert_called_once()
        mock_ax.set_title.assert_called_with("Residuals vs Fitted for y")

def test_plot_qq(mock_sem_fit):
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 3.0, 4.0]})
    
    with patch("matplotlib.pyplot.subplots") as mock_subplots:
        with patch("scipy.stats.probplot") as mock_probplot:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            ax = mock_sem_fit.plot(data, kind="qq")
            
            assert ax is mock_ax
            mock_probplot.assert_called_once()
            mock_ax.set_title.assert_called_with("Normal Q-Q Plot for y")
