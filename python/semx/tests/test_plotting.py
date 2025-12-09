import pytest
from unittest.mock import MagicMock, patch
from semx import SemFit, VariableKind, EdgeKind
from semx.plotting import plot_path

# Mock classes to simulate SemFit structure
class MockVariable:
    def __init__(self, name, kind):
        self.name = name
        self.kind = kind
        self.family = "gaussian" # Default

class MockEdge:
    def __init__(self, kind, source, target, parameter_id):
        self.kind = kind
        self.source = source
        self.target = target
        self.parameter_id = parameter_id

class MockModelIR:
    def __init__(self):
        self.variables = [
            MockVariable("x", VariableKind.Observed),
            MockVariable("y", VariableKind.Observed),
            MockVariable("f", VariableKind.Latent)
        ]
        self.edges = [
            MockEdge(EdgeKind.Regression, "x", "y", "beta1"),
            MockEdge(EdgeKind.Regression, "f", "y", "lambda1"),
            MockEdge(EdgeKind.Covariance, "x", "f", "phi1")
        ]

class MockFitResult:
    def __init__(self):
        self.optimization_result = MagicMock()
        self.optimization_result.parameters = [0.5, 0.8, 0.2]
        self.parameter_names = ["beta1", "lambda1", "phi1"]

@pytest.fixture
def mock_fit():
    ir = MockModelIR()
    res = MockFitResult()
    fit = SemFit(ir, res)
    return fit

def test_plot_path_calls_graphviz(mock_fit):
    """Test that plot_path constructs a graphviz Digraph with correct nodes and edges."""
    mock_graphviz = MagicMock()
    mock_dot = MagicMock()
    mock_graphviz.Digraph.return_value = mock_dot
    
    with patch.dict("sys.modules", {"graphviz": mock_graphviz}):
        plot_path(mock_fit, view=False)
        
        # Check if Digraph was initialized
        mock_graphviz.Digraph.assert_called_once()
        
        # Check nodes
        # We expect 3 nodes: x, y, f
        # x, y are observed (box), f is latent (ellipse)
        mock_dot.node.assert_any_call("x", "x", shape="box")
        mock_dot.node.assert_any_call("y", "y", shape="box")
        mock_dot.node.assert_any_call("f", "f", shape="ellipse")
        
        # Check edges
        # x -> y (beta1=0.5)
        mock_dot.edge.assert_any_call("x", "y", label="0.50", style="solid", dir="forward")
        # f -> y (lambda1=0.8)
        mock_dot.edge.assert_any_call("f", "y", label="0.80", style="solid", dir="forward")
        # x <-> f (phi1=0.2)
        mock_dot.edge.assert_any_call("x", "f", label="0.20", style="dashed", dir="both")

def test_plot_path_missing_graphviz(mock_fit):
    """Test that ImportError is raised if graphviz is missing."""
    with patch.dict("sys.modules", {"graphviz": None}):
        with pytest.raises(ImportError, match="graphviz"):
            plot_path(mock_fit)

def test_plot_survival(mock_fit):
    """Test plot_survival."""
    import pandas as pd
    import numpy as np
    from semx.plotting import plot_survival
    
    mock_fit.data = pd.DataFrame({"time": [10, 20], "x": [1, 2]})
    
    # Add survival variable
    time_var = MockVariable("time", VariableKind.Observed)
    time_var.family = "weibull"
    mock_fit.model_ir.variables.append(time_var)
    
    # Mock predict
    mock_fit.predict = MagicMock(return_value=pd.DataFrame({"time": [0.5, 1.0]}))
    
    # Mock parameters
    mock_fit.fit_result.optimization_result.parameters = [1.5]
    mock_fit.fit_result.parameter_names = ["psi_time_time"]
    
    mock_plt = MagicMock()
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    
    with patch.dict("sys.modules", {"matplotlib.pyplot": mock_plt}):
        plot_survival(mock_fit)
        
        mock_plt.subplots.assert_called()
        mock_ax.plot.assert_called()
