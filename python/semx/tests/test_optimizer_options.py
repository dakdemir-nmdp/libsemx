import pytest
import numpy as np
import pandas as pd
import semx

def test_optimizer_options_max_iterations():
    # Simple model
    np.random.seed(42)
    N = 100
    x = np.random.randn(N)
    y = 1.0 + 0.5 * x + np.random.randn(N) * 0.5
    df = pd.DataFrame({"x": x, "y": y})
    
    model = semx.Model(["y ~ x"], families={"y": "gaussian", "x": "gaussian"})
    
    # Fit with very few iterations
    fit = model.fit(df, max_iterations=1)
    
    # Should not converge or have few iterations
    # Note: LBFGS might do more evaluations than iterations, but iterations should be low.
    # We can check fit.optimization_result.iterations
    
    print(f"Iterations: {fit.optimization_result.iterations}")
    assert fit.optimization_result.iterations <= 2 # Allow small buffer
    
def test_optimizer_options_tolerance():
    # Simple model
    np.random.seed(42)
    N = 100
    x = np.random.randn(N)
    y = 1.0 + 0.5 * x + np.random.randn(N) * 0.5
    df = pd.DataFrame({"x": x, "y": y})
    
    model = semx.Model(["y ~ x"], families={"y": "gaussian", "x": "gaussian"})
    
    # Fit with loose tolerance
    fit_loose = model.fit(df, tolerance=1e-1)
    
    # Fit with tight tolerance
    fit_tight = model.fit(df, tolerance=1e-6)
    
    print(f"Loose iterations: {fit_loose.optimization_result.iterations}")
    print(f"Tight iterations: {fit_tight.optimization_result.iterations}")
    
    # Tight tolerance should generally take more or equal iterations
    assert fit_tight.optimization_result.iterations >= fit_loose.optimization_result.iterations

def test_optimizer_options_linesearch():
    # Simple model
    np.random.seed(42)
    N = 100
    x = np.random.randn(N)
    y = 1.0 + 0.5 * x + np.random.randn(N) * 0.5
    df = pd.DataFrame({"x": x, "y": y})
    
    model = semx.Model(["y ~ x"], families={"y": "gaussian", "x": "gaussian"})
    
    # Fit with different linesearch types
    # Just checking it doesn't crash
    fit_armijo = model.fit(df, linesearch_type="armijo")
    assert fit_armijo.optimization_result.converged
    
    fit_wolfe = model.fit(df, linesearch_type="wolfe")
    assert fit_wolfe.optimization_result.converged
    
    fit_strong = model.fit(df, linesearch_type="strong_wolfe")
    assert fit_strong.optimization_result.converged
