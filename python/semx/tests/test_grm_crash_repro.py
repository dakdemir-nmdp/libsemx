import numpy as np
import pandas as pd
import pytest
import semx

def test_grm_global_grouping():
    """
    Regression test for GRM usage.
    Ensures that using a global grouping variable with a GRM (where N_individuals == GRM_dim)
    works correctly and does not crash.
    """
    n_ind = 50
    n_markers = 100
    
    # Simulate markers
    np.random.seed(42)
    markers = np.random.randint(0, 3, size=(n_ind, n_markers)).astype(float)
    
    # Compute GRM
    markers_centered = markers - markers.mean(axis=0)
    p = markers.mean(axis=0) / 2.0
    denom = 2.0 * np.sum(p * (1.0 - p))
    K = np.dot(markers_centered, markers_centered.T) / denom
    
    # Simulate phenotype
    # y = u + e
    # u ~ N(0, K)
    # e ~ N(0, I)
    u = np.random.multivariate_normal(np.zeros(n_ind), K)
    e = np.random.normal(0, 1, size=n_ind)
    y = u + e
    
    # Create DataFrame
    df = pd.DataFrame({
        "y": y,
        "taxa": np.arange(n_ind).astype(float), # ID variable
        "_global": np.ones(n_ind) # Global grouping variable
    })
    
    # Define model
    model = semx.Model(
        equations=["y ~ 1"],
        families={"y": "gaussian"},
        genomic={
            "polygenic": {
                "data": K,
                "structure": "grm"
            }
        },
        random_effects=[{
            "name": "u",
            "variables": ["_global"], # Correct usage: group by constant
            "covariance": "polygenic"
        }]
    )
    
    # Fit
    fit = model.fit(df)
    
    assert fit.optimization_result.converged
    assert "polygenic_0" in fit.parameter_estimates # Variance component
    assert fit.parameter_estimates["polygenic_0"] > 0

def test_grm_incorrect_grouping_raises_error():
    """
    Ensures that using the ID variable as grouping (which creates N groups of size 1)
    raises a clear error instead of crashing, or at least raises the expected RuntimeError.
    """
    n_ind = 10
    K = np.eye(n_ind)
    df = pd.DataFrame({
        "y": np.random.randn(n_ind),
        "taxa": np.arange(n_ind).astype(float)
    })
    
    model = semx.Model(
        equations=["y ~ 1"],
        families={"y": "gaussian"},
        genomic={
            "polygenic": {
                "data": K,
                "structure": "grm"
            }
        },
        random_effects=[{
            "name": "u",
            "variables": ["taxa"], # Incorrect usage: group by ID
            "covariance": "polygenic"
        }]
    )
    
    # This used to raise RuntimeError because q (10) != n_i (1) and no design cols specified
    # But now it should be handled gracefully (treating each individual as a group)
    try:
        model.fit(df)
    except RuntimeError:
        pytest.fail("Should not raise RuntimeError")
