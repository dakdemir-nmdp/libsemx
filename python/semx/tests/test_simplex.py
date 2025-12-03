import numpy as np
import pandas as pd
import pytest
from semx import Model

def test_multi_kernel_simplex():
    # Create synthetic data
    np.random.seed(42)
    n = 100
    p = 5
    X = np.random.randn(n, p)
    
    # Create two kernels
    K1 = X @ X.T
    K2 = np.eye(n)
    
    Cov = 2.0 * (0.7 * K1 + 0.3 * K2)
    y = np.random.multivariate_normal(np.zeros(n), Cov)
    
    data = pd.DataFrame({"y": y})
    data["id"] = range(n)
    data["_all"] = 1 # Constant grouping for genomic effect
    
    model = Model(
        equations=["y ~ 1"],
        families={"y": "gaussian"},
        random_effects=[
            {
                "name": "re_polygenic",
                "variables": ["_all"],
                "covariance": "K_multi"
            }
        ],
        genomic={
            "K_multi": {
                "markers": [K1, K2],
                "structure": "multi_kernel_simplex",
                "precomputed": True
            }
        }
    )
    
    fit = model.fit(data)
    
    # Check weights
    weights_info = fit.get_covariance_weights("K_multi")
    assert weights_info is not None
    
    sigma_sq = weights_info["sigma_sq"]
    weights = weights_info["weights"]
    
    # Check structure
    assert sigma_sq > 0
    assert len(weights) == 2
    assert abs(sum(weights) - 1.0) < 1e-6
    
    # Check summary output
    summary = fit.summary()
    summary_str = str(summary)
    assert "Covariance Weights for 'K_multi':" in summary_str
    assert "Sigma^2:" in summary_str
    assert "Kernel 1:" in summary_str
