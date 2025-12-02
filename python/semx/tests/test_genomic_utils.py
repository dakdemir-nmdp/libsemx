import pytest
import numpy as np
import pandas as pd
from semx import extract_heritability, cv_genomic_prediction

def test_extract_heritability():
    class MockFit:
        parameter_estimates = {"G": 0.5, "E": 0.5}
        
    fit = MockFit()
    h2 = extract_heritability(fit, "G", "E")
    assert h2 == 0.5

def test_cv_genomic_prediction():
    class MockFitResult:
        def __init__(self, data):
            self.data = data
        def predict(self, data):
            # Return dummy predictions
            return pd.DataFrame({"y": np.zeros(len(data))})
            
    class MockModel:
        def fit(self, data, **kwargs):
            return MockFitResult(data)
            
    model = MockModel()
    data = pd.DataFrame({"y": [1, 2, 3, 4]})
    
    res = cv_genomic_prediction(model, data, "y", folds=2)
    assert "mean_cor" in res
    assert "mean_mse" in res
