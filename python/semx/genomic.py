from typing import Any, Dict, Optional, Union, Sequence
import numpy as np
import pandas as pd
from .model import Model, SemFit

def extract_heritability(fit: SemFit, genetic_component: str, residual_component: str) -> float:
    """Extract narrow-sense heritability from a fitted model.
    
    h^2 = Var(G) / (Var(G) + Var(E))
    
    Parameters
    ----------
    fit : SemFit
        The fitted model result.
    genetic_component : str
        The parameter ID corresponding to the genetic variance.
    residual_component : str
        The parameter ID corresponding to the residual variance.
        
    Returns
    -------
    float
        The heritability estimate.
    """
    params = fit.parameter_estimates
    
    var_g = params.get(genetic_component)
    if var_g is None:
        raise ValueError(f"Genetic component '{genetic_component}' not found in parameters.")
        
    var_e = params.get(residual_component)
    if var_e is None:
         raise ValueError(f"Residual component '{residual_component}' not found in parameters.")
         
    return var_g / (var_g + var_e)

def cv_genomic_prediction(
    model: Model,
    data: pd.DataFrame,
    outcome: str,
    folds: int = 5,
    seed: int = 42,
    optimizer_name: str = "lbfgs",
    options: Optional[Any] = None
) -> Dict[str, float]:
    """Perform k-fold cross-validation for genomic prediction using masking.
    
    Parameters
    ----------
    model : Model
        The model definition (with markers pre-loaded).
    data : pd.DataFrame
        The full dataset. Rows must match the order of markers in the model.
    outcome : str
        The outcome variable name to mask and predict.
    folds : int
        Number of folds.
    seed : int
        Random seed for fold generation.
    optimizer_name : str
        Optimizer to use.
    options : OptimizationOptions
        Optimizer options.
        
    Returns
    -------
    Dict[str, float]
        Metrics: mean_cor, mean_mse, sd_cor, sd_mse.
    """
    from typing import Any # Re-import if needed or use from outer scope
    
    np.random.seed(seed)
    n = len(data)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // folds
    
    correlations = []
    mses = []
    
    for i in range(folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < folds - 1 else n
        test_idx = indices[start:end]
        
        # Create masked data
        masked_data = data.copy()
        masked_data.loc[test_idx, outcome] = np.nan
        
        # Fit
        fit = model.fit(masked_data, options=options, optimizer_name=optimizer_name)
        
        # Predict
        preds = fit.predict(masked_data)
        
        y_true = data.loc[test_idx, outcome]
        y_pred = preds.loc[test_idx, outcome]
        
        # Compute metrics
        valid = ~np.isnan(y_pred) & ~np.isnan(y_true)
        if np.sum(valid) > 0:
            cor = np.corrcoef(y_true[valid], y_pred[valid])[0, 1]
            mse = np.mean((y_true[valid] - y_pred[valid])**2)
            correlations.append(cor)
            mses.append(mse)
            
    return {
        "mean_cor": float(np.mean(correlations)),
        "mean_mse": float(np.mean(mses)),
        "sd_cor": float(np.std(correlations)),
        "sd_mse": float(np.std(mses))
    }
