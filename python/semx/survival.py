import numpy as np
import pandas as pd
from typing import Union, Sequence, Mapping, List, Optional
from .model import SemFit

def predict_survival(
    fit: SemFit, 
    data: Union[Mapping[str, Sequence[float]], pd.DataFrame], 
    times: Sequence[float],
    outcome: str
) -> pd.DataFrame:
    """Predict survival probabilities S(t) for a given outcome.
    
    Parameters
    ----------
    fit : SemFit
        Fitted model.
    data : pd.DataFrame
        New data for prediction.
    times : Sequence[float]
        Time points to predict survival at.
    outcome : str
        Name of the survival outcome variable.
        
    Returns
    -------
    pd.DataFrame
        DataFrame where columns are time points and rows are observations.
        Values are S(t).
    """
    # Find variable definition
    var_def = next((v for v in fit.model_ir.variables if v.name == outcome), None)
    if not var_def:
        raise ValueError(f"Outcome '{outcome}' not found in model.")
        
    family = var_def.family
    
    # Predict linear predictor eta
    # fit.predict returns predictions for all endogenous variables
    # We need to ensure we get the linear predictor for 'outcome'
    preds = fit.predict(data)
    if outcome not in preds.columns:
        raise ValueError(f"Could not predict linear predictor for '{outcome}'")
        
    eta = preds[outcome].values
    
    # Get dispersion parameter (shape)
    # Try "psi_{outcome}_{outcome}" first
    dispersion_id = f"psi_{outcome}_{outcome}"
    dispersion = fit.parameter_estimates.get(dispersion_id)
    
    if dispersion is None:
        if family == "exponential":
            dispersion = 1.0
        else:
            # Try fuzzy match
            candidates = [k for k in fit.parameter_estimates if outcome in k and ("psi" in k or "shape" in k)]
            if len(candidates) == 1:
                dispersion = fit.parameter_estimates[candidates[0]]
            else:
                # Default to 1.0 if not found? Or raise?
                # For Weibull, shape=1 is Exponential.
                # Let's warn and use 1.0? No, raise.
                raise ValueError(f"Could not identify dispersion parameter for '{outcome}' (family: {family}). Candidates: {candidates}")
    
    k = dispersion
    
    results = {}
    for t in times:
        if t < 0:
            results[t] = np.nan
            continue
            
        # Weibull AFT: S(t) = exp( - (t * exp(-eta))^k )
        # Log-logistic: S(t) = 1 / (1 + (t * exp(-eta))^k)
        # Lognormal: S(t) = 1 - Phi( (log(t) - eta) / k ) ? (k is sigma)
        
        if family == "weibull" or family == "exponential":
            # z = (t * exp(-eta))^k
            # log(z) = k * (log(t) - eta)
            log_t = np.log(t) if t > 0 else -np.inf
            log_z = k * (log_t - eta)
            z = np.exp(log_z)
            S_t = np.exp(-z)
        elif family == "loglogistic":
            # S(t) = 1 / (1 + (t * exp(-eta))^k)
            log_t = np.log(t) if t > 0 else -np.inf
            log_z = k * (log_t - eta)
            z = np.exp(log_z)
            S_t = 1.0 / (1.0 + z)
        elif family == "lognormal":
            from scipy.stats import norm
            # T ~ Lognormal(mu=eta, sigma=k)
            # S(t) = 1 - CDF((log(t) - eta)/k)
            if t <= 0:
                S_t = 1.0
            else:
                z = (np.log(t) - eta) / k
                S_t = 1.0 - norm.cdf(z)
        else:
            raise ValueError(f"Unsupported family for survival prediction: {family}")
            
        results[t] = S_t
        
    return pd.DataFrame(results)

def predict_cif(
    fit: SemFit,
    data: Union[Mapping[str, Sequence[float]], pd.DataFrame],
    times: Sequence[float],
    event_outcome: str,
    competing_outcomes: Sequence[str]
) -> pd.DataFrame:
    """Predict Cumulative Incidence Function (CIF) for a specific event.
    
    CIF_j(t) = integral_0^t S_overall(u) * h_j(u) du
    
    Parameters
    ----------
    fit : SemFit
        Fitted model.
    data : pd.DataFrame
        New data.
    times : Sequence[float]
        Time points.
    event_outcome : str
        The outcome variable for the event of interest.
    competing_outcomes : Sequence[str]
        List of outcome variables for competing events.
        
    Returns
    -------
    pd.DataFrame
        CIF values.
    """
    # We need predictions for all outcomes
    all_outcomes = [event_outcome] + list(competing_outcomes)
    
    # Get parameters for all outcomes
    params = {} # outcome -> {eta: vector, k: scalar, family: str}
    
    preds = fit.predict(data)
    n_obs = len(preds)
    
    for out in all_outcomes:
        var_def = next((v for v in fit.model_ir.variables if v.name == out), None)
        if not var_def:
            raise ValueError(f"Outcome '{out}' not found")
        
        eta = preds[out].values
        
        dispersion_id = f"psi_{out}_{out}"
        dispersion = fit.parameter_estimates.get(dispersion_id)
        if dispersion is None:
             if var_def.family == "exponential":
                 dispersion = 1.0
             else:
                 candidates = [k for k in fit.parameter_estimates if out in k and ("psi" in k or "shape" in k)]
                 if len(candidates) == 1:
                     dispersion = fit.parameter_estimates[candidates[0]]
                 else:
                     raise ValueError(f"Dispersion for {out} not found")
                     
        params[out] = {"eta": eta, "k": dispersion, "family": var_def.family}

    # Numerical integration
    # We compute S_overall(u) and h_event(u) on a grid
    # Grid: 0 to max(times)
    max_time = max(times)
    # Use fine grid? Or just the requested times if they are dense?
    # Let's use a fixed grid for integration and interpolate?
    # Or just integrate step-wise between requested times (assuming sorted).
    
    sorted_times = sorted([t for t in times if t >= 0])
    if not sorted_times:
        return pd.DataFrame()
        
    # Add 0 if not present
    integration_points = sorted(list(set([0.0] + sorted_times)))
    
    cif_values = np.zeros((n_obs, len(integration_points)))
    
    # CIF(0) = 0
    
    for i in range(1, len(integration_points)):
        t_prev = integration_points[i-1]
        t_curr = integration_points[i]
        dt = t_curr - t_prev
        t_mid = (t_prev + t_curr) / 2.0
        
        # Compute S_overall(t_mid) and h_event(t_mid)
        
        # S_overall = product S_j
        S_overall = np.ones(n_obs)
        for out in all_outcomes:
            p = params[out]
            # Compute S_j(t_mid)
            # Reuse logic from predict_survival (inline for speed)
            k = p["k"]
            eta = p["eta"]
            fam = p["family"]
            
            if fam in ["weibull", "exponential"]:
                log_z = k * (np.log(t_mid) - eta)
                z = np.exp(log_z)
                S_j = np.exp(-z)
            elif fam == "loglogistic":
                log_z = k * (np.log(t_mid) - eta)
                z = np.exp(log_z)
                S_j = 1.0 / (1.0 + z)
            elif fam == "lognormal":
                from scipy.stats import norm
                z_score = (np.log(t_mid) - eta) / k
                S_j = 1.0 - norm.cdf(z_score)
            else:
                S_j = np.ones(n_obs) # Should not happen
            
            S_overall *= S_j
            
        # h_event(t_mid)
        p_evt = params[event_outcome]
        k = p_evt["k"]
        eta = p_evt["eta"]
        fam = p_evt["family"]
        
        if fam in ["weibull", "exponential"]:
            # h(t) = k/t * (t*exp(-eta))^k
            #      = k/t * z
            log_z = k * (np.log(t_mid) - eta)
            z = np.exp(log_z)
            h_evt = (k / t_mid) * z
        elif fam == "loglogistic":
            # h(t) = (k/t * z) / (1+z)
            log_z = k * (np.log(t_mid) - eta)
            z = np.exp(log_z)
            h_evt = (k / t_mid) * z / (1.0 + z)
        elif fam == "lognormal":
            from scipy.stats import norm
            # h(t) = pdf / S
            z_score = (np.log(t_mid) - eta) / k
            pdf = norm.pdf(z_score) / (t_mid * k)
            S_val = 1.0 - norm.cdf(z_score)
            h_evt = pdf / S_val
        else:
            h_evt = np.zeros(n_obs)
            
        # CIF increment
        d_cif = S_overall * h_evt * dt
        cif_values[:, i] = cif_values[:, i-1] + d_cif
        
    # Map back to requested times
    # We computed on integration_points which includes requested times
    # We just need to extract columns corresponding to requested times
    
    # Create mapping
    time_to_idx = {t: i for i, t in enumerate(integration_points)}
    
    final_results = {}
    for t in times:
        if t < 0:
            final_results[t] = np.nan
        else:
            idx = time_to_idx[t]
            final_results[t] = cif_values[:, idx]
            
    return pd.DataFrame(final_results)
