from typing import Optional, Any
from .model import SemFit
from ._libsemx import VariableKind, EdgeKind

def plot_path(fit: SemFit, filename: Optional[str] = None, view: bool = True) -> Any:
    """
    Generate a path diagram for the fitted model using Graphviz.
    
    Args:
        fit: The fitted model object.
        filename: Optional filename to save the rendered graph.
        view: Whether to open the rendered graph.
        
    Returns:
        A graphviz.Digraph object.
    """
    try:
        import graphviz
    except ImportError:
        raise ImportError("The 'graphviz' library is required for plotting. Install it with `pip install graphviz`.")

    dot = graphviz.Digraph(comment='Path Diagram')
    dot.attr(rankdir='LR')

    # Add nodes
    for var in fit.model_ir.variables:
        shape = 'box'
        if var.kind == VariableKind.Latent:
            shape = 'ellipse'
        elif var.kind == VariableKind.Grouping:
            shape = 'hexagon'
        
        dot.node(var.name, var.name, shape=shape)

    # Add edges
    estimates = fit.parameter_estimates
    drawn_covariances = set()
    
    for edge in fit.model_ir.edges:
        label = ""
        if edge.parameter_id in estimates:
            val = estimates[edge.parameter_id]
            label = f"{val:.2f}"
        
        style = 'solid'
        dir_ = 'forward'
        
        if edge.kind == EdgeKind.Covariance:
            dir_ = 'both'
            style = 'dashed'
            
            # Avoid duplicate covariance edges
            pair = tuple(sorted((edge.source, edge.target)))
            if pair in drawn_covariances:
                continue
            drawn_covariances.add(pair)
        
        dot.edge(edge.source, edge.target, label=label, style=style, dir=dir_)

    if filename:
        dot.render(filename, view=view)
        
    return dot

def plot_survival(fit: SemFit, newdata: Optional[Any] = None, time_grid: Optional[Any] = None) -> Any:
    """
    Generate survival curves for survival outcomes.
    
    Args:
        fit: The fitted model object.
        newdata: Optional DataFrame for prediction.
        time_grid: Optional time points.
        
    Returns:
        A matplotlib Figure or list of Figures.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
    except ImportError:
        raise ImportError("matplotlib, numpy, and pandas are required for plotting.")
        
    from .survival import predict_survival
        
    # Find survival variables
    surv_vars = [v for v in fit.model_ir.variables if v.family in ("weibull", "exponential", "loglogistic", "lognormal")]
    
    if not surv_vars:
        raise ValueError("No survival outcomes found in the model")
        
    # Use original data if newdata is not provided
    data = newdata if newdata is not None else fit.data
    
    if data is None:
        raise ValueError("No data available for prediction. Provide 'newdata' or ensure model was fitted with data.")
        
    plots = []
    
    for v in surv_vars:
        var_name = v.name
        family = v.family
        
        # Generate time grid
        if time_grid is None:
            if var_name in data.columns:
                max_t = data[var_name].max()
            else:
                max_t = 100.0
            times = np.linspace(0, max_t, 100)
        else:
            times = time_grid
            
        # Predict survival probabilities
        try:
            surv_probs = predict_survival(fit, data, times, var_name)
        except Exception as e:
            # Skip if prediction fails (e.g. missing parameters)
            print(f"Skipping {var_name}: {e}")
            continue
            
        fig, ax = plt.subplots()
        
        n_profiles = len(surv_probs)
        
        for i in range(n_profiles):
            surv = surv_probs.iloc[i].values
            
            label = f"Profile {i+1}" if n_profiles > 1 else "Survival"
            ax.plot(times, surv, label=label)
            
        ax.set_ylim(0, 1)
        ax.set_title(f"Survival Curve for {var_name}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival Probability")
        
        if n_profiles <= 10:
            ax.legend()
            
        plots.append(fig)
        
    if len(plots) == 1:
        return plots[0]
    else:
        return plots

def plot_reaction_norms(
    fit: SemFit, 
    x: str, 
    group: str, 
    response: str, 
    add_points: bool = True,
    ax: Optional[Any] = None
) -> Any:
    """
    Visualizes the interaction between genotype and environment by plotting
    reaction norms (phenotype vs environment) for each genotype.
    
    Args:
        fit: The fitted model object.
        x: The name of the environmental variable (x-axis).
        group: The name of the grouping variable (genotype/individual).
        response: The name of the response variable.
        add_points: Whether to add observed data points.
        ax: Optional matplotlib axes object.
        
    Returns:
        The matplotlib axes object.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
    except ImportError:
        raise ImportError("matplotlib, numpy, and pandas are required for plotting.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    data = fit.data
    if data is None:
        raise ValueError("Original data not found in SemFit object.")
    
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
        
    x_col = data[x]
    is_categorical = x_col.dtype == 'object' or isinstance(x_col.dtype, pd.CategoricalDtype)
    
    params = fit.parameter_estimates
    
    # Find random effect spec
    re_spec = None
    for re in fit.model_ir.random_effects:
        if re.variables and re.variables[0] == group:
            re_spec = re
            break
            
    if re_spec is None:
        raise ValueError(f"Random effects for group '{group}' not found.")
        
    re_values = fit.fit_result.random_effects.get(re_spec.id)
    if re_values is None:
        # Try covariance_id as fallback
        re_values = fit.fit_result.random_effects.get(re_spec.covariance_id)
        
    if re_values is None:
        raise ValueError(f"Random effect values for '{re_spec.id}' not found.")
        
    design_vars = re_spec.variables[1:]
    n_vars = len(design_vars)
    if n_vars == 0:
        raise ValueError("No design variables found for random effect.")
        
    n_groups = len(re_values) // n_vars
    re_mat = np.array(re_values).reshape(n_groups, n_vars)
    re_df = pd.DataFrame(re_mat, columns=design_vars)
    
    # Map rows to group levels
    group_levels = sorted(data[group].unique())
    if len(group_levels) != n_groups:
        # If mismatch, we can't reliably map. 
        # But usually it matches sorted unique levels.
        pass
    else:
        re_df.index = group_levels

    if is_categorical:
        levels = sorted(x_col.unique())
        n_levels = len(levels)
        
        # Fixed effects
        beta_int = params.get(f"beta_{response}_on__intercept", 0.0)
        fixed_means = np.zeros(n_levels)
        fixed_means[0] = beta_int
        
        for k in range(1, n_levels):
            lvl = levels[k]
            val = params.get(f"beta_{response}_on_{x}{lvl}", 0.0)
            fixed_means[k] = beta_int + val
            
        # Predictions
        preds = np.zeros((n_groups, n_levels))
        
        for i in range(n_groups):
            # Get random effects for this group
            u = re_mat[i, :]
            u_dict = dict(zip(design_vars, u))
            
            for k in range(n_levels):
                lvl = levels[k]
                val = fixed_means[k]
                
                # Add random intercept
                val += u_dict.get("_intercept", 0.0)
                
                # Add random slope/level
                # Try exact match or prefix match
                if lvl in u_dict:
                    val += u_dict[lvl]
                elif f"{x}{lvl}" in u_dict:
                    val += u_dict[f"{x}{lvl}"]
                    
                preds[i, k] = val
                
        # Plot
        ax.plot(range(n_levels), preds.T, color='blue', alpha=0.3)
        ax.plot(range(n_levels), fixed_means, color='red', linewidth=3, label='Population Mean')
        
        ax.set_xticks(range(n_levels))
        ax.set_xticklabels(levels)
        
        if add_points:
            # Map x to indices
            level_map = {l: i for i, l in enumerate(levels)}
            x_idx = x_col.map(level_map)
            # Jitter
            jitter = np.random.uniform(-0.1, 0.1, size=len(x_idx))
            ax.scatter(x_idx + jitter, data[response], color='gray', alpha=0.5, s=10)
            
    else:
        # Continuous
        beta_int = params.get(f"beta_{response}_on__intercept", 0.0)
        beta_slope = params.get(f"beta_{response}_on_{x}", 0.0)
        
        x_min, x_max = x_col.min(), x_col.max()
        x_grid = np.linspace(x_min, x_max, 100)
        
        # Plot lines
        for i in range(n_groups):
            u = re_mat[i, :]
            u_dict = dict(zip(design_vars, u))
            
            u_int = u_dict.get("_intercept", 0.0)
            u_slope = u_dict.get(x, 0.0)
            
            int_i = beta_int + u_int
            slope_i = beta_slope + u_slope
            
            y_grid = int_i + slope_i * x_grid
            ax.plot(x_grid, y_grid, color='blue', alpha=0.3)
            
        # Population mean
        ax.plot(x_grid, beta_int + beta_slope * x_grid, color='red', linewidth=3, label='Population Mean')
        
        if add_points:
            ax.scatter(x_col, data[response], color='gray', alpha=0.5)
            
    ax.set_xlabel(x)
    ax.set_ylabel(response)
    ax.legend()
    
    return ax
