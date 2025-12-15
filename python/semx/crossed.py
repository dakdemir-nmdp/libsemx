"""Convenience functions for crossed random effects models using Method of Moments."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ._libsemx import MoMSolver, MoMSolverOptions, MoMSolverResult


class MoMFit:
    """Result of fitting a crossed random effects model via Method of Moments.

    Attributes
    ----------
    beta : np.ndarray
        Fixed effects estimates.
    variance_components : np.ndarray
        Variance component estimates [σ²_u, σ²_v, σ²_e].
    n_groups_u : int
        Number of unique levels for first random effect.
    n_groups_v : int
        Number of unique levels for second random effect.
    converged : bool
        Whether the solver converged.
    message : str
        Convergence message.
    fixed_names : List[str]
        Names of fixed effects.
    u_name : str
        Name of first random grouping factor.
    v_name : str
        Name of second random grouping factor.
    """

    def __init__(
        self,
        result: MoMSolverResult,
        fixed_names: List[str],
        u_name: str,
        v_name: str
    ):
        self.beta = np.array(result.beta)
        self.variance_components = np.array(result.variance_components)
        self.n_groups_u = result.n_groups_u
        self.n_groups_v = result.n_groups_v
        self.converged = result.converged
        self.message = result.message
        self.fixed_names = fixed_names
        self.u_name = u_name
        self.v_name = v_name

    def __repr__(self) -> str:
        lines = ["Crossed Random Effects Model (Method of Moments)"]
        lines.append(f"Converged: {self.converged}")
        if not self.converged:
            lines.append(f"Message: {self.message}")
        lines.append("")
        lines.append("Fixed Effects:")
        for i, name in enumerate(self.fixed_names):
            lines.append(f"  {name:20s} {self.beta[i]:10.4f}")
        lines.append("")
        lines.append("Variance Components:")
        lines.append(f"  σ²_{self.u_name:17s} {self.variance_components[0]:10.4f}")
        lines.append(f"  σ²_{self.v_name:17s} {self.variance_components[1]:10.4f}")
        lines.append(f"  σ²_residual        {self.variance_components[2]:10.4f}")
        lines.append("")
        lines.append(f"Number of groups: {self.u_name}={self.n_groups_u}, {self.v_name}={self.n_groups_v}")
        return "\n".join(lines)

    def summary(self) -> Dict[str, Any]:
        """Return a dictionary with summary statistics."""
        return {
            "fixed_effects": dict(zip(self.fixed_names, self.beta)),
            "variance_components": {
                f"sigma2_{self.u_name}": self.variance_components[0],
                f"sigma2_{self.v_name}": self.variance_components[1],
                "sigma2_residual": self.variance_components[2],
            },
            "n_groups": {
                self.u_name: self.n_groups_u,
                self.v_name: self.n_groups_v,
            },
            "converged": self.converged,
            "message": self.message,
        }


def crossed_model(
    formula: str,
    data: pd.DataFrame,
    u: str,
    v: str,
    use_gls: bool = False,
    second_step: bool = False,
    verbose: bool = False,
    min_variance: float = 1e-10,
) -> MoMFit:
    """
    Fit a linear mixed model with two crossed random effects using Method of Moments.

    This function provides a fast O(N) solver for models of the form:

        y = Xβ + Z_u u + Z_v v + e

    where:
        - u ~ N(0, σ²_u I) is the first random effect
        - v ~ N(0, σ²_v I) is the second random effect
        - e ~ N(0, σ²_e I) is the residual error

    The solver uses the Method of Moments (MoM) approach based on U-statistics
    (Gao & Owen 2017, 2018), which scales linearly with the number of observations.
    This makes it particularly suitable for large-scale datasets with thousands or
    millions of observations.

    Parameters
    ----------
    formula : str
        Model formula for fixed effects (e.g., "y ~ x1 + x2").
    data : pd.DataFrame
        DataFrame containing the response, fixed effects, and grouping variables.
    u : str
        Name of the column for the first random effect grouping factor.
    v : str
        Name of the column for the second random effect grouping factor.
    use_gls : bool, optional
        If True, refine fixed effects using GLS after initial OLS estimation (default: False).
    second_step : bool, optional
        If True, perform a second MoM step using GLS residuals (default: False).
    verbose : bool, optional
        If True, print convergence details (default: False).
    min_variance : float, optional
        Minimum variance floor for variance components (default: 1e-10).

    Returns
    -------
    MoMFit
        Object containing fixed effects, variance components, and convergence info.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from semx import crossed_model
    >>>
    >>> # Simulate data with crossed random effects
    >>> np.random.seed(42)
    >>> n_students = 100
    >>> n_schools = 20
    >>> n_obs_per_cell = 5
    >>>
    >>> data = []
    >>> for student in range(n_students):
    >>>     for school in range(n_schools):
    >>>         for rep in range(n_obs_per_cell):
    >>>             data.append({
    >>>                 'student': student,
    >>>                 'school': school,
    >>>                 'x': np.random.randn(),
    >>>                 'y': 5.0 + 1.5 * np.random.randn() +
    >>>                      0.8 * np.random.randn() +  # student effect
    >>>                      0.6 * np.random.randn()     # school effect + residual
    >>>             })
    >>> df = pd.DataFrame(data)
    >>>
    >>> # Fit crossed random effects model
    >>> result = crossed_model("y ~ x", df, u="student", v="school")
    >>> print(result)

    Notes
    -----
    The MoM solver is most efficient for:
    - Large datasets (N > 10,000)
    - Balanced or mildly unbalanced designs
    - Two crossed random effects (no nesting)

    For more complex models (nested effects, non-Gaussian outcomes, multiple crossed
    effects), use the general `Model` class with REML or ML estimation.

    References
    ----------
    - Gao, K., & Owen, A. B. (2017). Efficient moment calculations for variance
      components in large unbalanced crossed random effects models.
      Electronic Journal of Statistics, 11(1), 1235-1296.
    - Gao, K., & Owen, A. B. (2018). Estimation and inference for very large
      linear mixed effects models. Statistica Sinica, 30(3), 1741-1771.
    """
    # Parse formula
    parts = formula.split("~")
    if len(parts) != 2:
        raise ValueError(f"Invalid formula: {formula}. Expected format 'y ~ x1 + x2'")

    response = parts[0].strip()
    predictors = [p.strip() for p in parts[1].split("+")]

    # Check that response and grouping variables exist
    if response not in data.columns:
        raise ValueError(f"Response variable '{response}' not found in data")
    if u not in data.columns:
        raise ValueError(f"Grouping variable '{u}' not found in data")
    if v not in data.columns:
        raise ValueError(f"Grouping variable '{v}' not found in data")

    # Extract response
    y = np.array(data[response].values, dtype=float)

    # Build design matrix
    X_cols = []
    fixed_names = []

    for pred in predictors:
        if pred == "1":
            # Intercept
            X_cols.append(np.ones(len(data)))
            fixed_names.append("(Intercept)")
        else:
            if pred not in data.columns:
                raise ValueError(f"Predictor '{pred}' not found in data")
            X_cols.append(data[pred].values)
            fixed_names.append(pred)

    X = np.column_stack(X_cols)

    # Convert grouping factors to integer indices
    u_unique = data[u].unique()
    v_unique = data[v].unique()
    u_map = {val: idx for idx, val in enumerate(sorted(u_unique))}
    v_map = {val: idx for idx, val in enumerate(sorted(v_unique))}

    u_indices = [u_map[val] for val in data[u].values]
    v_indices = [v_map[val] for val in data[v].values]

    # Set up options
    options = MoMSolverOptions()
    options.use_gls = use_gls
    options.second_step = second_step
    options.verbose = verbose
    options.min_variance = min_variance

    # Fit model
    result = MoMSolver.fit(y, X, u_indices, v_indices, options)

    # Wrap result
    return MoMFit(result, fixed_names, u, v)
