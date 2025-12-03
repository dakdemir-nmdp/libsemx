"""High-level Python front-end helpers described in blueprint §§8–10.

This module introduces a light-weight :class:`Model` wrapper that accepts
lavaan/lme4-inspired formulas and compiles them into the shared ``ModelIR``
structure understood by the C++ core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2

from _libsemx import (
    EdgeKind,
    GenomicRelationshipMatrix,
    ModelIR,
    ModelIRBuilder,
    VariableKind,
    compute_model_diagnostics,
    compute_modification_indices,
    compute_standardized_estimates,
)

__all__ = ["Model", "ModelSpecificationError", "SemFit"]


class ModelSpecificationError(ValueError):
    """Raised when formulas or metadata are inconsistent."""


@dataclass
class _VariableDef:
    name: str
    kind: VariableKind
    family: str
    label: str = ""
    measurement_level: str = ""


@dataclass
class _EdgeDef:
    kind: EdgeKind
    source: str
    target: str
    parameter_id: str


_VARIABLE_KIND_ALIASES: Dict[str, VariableKind] = {
    "observed": VariableKind.Observed,
    "latent": VariableKind.Latent,
    "grouping": VariableKind.Grouping,
}


class SemFit:
    """Wrapper for model fit results providing diagnostic methods."""

    def __init__(
        self,
        model_ir: ModelIR,
        fit_result: Any,
        sample_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_ir = model_ir
        self.fit_result = fit_result
        self.sample_stats = sample_stats or {}

    @property
    def optimization_result(self) -> Any:
        return self.fit_result.optimization_result

    @property
    def parameter_estimates(self) -> Dict[str, float]:
        """Map parameter IDs to their estimated values."""
        params = self.fit_result.optimization_result.parameters
        if hasattr(self.fit_result, "parameter_names") and self.fit_result.parameter_names:
            names = self.fit_result.parameter_names
        else:
            names = [p.id for p in self.model_ir.parameters]
        return dict(zip(names, params))

    @property
    def fit_indices(self) -> Dict[str, float]:
        """Return fit indices (Chi-square, CFI, TLI, RMSEA, SRMR)."""
        return {
            "chisq": self.fit_result.chi_square,
            "df": self.fit_result.df,
            "pvalue": self.fit_result.p_value,
            "cfi": self.fit_result.cfi,
            "tli": self.fit_result.tli,
            "rmsea": self.fit_result.rmsea,
            "srmr": self.fit_result.srmr,
            "aic": self.fit_result.aic,
            "bic": self.fit_result.bic,
        }

    def standardized_solution(self) -> Any:
        """Compute standardized parameter estimates (std.lv and std.all)."""
        if hasattr(self.fit_result, "parameter_names") and self.fit_result.parameter_names:
            names = self.fit_result.parameter_names
        else:
            names = [p.id for p in self.model_ir.parameters]
        values = self.fit_result.optimization_result.parameters
        return compute_standardized_estimates(self.model_ir, names, values)

    def get_covariance_weights(self, cov_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve weights for a multi-kernel covariance structure.
        
        Returns
        -------
        dict or None
            Dictionary with 'sigma_sq' and 'weights' (list of floats), or None
            if the covariance structure is not multi_kernel_simplex.
        """
        # Find the covariance spec
        cov_spec = next((c for c in self.model_ir.covariances if c.id == cov_id), None)
        if not cov_spec:
            raise ValueError(f"Covariance '{cov_id}' not found in model")

        if cov_spec.structure != "multi_kernel_simplex":
             return None

        # Get parameters
        params = self.parameter_estimates
        
        # The parameters are named cov_id_0, cov_id_1, ...
        # cov_id_0 is sigma_sq.
        # cov_id_1...k are thetas (softmax inputs).
        
        prefix = f"{cov_id}_"
        relevant_params = {k: v for k, v in params.items() if k.startswith(prefix)}
        
        if not relevant_params:
            return None
            
        # Sort by index
        sorted_keys = sorted(relevant_params.keys(), key=lambda x: int(x.split("_")[-1]))
        
        if len(sorted_keys) < 2:
            return None
            
        sigma_sq = relevant_params[sorted_keys[0]]
        thetas = [relevant_params[k] for k in sorted_keys[1:]]
        
        # Softmax
        max_theta = max(thetas)
        exps = [np.exp(t - max_theta) for t in thetas]
        sum_exps = sum(exps)
        weights = [e / sum_exps for e in exps]
        
        return {
            "sigma_sq": sigma_sq,
            "weights": weights
        }

    def diagnostics(self) -> Any:
        """Compute model diagnostics (residuals, SRMR)."""
        if not self.sample_stats:
            raise ValueError("Sample statistics not available for diagnostics")

        if hasattr(self.fit_result, "parameter_names") and self.fit_result.parameter_names:
            names = self.fit_result.parameter_names
        else:
            names = [p.id for p in self.model_ir.parameters]
        values = self.fit_result.optimization_result.parameters

        return compute_model_diagnostics(
            self.model_ir,
            names,
            values,
            self.sample_stats["means"],
            self.sample_stats["covariance"],
        )

    def modification_indices(self) -> Any:
        """Compute modification indices for missing paths."""
        if not self.sample_stats:
            raise ValueError("Sample statistics not available for modification indices")

        if hasattr(self.fit_result, "parameter_names") and self.fit_result.parameter_names:
            names = self.fit_result.parameter_names
        else:
            names = [p.id for p in self.model_ir.parameters]
        values = self.fit_result.optimization_result.parameters

        return compute_modification_indices(
            self.model_ir,
            names,
            values,
            self.sample_stats["covariance"],
            self.sample_stats["n_obs"],
        )

    def variance_components(self) -> pd.DataFrame:
        """Extract variance components (random effects)."""
        rows = []
        
        if hasattr(self.fit_result, "covariance_matrices"):
            cov_matrices = self.fit_result.covariance_matrices
            
            # Map covariance_id to random effect info
            re_map = {}
            for re in self.model_ir.random_effects:
                re_map[re.covariance_id] = re
                
            for cov_id, matrix in cov_matrices.items():
                if cov_id not in re_map:
                    continue
                
                re = re_map[cov_id]
                if not re.variables:
                    continue
                    
                group = re.variables[0]
                design_vars = re.variables[1:]
                
                # Handle implicit intercept for (1 | group)
                if not design_vars and len(matrix) == 1:
                    design_vars = ["(Intercept)"]
                
                # Matrix is flattened
                dim = len(design_vars)
                dim = len(design_vars)
                if len(matrix) != dim * dim:
                    # Mismatch, skip or warn
                    continue
                    
                mat = np.array(matrix).reshape(dim, dim)
                
                # Extract variances and correlations
                sds = np.sqrt(np.diag(mat))
                
                # Avoid division by zero
                sds_safe = sds.copy()
                sds_safe[sds_safe == 0] = 1.0
                corrs = mat / np.outer(sds_safe, sds_safe)
                
                for i, var1 in enumerate(design_vars):
                    rows.append({
                        "Group": group,
                        "Name1": var1,
                        "Name2": "",
                        "Variance": mat[i, i],
                        "Std.Dev": sds[i],
                        "Corr": np.nan
                    })
                    
                    for j in range(i + 1, dim):
                        var2 = design_vars[j]
                        rows.append({
                            "Group": group,
                            "Name1": var1,
                            "Name2": var2,
                            "Variance": mat[i, j],
                            "Std.Dev": np.nan,
                            "Corr": corrs[i, j]
                        })
        
        if not rows:
            return pd.DataFrame(columns=["Group", "Name1", "Name2", "Variance", "Std.Dev", "Corr"])
            
        return pd.DataFrame(rows)

    def summary(self) -> SemFitSummary:
        """Return a summary object containing fit statistics and parameter estimates."""
        return SemFitSummary(self)

    def predict(
        self, data: Union[Mapping[str, Sequence[float]], pd.DataFrame]
    ) -> pd.DataFrame:
        """Generate predictions for the given data (marginal)."""
        if hasattr(data, "to_dict"):
            df = pd.DataFrame(data.to_dict())
        else:
            df = pd.DataFrame(data)

        if "_intercept" not in df.columns:
            df["_intercept"] = 1.0

        params = self.parameter_estimates
        predictions = {}

        # Identify endogenous variables (targets of regressions)
        targets = set()
        for edge in self.model_ir.edges:
            if edge.kind == EdgeKind.Regression:
                targets.add(edge.target)

        for target in targets:
            pred = np.zeros(len(df))
            has_predictors = False
            for edge in self.model_ir.edges:
                if edge.kind == EdgeKind.Regression and edge.target == target:
                    if edge.source in df.columns:
                        val = params.get(edge.parameter_id, 0.0)
                        pred += val * df[edge.source]
                        has_predictors = True
            if has_predictors:
                predictions[target] = pred

        return pd.DataFrame(predictions)

    def plot(
        self,
        data: Union[Mapping[str, Sequence[float]], pd.DataFrame],
        kind: str = "residuals",
        ax: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot model diagnostics.

        Parameters
        ----------
        data : DataFrame or dict
            Data to use for predictions.
        kind : str, optional
            Type of plot: 'residuals' (default) or 'qq'.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, a new figure is created.
        **kwargs
            Additional arguments passed to the plotting function.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for plotting.")
            return None

        if ax is None:
            fig, ax = plt.subplots()

        preds = self.predict(data)
        if hasattr(data, "to_dict"):
            df = pd.DataFrame(data.to_dict())
        else:
            df = pd.DataFrame(data)

        # For simplicity, if multiple targets, we just plot the first one or overlay
        # Ideally, we should handle multiple targets better (e.g. subplots), but for now
        # let's pick the first one found in predictions.
        if preds.empty:
            print("No predictions available to plot.")
            return ax

        target = preds.columns[0]
        if target not in df.columns:
            print(f"Target variable '{target}' not found in data.")
            return ax

        y = df[target]
        y_hat = preds[target]
        resid = y - y_hat

        if kind == "residuals":
            ax.scatter(y_hat, resid, **kwargs)
            ax.set_xlabel(f"Fitted {target}")
            ax.set_ylabel(f"Residuals {target}")
            ax.set_title(f"Residuals vs Fitted for {target}")
            ax.axhline(0, color="red", linestyle="--")
        elif kind == "qq":
            import scipy.stats as stats
            stats.probplot(resid, dist="norm", plot=ax)
            ax.set_title(f"Normal Q-Q Plot for {target}")
        else:
            raise ValueError(f"Unknown plot kind: {kind}")

        return ax


class SemFitSummary:
    """Summary of model fit results."""

    def __init__(self, fit: SemFit) -> None:
        self.fit = fit
        self.fit_indices = fit.fit_indices
        self.parameters = self._build_parameter_table()

    def _build_parameter_table(self) -> pd.DataFrame:
        params = self.fit.fit_result.optimization_result.parameters
        ses = self.fit.fit_result.standard_errors

        if not ses:
            ses = [np.nan] * len(params)

        z_values = [p / se if se > 0 else np.nan for p, se in zip(params, ses)]
        p_values = [
            2 * (1 - norm.cdf(abs(z))) if not np.isnan(z) else np.nan for z in z_values
        ]

        if hasattr(self.fit.fit_result, "parameter_names") and self.fit.fit_result.parameter_names:
            param_ids = self.fit.fit_result.parameter_names
        else:
            param_ids = [p.id for p in self.fit.model_ir.parameters]

        data = {
            "Estimate": params,
            "Std.Error": ses,
            "z-value": z_values,
            "P(>|z|)": p_values,
        }
        return pd.DataFrame(data, index=param_ids)

    def __repr__(self) -> str:
        lines = []
        res = self.fit.optimization_result
        fit_res = self.fit.fit_result
        
        lines.append(f"Optimization converged: {res.converged}")
        lines.append(f"Iterations: {res.iterations}")
        lines.append(f"Log-likelihood: {-res.objective_value:.3f}")

        if not np.isnan(fit_res.chi_square):
             lines.append(f"Chi-square: {fit_res.chi_square:.3f} (df={fit_res.df:.0f})")
             if fit_res.df > 0:
                 pval = 1 - chi2.cdf(fit_res.chi_square, fit_res.df)
                 lines.append(f"P-value: {pval:.3f}")
        
        indices = []
        if not np.isnan(fit_res.cfi):
            indices.append(f"CFI: {fit_res.cfi:.3f}")
        if not np.isnan(fit_res.tli):
            indices.append(f"TLI: {fit_res.tli:.3f}")
        if not np.isnan(fit_res.rmsea):
            indices.append(f"RMSEA: {fit_res.rmsea:.3f}")
        if not np.isnan(fit_res.srmr):
            indices.append(f"SRMR: {fit_res.srmr:.3f}")
        
        if indices:
            lines.append(", ".join(indices))
            
        lines.append(f"AIC: {fit_res.aic:.1f}, BIC: {fit_res.bic:.1f}")
        lines.append("")
        lines.append(self.parameters.to_string())
        
        # Add variance components if any
        vc = self.fit.variance_components()
        if not vc.empty:
            lines.append("")
            lines.append("Variance Components:")
            lines.append(vc.to_string(index=False))
        
        # Add covariance weights if any
        for cov in self.fit.model_ir.covariances:
            if cov.structure == "multi_kernel_simplex":
                weights_info = self.fit.get_covariance_weights(cov.id)
                if weights_info:
                    lines.append("")
                    lines.append(f"Covariance Weights for '{cov.id}':")
                    lines.append(f"  Sigma^2: {weights_info['sigma_sq']:.4f}")
                    lines.append("  Weights:")
                    for i, w in enumerate(weights_info['weights']):
                        lines.append(f"    Kernel {i+1}: {w:.4f}")

        # Add variable metadata if any
        vars_with_meta = [v for v in self.fit.model_ir.variables if v.label or v.measurement_level]
        if vars_with_meta:
            lines.append("")
            lines.append("Variable Metadata:")
            meta_data = {
                "Variable": [v.name for v in vars_with_meta],
                "Label": [v.label for v in vars_with_meta],
                "Level": [v.measurement_level for v in vars_with_meta]
            }
            lines.append(pd.DataFrame(meta_data).to_string(index=False))
        
        return "\n".join(lines)



class Model:
    """Parse SEM-style formulas into a :class:`_libsemx.ModelIR` payload.

    Parameters
    ----------
    equations:
        Iterable of lavaan-style strings (``=~`` for loadings, ``~`` for
        regressions, ``~~`` for covariances).
    families:
        Mapping from observed variable names to outcome family identifiers
        (``gaussian``, ``binomial``, etc.).
    kinds:
        Optional mapping overriding the default variable kind inference. Valid
        values are ``observed``, ``latent``, or ``grouping``.
    """

    def __init__(
        self,
        equations: Iterable[str],
        *,
        families: Mapping[str, str],
        kinds: Optional[Mapping[str, str]] = None,
        covariances: Optional[Sequence[Mapping[str, Any]]] = None,
        genomic: Optional[Mapping[str, Any]] = None,
        random_effects: Optional[Sequence[Mapping[str, Any]]] = None,
        labels: Optional[Mapping[str, str]] = None,
        measurement_levels: Optional[Mapping[str, str]] = None,
    ) -> None:
        cleaned = [eq.strip() for eq in equations if eq and eq.strip()]
        if not cleaned:
            raise ModelSpecificationError("at least one equation is required")
        if not families:
            raise ModelSpecificationError("at least one observed family is required")

        self._equations = cleaned
        self._families = {k: v for k, v in families.items()}
        self._explicit_kinds = {
            name: self._normalize_kind(value) for name, value in (kinds or {}).items()
        }
        self._labels = labels or {}
        self._measurement_levels = measurement_levels or {}
        self._covariances = list(covariances or [])
        self._genomic_specs = self._normalize_genomic_specs(genomic or {})
        existing_cov_ids = {cov["name"] for cov in self._covariances}
        for cov_id, spec in self._genomic_specs.items():
            if cov_id in existing_cov_ids:
                existing = next(cov for cov in self._covariances if cov["name"] == cov_id)
                # Only check dimension if not a genomic/GRM structure (where dim=q, markers=N)
                # Use first marker matrix for dimension check
                first_markers = spec["markers"][0]
                if spec["structure"] not in ("grm", "genomic") and existing["dimension"] != first_markers.shape[0]:
                    raise ModelSpecificationError(
                        f"Genomic covariance '{cov_id}' dimension mismatch: "
                        f"{existing['dimension']} vs markers with {first_markers.shape[0]} rows"
                    )
                continue
            # Use first marker matrix for dimension
            first_markers = spec["markers"][0]
            self._covariances.append(
                {
                    "name": cov_id,
                    "structure": spec["structure"],
                    "dimension": first_markers.shape[0],
                }
            )
        self._random_effects = list(random_effects or [])
        self._variables: MutableMapping[str, _VariableDef] = {}
        self._edges: List[_EdgeDef] = []
        self._param_counts: Dict[str, int] = {}
        self._re_count = 0
        self._uses_intercept_column = False
        self._fixed_covariance_cache: Optional[Dict[str, List[List[float]]]] = None
        self._ir: Optional[ModelIR] = None
        self._parse_equations()
        self._register_random_effect_variables()

    def to_ir(self) -> ModelIR:
        """Materialize the shared IR structure (cached after first build)."""

        if self._ir is None:
            builder = ModelIRBuilder()
            for var in self._variables.values():
                builder.add_variable(var.name, var.kind, var.family, var.label, var.measurement_level)
            for edge in self._edges:
                builder.add_edge(edge.kind, edge.source, edge.target, edge.parameter_id)
            for cov in self._covariances:
                builder.add_covariance(cov["name"], cov["structure"], cov["dimension"])
            for re in self._random_effects:
                builder.add_random_effect(re["name"], re["variables"], re["covariance"])
            self._ir = builder.build()
        return self._ir

    def fixed_covariance_data(self) -> Dict[str, List[List[float]]]:
        """Return fixed covariance matrices derived from genomic marker inputs."""

        if not self._genomic_specs:
            return {}

        if self._fixed_covariance_cache is None:
            cache: Dict[str, List[List[float]]] = {}
            for cov_id, spec in self._genomic_specs.items():
                kernels = []
                for markers in spec["markers"]:
                    if spec["precomputed"]:
                        kernel = markers.flatten(order="C").tolist()
                    else:
                        kernel = GenomicRelationshipMatrix.vanraden(
                            markers.flatten(order="C").tolist(),
                            int(markers.shape[0]),
                            int(markers.shape[1]),
                            spec["center"],
                            spec["normalize"],
                        )
                    kernels.append(kernel)
                cache[cov_id] = kernels
            self._fixed_covariance_cache = cache
        return dict(self._fixed_covariance_cache)

    def fit(
        self,
        data: Any,
        options: Optional[Any] = None,
        optimizer_name: str = "lbfgs",
        fixed_covariance_data: Optional[Mapping[str, Sequence[Sequence[float]]]] = None,
        **kwargs
    ) -> SemFit:
        """Fit the model to data.

        Parameters
        ----------
        data : Any
            Pandas DataFrame or dictionary of lists.
        options : OptimizationOptions, optional
            Custom optimization options.
        optimizer_name : str, optional
            Name of the optimizer to use (default: "lbfgs").
        fixed_covariance_data : dict, optional
            Fixed covariance matrices.
        **kwargs
            Additional optimization options passed to OptimizationOptions if options is None.
            Supported: max_iterations, tolerance, learning_rate, m, past, delta, max_linesearch, linesearch_type.

        Returns
        -------
        SemFit
            The result of the fit, including parameter estimates, standard errors,
            fit statistics, and diagnostic methods.
        """
        from _libsemx import LikelihoodDriver, OptimizationOptions

        if options is None:
            options = OptimizationOptions()
            if "max_iterations" in kwargs:
                options.max_iterations = kwargs["max_iterations"]
            if "tolerance" in kwargs:
                options.tolerance = kwargs["tolerance"]
            if "learning_rate" in kwargs:
                options.learning_rate = kwargs["learning_rate"]
            if "m" in kwargs:
                options.m = kwargs["m"]
            if "past" in kwargs:
                options.past = kwargs["past"]
            if "delta" in kwargs:
                options.delta = kwargs["delta"]
            if "max_linesearch" in kwargs:
                options.max_linesearch = kwargs["max_linesearch"]
            if "linesearch_type" in kwargs:
                options.linesearch_type = kwargs["linesearch_type"]

        # Convert data to dict of lists for C++ if it's a DataFrame
        if hasattr(data, "to_dict"):
            data_dict = data.to_dict(orient="list")
        else:
            data_dict = dict(data)

        # Inject intercept column if needed
        if self._uses_intercept_column and "_intercept" not in data_dict:
            # Find length of data
            n_rows = 0
            if data_dict:
                n_rows = len(next(iter(data_dict.values())))
            data_dict["_intercept"] = [1.0] * n_rows

        # Compute sample stats for diagnostics
        ir = self.to_ir()
        all_vars = [v.name for v in ir.variables]
        var_to_idx = {name: i for i, name in enumerate(all_vars)}
        n_total = len(all_vars)

        # Create a DataFrame to easily compute cov/mean
        df = pd.DataFrame(data_dict)

        # Filter to only observed variables in the model
        observed_vars = [
            v.name for v in ir.variables if v.kind == VariableKind.Observed
        ]
        # Ensure all observed variables are in data
        missing_cols = [v for v in observed_vars if v not in df.columns]
        if missing_cols:
            raise ValueError(f"Data missing columns for variables: {missing_cols}")

        df_subset = df[observed_vars]

        # Compute stats (using pandas defaults: ignores NaNs in mean, pairwise for cov)
        means = df_subset.mean().values.tolist()
        cov_df = df_subset.cov()
        cov_values = cov_df.values
        n_obs = len(df_subset)

        # Construct full covariance matrix and means vector (including latents as 0)
        full_cov = np.zeros((n_total, n_total))
        full_means = np.zeros(n_total)

        # Fill means
        for i, name in enumerate(observed_vars):
            idx = var_to_idx[name]
            full_means[idx] = means[i]

        # Fill covariance
        for r_i, r_name in enumerate(observed_vars):
            r_idx = var_to_idx[r_name]
            for c_i, c_name in enumerate(observed_vars):
                c_idx = var_to_idx[c_name]
                full_cov[r_idx, c_idx] = cov_values[r_i, c_i]

        sample_stats = {
            "means": full_means.tolist(),
            "covariance": full_cov.flatten().tolist(),
            "n_obs": n_obs,
        }

        # Prepare status map for survival models
        status_map: Dict[str, List[float]] = {}
        if hasattr(self, "_survival_status_map"):
            for time_var, status_var in self._survival_status_map.items():
                if status_var not in df.columns:
                    raise ValueError(f"Status variable '{status_var}' for outcome '{time_var}' not found in data")
                status_map[time_var] = df[status_var].astype(float).tolist()

        driver = LikelihoodDriver()
        merged_fixed: Dict[str, List[List[float]]] = dict(fixed_covariance_data or {})
        for cov_id, kernel_list in self.fixed_covariance_data().items():
            merged_fixed.setdefault(cov_id, kernel_list)

        # Note: LikelihoodDriver.fit() signature in bindings needs to support 'status' argument
        # Currently it is: fit(model, data, options, optimizer_name, fixed_covariance_data)
        # But evaluate_model_loglik takes status.
        # We need to update the binding or pass it via data with a convention?
        # The C++ fit() method does NOT take status.
        # However, LikelihoodDriver::fit calls ModelObjective constructor.
        # ModelObjective constructor takes data.
        # It seems ModelObjective doesn't take status either in constructor?
        # Let's check ModelObjective constructor in C++.
        
        # In C++:
        # ModelObjective(const LikelihoodDriver& driver, const ModelIR& model, const std::unordered_map<std::string, std::vector<double>>& data, ...)
        # It does NOT take status.
        
        # But ModelObjective::value() calls driver_.evaluate_model_loglik(... status ...)
        # Where does it get status from?
        # It seems ModelObjective currently passes empty status {} in value().
        
        # This implies we need to update ModelObjective to accept and store status map!
        # And update LikelihoodDriver::fit to accept status map and pass it to ModelObjective.
        # And update bindings.
        
        # For now, I will assume I need to do those C++ updates.
        # I will pass status_map to fit() assuming I will update the binding.
        
        fit_res = driver.fit(ir, data_dict, options, optimizer_name, merged_fixed, status_map)

        return SemFit(ir, fit_res, sample_stats)

    @property
    def variables(self) -> Mapping[str, _VariableDef]:
        return dict(self._variables)

    @property
    def edges(self) -> Sequence[_EdgeDef]:
        return list(self._edges)

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _parse_equations(self) -> None:
        for eq in self._equations:
            if "=~" in eq:
                lhs, rhs = eq.split("=~", 1)
                self._add_loading(lhs.strip(), self._split_terms(rhs))
            elif "~~" in eq:
                lhs, rhs = eq.split("~~", 1)
                self._add_covariance(lhs.strip(), rhs.strip())
            elif "~" in eq:
                lhs, rhs = eq.split("~", 1)
                self._add_regression(lhs.strip(), self._split_terms(rhs))
            else:
                raise ModelSpecificationError(f"Unrecognized equation: {eq}")

    def _add_loading(self, latent: str, indicators: Sequence[str]) -> None:
        latent_var = self._ensure_variable(
            latent, VariableKind.Latent, enforce_kind=True, respect_explicit=False
        )
        if not indicators:
            raise ModelSpecificationError(f"Loading equation for {latent} is empty")
        for i, indicator in enumerate(indicators):
            target_var = self._ensure_variable(
                indicator, VariableKind.Observed, enforce_kind=True, respect_explicit=False
            )
            # Default marker variable strategy: fix first loading to 1.0
            param_id = "1.0" if i == 0 else self._next_parameter_id("lambda", latent_var.name, target_var.name)
            
            self._edges.append(
                _EdgeDef(
                    kind=EdgeKind.Loading,
                    source=latent_var.name,
                    target=target_var.name,
                    parameter_id=param_id,
                )
            )

    def _add_regression(self, target: str, predictors: Sequence[str]) -> None:
        # Check for Surv(time, status) syntax
        if target.startswith("Surv(") and target.endswith(")"):
            content = target[5:-1]
            parts = [p.strip() for p in content.split(",")]
            if len(parts) != 2:
                raise ModelSpecificationError(f"Surv() requires exactly 2 arguments (time, status), got: {target}")
            time_var_name, status_var_name = parts
            
            # Register time variable as the observed outcome
            target_var = self._ensure_variable(time_var_name, VariableKind.Observed)
            
            # Register status variable (implicitly observed, but not an outcome in the graph)
            # We don't add it to the graph as a node, but we need to ensure it exists in data
            # We can store it in a side-channel to be used during fit()
            if not hasattr(self, "_survival_status_map"):
                self._survival_status_map = {}
            self._survival_status_map[time_var_name] = status_var_name
            
        else:
            target_var = self._ensure_variable(target, VariableKind.Observed)

        if not predictors:
            raise ModelSpecificationError(f"Regression equation for {target} is empty")
        for predictor in predictors:
            mixed = self._parse_mixed_term(predictor)
            if mixed:
                self._add_random_effect(mixed)
                continue

            if predictor == "1":
                predictor = "_intercept"
                self._uses_intercept_column = True
                if "_intercept" not in self._families:
                    self._families["_intercept"] = "gaussian"
            elif predictor == "0":
                continue
            
            source_var = self._ensure_variable(predictor, VariableKind.Observed)
            self._edges.append(
                _EdgeDef(
                    kind=EdgeKind.Regression,
                    source=source_var.name,
                    target=target_var.name,
                    parameter_id=self._next_parameter_id("beta", source_var.name, target_var.name),
                )
            )

    def _add_covariance(self, left: str, right: str) -> None:
        a = self._ensure_variable(left, VariableKind.Observed)
        b = self._ensure_variable(right, VariableKind.Observed)
        ordered = tuple(sorted((a.name, b.name)))
        self._edges.append(
            _EdgeDef(
                kind=EdgeKind.Covariance,
                source=ordered[0],
                target=ordered[1],
                parameter_id=self._next_parameter_id("psi", ordered[0], ordered[1]),
            )
        )

    def _ensure_variable(
        self,
        name: str,
        default_kind: VariableKind,
        *,
        enforce_kind: Optional[bool] = None,
        respect_explicit: bool = True,
    ) -> _VariableDef:
        normalized = name.strip()
        if not normalized:
            raise ModelSpecificationError("variable names cannot be empty")
        explicit_kind = self._explicit_kinds.get(normalized)
        if not respect_explicit and explicit_kind is not None and explicit_kind != default_kind:
            raise ModelSpecificationError(
                f"Variable '{normalized}' cannot be forced to {explicit_kind.name.lower()} in this context"
            )
        desired_kind = explicit_kind if (respect_explicit and explicit_kind is not None) else default_kind
        enforce = enforce_kind if enforce_kind is not None else normalized in self._explicit_kinds
        if normalized in self._variables:
            var = self._variables[normalized]
            if enforce and var.kind != desired_kind:
                raise ModelSpecificationError(
                    f"Variable '{normalized}' was previously registered as {var.kind.name}"
                )
            return var

        kind = desired_kind
        family = self._families.get(normalized, "") if kind == VariableKind.Observed else ""
        if kind == VariableKind.Observed and not family:
            raise ModelSpecificationError(
                f"Observed variable '{normalized}' requires an outcome family"
            )
        var_def = _VariableDef(
            normalized, 
            kind, 
            family, 
            self._labels.get(normalized, ""), 
            self._measurement_levels.get(normalized, "")
        )
        self._variables[normalized] = var_def
        return var_def

    def _register_random_effect_variables(self) -> None:
        if not self._random_effects:
            return
        for effect in self._random_effects:
            variables = effect.get("variables") or []
            if not variables:
                continue
            grouping = variables[0]
            self._ensure_variable(grouping, VariableKind.Grouping, enforce_kind=True, respect_explicit=True)
            for var in variables[1:]:
                if var == "_intercept":
                    self._uses_intercept_column = True
                    if "_intercept" not in self._families:
                        self._families["_intercept"] = "gaussian"
                # Design variables default to observed; require a family mapping upstream.
                self._ensure_variable(var, VariableKind.Observed)

    def _next_parameter_id(self, prefix: str, source: str, target: str) -> str:
        base = f"{prefix}_{target}_on_{source}" if prefix != "psi" else f"{prefix}_{source}_{target}"
        count = self._param_counts.get(base, 0)
        self._param_counts[base] = count + 1
        if count == 0:
            return base
        return f"{base}_{count + 1}"

    @staticmethod
    def _normalize_genomic_specs(genomic: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        specs: Dict[str, Dict[str, Any]] = {}
        for cov_id, payload in genomic.items():
            raw_markers = payload.get("markers")
            if raw_markers is None:
                raise ModelSpecificationError(f"Genomic covariance '{cov_id}' requires a 'markers' matrix")
            
            structure = payload.get("structure", "grm")
            is_multi = "multi_kernel" in structure
            
            marker_list = []
            if is_multi:
                if not isinstance(raw_markers, list):
                     # It might be a single array, but user wants multi_kernel (maybe just 1 kernel)
                     # But usually multi_kernel implies list.
                     # If it's a numpy array, it's a single matrix.
                     if hasattr(raw_markers, "ndim") and raw_markers.ndim == 2:
                         marker_list = [np.asarray(raw_markers, dtype=float)]
                     else:
                         raise ModelSpecificationError(f"Multi-kernel covariance '{cov_id}' requires a list of marker matrices")
                else:
                     marker_list = [np.asarray(m, dtype=float) for m in raw_markers]
            else:
                # Single matrix expected
                if isinstance(raw_markers, list) and len(raw_markers) > 0 and hasattr(raw_markers[0], "ndim"):
                     # User passed list of matrices but structure is not multi_kernel?
                     # Assume first one or error?
                     # Let's assume single matrix.
                     marker_list = [np.asarray(raw_markers, dtype=float)]
                else:
                     marker_list = [np.asarray(raw_markers, dtype=float)]

            # Validate dimensions
            if not marker_list:
                 raise ModelSpecificationError(f"Genomic covariance '{cov_id}' has no markers")
                 
            dim = marker_list[0].shape[0]
            for m in marker_list:
                if m.ndim != 2:
                    raise ModelSpecificationError(f"Genomic covariance '{cov_id}' markers must be a non-empty 2D array")
                if m.shape[0] != dim:
                     raise ModelSpecificationError(f"Genomic covariance '{cov_id}' markers dimension mismatch")

            precomputed = bool(payload.get("precomputed", False))
            if precomputed:
                for m in marker_list:
                    if m.shape[0] != m.shape[1]:
                        raise ModelSpecificationError(
                            f"Genomic covariance '{cov_id}' precomputed kernel must be square"
                        )
            specs[cov_id] = {
                "markers": marker_list,
                "structure": structure,
                "center": bool(payload.get("center", True)),
                "normalize": bool(payload.get("normalize", True)),
                "precomputed": precomputed,
            }
        return specs

    def _parse_mixed_term(self, term: str) -> Optional[Dict[str, Any]]:
        # Check if it looks like (lhs | rhs)
        if not (term.startswith("(") and term.endswith(")") and "|" in term):
            return None

        content = term[1:-1]
        parts = content.split("|")
        if len(parts) != 2:
            raise ModelSpecificationError(f"Invalid mixed model term: {term}")

        lhs_str, rhs_str = parts[0].strip(), parts[1].strip()

        # Parse LHS (design variables)
        lhs_terms = [t.strip() for t in lhs_str.split("+") if t.strip()]

        has_intercept = True  # Default in R/lme4
        design_vars = []

        for t in lhs_terms:
            if t == "1":
                has_intercept = True
            elif t == "0":
                has_intercept = False
            else:
                design_vars.append(t)

        return {
            "grouping": rhs_str,
            "design": design_vars,
            "intercept": has_intercept,
        }

    def _add_random_effect(self, spec: Dict[str, Any]) -> None:
        group = spec["grouping"]
        design = spec["design"]
        intercept = spec["intercept"]

        # Ensure grouping variable
        self._ensure_variable(
            group, VariableKind.Grouping, enforce_kind=True, respect_explicit=True
        )

        ir_vars = [group]

        # Determine dimension and variables
        if not design:
            # Intercept only: (1 | group)
            if not intercept:
                raise ModelSpecificationError(
                    "Random effect term must have at least one variable (intercept or design)"
                )
            # Implicit intercept, dim=1
            dimension = 1
        else:
            # Random slopes
            if intercept:
                ir_vars.append("_intercept")
                self._uses_intercept_column = True
                # Ensure _intercept is registered
                if "_intercept" not in self._families:
                    self._families["_intercept"] = "gaussian"
                self._ensure_variable("_intercept", VariableKind.Observed)

            for dv in design:
                self._ensure_variable(dv, VariableKind.Observed)
                ir_vars.append(dv)

            dimension = len(ir_vars) - 1  # Subtract grouping var

        # Create covariance
        self._re_count += 1
        cov_name = f"cov_re_{self._re_count}"

        self._covariances.append(
            {
                "name": cov_name,
                "structure": "unstructured",  # Default for lme4
                "dimension": dimension,
            }
        )

        self._random_effects.append(
            {
                "name": f"re_{group}_{self._re_count}",
                "variables": ir_vars,
                "covariance": cov_name,
            }
        )

    @staticmethod
    def _split_terms(rhs: str) -> List[str]:
        terms = []
        current = []
        depth = 0
        for char in rhs:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1

            if char == "+" and depth == 0:
                terms.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        if current:
            terms.append("".join(current).strip())
        return [t for t in terms if t]

    @staticmethod
    def _normalize_kind(label: str) -> VariableKind:
        key = label.strip().lower()
        if key not in _VARIABLE_KIND_ALIASES:
            raise ModelSpecificationError(f"Unknown variable kind: {label}")
        return _VARIABLE_KIND_ALIASES[key]
