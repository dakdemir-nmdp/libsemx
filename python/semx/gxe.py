from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .model import Model


def gxe_model(
    formula: str,
    data: pd.DataFrame,
    genotype: str,
    environment: str,
    genomic: Optional[Union[np.ndarray, Dict[str, Any]]] = None,
    family: str = "gaussian",
) -> Tuple[Model, pd.DataFrame]:
    """
    Builds a Genotype-by-Environment (GxE) model from a high-level specification.

    This builder simplifies the construction of Mixed Models for Multi-Environment
    Trials (MET). It assumes the data is in "long" format (one observation per
    genotype per environment).

    The constructed model includes:
      1. Fixed effects specified in ``formula``.
      2. A random intercept for ``genotype`` (G).
      3. A random intercept for ``environment`` (E).
      4. A random intercept for the GxE interaction (G:E).

    If ``genomic`` is provided, the G term uses the provided relationship matrix.
    The GxE term currently defaults to a diagonal (independent) structure.

    Args:
        formula: A string specifying the fixed effects (e.g., "yield ~ 1").
        data: A pandas DataFrame containing the data.
        genotype: The name of the column containing genotype identifiers.
        environment: The name of the column containing environment identifiers.
        genomic: Optional. A numeric matrix (numpy array) representing the
            genomic relationship matrix (GRM) for the genotypes.
        family: The outcome family (default: "gaussian").

    Returns:
        A tuple (model, data) where:
        - model: A :class:`semx.Model` instance ready for fitting.
        - data: The input DataFrame, potentially augmented with an interaction column.
    """
    # Parse response variable
    if "~" not in formula:
        raise ValueError("Formula must contain '~'")
    response = formula.split("~")[0].strip()

    # Create interaction column if needed
    interaction_col = f"{genotype}:{environment}"
    # We modify a copy to avoid side effects on the user's dataframe
    data = data.copy()
    if interaction_col not in data.columns:
        data[interaction_col] = data[genotype].astype(str) + ":" + data[environment].astype(str)

    # Encode grouping variables to integers (0-based indices)
    # This is required because the C++ core expects numeric data.
    # We use sort=True to ensure deterministic mapping (e.g. matching sorted GRM).
    if data[genotype].dtype == object or data[genotype].dtype.name == "category":
        codes, uniques = pd.factorize(data[genotype], sort=True)
        data[genotype] = codes
        if genomic is not None and genomic.shape[0] != len(uniques):
            raise ValueError(
                f"Genomic matrix dimension ({genomic.shape[0]}) does not match number of genotypes ({len(uniques)})"
            )

    if data[environment].dtype == object or data[environment].dtype.name == "category":
        data[environment] = pd.factorize(data[environment], sort=True)[0]

    if data[interaction_col].dtype == object or data[interaction_col].dtype.name == "category":
        data[interaction_col] = pd.factorize(data[interaction_col], sort=True)[0]

    # Inject intercept column explicitly since we are bypassing the formula parser's auto-injection
    if "_intercept" not in data.columns:
        data["_intercept"] = 1.0

    # Construct random effects list
    random_effects = []
    covariances = []
    genomic_args = {}

    # 1. Genotype Random Effect
    re_g = {
        "name": f"u_{genotype}",
        "variables": [genotype, "_intercept"],  # [grouping, design_var]
        "group": genotype,
        # Covariance will be set below
    }
    
    if genomic is not None:
        # Define genomic covariance
        cov_name = "K_g"
        # semx.Model expects genomic dict: name -> {markers: ...}
        genomic_args[cov_name] = {"markers": genomic}
        re_g["covariance"] = cov_name
        
        # We must also register the covariance definition
        covariances.append({
            "name": cov_name,
            "structure": "genomic",
            "dimension": 1
        })
    else:
        # Default diagonal covariance
        cov_name = f"cov_{genotype}"
        re_g["covariance"] = cov_name
        covariances.append({
            "name": cov_name,
            "structure": "diagonal",
            "dimension": 1
        })

    random_effects.append(re_g)

    # 2. Environment Random Effect
    cov_name_e = f"cov_{environment}"
    re_e = {
        "name": f"u_{environment}",
        "variables": [environment, "_intercept"],
        "group": environment,
        "covariance": cov_name_e
    }
    covariances.append({
        "name": cov_name_e,
        "structure": "diagonal",
        "dimension": 1
    })
    random_effects.append(re_e)

    # 3. Interaction Random Effect
    cov_name_gxe = f"cov_{interaction_col}"
    re_gxe = {
        "name": f"u_{interaction_col}",
        "variables": [interaction_col, "_intercept"],
        "group": interaction_col,
        "covariance": cov_name_gxe
    }
    covariances.append({
        "name": cov_name_gxe,
        "structure": "diagonal",
        "dimension": 1
    })
    random_effects.append(re_gxe)

    # Prepare families
    families = {response: family}
    # Register _intercept family to avoid ModelSpecificationError
    families["_intercept"] = "gaussian"

    # Construct Model
    # We pass the fixed effects formula.
    # We pass the explicit random effects.
    model = Model(
        equations=[formula],
        families=families,
        covariances=covariances,
        random_effects=random_effects,
        genomic=genomic_args if genomic_args else None
    )

    return model, data
