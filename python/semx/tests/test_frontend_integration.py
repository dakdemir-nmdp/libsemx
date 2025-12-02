"""Integration tests for the semx.Model front-end."""

import pytest
from semx import Model, VariableKind

def test_frontend_roundtrip_laplace_glmm():
    """
    Round-trip a Binomial GLMM (Laplace) model.
    Corresponds to: y ~ 1 + (1 | cluster)
    """
    model = Model(
        equations=["y ~ 1 + cluster"],
        families={"y": "binomial"},
        kinds={"cluster": "grouping"},
        covariances=[
            {"name": "G_diag", "structure": "diagonal", "dimension": 1}
        ],
        random_effects=[
            {"name": "u_cluster", "variables": ["cluster"], "covariance": "G_diag"}
        ]
    )
    
    ir = model.to_ir()
    
    # Check variables
    var_map = {v.name: v for v in ir.variables}
    assert var_map["y"].family == "binomial"
    assert var_map["cluster"].kind == VariableKind.Grouping
    
    # Check covariances
    assert len(ir.covariances) == 1
    assert ir.covariances[0].id == "G_diag"
    assert ir.covariances[0].structure == "diagonal"
    
    # Check random effects
    assert len(ir.random_effects) == 1
    assert ir.random_effects[0].id == "u_cluster"
    assert ir.random_effects[0].variables == ["cluster"]
    assert ir.random_effects[0].covariance_id == "G_diag"

def test_frontend_roundtrip_survival():
    """
    Round-trip a Survival model (Weibull).
    Corresponds to: y ~ x
    """
    model = Model(
        equations=["y ~ x"],
        families={"y": "weibull", "x": "gaussian"},
    )
    ir = model.to_ir()
    
    # Check variables
    var_map = {v.name: v for v in ir.variables}
    assert var_map["y"].family == "weibull"
    assert var_map["x"].kind == VariableKind.Observed

def test_frontend_roundtrip_kronecker():
    """
    Round-trip a Kronecker covariance model.
    """
    model = Model(
        equations=["y ~ 1 + env + geno"],
        families={"y": "gaussian"},
        kinds={"env": "grouping", "geno": "grouping"},
        covariances=[
            {"name": "G_kron", "structure": "kronecker", "dimension": 1}
        ],
        random_effects=[
            {"name": "u_gxe", "variables": ["env", "geno"], "covariance": "G_kron"}
        ]
    )
    
    ir = model.to_ir()
    
    assert len(ir.covariances) == 1
    assert ir.covariances[0].id == "G_kron"
    assert ir.covariances[0].structure == "kronecker"
    
    assert len(ir.random_effects) == 1
    assert ir.random_effects[0].id == "u_gxe"
    assert ir.random_effects[0].variables == ["env", "geno"]
    assert ir.random_effects[0].covariance_id == "G_kron"
