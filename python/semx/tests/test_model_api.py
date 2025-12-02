"""Tests for the high-level semx.Model front-end."""

import pytest

from semx import EdgeKind, Model, ModelSpecificationError, VariableKind


def test_model_generates_ir_from_formulas():
    model = Model(
        equations=[
            "eta =~ y1 + y2",
            "y1 ~ eta + x1",
            "y2 ~ x2",
            "y1 ~~ y2",
        ],
        families={"y1": "gaussian", "y2": "gaussian", "x1": "gaussian", "x2": "gaussian"},
    )

    ir = model.to_ir()
    var_map = {spec.name: spec for spec in ir.variables}
    assert var_map["eta"].kind == VariableKind.Latent
    assert var_map["y1"].family == "gaussian"
    assert var_map["x1"].kind == VariableKind.Observed

    kinds = [edge.kind for edge in ir.edges]
    assert kinds.count(EdgeKind.Loading) == 2
    assert kinds.count(EdgeKind.Regression) == 3
    assert kinds.count(EdgeKind.Covariance) == 1


def test_model_respects_explicit_kinds():
    model = Model(
        equations=["y ~ 1 + cluster"],
        families={"y": "gaussian"},
        kinds={"cluster": "grouping"},
    )

    ir = model.to_ir()
    var_map = {spec.name: spec for spec in ir.variables}
    assert var_map["cluster"].kind == VariableKind.Grouping


def test_model_requires_families_for_observed_variables():
    with pytest.raises(ModelSpecificationError):
        Model(equations=["y ~ x"], families={"x": "gaussian"})


def test_model_rejects_incompatible_kinds():
    with pytest.raises(ModelSpecificationError):
        Model(
            equations=["eta =~ y1", "y1 =~ x1"],
            families={"y1": "gaussian", "x1": "gaussian"},
            kinds={"y1": "observed"},
        )


def test_model_parses_survival_syntax():
    model = Model(
        equations=["Surv(time, status) ~ age + treatment"],
        families={"time": "weibull", "age": "gaussian", "treatment": "binomial"},
    )
    
    ir = model.to_ir()
    var_map = {spec.name: spec for spec in ir.variables}
    
    # Check variables
    assert "time" in var_map
    assert var_map["time"].kind == VariableKind.Observed
    assert var_map["time"].family == "weibull"
    
    # Status variable is not in the graph as a node, but registered internally
    assert hasattr(model, "_survival_status_map")
    assert model._survival_status_map["time"] == "status"
    
    # Check edges
    edges = ir.edges
    assert len(edges) == 2
    assert edges[0].kind == EdgeKind.Regression
    assert edges[0].target == "time"
    assert edges[0].source in {"age", "treatment"}