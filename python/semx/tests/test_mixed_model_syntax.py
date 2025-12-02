
import pytest
from semx import Model, ModelSpecificationError, VariableKind, EdgeKind

def test_mixed_model_intercept_only():
    # y ~ x + (1 | cluster)
    model = Model(
        equations=["y ~ x + (1 | cluster)"],
        families={"y": "gaussian", "x": "gaussian"},
        kinds={"cluster": "grouping"}
    )
    
    ir = model.to_ir()
    
    # Check random effects
    assert len(ir.random_effects) == 1
    re = ir.random_effects[0]
    assert re.variables == ["cluster"]
    assert re.covariance_id is not None
    
    # Check covariance structure
    cov = next(c for c in ir.covariances if c.id == re.covariance_id)
    assert cov.structure == "unstructured" # Default for single random effect
    assert cov.dimension == 1

def test_mixed_model_random_slope():
    # y ~ x + (x | cluster)
    # This implies random intercept AND random slope for x
    model = Model(
        equations=["y ~ x + (x | cluster)"],
        families={"y": "gaussian", "x": "gaussian"},
        kinds={"cluster": "grouping"}
    )
    
    ir = model.to_ir()
    
    assert len(ir.random_effects) == 1
    re = ir.random_effects[0]
    # Should include cluster and the variable for x's random slope?
    # Actually, in libsemx IR, random effects are defined by the grouping variable(s) 
    # and the variables that have random coefficients.
    # Wait, let's check ModelIR structure.
    # RandomEffectSpec: id, variables (list of strings), covariance_id.
    # Usually variables[0] is the grouping variable.
    # If we have (1 | cluster), variables = ["cluster"].
    # If we have (x | cluster), variables = ["cluster", "x"].
    
    assert re.variables == ["cluster", "_intercept", "x"]
    
    cov = next(c for c in ir.covariances if c.id == re.covariance_id)
    assert cov.dimension == 2 # Intercept + Slope
    assert cov.structure == "unstructured" # Default for multiple REs in same group

def test_mixed_model_explicit_intercept_and_slope():
    # y ~ x + (1 + x | cluster)
    model = Model(
        equations=["y ~ x + (1 + x | cluster)"],
        families={"y": "gaussian", "x": "gaussian"},
        kinds={"cluster": "grouping"}
    )
    
    ir = model.to_ir()
    re = ir.random_effects[0]
    assert re.variables == ["cluster", "_intercept", "x"]
    
    cov = next(c for c in ir.covariances if c.id == re.covariance_id)
    assert cov.dimension == 2

def test_mixed_model_multiple_groups():
    # y ~ x + (1 | school) + (1 | class)
    model = Model(
        equations=["y ~ x + (1 | school) + (1 | class)"],
        families={"y": "gaussian", "x": "gaussian"},
        kinds={"school": "grouping", "class": "grouping"}
    )
    
    ir = model.to_ir()
    assert len(ir.random_effects) == 2
    
    re_school = next(r for r in ir.random_effects if "school" in r.variables)
    assert re_school.variables == ["school"]
    
    re_class = next(r for r in ir.random_effects if "class" in r.variables)
    assert re_class.variables == ["class"]

def test_mixed_model_no_intercept():
    # y ~ x + (0 + x | cluster)
    model = Model(
        equations=["y ~ x + (0 + x | cluster)"],
        families={"y": "gaussian", "x": "gaussian"},
        kinds={"cluster": "grouping"}
    )
    
    ir = model.to_ir()
    re = ir.random_effects[0]
    assert re.variables == ["cluster", "x"]
    
    cov = next(c for c in ir.covariances if c.id == re.covariance_id)
    assert cov.dimension == 1 # Only slope, no intercept
