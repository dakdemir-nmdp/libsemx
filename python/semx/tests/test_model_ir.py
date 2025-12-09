"""Tests for model IR functionality."""

import pytest

try:
    import semx as semx_cpp
except ImportError:
    pytest.skip("C++ extension not available", allow_module_level=True)

class TestModelIR:
    """Test suite for ModelIR mirroring C++ fixtures."""

    def test_builder_assembles_valid_model(self):
        """Mirror C++: ModelIRBuilder assembles valid model."""
        builder = semx_cpp.ModelIRBuilder()
        builder.add_variable("x", semx_cpp.VariableKind.Observed, "gaussian")
        builder.add_variable("y", semx_cpp.VariableKind.Observed, "gaussian")
        builder.add_variable("f", semx_cpp.VariableKind.Latent)
        
        builder.add_edge(semx_cpp.EdgeKind.Loading, "f", "x", "lam_x")
        builder.add_edge(semx_cpp.EdgeKind.Regression, "x", "y", "beta_xy")
        
        model = builder.build()
        
        assert len(model.variables) == 3
        assert model.variables[0].name == "x"
        assert model.variables[0].family == "gaussian"
        assert model.variables[1].name == "y"
        assert model.variables[1].family == "gaussian"
        
        assert len(model.edges) == 2
        assert model.edges[0].source == "f"
        assert model.edges[0].target == "x"
        assert model.edges[0].kind == semx_cpp.EdgeKind.Loading

    def test_builder_serializes_covariances_and_random_effects(self):
        """ModelIR exposes covariance and random-effect metadata to Python."""
        builder = semx_cpp.ModelIRBuilder()
        builder.add_variable("y", semx_cpp.VariableKind.Observed, "gaussian")
        builder.add_variable("cluster", semx_cpp.VariableKind.Grouping)
        builder.add_covariance("G_diag", "diagonal", 1)
        builder.add_random_effect("u_cluster", ["cluster"], "G_diag")

        model = builder.build()
        assert len(model.covariances) == 1
        assert model.covariances[0].id == "G_diag"
        assert model.covariances[0].structure == "diagonal"
        assert len(model.random_effects) == 1
        assert model.random_effects[0].id == "u_cluster"
        assert model.random_effects[0].variables == ["cluster"]
        assert model.random_effects[0].covariance_id == "G_diag"

    def test_builder_enforces_invariants(self):
        """Mirror C++: ModelIRBuilder enforces invariants."""
        # Duplicate variable names must fail
        builder = semx_cpp.ModelIRBuilder()
        builder.add_variable("x", semx_cpp.VariableKind.Observed, "gaussian")
        with pytest.raises(ValueError, match="duplicate variable name"):
            builder.add_variable("x", semx_cpp.VariableKind.Latent)

        # Observed outcomes must declare a family
        builder = semx_cpp.ModelIRBuilder()
        with pytest.raises(ValueError, match="requires outcome family"):
            builder.add_variable("y", semx_cpp.VariableKind.Observed)

        # Edges require parameter identifiers
        builder = semx_cpp.ModelIRBuilder()
        builder.add_variable("y", semx_cpp.VariableKind.Observed, "gaussian")
        builder.add_variable("x", semx_cpp.VariableKind.Latent)
        with pytest.raises(ValueError, match="parameter id must be non-empty"):
            builder.add_edge(semx_cpp.EdgeKind.Regression, "x", "y", "")

        # Covariance specs must be well-defined
        with pytest.raises(ValueError, match="covariance id must be non-empty"):
            builder.add_covariance("", "diagonal", 1)
        with pytest.raises(ValueError, match="covariance dimension must be positive"):
            builder.add_covariance("G", "diagonal", 0)
        builder.add_covariance("G", "diagonal", 1)

        # Random effects need variables and known covariance ids
        with pytest.raises(ValueError, match="must reference at least one variable"):
            builder.add_random_effect("u", [], "G")
        with pytest.raises(ValueError, match="unknown variable"):
            builder.add_random_effect("u", ["missing"], "G")
        with pytest.raises(ValueError, match="unknown covariance id"):
            builder.add_random_effect("u", ["x"], "missing")
