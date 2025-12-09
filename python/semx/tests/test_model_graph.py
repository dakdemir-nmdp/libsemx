"""Tests for model graph functionality exposed via ModelIRBuilder."""

import pytest

try:
    import semx as semx_cpp
except ImportError:  # pragma: no cover - exercised only when extension missing
    pytest.skip("C++ extension not available", allow_module_level=True)


class TestModelGraph:
    """Test suite for ModelGraph invariants surfaced through the builder."""

    def test_registers_observed_and_latent_variables(self):
        """ModelGraph records metadata for observed and latent variables."""
        builder = semx_cpp.ModelIRBuilder()
        builder.add_variable("y", semx_cpp.VariableKind.Observed, "gaussian")
        builder.add_variable("f", semx_cpp.VariableKind.Latent)
        builder.add_edge(semx_cpp.EdgeKind.Loading, "f", "y", "lambda_y")

        model = builder.build()

        assert [var.name for var in model.variables] == ["y", "f"]
        assert model.variables[0].family == "gaussian"
        assert model.variables[1].family == ""
        assert model.edges[0].source == "f"
        assert model.edges[0].target == "y"
        assert model.edges[0].parameter_id == "lambda_y"

    def test_rejects_duplicate_variable_names(self):
        """Python bindings propagate ModelGraph validation failures."""
        builder = semx_cpp.ModelIRBuilder()
        builder.add_variable("y", semx_cpp.VariableKind.Observed, "gaussian")

        with pytest.raises(ValueError, match="duplicate variable name"):
            builder.add_variable("y", semx_cpp.VariableKind.Observed, "gaussian")

    def test_observed_variable_requires_family(self):
        """Observed variables must provide an outcome family identifier."""
        builder = semx_cpp.ModelIRBuilder()

        with pytest.raises(ValueError, match="requires outcome family"):
            builder.add_variable("y", semx_cpp.VariableKind.Observed)

    def test_serializes_covariances_and_random_effects(self):
        """ModelGraph preserves insertion order for covariances and random effects."""
        builder = semx_cpp.ModelIRBuilder()
        builder.add_variable("y", semx_cpp.VariableKind.Observed, "gaussian")
        builder.add_variable("cluster", semx_cpp.VariableKind.Grouping)
        builder.add_covariance("G_diag", "diagonal", 1)
        builder.add_covariance("G_kron", "multi_kernel", 4)
        builder.add_random_effect("u_diag", ["cluster"], "G_diag")
        builder.add_random_effect("u_kron", ["cluster"], "G_kron")

        model = builder.build()

        assert [cov.id for cov in model.covariances] == ["G_diag", "G_kron"]
        assert model.covariances[0].structure == "diagonal"
        assert model.covariances[1].dimension == 4
        assert [re.id for re in model.random_effects] == ["u_diag", "u_kron"]
        assert model.random_effects[0].covariance_id == "G_diag"
        assert model.random_effects[1].variables == ["cluster"]

    def test_edge_requires_registered_variables(self):
        """Edges must reference variables that exist in the graph."""
        builder = semx_cpp.ModelIRBuilder()
        builder.add_variable("y", semx_cpp.VariableKind.Observed, "gaussian")

        with pytest.raises(ValueError, match="edge source not registered"):
            builder.add_edge(semx_cpp.EdgeKind.Regression, "f", "y", "beta")

        builder.add_variable("x", semx_cpp.VariableKind.Latent)
        with pytest.raises(ValueError, match="edge target not registered"):
            builder.add_edge(semx_cpp.EdgeKind.Regression, "x", "z", "beta")

    def test_edge_requires_parameter_identifier(self):
        """Edges must be tied to a named parameter for catalog lookup."""
        builder = semx_cpp.ModelIRBuilder()
        builder.add_variable("y", semx_cpp.VariableKind.Observed, "gaussian")
        builder.add_variable("x", semx_cpp.VariableKind.Latent)

        with pytest.raises(ValueError, match="parameter id must be non-empty"):
            builder.add_edge(semx_cpp.EdgeKind.Regression, "x", "y", "")

    def test_random_effect_validations(self):
        """Random effects must reference unique variables and known covariances."""
        builder = semx_cpp.ModelIRBuilder()
        builder.add_variable("cluster", semx_cpp.VariableKind.Grouping)
        builder.add_covariance("G", "diagonal", 1)

        with pytest.raises(ValueError, match="unknown variable"):
            builder.add_random_effect("u", ["missing"], "G")

        builder.add_variable("batch", semx_cpp.VariableKind.Grouping)
        with pytest.raises(ValueError, match="multiple times"):
            builder.add_random_effect("u", ["cluster", "cluster"], "G")

        with pytest.raises(ValueError, match="unknown covariance id"):
            builder.add_random_effect("u", ["cluster"], "missing")

    def test_build_requires_variables(self):
        """ModelGraph refuses to emit IR when empty, per blueprint §§2.2–3.2."""
        builder = semx_cpp.ModelIRBuilder()

        with pytest.raises(ValueError, match="must contain at least one variable"):
            builder.build()

    def test_parameter_specs_follow_insertion_order(self):
        """ModelIR surfaces ordered parameter specs for structural edges."""
        builder = semx_cpp.ModelIRBuilder()
        builder.add_variable("y", semx_cpp.VariableKind.Observed, "gaussian")
        builder.add_variable("x1", semx_cpp.VariableKind.Latent)
        builder.add_variable("x2", semx_cpp.VariableKind.Latent)

        builder.add_edge(semx_cpp.EdgeKind.Regression, "x1", "y", "beta_x1")
        builder.add_edge(semx_cpp.EdgeKind.Regression, "x2", "y", "beta_x2")
        builder.add_edge(semx_cpp.EdgeKind.Regression, "x1", "y", "beta_x1")
        builder.add_edge(semx_cpp.EdgeKind.Regression, "x2", "y", "0.5")

        model = builder.build()
        param_ids = [spec.id for spec in model.parameters]
        assert param_ids == ["beta_x1", "beta_x2"]
        assert all(spec.constraint == semx_cpp.ParameterConstraint.Free for spec in model.parameters)
        assert all(spec.initial_value == pytest.approx(0.0) for spec in model.parameters)