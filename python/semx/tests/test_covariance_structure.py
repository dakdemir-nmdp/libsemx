"""Tests for covariance structure functionality."""

from __future__ import annotations

import pytest

semx = pytest.importorskip("semx")


@pytest.mark.parametrize(
    ("structure", "dimension"),
    [
        ("compound_symmetry", 2),
        ("cs", 3),
        ("ar1", 4),
        ("toeplitz", 3),
        ("fa1", 3),
    ],
)
def test_model_ir_records_new_covariance_structures(structure: str, dimension: int) -> None:
    """ModelIRBuilder should serialize the new covariance identifiers for bindings."""

    builder = semx.ModelIRBuilder()
    builder.add_variable("y", semx.VariableKind.Observed, "gaussian")
    builder.add_variable("cluster", semx.VariableKind.Grouping)
    builder.add_edge(semx.EdgeKind.Regression, "cluster", "y", "beta")
    builder.add_covariance("G_new", structure, dimension)

    design_vars = ["cluster"]
    if dimension > 1:
        for idx in range(dimension):
            latent_name = f"z{idx}"
            builder.add_variable(latent_name, semx.VariableKind.Latent)
            design_vars.append(latent_name)

    builder.add_random_effect("u_cluster", design_vars, "G_new")

    model = builder.build()
    assert len(model.covariances) == 1
    cov = model.covariances[0]
    assert cov.id == "G_new"
    assert cov.structure == structure
    assert cov.dimension == dimension