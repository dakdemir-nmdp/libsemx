"""Tests for model IR functionality."""

import pytest

try:
    import _libsemx as semx_cpp
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

    def test_builder_enforces_invariants(self):
        """Mirror C++: ModelIRBuilder enforces invariants."""
        # In C++, this throws exceptions. We should check if pybind11 translates them.
        builder = semx_cpp.ModelIRBuilder()
        # Assuming C++ throws std::invalid_argument or similar, pybind11 maps it to ValueError or RuntimeError
        
        # Example: Duplicate variable
        builder.add_variable("x", semx_cpp.VariableKind.Observed, "gaussian")
        with pytest.raises(ValueError): # C++ throws invalid_argument, mapped to ValueError
             builder.add_variable("x", semx_cpp.VariableKind.Latent)
