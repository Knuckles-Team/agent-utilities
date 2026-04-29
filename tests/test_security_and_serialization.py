"""Tests for safe serialization functions (CONCEPT:AU-005 Serialization Safety).

Verifies the JSON-based safe_save_model and safe_load_model functions
provide equivalent functionality to the deprecated pickle-based alternatives
without the CWE-502 security risk.
"""
import json
import os
import warnings

import pytest


# CONCEPT:AU-005 Serialization Safety
@pytest.mark.concept("AU-005")
class TestSafeSerialization:
    """Test suite for the safe serialization API."""

    def test_safe_save_model_dict(self, tmp_path):
        """safe_save_model correctly serializes a plain dict to JSON."""
        from agent_utilities.base_utilities import safe_save_model

        data = {"key": "value", "nested": {"a": 1}}
        path = safe_save_model(data, file_name="test_dict", file_path=str(tmp_path))
        assert path.endswith(".json")
        assert os.path.exists(path)

        with open(path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_safe_load_model(self, tmp_path):
        """safe_load_model correctly deserializes a JSON file."""
        from agent_utilities.base_utilities import safe_load_model, safe_save_model

        original = {"agents": ["router", "planner"], "version": 42}
        path = safe_save_model(original, file_name="roundtrip", file_path=str(tmp_path))
        loaded = safe_load_model(path)
        assert loaded == original

    def test_safe_save_model_list(self, tmp_path):
        """safe_save_model handles list inputs."""
        from agent_utilities.base_utilities import safe_save_model

        data = [1, 2, {"three": 3}]
        path = safe_save_model(data, file_name="test_list", file_path=str(tmp_path))
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_deprecated_save_model_warns(self, tmp_path):
        """Legacy save_model emits a DeprecationWarning about CWE-502."""
        from agent_utilities.base_utilities import save_model

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            save_model({"test": True}, file_name="deprecated", file_path=str(tmp_path))
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "CWE-502" in str(w[0].message)

    def test_deprecated_load_model_warns(self, tmp_path):
        """Legacy load_model emits a DeprecationWarning about CWE-502."""
        from agent_utilities.base_utilities import load_model, save_model

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            path = save_model({"test": True}, file_name="dep_load", file_path=str(tmp_path))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_model(path)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "CWE-502" in str(w[0].message)
            assert result == {"test": True}


# CONCEPT:AU-003 Workspace Management
@pytest.mark.concept("AU-003")
class TestWorkspaceTraversalGuard:
    """Verify workspace path traversal protection."""

    def test_validate_rejects_traversal(self, tmp_path, monkeypatch):
        """validate_workspace_path blocks paths outside workspace."""
        from agent_utilities.workspace import validate_workspace_path
        import agent_utilities.workspace as ws_mod

        monkeypatch.setattr(ws_mod, "WORKSPACE_DIR", str(tmp_path))
        with pytest.raises(ValueError, match="outside the workspace"):
            validate_workspace_path(tmp_path / ".." / ".." / "etc" / "passwd")

    def test_validate_allows_internal_path(self, tmp_path, monkeypatch):
        """validate_workspace_path allows paths inside workspace."""
        from agent_utilities.workspace import validate_workspace_path
        import agent_utilities.workspace as ws_mod

        internal = tmp_path / "subdir" / "file.txt"
        internal.parent.mkdir(parents=True, exist_ok=True)
        internal.touch()
        monkeypatch.setattr(ws_mod, "WORKSPACE_DIR", str(tmp_path))
        result = validate_workspace_path(internal)
        assert result == internal.resolve()
