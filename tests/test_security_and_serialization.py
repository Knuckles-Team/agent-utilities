"""Tests for safe serialization functions (CONCEPT:ORCH-1.3 Serialization Safety).

Verifies the JSON-based safe_save_model and safe_load_model functions. The
pickle-based save_model/load_model were removed (No-Legacy, CWE-502); JSON is the
only serialization path.
"""

import json
import os

import pytest


# CONCEPT:ORCH-1.3 Serialization Safety
@pytest.mark.concept("CONCEPT:OS-5.0")
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

# CONCEPT:OS-5.0 Workspace Management
@pytest.mark.concept("CONCEPT:ORCH-1.0")
class TestWorkspaceTraversalGuard:
    """Verify workspace path traversal protection."""

    def test_validate_rejects_traversal(self, tmp_path, monkeypatch):
        """validate_workspace_path blocks paths outside workspace."""
        import agent_utilities.core.workspace as ws_mod
        from agent_utilities.core.workspace import validate_workspace_path

        monkeypatch.setattr(ws_mod, "WORKSPACE_DIR", str(tmp_path))
        with pytest.raises(ValueError, match="outside the workspace"):
            validate_workspace_path(tmp_path / ".." / ".." / "etc" / "passwd")

    def test_validate_allows_internal_path(self, tmp_path, monkeypatch):
        """validate_workspace_path allows paths inside workspace."""
        import agent_utilities.core.workspace as ws_mod
        from agent_utilities.core.workspace import validate_workspace_path

        internal = tmp_path / "subdir" / "file.txt"
        internal.parent.mkdir(parents=True, exist_ok=True)
        internal.touch()
        monkeypatch.setattr(ws_mod, "WORKSPACE_DIR", str(tmp_path))
        result = validate_workspace_path(internal)
        assert result == internal.resolve()
