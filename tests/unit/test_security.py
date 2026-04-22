import pytest
import os
from pathlib import Path
from agent_utilities.tool_guard import is_sensitive_tool, is_safe_tool
from agent_utilities.workspace import validate_workspace_path, get_agent_workspace
from agent_utilities.config import TOOL_GUARD_MODE

def test_is_safe_tool():
    assert is_safe_tool("read_file") is True
    assert is_safe_tool("list_directory") is True
    assert is_safe_tool("delete_file") is False
    assert is_safe_tool("write_file") is False

def test_is_sensitive_tool_strict():
    # Force strict mode for this test
    import agent_utilities.tool_guard as tg
    original_mode = tg.TOOL_GUARD_MODE
    tg.TOOL_GUARD_MODE = "strict"
    try:
        assert is_sensitive_tool("read_file") is False
        assert is_sensitive_tool("delete_file") is True
        assert is_sensitive_tool("any_unknown_tool") is True
    finally:
        tg.TOOL_GUARD_MODE = original_mode

def test_path_traversal_protection(tmp_path):
    # Mock workspace directory
    import agent_utilities.workspace as ws
    original_ws = ws.WORKSPACE_DIR
    ws.WORKSPACE_DIR = str(tmp_path)
    
    try:
        # Safe path
        safe_path = tmp_path / "test.txt"
        safe_path.touch()
        assert validate_workspace_path(safe_path) == safe_path.resolve()
        
        # Unsafe path (outside workspace)
        unsafe_path = Path("/etc/passwd")
        with pytest.raises(ValueError, match="Access denied"):
            validate_workspace_path(unsafe_path)
            
        # Unsafe path (traversal)
        traversal_path = tmp_path / ".." / "outside.txt"
        with pytest.raises(ValueError, match="Access denied"):
            validate_workspace_path(traversal_path)
    finally:
        ws.WORKSPACE_DIR = original_ws

def test_expanded_sensitive_patterns():
    # These should be sensitive even in default mode if they match patterns
    assert is_sensitive_tool("http_request") is True
    assert is_sensitive_tool("eval_code") is True
    assert is_sensitive_tool("subprocess_run") is True
    assert is_sensitive_tool("os_remove") is True
