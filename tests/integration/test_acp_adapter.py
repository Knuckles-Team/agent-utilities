from unittest.mock import patch

import pytest

from agent_utilities.acp_adapter import _ACP_INSTALLED, build_acp_config

pytestmark = pytest.mark.skipif(not _ACP_INSTALLED, reason="pydantic-acp not installed")


def test_build_acp_config_defaults():
    """Test build_acp_config with default values."""
    with patch("pathlib.Path.mkdir"):
        config = build_acp_config()
        # Check standard modes are there (ask, plan)
        # Assuming PrepareToolsBridge is the first bridge in capability_bridges
        bridge = next(
            b for b in config.capability_bridges if hasattr(b, "default_mode_id")
        )
        assert bridge.default_mode_id == "ask"
        assert any(m.id == "ask" for m in bridge.modes)
        assert any(m.id == "plan" for m in bridge.modes)
        # Verify approval bridge is enabled with persistent choices
        assert config.approval_bridge.enable_persistent_choices is True


def test_build_acp_config_custom_root(tmp_path):
    """Test build_acp_config with a custom session root."""
    session_root = tmp_path / "sessions"
    config = build_acp_config(session_root=session_root)
    assert session_root.exists()
    # verify FileSessionStore uses this root
    assert config.session_store.root == session_root


def test_build_acp_config_disable_features():
    """Test build_acp_config with disabled thinking and approvals."""
    with patch("pathlib.Path.mkdir"):
        config = build_acp_config(enable_approvals=False, enable_thinking=False)
        # Verifying bridges
        # ThinkingBridge should be absent
        from pydantic_acp import ThinkingBridge

        assert not any(isinstance(b, ThinkingBridge) for b in config.capability_bridges)

        # When disabled, it should use the default bridge (with enable_persistent_choices=False)
        # as seen in my inspection (pydantic-acp default)
        assert config.approval_bridge.enable_persistent_choices is False


def test_interaction_modes():
    """Test the logic of the ask and plan modes."""
    with patch("pathlib.Path.mkdir"):
        config = build_acp_config()
        bridge = next(b for b in config.capability_bridges if hasattr(b, "modes"))
        ask_mode = next(m for m in bridge.modes if m.id == "ask")
        plan_mode = next(m for m in bridge.modes if m.id == "plan")

        # Test prepare_func returns the tool_defs as is (default behavior in build_acp_config)
        tool_defs = ["tool1", "tool2"]
        assert ask_mode.prepare_func(None, tool_defs) == tool_defs
        assert plan_mode.prepare_func(None, tool_defs) == tool_defs
        assert plan_mode.plan_mode is True
