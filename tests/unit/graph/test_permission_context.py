from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_utilities.graph.builder import _build_graph_config
from agent_utilities.security.permissions_kernel import (
    PermissionBootstrapError,
    PermissionsKernel,
)


def _config_kwargs(kernel: PermissionsKernel, identity) -> dict[str, object]:
    return {
        "permissions_kernel": kernel,
        "agent_identity": identity,
    }


def _build(
    kwargs: dict[str, object], *, mcp_toolsets: list[object] | None = None
) -> dict[str, object]:
    return _build_graph_config(
        graph_nodes={},
        knowledge_engine=None,
        agent_subject="synthetic-graph",
        mcp_toolsets=(
            [MagicMock(spec=["list_tools"])] if mcp_toolsets is None else mcp_toolsets
        ),
        tag_prompts={},
        tag_env_vars={},
        mcp_url=None,
        mcp_config=None,
        router_model=None,
        agent_model=None,
        router_timeout=None,
        verifier_timeout=None,
        min_confidence=0.5,
        sub_agents=None,
        routing_strategy="hybrid",
        kwargs=kwargs,
    )


def test_graph_builder_injects_one_verified_permission_context() -> None:
    kernel = PermissionsKernel(signing_key="test-signing-authority-material-32b")
    identity = kernel.issue_identity("graph-runtime")

    config = _build(_config_kwargs(kernel, identity))

    assert config["permissions_kernel"] is kernel
    assert config["agent_identity"] is identity


def test_graph_builder_rejects_mismatched_permission_context() -> None:
    kernel = PermissionsKernel(signing_key="test-signing-authority-material-32b")
    other = PermissionsKernel(signing_key="other-signing-authority-material-32b")
    identity = other.issue_identity("graph-runtime")

    with pytest.raises(PermissionBootstrapError, match="invalid"):
        _build(_config_kwargs(kernel, identity))


def test_graph_builder_omits_context_only_without_executable_mcp() -> None:
    config = _build({}, mcp_toolsets=[])

    assert config["permissions_kernel"] is None
    assert config["agent_identity"] is None
