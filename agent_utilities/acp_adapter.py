from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Dict, List

from pydantic_ai import Agent

# Guarded imports for optional ACP features
try:
    from pydantic_acp import (
        AdapterConfig,
        FileSessionStore,
        NativeApprovalBridge,
        PrepareToolsBridge,
        PrepareToolsMode,
        ThinkingBridge,
    )

    _ACP_INSTALLED = True
except ImportError:
    _ACP_INSTALLED = False

    # Type stubs for when package is missing
    class AdapterConfig:
        pass

    class PrepareToolsMode:
        pass


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


def build_acp_config(
    session_root: Optional[Path] = None,
    enable_approvals: bool = True,
    enable_thinking: bool = True,
    modes: Optional[List[PrepareToolsMode]] = None,
) -> AdapterConfig:
    """Constructs a production-ready ACP AdapterConfig."""
    if not session_root:
        session_root = Path(os.getenv("ACP_SESSION_ROOT", ".acp-sessions"))

    try:
        session_root.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create ACP session directory {session_root}: {e}")

    if not modes:
        modes = [
            PrepareToolsMode(
                id="ask",
                name="Ask",
                description="Standard interaction mode.",
                prepare_func=lambda ctx, tool_defs: list(tool_defs),
            ),
            PrepareToolsMode(
                id="plan",
                name="Plan",
                description="Full graph planning and verification mode.",
                prepare_func=lambda ctx, tool_defs: list(tool_defs),
                plan_mode=True,
            ),
        ]

    bridges: List[Any] = [PrepareToolsBridge(default_mode_id="ask", modes=modes)]
    if enable_thinking:
        bridges.append(ThinkingBridge())

    # Wire in the workspace plan persistence provider to mirror ACP state to MEMORY.md
    from .acp_providers import get_workspace_persistence_provider

    plan_provider = get_workspace_persistence_provider()

    config_params: Dict[str, Any] = {
        "session_store": FileSessionStore(root=session_root),
        "capability_bridges": bridges,
        "plan_provider": plan_provider,  # Native ACP Provider for workspace mirroring
    }
    if enable_approvals:
        config_params["approval_bridge"] = NativeApprovalBridge(
            enable_persistent_choices=True
        )

    return AdapterConfig(**config_params)


def create_acp_app(agent: Agent, config: AdapterConfig):
    """Creates a mountable ACP ASGI application from an agent."""
    from pydantic_acp import create_acp_agent

    return create_acp_agent(agent=agent, config=config)


def is_acp_available() -> bool:
    """Checks if the pydantic-acp package is installed."""
    try:
        import pydantic_acp  # noqa: F401

        return True
    except ImportError:
        return False
