"""Public SDK declaration of agent-utilities package content.

Agent Utilities owns the operational governance shapes it contributes, while
epistemic-graph owns their active GraphSchema interpretation and validation.
This module declares package content through the connector SDK's single public
contract; it never parses, loads, validates, imports, or attaches shapes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import agent_utilities

if TYPE_CHECKING:
    from agent_connector_sdk.mcp.content import ConnectorContent


def agent_utilities_content() -> ConnectorContent:
    """Return AU's canonical, versioned SDK content declaration."""
    from agent_connector_sdk.mcp.content import ConnectorContent

    return ConnectorContent(
        connector="agent-utilities",
        package_root=Path(agent_utilities.__file__).resolve().parent,
        package_version=agent_utilities.__version__,
        manifest_path=None,
    )


AGENT_UTILITIES_CONTENT_PROVIDER = agent_utilities_content

__all__ = ["AGENT_UTILITIES_CONTENT_PROVIDER", "agent_utilities_content"]
