"""ACP custom providers for agent-utilities ecosystem.

Bridges ACP protocol features (planning, session state) to
local agent-utilities/workspace conventions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence, Optional
from pathlib import Path

if TYPE_CHECKING:
    from pydantic_acp.providers import PlanEntry
    from pydantic_acp.session.state import AcpSessionContext

logger = logging.getLogger(__name__)


class WorkspacePlanPersistenceProvider:
    """Mirrors ACP plan state into workspace markdown files (e.g. MEMORY.md)."""

    def __init__(
        self, workspace_root: Optional[Path] = None, plan_filename: str = "MEMORY.md"
    ):
        self.workspace_root = workspace_root or Path(".")
        self.plan_filename = plan_filename

    async def persist_plan_state(
        self, ctx: "AcpSessionContext", plan: Sequence["PlanEntry"]
    ) -> None:
        """Saves the current ACP plan state into a human-readable markdown file."""
        plan_path = self.workspace_root / self.plan_filename

        logger.info(f"Syncing ACP plan state to {plan_path}")

        lines = ["# Agent Plan (Auto-generated from ACP State)\n"]
        for entry in plan:
            status = " [x] " if entry.done else " [ ] "
            lines.append(f"-{status}{entry.description}")

        content = "\n".join(lines)

        try:
            with open(plan_path, "w") as f:
                f.write(content)
        except Exception as e:
            logger.error(f"Failed to persist ACP plan to {plan_path}: {e}")


def get_workspace_persistence_provider() -> "WorkspacePlanPersistenceProvider":
    """Factory function to get a workspace persistence provider."""
    from .server import WORKSPACE_DIR

    return WorkspacePlanPersistenceProvider(workspace_root=Path(WORKSPACE_DIR or "."))
