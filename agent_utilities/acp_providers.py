#!/usr/bin/python
# coding: utf-8
"""ACP Providers Module.

This module implements custom persistence providers for the ACP protocol.
It enables mirroring of ACP session state (like plans) into workspace-local
markdown files, ensuring human-readability and state persistence across restarts.
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
    """Persistence provider that mirrors ACP plan state to workspace files.

    This provider synchronizes the internal ACP plan state (tasks and their
    completion status) into a markdown file (e.g., MEMORY.md) within the
    agent's workspace.
    """

    def __init__(
        self, workspace_root: Optional[Path] = None, plan_filename: str = "MEMORY.md"
    ):
        self.workspace_root = workspace_root or Path(".")
        self.plan_filename = plan_filename

    async def persist_plan_state(
        self, ctx: "AcpSessionContext", plan: Sequence["PlanEntry"]
    ) -> None:
        """Persist the current ACP plan state to a markdown file.

        Args:
            ctx: The active ACP session context.
            plan: A sequence of plan entries to persist.

        """
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
    """Factory function to initialize a WorkspacePlanPersistenceProvider.

    Retrieves the current workspace directory from the server state and
    constructs a provider pointing to that location.

    Returns:
        An instance of WorkspacePlanPersistenceProvider.

    """
    from .server import WORKSPACE_DIR

    return WorkspacePlanPersistenceProvider(workspace_root=Path(WORKSPACE_DIR or "."))
