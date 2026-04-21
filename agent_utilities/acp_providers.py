#!/usr/bin/python
"""ACP Providers Module.

This module implements custom persistence providers for the ACP protocol.
It enables mirroring of ACP session state (like plans) into workspace-local
markdown files, ensuring human-readability and state persistence across restarts.

The :class:`WorkspacePlanPersistenceProvider` implements the
``NativePlanPersistenceProvider`` protocol so that every time ACP's native
plan tools mutate plan state, the result is also written to a workspace
markdown file (default: ``PLAN.md``).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from acp.schema import PlanEntry
    from pydantic_acp.session.state import AcpSessionContext

logger = logging.getLogger(__name__)


class WorkspacePlanPersistenceProvider:
    """Persistence provider that mirrors ACP native plan state to workspace files.

    Implements the ``NativePlanPersistenceProvider`` protocol from
    ``pydantic-acp``.  The protocol signature is::

        persist_plan_state(session, agent, entries, plan_markdown) -> None

    Each call overwrites the target markdown file with the current plan,
    rendering entries as a numbered checklist.
    """

    def __init__(
        self,
        workspace_root: Path | None = None,
        plan_filename: str = "PLAN.md",
    ):
        self.workspace_root = workspace_root or Path(".")
        self.plan_filename = plan_filename

    async def persist_plan_state(
        self,
        session: AcpSessionContext,
        agent: Any,
        entries: Sequence[PlanEntry],
        plan_markdown: str | None,
    ) -> None:
        """Persist the current ACP plan state to a workspace markdown file.

        Args:
            session: The active ACP session context.
            agent: The runtime agent instance (unused, required by protocol).
            entries: A sequence of ``PlanEntry`` objects from ACP's native
                plan runtime.  Each entry has ``content``, ``status``
                (pending | in_progress | completed), and ``priority``
                (high | medium | low).
            plan_markdown: Optional free-form markdown that accompanies the
                structured entries.

        """
        plan_path = self.workspace_root / "agent_data" / self.plan_filename

        logger.info(f"Syncing ACP plan state ({len(entries)} entries) to {plan_path}")

        _STATUS_ICON = {
            "completed": "[x]",
            "in_progress": "[~]",
            "pending": "[ ]",
        }

        lines: list[str] = []
        if plan_markdown:
            lines.append(plan_markdown.rstrip())
            lines.append("")

        if entries:
            lines.append("## Plan Entries\n")
            for idx, entry in enumerate(entries, start=1):
                icon = _STATUS_ICON.get(entry.status, "[ ]")
                priority_tag = f" `{entry.priority}`" if entry.priority else ""
                lines.append(f"{idx}. {icon}{priority_tag} {entry.content}")

        content = "\n".join(lines) + "\n"

        try:
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(content, encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to persist ACP plan to {plan_path}: {e}")


def get_workspace_persistence_provider() -> WorkspacePlanPersistenceProvider:
    """Factory function to initialize a WorkspacePlanPersistenceProvider.

    Retrieves the current workspace directory from the server state and
    constructs a provider pointing to that location.

    Returns:
        An instance of WorkspacePlanPersistenceProvider.

    """
    from .server import WORKSPACE_DIR

    return WorkspacePlanPersistenceProvider(workspace_root=Path(WORKSPACE_DIR or "."))
