#!/usr/bin/python
from __future__ import annotations

"""Agent Manager Dashboard — Single-Pane Governance View.

CONCEPT:AU-ECO.ui.agent-manager-dashboard — Agent Manager Dashboard

Provides a CLI and programmatic interface for the "agent manager" role —
a hybrid PM/engineer function dedicated to managing the agent ecosystem.

Reports: installed hooks, active skills, plugin bundles, AGENTS.md compliance,
permission policies, and usage statistics from the KG.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

__all__ = ["AgentManagerDashboard", "DashboardReport"]


@dataclass
class ComponentStatus:
    """Status of a single ecosystem component."""

    category: str  # hooks, skills, plugins, permissions, agents_md
    name: str
    status: str = "ok"  # ok, warning, error, stale
    detail: str = ""
    usage_count: int = 0
    last_used: str = ""


@dataclass
class DashboardReport:
    """Complete dashboard report."""

    timestamp: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    components: list[ComponentStatus] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)
    health_score: float = 1.0

    def to_markdown(self) -> str:
        lines = [
            f"# Agent Manager Dashboard — {self.timestamp}",
            "",
            f"**Health Score**: {self.health_score:.0%}",
            "",
            "## Summary\n",
            "| Category | OK | Warning | Error |",
            "|---|---|---|---|",
        ]

        cats: dict[str, dict[str, int]] = {}
        for c in self.components:
            cats.setdefault(c.category, {"ok": 0, "warning": 0, "error": 0, "stale": 0})
            cats[c.category][c.status] = cats[c.category].get(c.status, 0) + 1

        for cat, counts in sorted(cats.items()):
            lines.append(
                f"| {cat} | {counts.get('ok', 0)} | "
                f"{counts.get('warning', 0)} | {counts.get('error', 0)} |"
            )

        lines.append("")

        # Detail sections
        for category in sorted({c.category for c in self.components}):
            items = [c for c in self.components if c.category == category]
            lines.append(f"## {category.title()}\n")
            for item in items:
                icon = {"ok": "✅", "warning": "⚠️", "error": "❌", "stale": "🕐"}.get(
                    item.status, "❓"
                )
                detail = f" — {item.detail}" if item.detail else ""
                usage = f" (used {item.usage_count}x)" if item.usage_count else ""
                lines.append(f"- {icon} **{item.name}**{detail}{usage}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "health_score": self.health_score,
            "components": [
                {
                    "category": c.category,
                    "name": c.name,
                    "status": c.status,
                    "detail": c.detail,
                    "usage_count": c.usage_count,
                }
                for c in self.components
            ],
            "summary": self.summary,
        }


class AgentManagerDashboard:
    """Single-pane governance dashboard for agent ecosystem management.

    CONCEPT:AU-ECO.ui.agent-manager-dashboard — Agent Manager Dashboard

    Usage::

        dashboard = AgentManagerDashboard(workspace="/my/project", engine=kg)
        report = dashboard.run()
        print(report.to_markdown())
    """

    def __init__(
        self,
        workspace: str | Path = ".",
        engine: IntelligenceGraphEngine | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.engine = engine

    def run(self) -> DashboardReport:
        """Run a full dashboard check."""
        report = DashboardReport()

        report.components.extend(self._check_agents_md())
        report.components.extend(self._check_hooks())
        report.components.extend(self._check_plugins())
        report.components.extend(self._check_permissions())
        report.components.extend(self._check_skills_usage())
        report.components.extend(self._check_proposals())

        # Calculate health score
        total = len(report.components) or 1
        ok = sum(1 for c in report.components if c.status == "ok")
        report.health_score = ok / total

        report.summary = {
            "total": total,
            "ok": ok,
            "warnings": sum(1 for c in report.components if c.status == "warning"),
            "errors": sum(1 for c in report.components if c.status == "error"),
        }

        logger.info(
            "[ECO-4.8] Dashboard: health=%.0f%% (%d/%d ok)",
            report.health_score * 100,
            ok,
            total,
        )
        return report

    def _check_agents_md(self) -> list[ComponentStatus]:
        """Check AGENTS.md presence and freshness."""
        items: list[ComponentStatus] = []
        agents_md = self.workspace / "AGENTS.md"

        if agents_md.is_file():
            age = (time.time() - agents_md.stat().st_mtime) / 86400
            status = "ok" if age < 90 else "stale"
            items.append(
                ComponentStatus(
                    category="agents_md",
                    name="AGENTS.md",
                    status=status,
                    detail=f"Last modified {int(age)} days ago",
                )
            )
        else:
            items.append(
                ComponentStatus(
                    category="agents_md",
                    name="AGENTS.md",
                    status="error",
                    detail="File not found",
                )
            )

        # Check for CODEBASE.md
        codebase_md = self.workspace / "CODEBASE.md"
        items.append(
            ComponentStatus(
                category="agents_md",
                name="CODEBASE.md",
                status="ok" if codebase_md.is_file() else "warning",
                detail="Present" if codebase_md.is_file() else "Not generated yet",
            )
        )

        return items

    def _check_hooks(self) -> list[ComponentStatus]:
        """Check installed hooks."""
        items: list[ComponentStatus] = []
        hooks_dir = self.workspace / ".agents" / "hooks"

        if hooks_dir.exists():
            for hook in hooks_dir.iterdir():
                if hook.is_file():
                    items.append(
                        ComponentStatus(
                            category="hooks",
                            name=hook.name,
                            status="ok",
                        )
                    )
        else:
            items.append(
                ComponentStatus(
                    category="hooks",
                    name="hooks_directory",
                    status="warning",
                    detail="No hooks directory found",
                )
            )

        return items

    def _check_plugins(self) -> list[ComponentStatus]:
        """Check installed plugins."""
        items: list[ComponentStatus] = []
        plugins_dir = self.workspace / ".agents" / "plugins"

        if plugins_dir.exists():
            for pd in plugins_dir.iterdir():
                if pd.is_dir():
                    manifest = any(
                        (pd / n).exists()
                        for n in ("plugin.yaml", "plugin.yml", "plugin.json")
                    )
                    items.append(
                        ComponentStatus(
                            category="plugins",
                            name=pd.name,
                            status="ok" if manifest else "warning",
                            detail="Valid manifest" if manifest else "Missing manifest",
                        )
                    )

        return items

    def _check_permissions(self) -> list[ComponentStatus]:
        """Check permission policy."""
        items: list[ComponentStatus] = []
        perm_file = self.workspace / ".agents" / "permissions.json"

        if perm_file.is_file():
            try:
                data = json.loads(perm_file.read_text(encoding="utf-8"))
                deny_count = len(data.get("deny_paths", []))
                items.append(
                    ComponentStatus(
                        category="permissions",
                        name="permissions.json",
                        status="ok",
                        detail=f"{deny_count} deny rules",
                    )
                )
            except Exception:
                items.append(
                    ComponentStatus(
                        category="permissions",
                        name="permissions.json",
                        status="error",
                        detail="Invalid JSON",
                    )
                )
        else:
            items.append(
                ComponentStatus(
                    category="permissions",
                    name="permissions.json",
                    status="warning",
                    detail="No permission policy configured",
                )
            )

        return items

    def _check_skills_usage(self) -> list[ComponentStatus]:
        """Check skill usage from KG."""
        if not self.engine:
            return []
        items: list[ComponentStatus] = []
        try:
            results = self.engine.query_cypher(
                "MATCH (s) WHERE s.node_type = 'skill' "
                "OPTIONAL MATCH (s)<-[:USED_SKILL]-(e) "
                "RETURN s.name as name, count(e) as uses "
                "ORDER BY uses DESC LIMIT 20"
            )
            for row in results:
                name = row.get("name", "unknown")
                uses = row.get("uses", 0)
                items.append(
                    ComponentStatus(
                        category="skills",
                        name=name,
                        status="ok" if uses > 0 else "stale",
                        usage_count=uses,
                    )
                )
        except Exception as e:
            logger.debug("[ECO-4.8] Skill check failed: %s", e)
        return items

    def _check_proposals(self) -> list[ComponentStatus]:
        """Check pending AGENTS.md proposals."""
        items: list[ComponentStatus] = []
        proposals_dir = self.workspace / ".agents" / "proposed_updates"

        if proposals_dir.exists():
            files = list(proposals_dir.glob("*.md"))
            if files:
                items.append(
                    ComponentStatus(
                        category="proposals",
                        name="pending_updates",
                        status="warning",
                        detail=f"{len(files)} pending AGENTS.md proposals to review",
                    )
                )

        return items
