#!/usr/bin/python
from __future__ import annotations

"""Config Staleness Auditor — Periodic Review of Agent Configuration.

CONCEPT:ECO-4.21 — Configuration Staleness Auditor

Periodic audit (default 30 days) that reviews AGENTS.md, skills, and hooks
for staleness and proposes removals.  Identifies rules never triggered,
skills never used, and hooks compensating for resolved model limitations.

Integrates with ``maintenance_cron.py`` for scheduling.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

__all__ = [
    "ConfigStalenessAuditor",
    "StalenessReport",
    "StalenessItem",
]


@dataclass
class StalenessItem:
    """A single item assessed for staleness."""

    item_type: str  # agents_md_rule, skill, hook, plugin
    name: str
    last_used: str = ""
    usage_count: int = 0
    age_days: int = 0
    recommendation: str = "KEEP"  # KEEP, UPDATE, REMOVE
    reason: str = ""
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.item_type,
            "name": self.name,
            "last_used": self.last_used,
            "usage_count": self.usage_count,
            "age_days": self.age_days,
            "recommendation": self.recommendation,
            "reason": self.reason,
            "confidence": self.confidence,
        }


@dataclass
class StalenessReport:
    """Complete staleness audit report."""

    timestamp: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    items: list[StalenessItem] = field(default_factory=list)
    summary: str = ""

    @property
    def keep_items(self) -> list[StalenessItem]:
        return [i for i in self.items if i.recommendation == "KEEP"]

    @property
    def update_items(self) -> list[StalenessItem]:
        return [i for i in self.items if i.recommendation == "UPDATE"]

    @property
    def remove_items(self) -> list[StalenessItem]:
        return [i for i in self.items if i.recommendation == "REMOVE"]

    def to_markdown(self) -> str:
        lines = [
            f"# Configuration Staleness Audit — {self.timestamp}",
            "",
            f"> {self.summary}",
            "",
            "| Recommendation | Count |",
            "|---|---|",
            f"| KEEP | {len(self.keep_items)} |",
            f"| UPDATE | {len(self.update_items)} |",
            f"| REMOVE | {len(self.remove_items)} |",
            "",
        ]

        if self.remove_items:
            lines.append("## Recommended Removals\n")
            for item in self.remove_items:
                lines.append(
                    f"- **[{item.item_type}] {item.name}** — {item.reason} "
                    f"(confidence: {item.confidence:.0%})"
                )
            lines.append("")

        if self.update_items:
            lines.append("## Recommended Updates\n")
            for item in self.update_items:
                lines.append(f"- **[{item.item_type}] {item.name}** — {item.reason}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "items": [i.to_dict() for i in self.items],
            "summary": self.summary,
            "counts": {
                "keep": len(self.keep_items),
                "update": len(self.update_items),
                "remove": len(self.remove_items),
            },
        }


class ConfigStalenessAuditor:
    """Periodic staleness audit for agent configurations.

    CONCEPT:ECO-4.21 — Configuration Staleness Auditor

    Usage::

        auditor = ConfigStalenessAuditor(engine, workspace="/my/project")
        report = auditor.run_audit()
        print(report.to_markdown())
    """

    DEFAULT_INTERVAL_DAYS = 30
    STALE_THRESHOLD_DAYS = 60  # Items unused for 60+ days are candidates

    def __init__(
        self,
        engine: IntelligenceGraphEngine | None = None,
        workspace: str | Path = ".",
        interval_days: int = DEFAULT_INTERVAL_DAYS,
        stale_threshold_days: int = STALE_THRESHOLD_DAYS,
    ) -> None:
        self.engine = engine
        self.workspace = Path(workspace).resolve()
        self.interval_days = interval_days
        self.stale_threshold = stale_threshold_days

    def run_audit(self) -> StalenessReport:
        """Run a full staleness audit across all config types."""
        report = StalenessReport()

        # Audit AGENTS.md sections
        report.items.extend(self._audit_agents_md())

        # Audit installed skills
        report.items.extend(self._audit_skills())

        # Audit installed hooks
        report.items.extend(self._audit_hooks())

        # Audit installed plugins
        report.items.extend(self._audit_plugins())

        # Generate summary
        total = len(report.items)
        remove = len(report.remove_items)
        update = len(report.update_items)
        report.summary = (
            f"Audited {total} configuration items. "
            f"{remove} recommended for removal, {update} for update."
        )

        # Persist report
        self._write_report(report)
        self._persist_report(report)

        logger.info("[ECO-4.6] Staleness audit: %s", report.summary)
        return report

    def should_run(self) -> bool:
        """Check if enough time has passed since the last audit."""
        report_dir = self.workspace / ".agents" / "audits"
        if not report_dir.exists():
            return True

        reports = sorted(report_dir.glob("staleness_*.md"))
        if not reports:
            return True

        last_report = reports[-1]
        age = time.time() - last_report.stat().st_mtime
        return age > (self.interval_days * 86400)

    # -- Auditors --

    def _audit_agents_md(self) -> list[StalenessItem]:
        """Audit AGENTS.md sections for staleness."""
        items: list[StalenessItem] = []
        agents_md = self.workspace / "AGENTS.md"
        if not agents_md.is_file():
            return items

        import re

        content = agents_md.read_text(encoding="utf-8")
        sections = re.findall(r"^(#{1,3}\s+.+)$", content, re.MULTILINE)

        for section in sections:
            usage = self._query_section_usage(section)
            item = StalenessItem(
                item_type="agents_md_rule",
                name=section.strip("# ").strip(),
                usage_count=usage.get("count", 0),
                last_used=usage.get("last_used", ""),
                age_days=usage.get("age_days", 0),
            )

            if (
                usage.get("count", 0) == 0
                and usage.get("age_days", 0) > self.stale_threshold
            ):
                item.recommendation = "REMOVE"
                item.reason = f"Never referenced in {usage.get('age_days', 0)} days"
                item.confidence = 0.7
            elif usage.get("contradicted", False):
                item.recommendation = "UPDATE"
                item.reason = "Agent frequently worked around this rule"
                item.confidence = 0.6

            items.append(item)

        return items

    def _audit_skills(self) -> list[StalenessItem]:
        """Audit installed skills for usage."""
        items: list[StalenessItem] = []
        if not self.engine:
            return items

        try:
            results = self.engine.query_cypher(
                "MATCH (s) WHERE s.node_type = 'skill' "
                "OPTIONAL MATCH (s)<-[:USED_SKILL]-(e) "
                "RETURN s.node_id as id, s.name as name, count(e) as uses "
                "ORDER BY uses ASC"
            )
            for row in results:
                skill_name = str(row.get("name") or row.get("id") or "unknown")
                item = StalenessItem(
                    item_type="skill",
                    name=skill_name,
                    usage_count=int(row.get("uses", 0)),
                )
                if row.get("uses", 0) == 0:
                    item.recommendation = "REMOVE"
                    item.reason = "Never used since installation"
                    item.confidence = 0.5
                items.append(item)
        except Exception as e:
            logger.debug("[ECO-4.6] Skill audit failed: %s", e)

        return items

    def _audit_hooks(self) -> list[StalenessItem]:
        """Audit installed hooks."""
        items: list[StalenessItem] = []
        hooks_dir = self.workspace / ".agents" / "hooks"
        if not hooks_dir.exists():
            return items

        for hook_file in hooks_dir.iterdir():
            if hook_file.is_file():
                age = (time.time() - hook_file.stat().st_mtime) / 86400
                item = StalenessItem(
                    item_type="hook",
                    name=hook_file.name,
                    age_days=int(age),
                )
                if age > self.stale_threshold * 2:
                    item.recommendation = "UPDATE"
                    item.reason = f"Hook file unchanged for {int(age)} days"
                    item.confidence = 0.4
                items.append(item)

        return items

    def _audit_plugins(self) -> list[StalenessItem]:
        """Audit installed plugins."""
        items: list[StalenessItem] = []
        plugins_dir = self.workspace / ".agents" / "plugins"
        if not plugins_dir.exists():
            return items

        for plugin_dir in plugins_dir.iterdir():
            if plugin_dir.is_dir():
                age = (time.time() - plugin_dir.stat().st_mtime) / 86400
                items.append(
                    StalenessItem(
                        item_type="plugin",
                        name=plugin_dir.name,
                        age_days=int(age),
                        recommendation="KEEP",
                    )
                )

        return items

    # -- Helpers --

    def _query_section_usage(self, section: str) -> dict[str, Any]:
        """Query KG for how often a section was referenced."""
        if not self.engine:
            return {"count": -1, "age_days": 0}
        try:
            results = self.engine.query_cypher(
                "MATCH (r) WHERE r.node_type = 'agents_md_proposal' "
                "AND r.section CONTAINS $section "
                "RETURN count(r) as cnt",
                {"section": section.strip("# ").strip()[:50]},
            )
            count = results[0].get("cnt", 0) if results else 0
            return {"count": count, "age_days": 0}
        except Exception:
            return {"count": -1, "age_days": 0}

    def _write_report(self, report: StalenessReport) -> None:
        """Write report to disk."""
        report_dir = self.workspace / ".agents" / "audits"
        report_dir.mkdir(parents=True, exist_ok=True)
        date = time.strftime("%Y-%m-%d", time.gmtime())
        fp = report_dir / f"staleness_{date}.md"
        fp.write_text(report.to_markdown(), encoding="utf-8")

    def _persist_report(self, report: StalenessReport) -> None:
        """Persist report summary to KG."""
        if not self.engine:
            return
        try:
            self.engine.add_node(
                f"staleness_audit_{report.timestamp[:10]}",
                "staleness_audit",
                {
                    "name": f"Staleness Audit {report.timestamp[:10]}",
                    "description": report.summary,
                    "total_items": len(report.items),
                    "remove_count": len(report.remove_items),
                    "update_count": len(report.update_items),
                    "timestamp": report.timestamp,
                },
            )
        except Exception as e:
            logger.debug("[ECO-4.6] KG persist failed: %s", e)
