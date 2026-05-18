#!/usr/bin/python
from __future__ import annotations

"""KG -> Markdown Memory Materializer.

CONCEPT:KG-2.10 -- Observational Memory Bridge

Renders KG memory state into human-inspectable Markdown files (materialized views).
The KG is the single source of truth; Markdown files are projections that can be
inspected, edited, and used as startup context for external agents.

Files stored at ~/.local/share/agent-utilities/memory/ (XDG-compliant).
"""

import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...core.paths import data_dir

if TYPE_CHECKING:
    from ..core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

__all__ = [
    "MemoryMaterializer",
    "memory_dir",
    "materialize_memory",
    "ingest_memory_edits",
]

PRIORITY_EMOJI = {
    "critical": "\U0001f534",
    "important": "\U0001f7e1",
    "normal": "\U0001f7e2",
    "high": "\U0001f534",
    "medium": "\U0001f7e1",
    "low": "\U0001f7e2",
}
MEMORY_FILES = ["observations.md", "reflections.md", "profile.md", "active.md"]
CURSOR_FILE = ".memory_cursor.json"


def memory_dir() -> Path:
    """Return the XDG memory directory for materialized views."""
    import os

    override = os.environ.get("AGENT_UTILITIES_MEMORY_DIR")
    if override:
        return Path(override).expanduser()
    return data_dir() / "memory"


def _ensure_memory_dir() -> Path:
    d = memory_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


class MemoryMaterializer:
    """Renders KG memory state into beautiful, inspectable Markdown files.

    CONCEPT:KG-2.10 -- Observational Memory Bridge
    """

    def __init__(self, engine: IntelligenceGraphEngine) -> None:
        self.engine = engine
        self.base_dir = _ensure_memory_dir()
        self._cursor_path = self.base_dir / CURSOR_FILE

    def materialize(self) -> dict[str, Path]:
        """Render all 4 memory files from current KG state."""
        paths: dict[str, Path] = {}
        for name, renderer in [
            ("observations.md", self._render_observations),
            ("reflections.md", self._render_reflections),
            ("profile.md", self._render_profile),
            ("active.md", self._render_active),
        ]:
            path = self.base_dir / name
            path.write_text(renderer(), encoding="utf-8")
            paths[name] = path

        self._save_cursor(
            {
                name: hashlib.md5(p.read_bytes(), usedforsecurity=False).hexdigest()
                for name, p in paths.items()
            }
        )
        logger.info(
            "[KG-2.10] Materialized %d memory files to %s", len(paths), self.base_dir
        )
        return paths

    def detect_edits(self) -> list[str]:
        """Detect which materialized files have been manually edited."""
        cursor = self._load_cursor()
        if not cursor:
            return []
        edited = []
        for name in MEMORY_FILES:
            path = self.base_dir / name
            if not path.exists():
                continue
            if hashlib.md5(
                path.read_bytes(), usedforsecurity=False
            ).hexdigest() != cursor.get(name, ""):
                edited.append(name)
        return edited

    def ingest_edits(self) -> dict[str, int]:
        """Detect and ingest manual edits from materialized files back to KG."""
        edited = self.detect_edits()
        if not edited:
            return {}
        results: dict[str, int] = {}
        for name in edited:
            path = self.base_dir / name
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8")
            if name == "profile.md":
                results[name] = self._ingest_profile_edits(content)
            elif name == "reflections.md":
                results[name] = self._ingest_reflection_edits(content)
            elif name == "observations.md":
                results[name] = self._ingest_observation_edits(content)
            else:
                results[name] = 0
        self.materialize()
        return results

    # -- Rendering --

    def _render_observations(self) -> str:
        lines = [
            "# Observations",
            "",
            "<!-- Materialized from agent-utilities Knowledge Graph. -->",
            "<!-- Edit this file to correct observations -- changes sync back to KG. -->",
            "",
        ]
        observations = self._query_nodes("observation", 100)
        if not observations:
            return "\n".join(lines) + "*No observations recorded yet.*\n"
        by_date: dict[str, list[dict]] = {}
        for obs in observations:
            ts = obs.get("timestamp", "")[:10] or datetime.now(UTC).strftime("%Y-%m-%d")
            by_date.setdefault(ts, []).append(obs)
        for date in sorted(by_date, reverse=True):
            lines.append(f"## {date}")
            lines.append("")
            for obs in by_date[date]:
                p = obs.get("priority", "normal")
                emoji = PRIORITY_EMOJI.get(p, "\U0001f7e2")
                content = obs.get("content", obs.get("description", ""))
                src = obs.get("source", "")
                lines.append(f"- {emoji} {content}" + (f" [{src}]" if src else ""))
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _render_reflections(self) -> str:
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            "# Reflections",
            "",
            f"*Last updated: {now}*",
            "",
            "<!-- Materialized from agent-utilities Knowledge Graph. -->",
            "",
        ]
        refs = self._query_nodes("reflection", 50)
        if not refs:
            return "\n".join(lines) + "*No reflections recorded yet.*\n"
        by_cat: dict[str, list[dict]] = {}
        for r in refs:
            by_cat.setdefault(r.get("category", "General"), []).append(r)
        for cat in sorted(by_cat):
            lines.extend([f"## {cat.title()}", ""])
            for r in by_cat[cat]:
                c = r.get("content", r.get("description", ""))
                conf = r.get("confidence", 1.0)
                tag = f" (confidence: {conf:.0%})" if conf < 1.0 else ""
                lines.append(f"- {c}{tag}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _render_profile(self) -> str:
        lines = [
            "# User Profile",
            "",
            "<!-- Materialized from agent-utilities Knowledge Graph. -->",
            "",
        ]
        lines.extend(["## Core Identity", ""])
        identity = self._query_identity()
        if identity:
            for k, v in identity.items():
                lines.append(f"- **{k.replace('_', ' ').title()}:** {v}")
        else:
            lines.append("- *No identity information recorded yet.*")
        lines.append("")
        prefs = self._query_nodes("preference", 50)
        lines.extend(["## Preferences & Opinions", ""])
        if prefs:
            for p in prefs:
                v = p.get("value", p.get("description", ""))
                lines.append(f"- {v}")
        else:
            lines.append("- *No preferences recorded yet.*")
        lines.append("")
        facts = self._query_nodes("fact", 30)
        if facts:
            lines.extend(["## Key Facts & Context", ""])
            for f in facts:
                lines.append(f"- {f.get('content', f.get('description', ''))}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _render_active(self) -> str:
        lines = [
            "# Active Context",
            "",
            "<!-- Materialized from agent-utilities Knowledge Graph. -->",
            "",
        ]
        goals = self._query_active_goals()
        if goals:
            lines.extend(["## Active Goals", ""])
            for g in goals:
                lines.append(
                    f"- \U0001f3af {g.get('goal_text', g.get('description', ''))}"
                )
            lines.append("")
        episodes = self._query_nodes("episode", 10)
        if episodes:
            lines.extend(["## Recent Sessions", ""])
            for ep in episodes:
                s = ep.get("summary", ep.get("description", ""))
                ts = ep.get("timestamp", "")[:16]
                lines.append(f"- **{ts}**: {s}")
            lines.append("")
        threads = self._query_nodes("thread", 5)
        if threads:
            lines.extend(["## Current Session Snapshot", ""])
            for t in threads:
                lines.append(f"- {t.get('title', t.get('name', 'Untitled'))}")
            lines.append("")
        if len(lines) <= 4:
            lines.append("*No active context recorded yet.*\n")
        return "\n".join(lines).rstrip() + "\n"

    # -- KG Query Helpers --

    def _query_nodes(self, node_type: str, limit: int) -> list[dict[str, Any]]:
        if not self.engine.backend:
            return self._query_nx(node_type, limit)
        label = node_type.title()
        try:
            res = self.engine.backend.execute(
                f"MATCH (n:{label}) RETURN n ORDER BY n.timestamp DESC LIMIT $limit",
                {"limit": limit},
            )
            return [r["n"] for r in res if "n" in r]
        except Exception:
            return self._query_nx(node_type, limit)

    def _query_identity(self) -> dict[str, Any]:
        if not self.engine.backend:
            for _, a in self.engine.graph.nodes(data=True):
                if a.get("type") == "user":
                    return {
                        k: v
                        for k, v in a.items()
                        if k not in ("type", "embedding", "ewc_fisher_diag", "id") and v
                    }
            return {}
        try:
            res = self.engine.backend.execute("MATCH (n:User) RETURN n LIMIT 1", {})
            if res:
                n = res[0].get("n", {})
                return {
                    k: v
                    for k, v in n.items()
                    if k not in ("type", "embedding", "ewc_fisher_diag", "id") and v
                }
        except Exception:
            pass  # nosec B110
        return {}

    def _query_active_goals(self) -> list[dict[str, Any]]:
        if not self.engine.backend:
            return [
                dict(a)
                for _, a in self.engine.graph.nodes(data=True)
                if a.get("type") == "goal" and a.get("status") == "active"
            ][:10]
        try:
            res = self.engine.backend.execute(
                "MATCH (n:Goal) WHERE n.status = 'active' RETURN n LIMIT 10", {}
            )
            return [r["n"] for r in res if "n" in r]
        except Exception:
            return []

    def _query_nx(self, node_type: str, limit: int) -> list[dict[str, Any]]:
        results = []
        for _, a in self.engine.graph.nodes(data=True):
            if a.get("type") == node_type:
                results.append(dict(a))
                if len(results) >= limit:
                    break
        return results

    # -- Ingestion (Markdown -> KG) --

    def _ingest_profile_edits(self, content: str) -> int:
        count = 0
        for m in re.finditer(
            r"^- (?:[\U0001f534\U0001f7e1\U0001f7e2] )?(.+)$", content, re.MULTILINE
        ):
            v = m.group(1).strip()
            if v and not v.startswith("*"):
                pid = f"pref_{hashlib.md5(v.encode(), usedforsecurity=False).hexdigest()[:8]}"
                self.engine.add_node(
                    pid,
                    "preference",
                    {
                        "name": v[:80],
                        "value": v,
                        "category": "user_edited",
                        "description": v,
                        "importance_score": 0.7,
                    },
                )
                count += 1
        return count

    def _ingest_reflection_edits(self, content: str) -> int:
        count = 0
        for m in re.finditer(
            r"^- (.+?)(?:\s*\(confidence: [\d.]+%\))?$", content, re.MULTILINE
        ):
            t = m.group(1).strip()
            if t and not t.startswith("*"):
                rid = f"ref_{hashlib.md5(t.encode(), usedforsecurity=False).hexdigest()[:8]}"
                self.engine.add_node(
                    rid,
                    "reflection",
                    {
                        "name": t[:80],
                        "content": t,
                        "description": t,
                        "confidence": 0.8,
                        "importance_score": 0.6,
                    },
                )
                count += 1
        return count

    def _ingest_observation_edits(self, content: str) -> int:
        count = 0
        current_date = ""
        for line in content.splitlines():
            dm = re.match(r"^## (\d{4}-\d{2}-\d{2})$", line)
            if dm:
                current_date = dm.group(1)
                continue
            om = re.match(
                r"^- (?:[\U0001f534\U0001f7e1\U0001f7e2] )?(.+?)(?:\s*\[.+\])?$", line
            )
            if om and current_date:
                t = om.group(1).strip()
                if t and not t.startswith("*"):
                    oid = f"obs_{hashlib.md5(t.encode(), usedforsecurity=False).hexdigest()[:8]}"
                    priority = (
                        "critical"
                        if "\U0001f534" in line
                        else ("important" if "\U0001f7e1" in line else "normal")
                    )
                    self.engine.add_node(
                        oid,
                        "observation",
                        {
                            "name": t[:80],
                            "content": t,
                            "description": t,
                            "priority": priority,
                            "timestamp": f"{current_date}T00:00:00Z",
                            "importance_score": 0.5,
                        },
                    )
                    count += 1
        return count

    # -- Cursor --

    def _load_cursor(self) -> dict[str, str]:
        if not self._cursor_path.exists():
            return {}
        try:
            return json.loads(self._cursor_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_cursor(self, cursor: dict[str, str]) -> None:
        cursor["_materialized_at"] = datetime.now(UTC).isoformat()
        self._cursor_path.write_text(json.dumps(cursor, indent=2), encoding="utf-8")


def materialize_memory(engine: IntelligenceGraphEngine) -> dict[str, Path]:
    """Convenience: materialize all memory files from the given engine."""
    return MemoryMaterializer(engine).materialize()


def ingest_memory_edits(engine: IntelligenceGraphEngine) -> dict[str, int]:
    """Convenience: detect and ingest manual edits from materialized files."""
    return MemoryMaterializer(engine).ingest_edits()
