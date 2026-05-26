#!/usr/bin/python
from __future__ import annotations

"""Budgeted Startup Context Builder.

CONCEPT:KG-2.1 -- Observational Memory Bridge

Produces a deterministic, budget-bounded startup payload for injecting into
external agent hooks (Claude Code SessionStart, Codex startup, Grok Build, etc.).

Architecture:
    KG -> HybridRetriever -> priority-scored chunks -> budget trim -> StartupPayload
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .memory_materializer import MemoryMaterializer

if TYPE_CHECKING:
    from ..core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

DEFAULT_BUDGET_CHARS = 24000
MIN_BUDGET_CHARS = 2000


@dataclass(frozen=True)
class StartupChunk:
    """A prioritized chunk of startup context."""

    source: str
    heading: str
    body: str
    handle: str
    priority: int

    @property
    def size(self) -> int:
        return len(self.body)


@dataclass(frozen=True)
class StartupPayload:
    """Budgeted startup payload with expansion handles."""

    text: str
    budget_chars: int
    included_handles: list[str]
    overflow: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "budget_chars": self.budget_chars,
            "included_handles": self.included_handles,
            "overflow": self.overflow,
        }


class StartupContextBuilder:
    """Builds budgeted startup context from KG memory.

    CONCEPT:KG-2.1 -- Observational Memory Bridge

    Queries the KG via HybridRetriever, scores and ranks chunks by relevance
    to the current agent/task/cwd, then assembles a budget-bounded Markdown
    payload with overflow handles for deeper recall.
    """

    def __init__(self, engine: IntelligenceGraphEngine) -> None:
        self.engine = engine
        self.materializer = MemoryMaterializer(engine)

    def build_payload(
        self,
        *,
        budget_chars: int | None = None,
        cwd: str | None = None,
        task: str | None = None,
        agent: str | None = None,
        team: str | None = None,
    ) -> StartupPayload:
        """Build a deterministic, budgeted startup payload.

        Args:
            budget_chars: Maximum characters for the payload (default: 24000).
            cwd: Current working directory for routing context.
            task: Current task description for routing context.
            agent: Agent name (claude, codex, grok, etc.) for routing.
            team: Team name for loading team-specific conventions from KG.

        Returns:
            StartupPayload with bounded text and expansion handles.
        """
        budget = max(budget_chars or DEFAULT_BUDGET_CHARS, MIN_BUDGET_CHARS)

        # Ensure materialized files exist
        self.materializer.materialize()
        base_dir = self.materializer.base_dir

        # Build chunks from materialized files
        chunks = self._build_chunks(base_dir, cwd=cwd, task=task, agent=agent)

        # Inject team-specific context from KG
        if team:
            team_chunks = self._load_team_context(team)
            chunks.extend(team_chunks)

        # Inject layered AGENTS.md from CWD
        if cwd:
            layered_chunk = self._load_layered_agents_md(cwd)
            if layered_chunk:
                chunks.append(layered_chunk)

        # Assemble payload
        header = self._build_header(budget, cwd=cwd, task=task, agent=agent)
        footer = self._build_footer(task=task, cwd=cwd)
        used = len(header) + len(footer)
        selected: list[StartupChunk] = []
        overflow: list[StartupChunk] = []

        for chunk in sorted(chunks, key=lambda c: (-c.priority, c.source, c.heading)):
            chunk_text = "\n\n" + chunk.body.strip()
            if used + len(chunk_text) <= budget:
                selected.append(chunk)
                used += len(chunk_text)
            else:
                overflow.append(chunk)

        parts = [header.rstrip()]
        parts.extend(c.body.strip() for c in selected)
        if overflow:
            parts.append(self._overflow_section(overflow))
        parts.append(footer.rstrip())
        text = "\n\n".join(p for p in parts if p).rstrip() + "\n"

        if len(text) > budget:
            text = self._hard_trim(text, budget)

        return StartupPayload(
            text=text,
            budget_chars=budget,
            included_handles=[c.handle for c in selected],
            overflow=[
                {
                    "handle": c.handle,
                    "heading": c.heading,
                    "source": c.source,
                    "chars": c.size,
                }
                for c in overflow
            ],
        )

    def recall_handle(self, handle: str) -> str:
        """Expand a startup recall handle into Markdown."""
        base_dir = self.materializer.base_dir
        if handle == "startup:profile":
            p = base_dir / "profile.md"
            return p.read_text() if p.exists() else ""
        if handle == "startup:active":
            p = base_dir / "active.md"
            return p.read_text() if p.exists() else ""
        chunks = self._build_chunks(base_dir)
        for chunk in chunks:
            if chunk.handle == handle:
                return chunk.body.rstrip() + "\n"
        raise KeyError(f"Unknown handle: {handle}")

    def recall_query(self, query: str, limit: int = 8) -> list[str]:
        """Search KG memory with formatted output for agent consumption."""
        try:
            results = self.engine.search_hybrid(query, top_k=limit)
            return [
                f"- **{r.get('name', r.get('id', ''))}**: {r.get('description', r.get('content', ''))}"
                for r in results
            ]
        except Exception as e:
            logger.debug("Recall query failed: %s", e)
            return []

    # -- Internal --

    def _build_chunks(
        self,
        base_dir: Path,
        *,
        cwd: str | None = None,
        task: str | None = None,
        agent: str | None = None,
    ) -> list[StartupChunk]:
        chunks: list[StartupChunk] = []
        for source_name, filename in [
            ("profile", "profile.md"),
            ("active", "active.md"),
        ]:
            path = base_dir / filename
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            for heading, body in self._split_h2(text):
                handle = f"startup:{source_name}:{self._slug(heading)}"
                priority = self._chunk_priority(
                    source_name, heading, body, cwd=cwd, task=task, agent=agent
                )
                chunks.append(
                    StartupChunk(
                        source=source_name,
                        heading=heading,
                        body=body,
                        handle=handle,
                        priority=priority,
                    )
                )
        return chunks

    def _split_h2(self, text: str) -> list[tuple[str, str]]:
        if not text.strip():
            return []
        chunks: list[tuple[str, list[str]]] = []
        current: list[str] | None = None
        heading = ""
        for line in text.splitlines():
            if line.startswith("## "):
                heading = line[3:].strip()
                current = [line]
                chunks.append((heading, current))
            elif current is not None:
                current.append(line)
        return [(h, "\n".join(b).strip()) for h, b in chunks if "\n".join(b).strip()]

    def _chunk_priority(
        self,
        source: str,
        heading: str,
        body: str,
        *,
        cwd: str | None,
        task: str | None,
        agent: str | None,
    ) -> int:
        h = heading.lower()
        priority = 4
        if source == "profile":
            priority = 7
        if "core identity" in h:
            priority = 8
        if "preference" in h:
            priority = 10
        if "active" in h and "goal" in h:
            priority = 9
        if "current session" in h:
            priority = 9
        terms = self._route_terms(cwd=cwd, task=task, agent=agent)
        if terms and any(t in body.lower() or t in h for t in terms):
            priority += 5
        return priority

    def _route_terms(
        self, *, cwd: str | None, task: str | None, agent: str | None
    ) -> list[str]:
        terms: list[str] = []
        if cwd:
            p = Path(cwd)
            terms.extend(part.lower() for part in (p.name, p.parent.name) if part)
        if task:
            terms.extend(
                w.lower() for w in re.findall(r"[a-zA-Z0-9_.-]+", task) if len(w) >= 3
            )
        if agent:
            terms.append(agent.lower())
        return [t for t in terms if len(t) >= 3]

    def _slug(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "root"

    def _build_header(
        self, budget: int, *, cwd: str | None, task: str | None, agent: str | None
    ) -> str:
        lines = [
            "# agent-utilities Startup Context",
            "",
            "<!-- Budgeted startup payload from KG-2.10 Memory Bridge. -->",
            "",
            "## Startup Routing",
            "",
            f"- Budget: {budget} chars",
        ]
        if agent:
            lines.append(f"- Agent: {agent}")
        if cwd:
            lines.append(f"- CWD: {cwd}")
        if task:
            lines.append(f"- Task: {task}")
        return "\n".join(lines)

    def _build_footer(self, *, task: str | None, cwd: str | None) -> str:
        query = task or (Path(cwd).name if cwd else "current work")
        return "\n".join(
            [
                "## Recall",
                "",
                "- Expand profile: `agent-utilities recall --handle startup:profile`",
                "- Expand active: `agent-utilities recall --handle startup:active`",
                f'- Search memory: `agent-utilities recall --query "{query}" --limit 8`',
            ]
        )

    def _overflow_section(self, chunks: list[StartupChunk]) -> str:
        lines = [
            "## Startup Overflow",
            "",
            "The following context was omitted from the startup budget:",
        ]
        for c in chunks:
            lines.append(f"- `{c.handle}` ({c.size} chars, {c.source}: {c.heading})")
        return "\n".join(lines)

    def _load_team_context(self, team: str) -> list[StartupChunk]:
        """Load team-specific context from KG TeamConfigNode nodes.

        CONCEPT:ECO-4.16 — Team-Specific Startup Context

        Queries the KG for ``TeamConfigNode`` or ``team_config`` nodes
        matching the team name and converts their conventions/preferences
        into startup chunks with high priority.
        """
        chunks: list[StartupChunk] = []
        try:
            results = self.engine.query_cypher(
                "MATCH (t) WHERE (t.node_type = 'team_config' OR "
                "t.node_type = 'TeamConfigNode') AND t.name CONTAINS $team "
                "RETURN t.name as name, t.description as desc, "
                "t.conventions as conventions, t.preferences as prefs",
                {"team": team},
            )
            for row in results:
                body_parts = [f"## Team: {row.get('name', team)}"]
                if row.get("desc"):
                    body_parts.append(str(row["desc"]))
                if row.get("conventions"):
                    body_parts.append(f"### Conventions\n{row['conventions']}")
                if row.get("prefs"):
                    body_parts.append(f"### Preferences\n{row['prefs']}")
                body = "\n\n".join(body_parts)
                chunks.append(
                    StartupChunk(
                        source="team",
                        heading=f"Team: {team}",
                        body=body,
                        handle=f"startup:team:{self._slug(team)}",
                        priority=8,  # High priority — team conventions matter
                    )
                )
        except Exception as e:
            logger.debug("Team context load failed for '%s': %s", team, e)
        return chunks

    def _load_layered_agents_md(self, cwd: str) -> StartupChunk | None:
        """Load hierarchically layered AGENTS.md content from CWD.

        CONCEPT:KG-2.1 — Layered Project-Aware Context

        Uses ``load_agents_md_layered`` to walk upward from CWD and collect
        all AGENTS.md files, assembling them root-first.
        """
        try:
            from ..core.agents_md import load_agents_md_layered

            content = load_agents_md_layered(cwd)
            if content:
                return StartupChunk(
                    source="agents_md",
                    heading="Project Rules (Layered)",
                    body=f"## Project Rules (AGENTS.md)\n\n{content}",
                    handle="startup:agents_md:layered",
                    priority=9,  # Very high — project rules are critical
                )
        except Exception as e:
            logger.debug("Layered AGENTS.md load failed: %s", e)
        return None

    def _hard_trim(self, text: str, budget: int) -> str:
        marker = (
            "\n\n## Startup Payload Truncated\n\n"
            "- Increase `--budget-chars` or use recall handles for expansion.\n"
        )
        keep = max(budget - len(marker), 0)
        return text[:keep].rstrip() + marker


def build_startup_payload(
    engine: IntelligenceGraphEngine,
    **kwargs: Any,
) -> StartupPayload:
    """Convenience: build startup payload from engine."""
    return StartupContextBuilder(engine).build_payload(**kwargs)
