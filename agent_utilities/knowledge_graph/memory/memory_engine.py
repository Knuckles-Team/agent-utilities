from __future__ import annotations

# --- FROM memory_engine.py ---
"""CONCEPT:AU-KG.memory.tiered-memory-caching — Unified Memory Manager.

Single entry point for the full memory lifecycle:
  startup → active context → compaction → consolidation → retrieval

Coordinates all memory subsystems without replacing them.  Each
subsystem remains the implementation; this manager provides the
lifecycle orchestration API on top.

Subsystems coordinated:
    - ``StartupContextBuilder``: Initial context assembly
    - ``ContextCompactor``: Active window management (3 strategies)
    - ``MementoCompressor``: LLM-based state compression
    - ``ConsolidationEngine``: Episode → Preference/Principle promotion
    - ``SemanticCompactor``: Trace compaction (prevents graph explosion)
    - ``MemoryRetriever``: KG-based semantic recall

Usage::

    mgr = MemoryLifecycleManager(engine=kg_engine)

    # Startup: assemble initial context within budget
    ctx = mgr.build_startup_context(query="debug auth flow", budget=8000)

    # Active session: compact when context grows too large
    result = mgr.compact_if_needed(messages, strategy="summarize_tools")

    # Long session: compress state to dense memento
    memento = mgr.compress_to_memento(messages, source="agent_runner")

    # Post-session: promote episodes to long-term knowledge
    await mgr.consolidate()

    # Recall: retrieve relevant memories for a new query
    memories = mgr.retrieve(query="auth flow", k=5)
"""


import logging
from typing import TYPE_CHECKING, Any

from agent_utilities.core.config import setting

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    from .agent_context import CompactedResult, CompactionStrategy

logger = logging.getLogger(__name__)


class MemoryEngine:
    """Coordinates memory subsystems through startup → compaction → consolidation → retrieval.

    CONCEPT:AU-KG.memory.tiered-memory-caching — Memory Lifecycle Manager

    Provides a facade over the independent memory implementations,
    ensuring they are used in the correct lifecycle order and with
    consistent configuration.

    Args:
        engine: Optional ``IntelligenceGraphEngine`` for KG-backed
            persistence and retrieval.
        max_tokens: Default token budget for context compaction.
        auto_compaction_ratio: Fraction of max_tokens that triggers
            automatic compaction (default 0.8).
    """

    def __init__(
        self,
        engine: IntelligenceGraphEngine | None = None,
        *,
        max_tokens: int = 8000,
        auto_compaction_ratio: float = 0.8,
    ) -> None:
        self.engine = engine
        self.max_tokens = max_tokens
        self.auto_compaction_ratio = auto_compaction_ratio

        # Lazy-initialized subsystem instances
        self._compactor: Any = None
        self._startup_builder: Any = None

    # ── Subsystem Access ─────────────────────────────────────────────

    @property
    def compactor(self) -> Any:
        """Lazy-initialized ContextCompactor."""
        if self._compactor is None:
            from .agent_context import ContextCompactor

            self._compactor = ContextCompactor(
                max_tokens=self.max_tokens,
                auto_compaction_ratio=self.auto_compaction_ratio,
            )
        return self._compactor

    @property
    def startup_builder(self) -> Any:
        """Lazy-initialized StartupContextBuilder."""
        if self._startup_builder is None and self.engine is not None:
            # StartupContextBuilder is defined in this module (below); the prior
            # ``from .startup_context import`` was a broken self-reference to a module
            # that never existed.
            self._startup_builder = StartupContextBuilder(engine=self.engine)
        return self._startup_builder

    # ── Phase 1: Startup Context ─────────────────────────────────────

    def build_startup_context(
        self,
        query: str,
        budget: int = 8000,
    ) -> StartupPayload:
        """Assemble initial context within a token budget.

        Uses ``StartupContextBuilder`` to gather KG memories, recent
        mementos, and project context into a budgeted payload for
        the agent's first turn.

        Args:
            query: The user query to tailor context retrieval.
            budget: Maximum token budget for the startup payload.

        Returns:
            A ``StartupPayload`` with assembled context and metadata.
        """
        if self.engine is None:
            return StartupPayload(
                text="",
                budget_chars=budget,
                included_handles=[],
                overflow=[],
            )

        return build_startup_payload(
            engine=self.engine,
            query=query,
            budget=budget,
        )

    # ── Phase 2: Active Context Management ───────────────────────────

    def compact_if_needed(
        self,
        messages: list[dict[str, Any]],
        strategy: CompactionStrategy | str = "summarize_tools",
    ) -> CompactedResult | None:
        """Compact active context if the auto-compaction threshold is exceeded.

        Args:
            messages: Current conversation messages.
            strategy: Compaction strategy to use (``summarize_tools``,
                ``drop_middle``, or ``progressive``).

        Returns:
            A ``CompactedResult`` if compaction was performed, else ``None``.
        """
        if self.compactor.should_compact(messages):
            result = self.compactor.compact(messages, strategy=strategy)
            logger.info(
                "[UnifiedMemory] Compacted context: %d → %d tokens (%s)",
                result.tokens_before,
                result.tokens_after,
                result.strategy_used,
            )
            return result
        return None

    def force_compact(
        self,
        messages: list[dict[str, Any]],
        strategy: CompactionStrategy | str = "summarize_tools",
    ) -> CompactedResult:
        """Force compaction regardless of threshold.

        Args:
            messages: Current conversation messages.
            strategy: Compaction strategy to use.

        Returns:
            A ``CompactedResult`` with compaction details.
        """
        return self.compactor.compact(messages, strategy=strategy)

    # ── Phase 3: State Compression ───────────────────────────────────

    def compress_to_memento(
        self,
        messages: list[dict[str, str]],
        *,
        source: str = "agent_runner",
        dry_run: bool = False,
    ) -> str | None:
        """Compress a block of messages into a dense memento.

        Uses LLM-based ``MementoCompressor`` to generate a state
        snapshot that can replace the raw message block for
        long-running sessions.

        Args:
            messages: The block of raw messages to compress.
            source: The source agent or component name.
            dry_run: If True, generate memento but don't persist to KG.

        Returns:
            The generated memento string, or None if compression failed.
        """
        if self.engine is None:
            return None

        from .memento_compressor import compress_to_memento

        return compress_to_memento(
            self.engine,
            messages,
            source=source,
            dry_run=dry_run,
        )

    def get_recent_mementos(
        self,
        source: str,
        limit: int = 5,
    ) -> list[str]:
        """Retrieve the most recent mementos for a given source.

        Args:
            source: The source agent name.
            limit: Maximum number of mementos to retrieve.

        Returns:
            List of memento content strings.
        """
        if self.engine is None:
            return []

        from .memento_compressor import get_recent_mementos

        return get_recent_mementos(self.engine, source=source, limit=limit)

    # ── Phase 4: Consolidation ───────────────────────────────────────

    def consolidate(
        self,
        *,
        dry_run: bool = True,
    ) -> list[Any]:
        """Promote episodes to preferences and principles.

        Runs the ``ConsolidationEngine`` to identify episode clusters
        that should be promoted to higher-tier knowledge structures
        in the Knowledge Graph.

        Args:
            dry_run: If True, returns proposals without persisting.

        Returns:
            List of ``ConsolidationProposal`` objects describing promotions.
        """
        if self.engine is None:
            return []

        from .optimization_engine import (
            DecisionToPrincipleRule,
            EpisodeToPreferenceRule,
            SynthesisEngine,
            TraceToSkillRule,
        )

        se = SynthesisEngine(engine=self.engine)
        se.register(EpisodeToPreferenceRule())
        se.register(DecisionToPrincipleRule())
        se.register(TraceToSkillRule())
        return se.run(dry_run=dry_run)

    # ── Phase 5: Trace Compaction ────────────────────────────────────

    def compact_traces(
        self,
        agent_id: str = "",
        threshold: int = 10,
    ) -> int:
        """Compact memory traces to prevent graph explosion (CONCEPT:EG-KG.compute.compiled-semantic-reasoner).

        Delegates to :class:`MemoryHygiene`, which decays stale memory nodes
        (soft bi-temporal close) and applies semantic-merge of near-duplicate
        traces — the real "merge or prune trace nodes" mechanism.

        Args:
            agent_id: Advisory scope label (hygiene runs graph-wide); used for logging.
            threshold: Advisory; the live mechanism is decay+semantic-merge, not a
                fixed node-count cutoff.

        Returns:
            Number of compacted nodes (archived + merged).
        """
        if self.engine is None:
            return 0

        from .hygiene import MemoryHygiene

        logger.debug(
            "compact_traces(agent=%s, threshold=%s) → KG-2.17 hygiene pass",
            agent_id,
            threshold,
        )
        report = MemoryHygiene(self.engine).run()
        return int(report.get("archived", 0)) + int(report.get("merged", 0))

    # ── Phase 6: Retrieval ───────────────────────────────────────────

    def retrieve_self_model(self) -> dict[str, Any] | None:
        """Retrieve the current agent self-model from the Knowledge Graph.

        Uses ``MemoryRetriever.get_current()`` to load the latest
        versioned self-model snapshot.

        Returns:
            A dict with self-model data, or None if unavailable.
        """
        if not self.engine:
            return None

        try:
            from ..retrieval.memory_retriever import MemoryRetriever

            mr = MemoryRetriever(engine=self.engine)
            current = mr.get_current()
            if current:
                return current.model_dump()
            return None
        except ImportError:
            logger.debug("MemoryRetriever not available")
            return None
        except Exception as e:
            logger.debug("Self-model retrieval failed: %s", e)
            return None

    def query_capabilities(self, domain: str) -> dict[str, float]:
        """Query the self-model for capability scores in a domain.

        Args:
            domain: The domain to query (e.g., "gitlab", "servicenow").

        Returns:
            Dict with ``success_rate``, ``confidence``, ``proficiency``.
        """
        if not self.engine:
            return {"success_rate": 0.0, "confidence": 0.0, "proficiency": 0.0}

        try:
            from ..retrieval.memory_retriever import MemoryRetriever

            mr = MemoryRetriever(engine=self.engine)
            return mr.query_capabilities(domain)
        except Exception as e:
            logger.debug("Capability query failed: %s", e)
            return {"success_rate": 0.0, "confidence": 0.0, "proficiency": 0.0}


# --- FROM memory_materializer.py ---
#!/usr/bin/python

"""KG -> Markdown Memory Materializer.

CONCEPT:AU-KG.memory.tiered-memory-caching -- Observational Memory Bridge

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
from typing import TYPE_CHECKING

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

    override = setting("AGENT_UTILITIES_MEMORY_DIR")
    if override:
        return Path(override).expanduser()
    return data_dir() / "memory"


def _ensure_memory_dir() -> Path:
    d = memory_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


class MemoryMaterializer:
    """Renders KG memory state into beautiful, inspectable Markdown files.

    CONCEPT:AU-KG.memory.tiered-memory-caching -- Observational Memory Bridge
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
            "[KG-2.7] Materialized %d memory files to %s", len(paths), self.base_dir
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


# --- FROM startup_context.py ---
#!/usr/bin/python

"""Budgeted Startup Context Builder.

CONCEPT:AU-KG.memory.tiered-memory-caching -- Observational Memory Bridge

Produces a deterministic, budget-bounded startup payload for injecting into
external agent hooks (Claude Code SessionStart, Codex startup, Grok Build, etc.).

Architecture:
    KG -> HybridRetriever -> priority-scored chunks -> budget trim -> StartupPayload
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

DEFAULT_BUDGET_CHARS = 24000
MIN_BUDGET_CHARS = 2000


# CONCEPT:AU-KG.memory.ground-truth-preamble-declaring -- Ground-Truth Context Authority.
# Assimilated from memory-os Layer 7 "Ground Truth Hierarchy" (ClaudioDrews/memory-os@a4ca094,
# layers/07-ground-truth.md): injected memory is declared authoritative so the agent stops
# re-fetching/rediscovering context already in its prompt ("memory-zero behavior"). agent-utilities
# makes it structural -- authority is a graph-grounded tier on each chunk, not a flat prompt rule.
AUTHORITY_ADVISORY = "advisory"
AUTHORITY_STANDARD = "standard"
AUTHORITY_AUTHORITATIVE = "authoritative"
# Priority boost applied to authoritative chunks so durable, injected memory ranks above hints.
AUTHORITY_BOOST = 6


def _authority_for(source: str, heading: str) -> str:
    """Classify a startup chunk's authority tier (CONCEPT:AU-KG.memory.ground-truth-preamble-declaring).

    Durable, user/agent-curated memory (profile facts, preferences, identity, active goals, team
    conventions, layered project rules) is *authoritative*; transient recall hints are *advisory*.
    """
    h = heading.lower()
    if source in ("profile", "team", "agents_md"):
        return AUTHORITY_AUTHORITATIVE
    if any(
        k in h
        for k in (
            "core identity",
            "preference",
            "active goal",
            "current session",
            "rule",
        )
    ):
        return AUTHORITY_AUTHORITATIVE
    if source in ("recall", "overflow", "hint"):
        return AUTHORITY_ADVISORY
    return AUTHORITY_STANDARD


@dataclass(frozen=True)
class StartupChunk:
    """A prioritized chunk of startup context."""

    source: str
    heading: str
    body: str
    handle: str
    priority: int
    source_authority: str = AUTHORITY_STANDARD  # CONCEPT:AU-KG.memory.ground-truth-preamble-declaring

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

    CONCEPT:AU-KG.memory.tiered-memory-caching -- Observational Memory Bridge

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
        # CONCEPT:AU-KG.memory.ground-truth-preamble-declaring -- Ground-Truth preamble declaring authoritative sources up front.
        auth_sources = sorted(
            {c.source for c in chunks if c.source_authority == AUTHORITY_AUTHORITATIVE}
        )
        preamble = self._build_authority_preamble(auth_sources)
        used = len(header) + len(footer) + (len(preamble) + 2 if preamble else 0)
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
        if preamble:
            parts.append(preamble)
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
                        source_authority=_authority_for(source_name, heading),
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
        # CONCEPT:AU-KG.memory.ground-truth-preamble-declaring -- authoritative (durable, injected) memory outranks advisory hints.
        if _authority_for(source, heading) == AUTHORITY_AUTHORITATIVE:
            priority += AUTHORITY_BOOST
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
            "<!-- Budgeted startup payload from KG-2.7 Memory Bridge. -->",
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

    def _build_authority_preamble(self, auth_sources: list[str]) -> str:
        """Emit the Ground-Truth Hierarchy preamble (CONCEPT:AU-KG.memory.ground-truth-preamble-declaring).

        Declares that the memory injected below is authoritative and must be used directly, so the
        agent stops re-fetching context it was already given. Assimilated from memory-os Layer 7;
        made structural here -- the named sources are graph-grounded (bi-temporal + trust ranked),
        not an unverifiable prompt assertion. Returns "" when no authoritative sources are present.
        """
        if not auth_sources:
            return ""
        labels = {
            "profile": "user profile & preferences",
            "team": "team conventions",
            "agents_md": "project rules (AGENTS.md)",
        }
        named = ", ".join(labels.get(s, s) for s in auth_sources)
        return (
            "## Ground Truth Hierarchy (authoritative)\n"
            "\n"
            "The memory injected below is **authoritative** for this session. Treat it as already-"
            "known fact: do not re-fetch, re-search, or re-derive information that is already "
            "present here, and when it conflicts with your prior assumptions, the injected memory "
            "wins. Runtime tool output you obtain this turn outranks it; everything else does not.\n"
            f"\n- Authoritative sources present: {named}.\n"
        )

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

        CONCEPT:AU-KG.memory.team-startup-context — Team-Specific Startup Context

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
                        priority=8 + AUTHORITY_BOOST,  # High + authoritative (KG-2.14)
                        source_authority=AUTHORITY_AUTHORITATIVE,
                    )
                )
        except Exception as e:
            logger.debug("Team context load failed for '%s': %s", team, e)
        return chunks

    def _load_layered_agents_md(self, cwd: str) -> StartupChunk | None:
        """Load hierarchically layered AGENTS.md content from CWD.

        CONCEPT:AU-KG.memory.tiered-memory-caching — Layered Project-Aware Context

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
                    priority=9 + AUTHORITY_BOOST,  # Very high + authoritative (KG-2.14)
                    source_authority=AUTHORITY_AUTHORITATIVE,
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


import time


class EvolvingMemoryAPI:
    """Temporally-Aware Epistemic Memory API.

    Provides core logic for storing and querying facts with temporal decay and confidence weighting.
    """

    def __init__(self, engine: Any):
        self.engine = engine

    def store_new_memory(
        self,
        user_id: str,
        content: str,
        confidence: float = 1.0,
        valid_until: int | None = None,
    ) -> str:
        import uuid

        node_id = f"fact_{user_id}_{uuid.uuid4().hex[:8]}"

        now = int(time.time())
        props = {
            "content": content,
            "confidence": confidence,
            "valid_from": now,
        }
        if valid_until is not None:
            props["valid_until"] = valid_until

        self.engine.add_node(node_id, "fact", props)
        self.engine.add_node(user_id, "user", {})

        self.engine.link_nodes(
            source_id=user_id, target_id=node_id, rel_type="HAS_FACT", properties=props
        )
        return node_id

    def retrieve_personalized_context(
        self, user_id: str, query: str, top_k: int = 10, max_hops: int = 1
    ) -> list[dict[str, Any]]:
        try:
            results = self.engine.search_hybrid(query, top_k=top_k)
        except Exception:
            results = self.engine.query_cypher(
                f"MATCH (n:fact {{user_id: '{user_id}'}}) RETURN n"
            )

        now = int(time.time())
        valid_results = []
        for r in results:
            if r.get("user_id") != user_id and r.get("user_id") is not None:
                continue
            if r.get("valid_until") and r.get("valid_until") < now:
                continue
            valid_results.append(r)

            if max_hops > 0:
                neighbors = self.engine.get_blast_radius(r["id"], max_hops=max_hops)
                for neighbor in neighbors:
                    if neighbor not in valid_results:
                        valid_results.append(neighbor)

        return valid_results
