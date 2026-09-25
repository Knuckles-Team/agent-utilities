"""CONCEPT:AU-KG.memory.drive-one-agent-native — Agent-native memory lifecycle loop (AU-side driver).

The epistemic-graph ENGINE ships the deterministic agent-native-memory *primitives*
(called by AU): the summary-node tier ``create_summary_node`` / ``summary_children``
(EG-220), episodic→semantic ``consolidate`` (EG-221), and the maintenance
``reinforce`` / ``decay`` / ``evict`` / ``maintain`` ops (EG-222). This module is the
AU-side LIFECYCLE LOOP that DRIVES them — the *policy* half the engine deliberately
leaves to the agent:

  * **select** a bounded working set + the cluster of episodic memories ripe for
    consolidation (``select_consolidation_candidates``) — "localized maintenance"
    per the paper: we operate on a selected working set, never a global scan;
  * **decide** when to summarize / consolidate / maintain (``tick``);
  * **generate the summary TEXT** for a summary node with the LLM
    (``run_summarization`` — the engine STORES the text, AU PRODUCES it);
  * **call the engine primitives** (``run_summarization`` →
    ``create_summary_node``, ``run_consolidation`` → ``consolidate``,
    ``run_maintenance`` → ``maintain`` = decay+evict).

Design notes:

  * **Additive + off by default.** ``MemoryLifecycleConfig.enabled`` defaults to
    ``False`` (env ``AGENT_UTILITIES_MEMORY_LIFECYCLE``); ``tick`` is a no-op until
    enabled, so registering the schedule (``deploy/schedules.yml``, disabled) is
    inert until an operator turns it on.
  * **Reuses the existing engine client + LLM client.** Reads the working set via
    the same ``engine.backend.execute`` Cypher-subset path as
    :class:`~agent_utilities.knowledge_graph.memory.hygiene.MemoryHygiene`
    (CONCEPT:EG-KG.compute.compiled-semantic-reasoner), and generates summary text via the shared
    ``memento_compressor._memento_llm`` one-shot completion helper. Native
    mutations use the session-routed EG graph client. A missing client is
    reported as an error rather than silently skipping a requested write.
  * **Safe + idempotent.** ``tick`` never raises into the scheduler and dedupes an
    already-processed cluster (per stable signature) so a repeat tick does not
    re-summarize the same episodes.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# Memory nodes in these states are already retired / summarised and are never
# eligible for a fresh consolidation (mirrors hygiene's soft-close statuses).
_INACTIVE_STATUSES = frozenset(
    {"ARCHIVED", "MERGED", "CONSOLIDATED", "SUMMARIZED", "RETIRED", "DELETED"}
)


def _truthy(val: str | None) -> bool:
    return str(val or "").strip().lower() in ("1", "true", "yes", "on")


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    return _truthy(val) if val is not None else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


# Default system prompt for summary-node text generation. Terse, information-dense,
# no invented facts — the summary must let an agent reason without the raw episodes.
_SUMMARY_SYSTEM_PROMPT = (
    "You are the memory-consolidation summarizer for an autonomous agent. Compress "
    "the cluster of related EPISODIC memories below into ONE terse, information-dense "
    "SEMANTIC summary that preserves the durable facts, entities, decisions and "
    "relationships, and drops incidental detail. Invent nothing. Output only the "
    "summary prose — no preamble, no bullet headers."
)


@dataclass
class MemoryLifecycleConfig:
    """Policy knobs for the lifecycle loop (CONCEPT:AU-KG.memory.drive-one-agent-native).

    All fields are additive and conservative; ``enabled`` gates the whole loop off
    by default so the component is inert until an operator opts in.
    """

    enabled: bool = False
    # Bound the localized working set read each tick (never a global scan).
    max_working_set: int = 500
    # A cluster is ripe for consolidation once it has >= this many episodic
    # memories whose oldest member is >= ``min_cluster_age_hours`` old.
    min_cluster_size: int = 3
    min_cluster_age_hours: float = 6.0
    # Cap the number of episodes fed into one summary (token budget + latency).
    max_cluster_size: int = 32
    # Node property clustered on (episodes about the same entity/topic consolidate
    # together); falls back to ``category`` then ``"general"``.
    cluster_key: str = "target_entity"
    # Summary-node text is truncated to this many chars before it is stored.
    summary_max_chars: int = 4000
    # Maintenance (EG-222) tuning passed straight to the engine ``maintain`` op.
    decay_half_life_secs: float = 604_800.0  # 7 days (Ebbinghaus default, KG-2.16)
    decay_floor: float = 0.05
    evict_max_nodes: int = 0  # 0 ⇒ engine default / no hard cap
    system_prompt: str = _SUMMARY_SYSTEM_PROMPT
    # Extra metadata stamped onto summary nodes.
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> MemoryLifecycleConfig:
        """Build a config from ``AGENT_UTILITIES_MEMORY_LIFECYCLE*`` env vars."""
        return cls(
            enabled=_env_bool("AGENT_UTILITIES_MEMORY_LIFECYCLE"),
            max_working_set=_env_int(
                "AGENT_UTILITIES_MEMORY_LIFECYCLE_WORKING_SET", 500
            ),
            min_cluster_size=_env_int(
                "AGENT_UTILITIES_MEMORY_LIFECYCLE_MIN_CLUSTER", 3
            ),
            min_cluster_age_hours=_env_float(
                "AGENT_UTILITIES_MEMORY_LIFECYCLE_MIN_AGE_HOURS", 6.0
            ),
            max_cluster_size=_env_int(
                "AGENT_UTILITIES_MEMORY_LIFECYCLE_MAX_CLUSTER", 32
            ),
            decay_half_life_secs=_env_float(
                "AGENT_UTILITIES_MEMORY_LIFECYCLE_HALF_LIFE_SECS", 604_800.0
            ),
            decay_floor=_env_float(
                "AGENT_UTILITIES_MEMORY_LIFECYCLE_DECAY_FLOOR", 0.05
            ),
            evict_max_nodes=_env_int("AGENT_UTILITIES_MEMORY_LIFECYCLE_EVICT_MAX", 0),
        )


def _age_hours(node: dict[str, Any], now: datetime) -> float:
    for key in ("created_at", "storage_time", "event_time", "updated_at"):
        raw = node.get(key)
        if not raw:
            continue
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return max(0.0, (now - dt).total_seconds() / 3600.0)
    return 0.0


class MemoryLifecycle:
    """AU-side driver for the engine agent-native-memory primitives (CONCEPT:AU-KG.memory.drive-one-agent-native).

    Instantiate with the same ``engine`` object the scheduler hands maintenance
    handlers (it exposes ``.backend`` and — directly or via the compute engine —
    the memory primitives). ``llm`` is an injectable one-shot completion callable
    ``(system_prompt, user_content) -> str | None`` (defaults to the shared
    ``memento_compressor._memento_llm``); tests inject a stub instead of a live model.
    """

    def __init__(
        self,
        engine: Any,
        config: MemoryLifecycleConfig | None = None,
        llm: Callable[[str, str], str | None] | None = None,
    ) -> None:
        self.engine = engine
        self.config = config or MemoryLifecycleConfig.from_env()
        self._llm = llm
        # Stable signatures of clusters already summarised+consolidated by this
        # instance — makes a repeated tick idempotent even if the engine has not
        # yet flipped the source episodes out of the working set.
        self._processed: set[str] = set()

    # ── LLM summary generation ────────────────────────────────────────────────
    def _generate_summary(self, system_prompt: str, user_content: str) -> str | None:
        if self._llm is not None:
            return self._llm(system_prompt, user_content)
        # Reuse the shared, resilient one-shot completion helper (CONCEPT:AU-KG.memory.mementified-context).
        from .memento_compressor import _memento_llm

        return _memento_llm(system_prompt, user_content)

    # ── One typed EG client path ───────────────────────────────────────────────
    def _memory_client(self) -> Any:
        """Return the session-routed EG graph namespace, never a raw wire escape."""
        backend = getattr(self.engine, "backend", None)
        graph = getattr(backend, "graph", None)
        client = getattr(graph, "client", None)
        memory = getattr(client, "graph", None)
        if memory is None:
            raise RuntimeError("EG memory client is unavailable")
        return memory

    # ── Working-set read (localized, bounded) ─────────────────────────────────
    def _read_working_set(self, now: datetime) -> list[dict[str, Any]]:
        """Read a BOUNDED working set of memory nodes (never a global scan).

        Mirrors the hygiene scan (CONCEPT:EG-KG.compute.compiled-semantic-reasoner): reads ``:Memory`` nodes with
        content via the engine backend's Cypher-subset ``execute``, capped at
        ``config.max_working_set``.
        """
        backend = getattr(self.engine, "backend", None)
        if backend is None or not hasattr(backend, "execute"):
            return []
        limit = max(1, int(self.config.max_working_set))
        try:
            rows = backend.execute(
                "MATCH (n:Memory) WHERE n.content IS NOT NULL "
                f"RETURN n.id as id, n as data LIMIT {limit}"
            )
        except Exception as e:  # noqa: BLE001 — backend variance, degrade to empty
            logger.debug("[KG-2.307] working-set read failed: %s", e)
            return []
        nodes: list[dict[str, Any]] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            data = row.get("data")
            if isinstance(data, dict):
                d = dict(data)
                d["id"] = row.get("id") or d.get("id")
                nodes.append(d)
        return nodes

    # ── Candidate selection ───────────────────────────────────────────────────
    def select_consolidation_candidates(
        self,
        nodes: list[dict[str, Any]] | None = None,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Return the single ripest cluster of episodic memories, or ``[]``.

        Localized maintenance (CONCEPT:AU-KG.memory.drive-one-agent-native): groups the ACTIVE ``episodic``
        memories in the working set by ``cluster_key`` (→ ``category`` → ``general``)
        and returns the group that is ripe — at least ``min_cluster_size`` episodes
        whose oldest member is at least ``min_cluster_age_hours`` old — preferring
        the largest (then oldest) such group. Members are returned in a stable
        (oldest-first, then id) order so the cluster signature is deterministic.
        """
        now = now or datetime.now(UTC)
        if nodes is None:
            nodes = self._read_working_set(now)

        groups: dict[str, list[dict[str, Any]]] = {}
        for n in nodes:
            if str(n.get("memory_type", "")).lower() != "episodic":
                continue
            if str(n.get("status", "ACTIVE")).upper() in _INACTIVE_STATUSES:
                continue
            if not (n.get("content") or n.get("name")):
                continue
            key = str(n.get(self.config.cluster_key) or n.get("category") or "general")
            groups.setdefault(key, []).append(n)

        best: list[dict[str, Any]] = []
        best_oldest = 0.0
        for group in groups.values():
            if len(group) < self.config.min_cluster_size:
                continue
            oldest = max(_age_hours(g, now) for g in group)
            if oldest < self.config.min_cluster_age_hours:
                continue
            if len(group) > len(best) or (
                len(group) == len(best) and oldest > best_oldest
            ):
                best, best_oldest = group, oldest

        if not best:
            return []
        best.sort(key=lambda g: (-_age_hours(g, now), str(g.get("id", ""))))
        return best[: max(1, int(self.config.max_cluster_size))]

    @staticmethod
    def _cluster_ids(cluster: list[dict[str, Any]]) -> list[str]:
        return [str(n.get("id", "")) for n in cluster if n.get("id")]

    def _signature(self, cluster: list[dict[str, Any]]) -> str:
        joined = "|".join(sorted(self._cluster_ids(cluster)))
        return hashlib.sha256(joined.encode()).hexdigest()

    def _cluster_text(self, cluster: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for n in cluster:
            text = str(n.get("content") or n.get("name") or "").strip()
            if not text:
                continue
            tags = n.get("tags") or []
            tag_s = f" [tags: {', '.join(map(str, tags))}]" if tags else ""
            parts.append(f"- ({n.get('id', '?')}){tag_s} {text}")
        return "\n".join(parts)

    # ── Phase: summarization → engine create_summary_node (EG-220) ────────────
    def run_summarization(self, cluster: list[dict[str, Any]]) -> dict[str, Any]:
        """LLM-generate the summary TEXT for a cluster and store it as a summary node.

        AU produces the text (CONCEPT:AU-KG.memory.drive-one-agent-native); the engine ``create_summary_node``
        (EG-220) stores it as the parent of the cluster's episodes.
        """
        ids = self._cluster_ids(cluster)
        if not ids:
            return {"status": "skipped", "reason": "empty_cluster"}
        summary_text = self._generate_summary(
            self.config.system_prompt,
            "Consolidate these related episodic memories into one summary:\n"
            + self._cluster_text(cluster),
        )
        if not summary_text or not summary_text.strip():
            return {"status": "skipped", "reason": "no_summary_text", "child_ids": ids}
        summary_text = summary_text.strip()[: self.config.summary_max_chars]
        summary_id = self._memory_client().create_summary_node(
            child_ids=ids,
            summary_text=summary_text,
            metadata={
                "source": "memory_lifecycle",
                "concept": "AU-KG.memory.drive-one-agent-native",
                **dict(self.config.extra_metadata),
            },
        )
        return {
            "status": "ok",
            "summary_id": summary_id,
            "summary_text": summary_text,
            "child_ids": ids,
        }

    # ── Phase: consolidation → engine consolidate (EG-221) ────────────────────
    def run_consolidation(
        self,
        cluster: list[dict[str, Any]],
        summary_id: str,
        summary_text: str,
    ) -> dict[str, Any]:
        """Fold the episodic cluster into semantic memory via the engine (EG-221)."""
        ids = self._cluster_ids(cluster)
        if not ids:
            return {"status": "skipped", "reason": "empty_cluster"}
        if not summary_id or not summary_text:
            raise ValueError(
                "EG summary ID and text are required for memory consolidation"
            )
        res = self._memory_client().consolidate_memories(
            episodic_ids=ids, summary_id=summary_id, summary_text=summary_text
        )
        return {"status": "ok", "node_ids": ids, "result": res}

    # ── Phase: maintenance → engine maintain = decay + evict (EG-222) ─────────
    def run_maintenance(
        self,
        working_set: list[dict[str, Any]],
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Run localized decay+evict over the WORKING SET via the engine (EG-222).

        "Localized maintenance": the ``maintain`` op is scoped to the selected
        working-set node ids, not a global sweep.
        """
        now = now or datetime.now(UTC)
        ids = [str(n.get("id", "")) for n in working_set if n.get("id")]
        if not ids:
            return {"status": "skipped", "reason": "empty_working_set"}
        if self.config.evict_max_nodes > 0:
            raise ValueError("EG Maintain supports a threshold, not evict_max_nodes")
        res = self._memory_client().maintain_memories(
            ids=ids,
            now_ms=int(now.timestamp() * 1000),
            half_life_ms=int(self.config.decay_half_life_secs * 1000),
            evict_threshold=self.config.decay_floor,
            delete=False,
        )
        return {"status": "ok", "node_count": len(ids), "result": res}

    # ── The scheduled entry point ─────────────────────────────────────────────
    def tick(self, now: datetime | None = None) -> dict[str, Any]:
        """One lifecycle cycle — the scheduled maintenance task (CONCEPT:AU-KG.memory.drive-one-agent-native).

        Safe + idempotent: no-op when disabled, never raises into the scheduler, and
        skips re-summarising a cluster already processed by this instance.
        """
        if not self.config.enabled:
            return {"status": "disabled"}
        now = now or datetime.now(UTC)
        result: dict[str, Any] = {
            "status": "ok",
            "summarized": 0,
            "consolidated": 0,
            "maintained": 0,
        }
        try:
            working_set = self._read_working_set(now)
            result["working_set"] = len(working_set)

            cluster = self.select_consolidation_candidates(working_set, now)
            if cluster:
                sig = self._signature(cluster)
                if sig in self._processed:
                    result["skipped"] = "already_processed"
                else:
                    summ = self.run_summarization(cluster)
                    result["summarization"] = summ
                    if summ.get("status") == "ok":
                        result["summarized"] = 1
                        cons = self.run_consolidation(
                            cluster, summ["summary_id"], summ["summary_text"]
                        )
                        result["consolidation"] = cons
                        if cons.get("status") == "ok":
                            result["consolidated"] = 1
                        # Mark processed once we have generated+stored the summary,
                        # so a repeat tick over the same episodes is a no-op.
                        self._processed.add(sig)

            maint = self.run_maintenance(working_set, now)
            result["maintenance"] = maint
            if maint.get("status") == "ok":
                result["maintained"] = int(maint.get("node_count", 0) or 0)
            return result
        except Exception as e:  # noqa: BLE001 — must never raise into the scheduler
            logger.error("[KG-2.307] memory lifecycle tick failed: %s", e)
            return {"status": "error", "error": str(e)}


def run_memory_lifecycle(
    engine: Any,
    *,
    now: datetime | None = None,
    config: MemoryLifecycleConfig | None = None,
) -> dict[str, Any]:
    """CLI/daemon entry point for one memory-lifecycle cycle (CONCEPT:AU-KG.memory.drive-one-agent-native).

    The scheduler dispatches here (kind=skill, ref=``memory-lifecycle``,
    action=``maintain``). Gated off unless ``AGENT_UTILITIES_MEMORY_LIFECYCLE`` is
    set, so it is inert even if the (default-disabled) schedule is turned on.
    """
    return MemoryLifecycle(engine, config=config).tick(now=now)
