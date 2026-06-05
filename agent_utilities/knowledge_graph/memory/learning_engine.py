"""CONCEPT:KG-2.13 — Background Learning Engine.

Assimilates Quarq Agent's asynchronous targeted-edit learner (agent-oss/agent.py:99-160,
2951-3007, 3303/3646) into agent-utilities. The learner does **targeted ADD / UPDATE / DELETE**
fact edits — not raw transcript dumps — under a concurrency semaphore with bounded exponential
backoff and a sync barrier, writing **bi-temporal graph mutations** (KG-2.11) so an UPDATE
supersedes (history preserved) and a DELETE is soft.

Pure helpers (`resolve_relative_dates`, `parse_memory_edits`, `apply_edits`) are LLM-free and
fully unit-testable; the LLM extraction (`extract_edits`) uses the ORCH-1.27 ``learner`` role.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

# Quarq concurrency / backoff controls (agent-oss/agent.py:99,134-157). Unlike Quarq's *infinite*
# retry loop, backoff is bounded by ``max_attempts`` so the unit suite (≤60s gate) cannot hang.
LEARNING_CONCURRENCY = 4
BACKOFF_INITIAL = 2.0
BACKOFF_MAX = 60.0


class MemoryEdit(BaseModel):
    """A single targeted memory edit emitted by the learner.

    CONCEPT:KG-2.13 (+ memory-os typed-extraction enhancement, ClaudioDrews/memory-os@a4ca094,
    icarus/hooks.py): beyond ADD/UPDATE/DELETE, an edit carries a typed classification, a
    training-value signal, an outcome-grounding gate, and the evidence ids it was derived from, so
    only grounded, high-value learning is persisted.
    """

    action: Literal["ADD", "UPDATE", "DELETE"]
    id: str = ""
    content: str = ""
    memory_type: Literal["semantic", "episodic", "procedural"] = "semantic"
    target_entity: str = ""
    event_time: str | None = None
    entry_type: Literal["decision", "resolution", "note", "fact"] = "fact"
    training_value: Literal["high", "normal", "low"] = "normal"
    outcome_gate: bool = True
    evidence_ids: list[str] = Field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def resolve_relative_dates(text: str, *, now: str | None = None) -> str:
    """Resolve common relative date words to absolute ISO dates at learn time.

    CONCEPT:KG-2.13 — mirrors Quarq's learn-time relative→absolute rule (agent.py:3114-3161):
    "yesterday", "today", "tomorrow", and "N days/weeks/months ago" become absolute dates so the
    stored ``event_time`` is a real instant, never a floating phrase. Vague recency ("recently",
    "a while ago") is intentionally left untouched (it is report-time context, not an event date).
    """
    base = datetime.fromisoformat((now or _now_iso()).replace("Z", "+00:00"))

    def _d(delta_days: int) -> str:
        return (base + timedelta(days=delta_days)).strftime("%Y-%m-%d")

    out = text
    out = re.sub(r"\byesterday\b", _d(-1), out, flags=re.IGNORECASE)
    out = re.sub(r"\btoday\b", _d(0), out, flags=re.IGNORECASE)
    out = re.sub(r"\btomorrow\b", _d(1), out, flags=re.IGNORECASE)

    unit_days = {"day": 1, "week": 7, "month": 30, "year": 365}

    def _ago(m: re.Match[str]) -> str:
        n = int(m.group(1))
        unit = m.group(2).lower().rstrip("s")
        return _d(-n * unit_days.get(unit, 1))

    out = re.sub(
        r"\b(\d+)\s+(day|days|week|weeks|month|months|year|years)\s+ago\b",
        _ago,
        out,
        flags=re.IGNORECASE,
    )
    return out


def parse_memory_edits(raw: str) -> list[MemoryEdit]:
    """Parse a learner LLM response into validated edits, defensively (never raises).

    Accepts ``{"actions": [...]}`` or a bare JSON list, embedded in prose/fences. Invalid rows
    are skipped. Returns ``[]`` on total failure.
    """
    try:
        obj_match = re.search(r"\{.*\}|\[.*\]", raw, re.DOTALL)
        if not obj_match:
            return []
        data = json.loads(obj_match.group())
        rows = data.get("actions", []) if isinstance(data, dict) else data
        edits: list[MemoryEdit] = []
        for row in rows:
            try:
                edits.append(MemoryEdit.model_validate(row))
            except Exception as e:  # noqa: BLE001 - skip malformed individual rows
                logger.debug("skipping malformed memory edit row: %s", e)
                continue
        return edits
    except (ValueError, TypeError, json.JSONDecodeError):
        return []


async def with_backoff(
    fn: Any,
    *,
    initial: float = BACKOFF_INITIAL,
    max_delay: float = BACKOFF_MAX,
    max_attempts: int = 5,
) -> Any:
    """Await ``fn()`` with bounded exponential backoff (2→60s, capped at ``max_attempts``).

    Returns the result, or re-raises the last exception once attempts are exhausted. Bounded
    (unlike Quarq's infinite loop) so background learning can never wedge CI.
    """
    delay = initial
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except Exception as e:  # noqa: BLE001 - persistent-queue semantics
            last_exc = e
            if attempt >= max_attempts:
                break
            logger.warning(
                "[KG-2.13] learn attempt %d failed (%s); retrying in %.1fs",
                attempt,
                e,
                delay,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)
    if last_exc:
        raise last_exc
    return None


class BackgroundLearner:
    """Async, semaphore-bounded targeted-edit learner (CONCEPT:KG-2.13)."""

    def __init__(
        self,
        engine: IntelligenceGraphEngine,
        *,
        concurrency: int = LEARNING_CONCURRENCY,
    ) -> None:
        self.engine = engine
        self._sem = asyncio.Semaphore(concurrency)
        self._pending: set[asyncio.Task[Any]] = set()

    # ── edit application (pure-ish: only touches the engine) ──────────────────

    def apply_edits(
        self, edits: list[MemoryEdit], *, now: str | None = None
    ) -> dict[str, int]:
        """Apply ADD/UPDATE/DELETE edits as bi-temporal graph mutations.

        - ADD: create a ``MemoryNode`` stamped with event_time/storage_time/valid_from (KG-2.11).
        - UPDATE: stamp the existing node's ``valid_to`` (supersede) and write the new content.
        - DELETE: **soft** — set ``status="REMOVED"`` + close ``valid_to`` (history preserved).

        Returns counts ``{"added": n, "updated": n, "deleted": n, "skipped": n}``.
        """
        from ...models.knowledge_graph import MemoryNode
        from ..core.bitemporal import stamp_bitemporal

        stamp = now or _now_iso()
        counts = {"added": 0, "updated": 0, "deleted": 0, "skipped": 0, "gated": 0}

        for edit in edits:
            try:
                # CONCEPT:KG-2.13 — outcome-grounding gate: a decision/resolution that claims
                # grounding must cite evidence ids to be persisted; otherwise it is dropped (not
                # stored as an unverified fact). Notes/facts are exempt.
                if (
                    edit.action == "ADD"
                    and edit.outcome_gate
                    and edit.entry_type in ("decision", "resolution")
                    and not edit.evidence_ids
                ):
                    counts["gated"] += 1
                    continue
                if edit.action == "ADD" and edit.content:
                    props = stamp_bitemporal({}, event_time=edit.event_time, now=stamp)
                    node = MemoryNode(
                        id=edit.id or f"mem_{abs(hash((edit.content, stamp)))}",
                        name=edit.content[:80],
                        content=edit.content,
                        memory_type=edit.memory_type,
                        target_entity=edit.target_entity,
                        tags=[f"type:{edit.entry_type}", f"train:{edit.training_value}"],
                        event_time=props["event_time"],
                        storage_time=props["storage_time"],
                        valid_from=props["valid_from"],
                        valid_to=props["valid_to"],
                    )
                    self.engine.add_memory_node(node)
                    # Write GROUNDED_BY edges to the evidence the edit was derived from.
                    for ev in edit.evidence_ids:
                        try:
                            self.engine.link_nodes(node.id, ev, "GROUNDED_BY")
                        except Exception:  # noqa: BLE001 - evidence linking is best-effort
                            pass
                    counts["added"] += 1
                elif edit.action == "UPDATE" and edit.id:
                    existing = self.engine.get_memory_node(edit.id)
                    if existing is None:
                        counts["skipped"] += 1
                        continue
                    props = stamp_bitemporal({}, event_time=edit.event_time, now=stamp)
                    if edit.content:
                        existing.content = edit.content
                        existing.name = edit.content[:80]
                    existing.event_time = props["event_time"]
                    existing.storage_time = stamp
                    self.engine.update_memory_node(edit.id, existing)
                    counts["updated"] += 1
                elif edit.action == "DELETE" and edit.id:
                    existing = self.engine.get_memory_node(edit.id)
                    if existing is None:
                        counts["skipped"] += 1
                        continue
                    existing.status = "REMOVED"
                    existing.valid_to = stamp
                    self.engine.update_memory_node(edit.id, existing)
                    counts["deleted"] += 1
                else:
                    counts["skipped"] += 1
            except Exception as e:  # noqa: BLE001 - one bad edit must not abort the batch
                logger.warning("[KG-2.13] edit %s failed: %s", edit.action, e)
                counts["skipped"] += 1
        return counts

    # ── async scheduling + sync barrier ──────────────────────────────────────

    async def learn_async(
        self, edits: list[MemoryEdit], *, now: str | None = None
    ) -> dict[str, int]:
        """Apply edits under the concurrency semaphore with bounded backoff."""

        async def _run() -> dict[str, int]:
            async with self._sem:
                return await asyncio.to_thread(self.apply_edits, edits, now=now)

        return await with_backoff(_run)

    def schedule(
        self, edits: list[MemoryEdit], *, now: str | None = None
    ) -> asyncio.Task[Any]:
        """Fire-and-forget a learn task, tracked for the sync barrier (Quarq agent.py:2978-2981)."""
        task = asyncio.ensure_future(self.learn_async(edits, now=now))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return task

    async def await_pending(self) -> None:
        """Sync barrier — drain all in-flight learn tasks (Quarq agent.py:1791-1796)."""
        if self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)


def extract_edits(
    engine: IntelligenceGraphEngine,
    transcript: str,
    *,
    now: str | None = None,
) -> list[MemoryEdit]:
    """Extract targeted edits from a transcript via the ORCH-1.27 ``learner`` role (best-effort)."""
    from ...core.model_factory import create_model

    resolved = resolve_relative_dates(transcript, now=now)
    system_prompt = (
        "You are a memory learner. Read the interaction and emit ONLY a JSON object "
        '{"actions": [{"action": "ADD|UPDATE|DELETE", "id": "<for UPDATE/DELETE>", '
        '"content": "<atomic fact>", "memory_type": "semantic|episodic|procedural", '
        '"target_entity": "<for procedural rules>", "event_time": "<ISO date if known>", '
        '"entry_type": "decision|resolution|note|fact", '
        '"training_value": "high|normal|low", '
        '"evidence_ids": ["<ids of memories/messages this is grounded in>"]}]}. '
        "Add new facts, update outdated ones, delete contradictions. Classify each entry: a "
        "decision/resolution MUST cite evidence_ids and have a confirmed outcome, or it will be "
        "dropped. Tag high training_value only when the entry is a durable, reusable lesson. "
        "Preserve every number with its owner/action/event/exactness. Do not dump raw text."
    )
    try:
        from pydantic_ai import Agent

        model = create_model(role="learner")
        agent = Agent(model=model, system_prompt=system_prompt)
        result: Any = agent.run_sync(resolved)
        raw = str(getattr(result, "output", None) or getattr(result, "data", ""))
    except Exception as e:  # pragma: no cover - learner is best-effort
        logger.debug("[KG-2.13] learner extraction failed: %s", e)
        return []
    return parse_memory_edits(raw)


def run_learner(
    engine: IntelligenceGraphEngine,
    transcript: str,
    *,
    now: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Synchronous learn entry point for the CLI: extract edits then apply them.

    Returns ``{"edits": [...], "counts": {...}}`` (counts empty on ``dry_run``).
    """
    edits = extract_edits(engine, transcript, now=now)
    if dry_run:
        return {"edits": [e.model_dump() for e in edits], "counts": {}}
    counts = BackgroundLearner(engine).apply_edits(edits, now=now)
    return {"edits": [e.model_dump() for e in edits], "counts": counts}
