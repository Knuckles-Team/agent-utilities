#!/usr/bin/python
from __future__ import annotations

"""Scope-aware ``HistorySource`` over epistemic-graph's canonical RunTrace/ToolCall
provenance (PA-R1.8, pydantic-ai-harness reconciliation Track 5).

Upstream ships persisted-run search as ``StepPersistence(store=SqliteStepStore(...))``
+ ``ConversationSearch(SnapshotHistorySource(store))`` — a BM25 index over every run a
``HistorySource`` enumerates, exposed to the model as one tool,
``search_conversation_history`` (``pydantic_ai_harness.conversation_search``, verified
against the installed ``pydantic-ai-harness==0.14.0`` at
``.venv/lib/python3.14/site-packages/pydantic_ai_harness/conversation_search/``). Its own
docs name the frontier: the ``HistorySource.list_runs()`` protocol method takes **no
filter arguments at all** — the toolset's only access control is the capability-level
``scope: 'all' | 'conversation'`` switch (``ConversationSearchToolset._load_sections``:
``runs = await self._source.list_runs()`` unfiltered, then optionally narrowed to one
``conversation_id``). That is binary, not hierarchical: a sub-agent handed
``scope='all'`` reaches every run in the store; handed ``scope='conversation'`` it reaches
every run in the *whole* conversation, including sibling sub-agents and its own parent's
other business. There is no shape for "only my own subtree."

We already have the substrate this needs. ``:RunTrace``/``:ToolCall`` provenance
(``observability.trace_ontology``) already carries ``parent_run_id`` on every node (see
``kg_audit_sink.ToolCallRecord``/``RunAuditRecord``) — a real, already-written delegation
edge, not an invented one. And ``GraphSession`` (``knowledge_graph.core.session``) is our
one fail-closed identity/tenant/scope currency. :class:`ScopedEgHistorySource` composes
them: it implements the native ``HistorySource`` protocol structurally (no import of
``pydantic_ai_harness`` needed for that — it is duck-typed, and the module docstring of
``conversation_search._source`` explicitly invites a new substrate to implement it "via
replay"), but its own ``list_runs()``/``run_history()`` are bound, at construction, to one
``GraphSession`` and one ``root_run_id``. "All the runs this source enumerates" then means
"all the runs at or below ``root_run_id`` in the ``parent_run_id`` tree" — exactly the
missing access control the article names. A top-level orchestrator's own run has no
ancestors, so its subtree is everything below it (the old "search everything" behavior,
now *earned* by position in the tree rather than granted unconditionally); a delegated
sub-agent's subtree is only its own descendants. One mechanism serves both.

Fail-closed, explicitly:
  * :meth:`ScopedEgHistorySource.__init__` calls ``session.require_scope("kg:read")``
    immediately — a session that cannot read the KG at all never gets a source constructed.
  * :meth:`ScopedEgHistorySource.list_runs` returns ``[]`` (never "everything") the moment
    the engine is unavailable, the root run cannot be resolved, or the closure computation
    fails for any reason — it does NOT fall back to an unscoped enumeration.
  * :meth:`ScopedEgHistorySource.run_history` re-verifies ``run_id`` is a member of the
    CURRENT scoped closure before reading anything — it does not trust a caller-supplied
    ``run_id`` merely because ``list_runs`` was scoped once. This matters because
    ``HistorySource`` is a public protocol: a caller can invoke ``run_history`` directly
    without going through the toolset's own ``list_runs``-then-filter path.

Scaling note: the closure is computed by pulling every ``:RunTrace`` row's
``(run_id, parent_run_id)`` pair and doing an in-process BFS, not a native recursive
Cypher traversal — the smallest correct thing that keeps this an ADAPTER over the
existing ontology rather than a new store or a new query surface. A tenant with a very
large run population should follow up with a bounded/paginated or native-recursive
version; tracked as a known limitation, not hidden.

Raw content availability is the other honest limit worth naming: canonical
``trace_ontology.tool_call_properties``/``trace_properties`` blank ``args``/``result``/
``error`` to privacy-sanitized digests by design (``PersistencePrivacyGuard`` — see
``trace_ontology.py``). The ONLY place actual searchable text survives in the graph is
the ``audit_arguments``/``audit_result``/``audit_error`` fields :class:`KgAuditSink`
writes (through its pluggable, default-no-op ``redactor``). :meth:`run_history` therefore
reads through that same channel (via the shared ``tool_call_records_from_rows`` /
``tool_call_rows_by_trace_node_id`` helpers ``kg_audit_sink`` now exports) — a run that
was never recorded through ``AuditLog``/``KgAuditSink`` contributes no searchable text,
which is a direct, intentional consequence of the platform's existing privacy default,
not a bug in this adapter.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.capabilities.kg_audit_sink import (
    KgAuditSink,
    parse_timestamp,
    tool_call_records_from_rows,
    tool_call_rows_by_trace_node_id,
)
from agent_utilities.knowledge_graph.core.session import GraphSession
from agent_utilities.observability.trace_ontology import (
    TRACE_NODE_LABEL,
)
from agent_utilities.observability.trace_ontology import (
    trace_id as canonical_trace_id,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EgStepStore",
    "ScopedEgHistorySource",
    "build_conversation_search_capability",
]


def _run_tree_rows(engine: Any) -> list[dict[str, Any]]:
    """Read every ``(run_id, parent_run_id)`` pair the graph currently holds.

    Best-effort like every other ``context_plane.read_rows`` caller: a backend hiccup
    yields ``[]``, which composes correctly here because an empty adjacency set makes
    the BFS below resolve to "only the root, if it exists" — never "everything".
    """
    from agent_utilities.knowledge_graph.retrieval.context_plane import read_rows

    return read_rows(
        engine,
        f"MATCH (t:{TRACE_NODE_LABEL}) "
        "RETURN t.run_id AS run_id, t.parent_run_id AS parent_run_id",
        {},
    )


def _scoped_closure(engine: Any, root_run_id: str) -> set[str]:
    """Return the canonical-id closure of ``root_run_id`` and its transitive descendants.

    ``t.run_id`` on a ``:RunTrace`` node is already the canonical ``trace_id(...)`` form
    (``trace_properties`` stamps it that way); ``t.parent_run_id`` is stored RAW (see
    ``kg_audit_sink.ToolCallRecord``/``RunAuditRecord`` writers) — an intentional
    asymmetry inherited from the existing ontology, not introduced here. Each row's
    ``parent_run_id`` is therefore re-canonicalized before comparison so parent/child
    edges line up in one id space.
    """
    root_canonical = canonical_trace_id(root_run_id)
    if engine is None:
        return set()
    try:
        rows = _run_tree_rows(engine)
    except Exception as exc:
        # A closure read must fail closed (scope to nothing), never raise open into a
        # caller that might otherwise fall back to an unscoped read. Deliberately
        # broad: any failure mode here (bad engine, malformed rows, backend outage)
        # gets the same fail-closed treatment, logged with its cause preserved.
        logger.warning(
            "ScopedEgHistorySource: run-tree read failed (%s: %s); scoping to no runs",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return set()

    children_by_parent: dict[str, set[str]] = defaultdict(set)
    known: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        child = row.get("run_id")
        if not child:
            continue
        child = str(child)
        known.add(child)
        raw_parent = row.get("parent_run_id")
        if raw_parent:
            parent_canonical = canonical_trace_id(str(raw_parent))
            children_by_parent[parent_canonical].add(child)

    allowed: set[str] = set()
    if root_canonical in known:
        allowed.add(root_canonical)
    else:
        # The root run itself may not have accumulated a :RunTrace row yet (e.g. it is
        # still in flight) — that is not a reason to see nothing below it once children
        # do exist, but it IS a reason to never silently widen: seed with the root id
        # and let the BFS attach only rows that actually declare it as their parent.
        allowed.add(root_canonical)
    frontier = [root_canonical]
    while frontier:
        current = frontier.pop()
        for child in children_by_parent.get(current, ()):
            if child not in allowed:
                allowed.add(child)
                frontier.append(child)
    return allowed


@dataclass
class ScopedEgHistorySource:
    """Structural ``pydantic_ai_harness.conversation_search.HistorySource`` over eg.

    Bound at construction to one ``GraphSession`` (tenant + scope authority) and one
    ``root_run_id`` (the delegation-tree position that bounds "at or below my own
    rank"). See module docstring for the full design rationale and the two named
    fail-closed properties.
    """

    engine: Any
    session: GraphSession
    root_run_id: str
    _sink: KgAuditSink = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Fail closed at construction: a session that cannot even read the KG must
        # never get a working source handed to a model.
        self.session.require_scope("kg:read")
        if not str(self.root_run_id or "").strip():
            raise ValueError(
                "ScopedEgHistorySource requires a non-empty root_run_id to bound "
                "history visibility — there is no unscoped construction path"
            )
        self._sink = KgAuditSink(engine=self.engine)

    def _closure(self) -> set[str]:
        return _scoped_closure(self.engine, self.root_run_id)

    async def list_runs(self) -> list[Any]:
        """Return every run at or below ``root_run_id``, sorted by ``started_at``.

        Native ``HistorySource.list_runs()`` takes no filter arguments and is
        documented upstream to mean "every persisted run the source enumerates" — this
        is that contract, honestly kept: what THIS source enumerates is already
        rank-scoped, so "every run" here never reaches a sibling subtree or an
        ancestor's other business.
        """
        from pydantic_ai_harness.step_persistence import RunRecord

        allowed = self._closure()
        if not allowed:
            return []
        if self.engine is None:
            return []
        from agent_utilities.knowledge_graph.retrieval.context_plane import read_rows

        rows = read_rows(
            self.engine,
            f"MATCH (t:{TRACE_NODE_LABEL}) WHERE t.run_id IN $ids "
            "RETURN t.run_id AS run_id, t.conversation_id AS conversation_id, "
            "t.parent_run_id AS parent_run_id, t.agent_name AS agent_name, "
            "t.timestamp AS timestamp",
            {"ids": sorted(allowed)},
        )
        records: list[RunRecord] = []
        for row in rows:
            if not isinstance(row, dict) or not row.get("run_id"):
                continue
            records.append(
                RunRecord(
                    run_id=str(row["run_id"]),
                    conversation_id=row.get("conversation_id") or None,
                    parent_run_id=row.get("parent_run_id") or None,
                    agent_name=row.get("agent_name") or None,
                    started_at=parse_timestamp(row.get("timestamp")),
                )
            )
        records.sort(key=lambda r: r.started_at)
        return records

    async def run_history(self, *, run_id: str) -> list[Any]:
        """Return one run's durable message record, reconstructed from ``:ToolCall``.

        Re-derives the scoped closure and checks membership BEFORE reading anything —
        does not trust that ``run_id`` reached here via ``list_runs`` (see module
        docstring: ``HistorySource`` is a public protocol another caller can invoke
        directly). An out-of-scope or unknown ``run_id`` returns ``[]``, matching the
        native contract's "unknown run_id yields []" — the caller cannot distinguish
        "does not exist" from "not visible to you", which is the same non-enumerable
        failure shape ``ConversationSearchToolset`` already relies on for ``run_id``
        misses.
        """
        allowed = self._closure()
        if run_id not in allowed:
            return []
        if self.engine is None:
            return []
        rows = tool_call_rows_by_trace_node_id(self.engine, run_id)
        records = tool_call_records_from_rows(rows, run_id=run_id)
        return _records_to_model_messages(records)


def _records_to_model_messages(records: list[Any]) -> list[Any]:
    """Reconstruct a best-effort ``list[ModelMessage]`` from recorded tool calls.

    Only the raw-content channel described in the module docstring
    (``audit_arguments``/``audit_result``/``audit_error``) is searchable; each
    ``ToolCallRecord`` becomes one ``ModelResponse(ToolCallPart)`` (the call) and, when
    a result or error was captured, one ``ModelRequest(ToolReturnPart)`` (the outcome)
    — enough structure for the upstream BM25 toolset's message renderer
    (``_format_messages``) to produce indexable text, without claiming byte-fidelity
    with an original model conversation this substrate never captured.
    """
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        ToolCallPart,
        ToolReturnPart,
    )

    messages: list[Any] = []
    for record in records:
        messages.append(
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=record.tool_name,
                        args=record.arguments,
                        tool_call_id=record.tool_call_id,
                    )
                ]
            )
        )
        if record.error is not None:
            content = f"[error] {record.error}"
        elif record.result is not None:
            content = record.result
        else:
            continue
        messages.append(
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name=record.tool_name,
                        content=content,
                        tool_call_id=record.tool_call_id,
                    )
                ]
            )
        )
    return messages


class EgStepStore:
    """Partial ``pydantic_ai_harness.step_persistence.StepStore`` facade over eg.

    Deliberately partial, and loud about it. The run/event ledger
    (``register_run``/``get_run``/``list_runs``/``append_event``/``list_events``) maps
    cleanly onto the SAME ``:RunTrace``/``:ToolCall`` writes ``KgAuditSink`` already
    performs — this facade is a thin translation, not a new store.

    Snapshot/resume (``save_snapshot``/``latest_snapshot``) and the tool-effect ledger
    (``record_tool_effect``/``get_tool_effect``/``list_unresolved_tool_effects``) are
    NOT implemented: eg has no substrate today for a full-fidelity
    ``list[ModelMessage]`` snapshot or a side-effect-status ledger, and resume-from-
    snapshot is a correctness-critical feature — silently no-op'ing it would be a
    silent data-loss bug on replay, not a safe default. These methods raise
    ``NotImplementedError`` with a clear message rather than pretend to support resume.
    Building real snapshot/effect persistence is a separate, larger piece of work
    (new node types, since raw ``ModelMessage`` content and effect status are not
    captured by the canonical trace ontology today) — out of scope here.
    """

    _UNSUPPORTED = (
        "{method} is not implemented: epistemic-graph has no substrate yet for "
        "{what}. Silently no-op'ing this would be a silent-failure risk (resume/"
        "replay correctness), so it fails loud instead of fails quiet."
    )

    def __init__(self, engine: Any, session: GraphSession) -> None:
        session.require_scope("kg:write")
        self._engine = engine
        self._session = session
        self._sink = KgAuditSink(engine=engine)

    async def register_run(self, record: Any) -> None:
        from agent_utilities.capabilities.kg_audit_sink import RunAuditRecord

        await self._sink.record_run(
            RunAuditRecord(
                run_id=record.run_id,
                outcome="completed",
                conversation_id=record.conversation_id,
                parent_run_id=record.parent_run_id,
                agent_name=record.agent_name,
                started_at=record.started_at,
                ended_at=record.started_at,
            )
        )

    async def get_run(self, *, run_id: str) -> Any:
        from pydantic_ai_harness.step_persistence import RunRecord

        got = await self._sink.get_run(run_id=run_id)
        if got is None:
            return None
        return RunRecord(
            run_id=run_id,
            conversation_id=got.conversation_id,
            parent_run_id=got.parent_run_id,
            agent_name=got.agent_name,
            started_at=got.started_at,
        )

    async def list_runs(
        self,
        *,
        parent_run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[Any]:
        # A raw run/conversation-id sweep without a GraphSession-scoped root is exactly
        # the unscoped "whole store" shape this program's charter says not to add — a
        # write-side StepStore has no `root_run_id` concept to bound it by, so this
        # deliberately stays narrow (single run_id lookups only) rather than growing
        # into a second, unscoped read path next to ScopedEgHistorySource.
        raise NotImplementedError(
            "EgStepStore.list_runs is intentionally unimplemented: use "
            "ScopedEgHistorySource for scoped read access instead of adding a second, "
            "unscoped enumeration path."
        )

    async def append_event(self, event: Any) -> None:
        from agent_utilities.capabilities.kg_audit_sink import ToolCallRecord

        if event.kind not in {
            "tool_call_started",
            "tool_call_completed",
            "tool_call_failed",
        }:
            # run_started/run_completed/model_request_* have no ToolCall-shaped home
            # in the canonical ontology yet; dropping them (not raising) keeps
            # StepPersistence's per-boundary emission from crashing a live run over a
            # write path this facade only partially covers.
            return
        await self._sink.record_tool_call(
            ToolCallRecord(
                run_id=event.run_id,
                tool_call_id=event.tool_call_id or "",
                tool_name=event.tool_name or "",
                arguments="",
                error=event.error,
                started_at=event.timestamp,
                ended_at=event.timestamp if event.kind != "tool_call_started" else None,
                conversation_id=event.conversation_id,
                parent_run_id=event.parent_run_id,
                agent_name=event.agent_name,
            )
        )

    async def list_events(self, *, run_id: str) -> list[Any]:
        raise NotImplementedError(
            self._UNSUPPORTED.format(
                method="EgStepStore.list_events",
                what="a StepEvent-shaped read distinct from ToolCallRecord",
            )
        )

    async def save_snapshot(self, snapshot: Any) -> None:
        raise NotImplementedError(
            self._UNSUPPORTED.format(
                method="EgStepStore.save_snapshot",
                what="a full-fidelity list[ModelMessage] snapshot store",
            )
        )

    async def latest_snapshot(
        self, *, run_id: str, include_interrupted: bool = False
    ) -> Any:
        raise NotImplementedError(
            self._UNSUPPORTED.format(
                method="EgStepStore.latest_snapshot",
                what="a full-fidelity list[ModelMessage] snapshot store",
            )
        )

    async def record_tool_effect(self, record: Any) -> None:
        raise NotImplementedError(
            self._UNSUPPORTED.format(
                method="EgStepStore.record_tool_effect",
                what="a tool-effect status ledger",
            )
        )

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> Any:
        raise NotImplementedError(
            self._UNSUPPORTED.format(
                method="EgStepStore.get_tool_effect",
                what="a tool-effect status ledger",
            )
        )

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[Any]:
        raise NotImplementedError(
            self._UNSUPPORTED.format(
                method="EgStepStore.list_unresolved_tool_effects",
                what="a tool-effect status ledger",
            )
        )


def build_conversation_search_capability(
    engine: Any,
    session: GraphSession,
    *,
    root_run_id: str,
    scope: str = "all",
) -> Any:
    """Build a native ``ConversationSearch`` capability wired to :class:`ScopedEgHistorySource`.

    Thin composition helper — wrap, don't fork (the program charter's rule): the BM25
    ranking, rendering, and tool contract stay the native
    ``pydantic_ai_harness.conversation_search.ConversationSearch`` implementation
    unmodified. Only the corpus source is ours, and it is already rank-scoped, so this
    helper's ``scope`` parameter composes on top rather than replacing the enforcement:
    ``'conversation'`` still additionally narrows to the calling run's own
    ``conversation_id`` within the (already-narrower) rank-scoped set.
    """
    from pydantic_ai_harness.conversation_search import ConversationSearch

    source = ScopedEgHistorySource(
        engine=engine, session=session, root_run_id=root_run_id
    )
    return ConversationSearch(source=source, scope=scope)
