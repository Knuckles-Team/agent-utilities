#!/usr/bin/python
from __future__ import annotations

"""KG-backed ``AuditSink``: adopts the pydantic-ai-harness ``audit_log`` abstraction.

CONCEPT:AU-KG.audit.kg-native-audit-sink — PA-R1.7 (pydantic-ai-harness reconciliation,
contribution C-1 ``pydantic-ai-kg-provenance``). Upstream's ``audit_log`` capability
(``open-source-libraries/pydantic-ai-harness`` @ branch ``contrib/audit-log``) defines an
``AuditSink`` Protocol — ``record_tool_call`` / ``record_run`` / ``list_tool_calls`` /
``get_run`` — deliberately de-opinionated: it ships no persistent backend and no default
redaction policy, only the shape plus trivial in-memory/JSONL references. It is a fork, not
an au dependency, so this module implements the SAME Protocol shape locally rather than
importing it: :class:`AuditSink`, :class:`ToolCallRecord`, and :class:`RunAuditRecord` are
field-for-field mirrors of ``pydantic_ai_harness.audit_log``'s ``_sink.py`` / ``_types.py``.
When the fork lands upstream as a real dependency, only the Protocol re-export need change —
:class:`KgAuditSink` already satisfies it structurally (``@runtime_checkable``).

:class:`KgAuditSink` is the KG-backed implementation: it writes the SAME canonical
``:RunTrace`` / ``:ToolCall`` provenance nodes (``observability.trace_ontology``,
KG-2.296) that ``orchestration/agent_runner.py`` and ``workflows/runner.py`` already write
— reusing ``trace_properties`` / ``tool_call_properties`` for the canonical, privacy-guarded
fields (status, timing, sequence, content digests) rather than forking a second node shape —
so au's audit trail is KG-native and queryable with the same
``RunTrace -[:USED_TOOL]-> ToolCall`` Cypher shape every other provenance consumer already
uses. Alongside those canonical fields it stamps ``audit_arguments`` / ``audit_result`` /
``audit_error``: the ``AuditSink`` record's own text, passed through a pluggable
``redactor`` hook that defaults to no-op (``identity_redactor``) — the de-opinionation
principle applies to this sink specifically: it takes no position on what counts as a
secret, matching the harness's own ``identity_redactor`` default, and composes with (does
not replace) the canonical fields' existing ``PersistencePrivacyGuard`` digesting.

:class:`AuditLog` is the capability that drives the sink from a live run — it mirrors the
harness's own ``AuditLog`` capability's hook shape (``for_run`` / ``wrap_run`` /
``before_tool_execute`` / ``after_tool_execute`` / ``on_tool_execute_error`` / ``after_run``
/ ``on_run_error``) so a future swap to the upstream fork is a resolver change, not a
rewrite. Its default ``sink_resolver`` reads ``ctx.deps.graph_engine`` — the SAME lazy
per-run engine lookup ``StuckLoopDetection`` / ``ContextLimitWarner`` / ``MementoCompaction``
/ ``ToolOutputEviction`` / ``TeamCapability`` already use — so it is safe to attach
unconditionally: with no engine on ``ctx.deps`` it resolves a ``KgAuditSink(engine=None)``,
whose writes are no-ops, exactly like every other default reliability capability's own
"only acts when its trigger fires" contract (``capabilities/composition.py``).

Deliberately out of scope (left to au's existing, separate infrastructure rather than
reimplemented here — anti-sprawl): cross-run delegation-chain correlation (au's
``GraphSession`` / correlation traceparent already own that), and the identity/tenant
stamping ``orchestration/agent_runner.py``'s own ``_stamp_run_identity`` applies to its own
post-hoc provenance pass. When an agent's run is ALSO covered by that pass (the direct
single-server loop), both paths write the same canonical node ids — an idempotent upsert,
not a duplicate — this capability's marginal value is reaching the agents built directly
through ``core/contextual_model.create_context_agent`` that pass never covers (the same
A-0 gap ``capabilities/composition.py`` closed for compaction/stuck-loop/eviction).
"""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import uuid4

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability

logger = logging.getLogger(__name__)

RunOutcome = Literal["completed", "failed"]
"""Terminal outcome of an agent run recorded in a :class:`RunAuditRecord`."""

Redactor = Callable[[str, object], object]
"""Per-field redaction hook: ``(field_name, value) -> value``. Defaults to no redaction —
this sink takes no position on what counts as a secret (mirrors the harness AuditLog's
``identity_redactor`` default); pass your own for your deployment's policy."""


def identity_redactor(_field: str, value: object) -> object:
    """Default :data:`Redactor`: every value passes through unchanged."""
    return value


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _safe_str(value: Any) -> str:
    """``str()`` a tool result/error, substituting a placeholder if ``__str__`` itself raises."""
    try:
        return str(value)
    except Exception:  # noqa: BLE001 — a broken __str__ must not break audit recording
        return "<unrepresentable>"


def _parse_timestamp(value: Any) -> datetime:
    """Parse the ``trace_ontology`` ``%Y-%m-%dT%H:%M:%SZ`` timestamp back to a ``datetime``."""
    if isinstance(value, str) and value:
        try:
            return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        except ValueError:  # noqa: BLE001 — malformed/legacy timestamp text falls back to "now"; read-back display only, never a control-flow or audit-integrity decision
            pass
    return _utcnow()


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(kw_only=True)
class ToolCallRecord:
    """The audit record for one tool call. Mirrors ``pydantic_ai_harness.audit_log.ToolCallRecord``."""

    run_id: str
    tool_call_id: str
    tool_name: str
    arguments: str
    result: str | None = None
    error: str | None = None
    started_at: datetime = field(default_factory=_utcnow)
    ended_at: datetime | None = None
    conversation_id: str | None = None
    parent_run_id: str | None = None
    agent_name: str | None = None


@dataclass(kw_only=True)
class RunAuditRecord:
    """The audit record for one agent run. Mirrors ``pydantic_ai_harness.audit_log.RunAuditRecord``."""

    run_id: str
    outcome: RunOutcome
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    error: str | None = None
    conversation_id: str | None = None
    parent_run_id: str | None = None
    agent_name: str | None = None
    started_at: datetime = field(default_factory=_utcnow)
    ended_at: datetime = field(default_factory=_utcnow)


@runtime_checkable
class AuditSink(Protocol):
    """The adopted Protocol shape — see module docstring.

    Any backend (this KG sink, a file, a DB) implements these four async methods.
    ``record_tool_call`` / ``record_run`` are append/write; ``list_tool_calls`` returns a
    run's tool calls in recorded order; ``get_run`` returns the run's latest record, or
    ``None`` when the run is absent.
    """

    async def record_tool_call(
        self, record: ToolCallRecord
    ) -> None: ...  # pragma: no cover

    async def record_run(self, record: RunAuditRecord) -> None: ...  # pragma: no cover

    async def list_tool_calls(
        self, *, run_id: str
    ) -> list[ToolCallRecord]: ...  # pragma: no cover

    async def get_run(
        self, *, run_id: str
    ) -> RunAuditRecord | None: ...  # pragma: no cover


class KgAuditSink:
    """:class:`AuditSink` backed by the epistemic-graph's canonical RunTrace/ToolCall provenance.

    Writes through ``observability.trace_ontology``'s ``trace_properties`` /
    ``tool_call_properties`` builders — the SAME canonical, privacy-guarded shape every other
    provenance writer in au uses — so records materialize as ordinary ``:RunTrace`` /
    ``:ToolCall`` nodes, queryable with the standard ``RunTrace -[:USED_TOOL]-> ToolCall``
    Cypher shape (``TRACE_USED_TOOL_EDGE``). ``engine=None`` (no KG available) makes every
    method a no-op / empty-read, matching the ``if not engine`` idiom the rest of au's
    engine-touching capabilities use.
    """

    def __init__(
        self, engine: Any = None, *, redactor: Redactor = identity_redactor
    ) -> None:
        self._engine = engine
        self._redactor = redactor
        self._sequence_lock = threading.Lock()
        self._sequence: dict[str, int] = {}

    def _next_sequence(self, run_id: str) -> int:
        with self._sequence_lock:
            seq = self._sequence.get(run_id, 0)
            self._sequence[run_id] = seq + 1
            return seq

    async def record_tool_call(self, record: ToolCallRecord) -> None:
        """Write one ``:ToolCall`` node linked to its run's ``:RunTrace``."""
        if self._engine is None:
            return
        from agent_utilities.observability.trace_ontology import (
            TOOL_CALL_NODE_LABEL,
            TRACE_USED_TOOL_EDGE,
            tool_call_properties,
        )
        from agent_utilities.observability.trace_ontology import (
            trace_id as canonical_trace_id,
        )

        trace_node_id = canonical_trace_id(record.run_id)
        sequence = self._next_sequence(record.run_id)
        tc_id = f"toolcall:{trace_node_id.removeprefix('trace:')}:{sequence}"
        ts = (record.ended_at or _utcnow()).strftime("%Y-%m-%dT%H:%M:%SZ")
        props = tool_call_properties(
            run_id=record.run_id,
            tool_name=record.tool_name,
            args=record.arguments,
            result=record.result or "",
            error=record.error or "",
            status="ok" if record.error is None else "error",
            sequence=sequence,
            timestamp=ts,
        )
        props["tool_call_id"] = record.tool_call_id
        props["audit_arguments"] = _safe_str(
            self._redactor("arguments", record.arguments)
        )
        if record.result is not None:
            props["audit_result"] = _safe_str(self._redactor("result", record.result))
        if record.error is not None:
            props["audit_error"] = _safe_str(self._redactor("error", record.error))
        if record.agent_name:
            props["agent_name"] = record.agent_name
        if record.conversation_id:
            props["conversation_id"] = record.conversation_id
        if record.parent_run_id:
            props["parent_run_id"] = record.parent_run_id
        if record.ended_at is not None:
            props["duration_ms"] = max(
                0.0, (record.ended_at - record.started_at).total_seconds() * 1000
            )
        try:
            self._engine.add_node(tc_id, TOOL_CALL_NODE_LABEL, properties=props)
            self._engine.link_nodes(trace_node_id, tc_id, TRACE_USED_TOOL_EDGE)
        except Exception:  # noqa: BLE001 — a provenance write must never fail the run
            logger.warning(
                "KgAuditSink failed to record tool call %r for run %r",
                record.tool_call_id,
                record.run_id,
                exc_info=True,
            )

    async def record_run(self, record: RunAuditRecord) -> None:
        """Write one ``:RunTrace`` node for the run's terminal outcome."""
        if self._engine is None:
            return
        from agent_utilities.observability.trace_ontology import (
            TRACE_NODE_LABEL,
            trace_properties,
        )
        from agent_utilities.observability.trace_ontology import (
            trace_id as canonical_trace_id,
        )

        trace_node_id = canonical_trace_id(record.run_id)
        duration_ms = max(
            0.0, (record.ended_at - record.started_at).total_seconds() * 1000
        )
        props = trace_properties(
            run_id=record.run_id,
            agent_name=record.agent_name or "",
            task="",
            status=record.outcome,
            timestamp=record.ended_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            error=record.error,
            duration_ms=duration_ms,
        )
        if record.error is not None:
            props["audit_error"] = _safe_str(self._redactor("error", record.error))
        if record.agent_name:
            props["agent_name"] = record.agent_name
        if record.conversation_id:
            props["conversation_id"] = record.conversation_id
        if record.parent_run_id:
            props["parent_run_id"] = record.parent_run_id
        if record.input_tokens is not None:
            props["input_tokens"] = int(record.input_tokens)
        if record.output_tokens is not None:
            props["output_tokens"] = int(record.output_tokens)
        if record.total_tokens is not None:
            props["total_tokens"] = int(record.total_tokens)
        try:
            self._engine.add_node(trace_node_id, TRACE_NODE_LABEL, properties=props)
        except Exception:  # noqa: BLE001 — a provenance write must never fail the run
            logger.warning(
                "KgAuditSink failed to record run %r", record.run_id, exc_info=True
            )

    async def list_tool_calls(self, *, run_id: str) -> list[ToolCallRecord]:
        """Return ``run_id``'s tool calls in recorded (sequence) order."""
        if self._engine is None:
            return []
        from agent_utilities.knowledge_graph.retrieval.context_plane import read_rows
        from agent_utilities.observability.trace_ontology import (
            TRACE_USED_TOOL_EDGE,
        )
        from agent_utilities.observability.trace_ontology import (
            trace_id as canonical_trace_id,
        )

        trace_node_id = canonical_trace_id(run_id)
        rows = read_rows(
            self._engine,
            f"MATCH (:RunTrace {{id: $tid}})-[:{TRACE_USED_TOOL_EDGE}]->(t:ToolCall) "
            "RETURN t.tool_call_id AS tool_call_id, t.tool_name AS tool_name, "
            "t.audit_arguments AS audit_arguments, t.audit_result AS audit_result, "
            "t.audit_error AS audit_error, t.timestamp AS timestamp, "
            "t.agent_name AS agent_name, t.conversation_id AS conversation_id, "
            "t.parent_run_id AS parent_run_id, t.sequence AS sequence "
            "ORDER BY t.sequence ASC",
            {"tid": trace_node_id},
        )
        records: list[ToolCallRecord] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            error = row.get("audit_error") or None
            ts = _parse_timestamp(row.get("timestamp"))
            records.append(
                ToolCallRecord(
                    run_id=run_id,
                    tool_call_id=str(row.get("tool_call_id") or ""),
                    tool_name=str(row.get("tool_name") or ""),
                    arguments=str(row.get("audit_arguments") or ""),
                    result=(row.get("audit_result") or None) if error is None else None,
                    error=error,
                    started_at=ts,
                    ended_at=ts,
                    conversation_id=row.get("conversation_id") or None,
                    parent_run_id=row.get("parent_run_id") or None,
                    agent_name=row.get("agent_name") or None,
                )
            )
        return records

    async def get_run(self, *, run_id: str) -> RunAuditRecord | None:
        """Return ``run_id``'s latest ``RunAuditRecord``, or ``None`` when absent."""
        if self._engine is None:
            return None
        from agent_utilities.knowledge_graph.retrieval.context_plane import read_rows
        from agent_utilities.observability.trace_ontology import (
            trace_id as canonical_trace_id,
        )

        trace_node_id = canonical_trace_id(run_id)
        rows = read_rows(
            self._engine,
            "MATCH (t:RunTrace {id: $tid}) RETURN t.status AS status, "
            "t.audit_error AS audit_error, t.input_tokens AS input_tokens, "
            "t.output_tokens AS output_tokens, t.total_tokens AS total_tokens, "
            "t.conversation_id AS conversation_id, t.parent_run_id AS parent_run_id, "
            "t.agent_name AS agent_name, t.timestamp AS timestamp LIMIT 1",
            {"tid": trace_node_id},
        )
        if not rows or not isinstance(rows[0], dict):
            return None
        row = rows[0]
        status = str(row.get("status") or "").lower()
        outcome: RunOutcome = "failed" if status in ("failed", "error") else "completed"
        ts = _parse_timestamp(row.get("timestamp"))
        return RunAuditRecord(
            run_id=run_id,
            outcome=outcome,
            input_tokens=_int_or_none(row.get("input_tokens")),
            output_tokens=_int_or_none(row.get("output_tokens")),
            total_tokens=_int_or_none(row.get("total_tokens")),
            error=row.get("audit_error") or None,
            conversation_id=row.get("conversation_id") or None,
            parent_run_id=row.get("parent_run_id") or None,
            agent_name=row.get("agent_name") or None,
            started_at=ts,
            ended_at=ts,
        )


def _default_kg_sink_resolver(ctx: RunContext[Any]) -> AuditSink:
    """Default ``sink_resolver``: a fresh :class:`KgAuditSink` off ``ctx.deps.graph_engine``.

    The same lazy per-run engine lookup ``StuckLoopDetection`` / ``MementoCompaction`` /
    ``ToolOutputEviction`` / ``TeamCapability`` already use (``getattr(ctx.deps,
    "graph_engine", None)``). No engine on ``ctx.deps`` yields ``KgAuditSink(engine=None)``,
    which no-ops on every write — safe to attach unconditionally.
    """
    return KgAuditSink(engine=getattr(ctx.deps, "graph_engine", None))


@dataclass
class AuditLog(AbstractCapability[Any]):
    """Records a structured audit trail of tool calls and run outcomes to an :class:`AuditSink`.

    Adopts the pydantic-ai-harness ``audit_log.AuditLog`` capability's hook shape (see module
    docstring) with au's KG-native sink resolved by default. Where OpenTelemetry emits spans
    for observability, this captures the *content* of each call as a durable, queryable KG
    record.

    ``sink_resolver`` is keyed on the run context so each run's ``KgAuditSink`` binds the
    RIGHT engine (``ctx.deps.graph_engine``) rather than one fixed at construction time —
    matching how this capability is attached once, in ``capabilities/composition.py``, and
    reused across every agent/run that inherits the default capability set. Pass your own
    ``sink_resolver`` (or a fixed ``sink=``) to redirect audit records elsewhere (e.g. a test
    double, or a non-KG backend), and ``redactor`` to strip specific fields before they reach
    the sink — the capability, like :class:`KgAuditSink` itself, applies no redaction by
    default.
    """

    sink: AuditSink | None = None
    """Fixed sink, used when ``sink_resolver`` is ``None``. Unset means no-op (matches
    ``KgAuditSink(engine=None)``)."""

    sink_resolver: Callable[[RunContext[Any]], AuditSink] | None = (
        _default_kg_sink_resolver
    )
    """Per-run sink resolver, keyed on the run context. ``None`` falls back to ``sink``."""

    redactor: Redactor = identity_redactor
    """Per-field redaction policy ``(field_name, value) -> value``, applied to the raw
    arguments/result/error text before it reaches the sink. Defaults to no redaction."""

    agent_name: str | None = None
    """Logical agent name stamped on every record for cross-run attribution."""

    _started_at: datetime = field(
        default_factory=_utcnow, init=False, repr=False, compare=False
    )
    _pending: dict[str, datetime] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _run_id_fallback: str | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not serializable: ``sink``/``sink_resolver``/``redactor`` may carry closures."""
        return None

    async def for_run(self, ctx: RunContext[Any]) -> AuditLog:
        """Clone with per-run state: the resolved sink and a fresh ``run_id`` fallback."""
        sink = self.sink_resolver(ctx) if self.sink_resolver is not None else self.sink
        clone = replace(self, sink=sink)
        clone._run_id_fallback = ctx.run_id or uuid4().hex
        return clone

    def _run_id(self, ctx: RunContext[Any]) -> str:
        return ctx.run_id or self._run_id_fallback or uuid4().hex

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: Any,
        tool_def: Any,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Stamp the tool call's start time so the record carries its duration."""
        self._pending[call.tool_call_id] = _utcnow()
        return args

    async def after_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: Any,
        tool_def: Any,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        """Record the completed tool call and return its result unchanged."""
        await self._record_tool_call(
            ctx, call=call, tool_def=tool_def, args=args, result=result, error=None
        )
        return result

    async def on_tool_execute_error(
        self,
        ctx: RunContext[Any],
        *,
        call: Any,
        tool_def: Any,
        args: dict[str, Any],
        error: Exception,
    ) -> Any:
        """Record the failed tool call, then re-raise so behavior is unchanged."""
        await self._record_tool_call(
            ctx, call=call, tool_def=tool_def, args=args, result=None, error=error
        )
        raise error

    async def _record_tool_call(
        self,
        ctx: RunContext[Any],
        *,
        call: Any,
        tool_def: Any,
        args: dict[str, Any],
        result: Any,
        error: Exception | None,
    ) -> None:
        if self.sink is None:
            return
        record = ToolCallRecord(
            run_id=self._run_id(ctx),
            tool_call_id=call.tool_call_id,
            tool_name=tool_def.name,
            arguments=_safe_str(self.redactor("arguments", args)),
            result=None if error is not None else _safe_str(result),
            error=None if error is None else _safe_str(repr(error)),
            started_at=self._pending.pop(call.tool_call_id, _utcnow()),
            ended_at=_utcnow(),
            conversation_id=ctx.conversation_id,
            agent_name=self.agent_name,
        )
        try:
            await self.sink.record_tool_call(record)
        except Exception:  # noqa: BLE001 — a sink failure must not affect the tool outcome
            logger.warning(
                "AuditLog sink failed to record tool call %r for run %r",
                call.tool_call_id,
                record.run_id,
                exc_info=True,
            )

    async def after_run(self, ctx: RunContext[Any], *, result: Any) -> Any:
        """Record the run as completed with its final token usage, then return it unchanged."""
        await self._record_run(ctx, outcome="completed", error=None)
        return result

    async def on_run_error(self, ctx: RunContext[Any], *, error: BaseException) -> Any:
        """Record the run as failed with its error, then re-raise."""
        await self._record_run(ctx, outcome="failed", error=error)
        raise error

    async def _record_run(
        self, ctx: RunContext[Any], *, outcome: RunOutcome, error: BaseException | None
    ) -> None:
        if self.sink is None:
            return
        usage = ctx.usage
        record = RunAuditRecord(
            run_id=self._run_id(ctx),
            outcome=outcome,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            error=None if error is None else _safe_str(repr(error)),
            conversation_id=ctx.conversation_id,
            agent_name=self.agent_name,
            started_at=self._started_at,
            ended_at=_utcnow(),
        )
        try:
            await self.sink.record_run(record)
        except Exception:  # noqa: BLE001 — a sink failure must not affect the run outcome
            logger.warning(
                "AuditLog sink failed to record run %r", record.run_id, exc_info=True
            )


__all__ = [
    "AuditLog",
    "AuditSink",
    "KgAuditSink",
    "Redactor",
    "RunAuditRecord",
    "RunOutcome",
    "ToolCallRecord",
    "identity_redactor",
]
