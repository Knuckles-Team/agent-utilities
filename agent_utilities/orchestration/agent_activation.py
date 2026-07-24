#!/usr/bin/python
from __future__ import annotations

"""Agents-as-data activation layer (CONCEPT:AU-ORCH.dispatch.agents-as-data-activation, ADR-6 / W2.3).

**A million agents are ROWS, not processes.** A dormant agent is a durable
``eg-statechart`` ``MachineInstance`` (its ``dormant ⇄ active`` lifecycle) plus a
per-tenant graph node (placement + mailbox) — **no connection, no thread, no heartbeat
while dormant.** This module is the composition — three EXISTING stores + one worker
loop, no new scheduler framework and no actor library (ADR-6 non-goals):

1. **statecharts.redb** (``Method::Statechart``) — the agent-instance ``MachineInstance``
   is the durable lifecycle authority. Its template is the canonical
   :data:`AGENT_LIFECYCLE_DEF` (byte-mirror of eg's ``agent_lifecycle_statechart.rs``).
2. **WorkItem CAS** (the ADR-5 substrate) — each *activation* is a WorkItem
   (:data:`WORK_ITEM_KIND`). Its redb CAS row is the sole lease authority (who may run);
   its own statechart is the activation lifecycle. **The lease IS the liveness signal** —
   a dead worker's lease expires and the activation re-queues (bounded retries →
   dead_letter), so there is no heartbeat on the dormant instance row.
3. **graphs** — the ``:AgentInstance`` node (dormant registry row + mailbox + a
   best-effort mirror of the authoritative lifecycle state), the ``:AgentMessage`` mailbox,
   and the ``:RunTrace`` / ``:ToolCall`` provenance.

The activation flow::

    event (CDC / timer / broker / direct call)
      └─ deliver_activation: append :AgentMessage to the mailbox
                             + submit a WorkItem (QoS lane from the source)
    worker pool (stateless; one process = one worker, N processes = the pool)
      └─ claim_next  ── CAS lease (the ADR-5 authority; also the liveness signal)
      └─ statechart activate  ── dormant → active (OCC = the instance concurrency guard)
      └─ ADR-4 identity chain  ── build_spawn_delegation + run-token + use_delegation
      └─ priority_scope(class) ── engine calls ride the W2.4 QoS priority claim
      └─ drain mailbox + run   ── the injectable executor (default: provenance-ack)
      └─ :RunTrace + :ToolCall  ── the delegation chain recorded as provenance
      └─ heartbeat while active ── renews the lease AND revalidates delegation expiry
      └─ commit_result         ── the WorkItem lifecycle authority commits terminal
      └─ statechart deactivate ── active → dormant; save state; release (implicit)

Backpressure (ADR-6 §5): the activation's QoS class is derived from its SOURCE
(:func:`activation_priority_class`) — a direct/interactive call preempts a
CDC-from-ingest one — and rides both the WorkItem ``prio_bucket`` (claim ordering) and
the :class:`PriorityClass` the worker binds while running (the W2.4 engine QoS lanes).
"""

import contextlib
import json
import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from agent_utilities.core.resource_priority import PriorityClass, priority_scope
from agent_utilities.orchestration import work_item as _wi
from agent_utilities.security import delegation as _delegation

logger = logging.getLogger(__name__)

__all__ = [
    "AGENT_LIFECYCLE_DEF",
    "STATE_DORMANT",
    "STATE_ACTIVE",
    "STATE_TERMINATED",
    "EV_ACTIVATE",
    "EV_DEACTIVATE",
    "EV_TERMINATE",
    "WORK_ITEM_KIND",
    "AgentActivationError",
    "ActivationSource",
    "ActivationContext",
    "ActivationResult",
    "activation_priority_class",
    "agent_lifecycle_def_id",
    "register_agent_instance",
    "get_agent_instance",
    "list_agent_instances",
    "terminate_agent_instance",
    "deliver_activation",
    "drain_mailbox",
    "set_activation_executor",
    "process_one_activation",
    "run_activation_worker_loop",
    "start_activation_worker_pool",
    "main",
]


# ── the agent-lifecycle statechart def (byte-mirror of agent_lifecycle_statechart.rs) ──
#
# Authored as a plain dict shaped EXACTLY like Rust's ``eg_statechart::model::StatechartDef``
# (fields name/schema_version/states/alphabet/transitions/initial/finals/meta; guards
# externally-tagged on ``op`` per Rust's ``#[serde(tag="op")]``; ``meta`` empty so the
# server-computed content-addressed ``def_id`` equals eg's ``AGENT_LIFECYCLE_DEF_ID``).
# Mirrors ``research.loops.LOOP_STATECHART_DEF`` (the accepted W2.5 idiom).

STATE_DORMANT = "dormant"
STATE_ACTIVE = "active"
STATE_TERMINATED = "terminated"

EV_ACTIVATE = "activate"
EV_DEACTIVATE = "deactivate"
EV_TERMINATE = "terminate"

#: WorkItem ``kind`` for an activation; the worker claims only this kind.
WORK_ITEM_KIND = "agent_activation"

_INSTANCE_LABEL = "AgentInstance"
_MESSAGE_LABEL = "AgentMessage"
_RUNTRACE_LABEL = "RunTrace"
_TOOLCALL_LABEL = "ToolCall"
_HAS_MAILBOX_MESSAGE = "has_mailbox_message"
_ACTIVATION_OF = "activation_of"
_TRACE_OF = "run_trace_of"
_TOOLCALL_OF = "tool_call_of"


def _always() -> dict[str, Any]:
    return {"op": "always"}


def _build_agent_lifecycle_statechart_def() -> dict[str, Any]:
    """Build the canonical agent-instance lifecycle chart (ADR-6 §1).

    The four transitions are declared in the SAME order as the Rust
    ``agent_lifecycle_statechart_def()`` so the two encode byte-identically and the
    content-addressed ``def_id`` matches (the parity anchor).
    """
    return {
        "name": "agent_lifecycle",
        "schema_version": 1,
        "states": [
            {"id": STATE_DORMANT},
            {"id": STATE_ACTIVE},
            {"id": STATE_TERMINATED},
        ],
        "alphabet": [EV_ACTIVATE, EV_DEACTIVATE, EV_TERMINATE],
        "transitions": [
            {
                "from": STATE_DORMANT,
                "event": EV_ACTIVATE,
                "guard": _always(),
                "to": STATE_ACTIVE,
            },
            {
                "from": STATE_ACTIVE,
                "event": EV_DEACTIVATE,
                "guard": _always(),
                "to": STATE_DORMANT,
            },
            {
                "from": STATE_DORMANT,
                "event": EV_TERMINATE,
                "guard": _always(),
                "to": STATE_TERMINATED,
            },
            {
                "from": STATE_ACTIVE,
                "event": EV_TERMINATE,
                "guard": _always(),
                "to": STATE_TERMINATED,
            },
        ],
        "initial": STATE_DORMANT,
        "finals": [STATE_TERMINATED],
        "meta": {},
    }


#: The canonical dormant-agent lifecycle template. Registered once (content-addressed /
#: idempotent) via :func:`agent_lifecycle_def_id`.
AGENT_LIFECYCLE_DEF: dict[str, Any] = _build_agent_lifecycle_statechart_def()


class AgentActivationError(RuntimeError):
    """Raised when the activation layer cannot complete a required engine step.

    Fails LOUD (AU-P0-3 discipline) rather than silently degrading — a registration or
    activation that could not reach its authoritative store must not report success.
    """


def agent_lifecycle_def_id(engine: Any) -> str:
    """Register the agent-lifecycle chart, returning its content-addressed id.

    ``engine.statechart.define`` is content-addressed / idempotent server-side (mirrors
    ``research.loops.loop_def_id``), so calling it defensively every time is correct.
    """
    return str(engine.statechart.define(AGENT_LIFECYCLE_DEF))


# ── activation source → QoS lane (ADR-6 §5 backpressure) ─────────────────────────────


class ActivationSource(StrEnum):
    """Where an activation event came from — determines its QoS admission class.

    ADR-6 §5: a direct/interactive call preempts an ingestion-triggered one. Mapped to
    the ONE :class:`PriorityClass` currency (:func:`activation_priority_class`) so the
    activation's WorkItem ordering AND the engine calls the worker makes while running it
    both ride the same W2.4 QoS lane end to end.
    """

    #: A direct / interactive call (a live user or agent invoking this instance now).
    DIRECT = "direct"
    #: A scheduled timer / cron tick.
    TIMER = "timer"
    #: An orchestration/delegation-driven activation.
    ORCHESTRATION = "orchestration"
    #: A broker message (another service/agent published an event this instance subscribes to).
    BROKER = "broker"
    #: A change-data-capture match produced by background ingestion — the yielding lane.
    CDC = "cdc"


_SOURCE_PRIORITY: dict[ActivationSource, PriorityClass] = {
    ActivationSource.DIRECT: PriorityClass.INTERACTIVE,
    ActivationSource.TIMER: PriorityClass.ORCHESTRATION,
    ActivationSource.ORCHESTRATION: PriorityClass.ORCHESTRATION,
    ActivationSource.BROKER: PriorityClass.ORCHESTRATION,
    ActivationSource.CDC: PriorityClass.BACKGROUND_INGESTION,
}


def activation_priority_class(source: ActivationSource | str) -> PriorityClass:
    """Map an activation source to its QoS admission class (ADR-6 §5).

    ``direct`` ⇒ INTERACTIVE, ``timer``/``orchestration``/``broker`` ⇒ ORCHESTRATION,
    ``cdc`` ⇒ BACKGROUND_INGESTION. An unknown source degrades to ORCHESTRATION (au's
    untagged-context default: high, never starved — never silently deprioritised).
    """
    try:
        src = ActivationSource(str(source))
    except ValueError:
        logger.warning(
            "[agents-as-data] unknown activation source %r; defaulting to ORCHESTRATION",
            source,
        )
        return PriorityClass.ORCHESTRATION
    return _SOURCE_PRIORITY[src]


# ── the agent-instance registry (dormant rows) ───────────────────────────────────────


def _authority(engine: Any) -> Any:
    """Resolve the graph authority (mirrors ``work_item._authority``)."""
    view = getattr(engine, "_work_item_engine", None)
    return view if view is not None else engine


def new_instance_id() -> str:
    """A full 128-bit agent-instance id (AU-P0-3: no truncated ids)."""
    return f"agent-instance:{uuid.uuid4().hex}"


def get_agent_instance(engine: Any, instance_id: str) -> dict[str, Any] | None:
    """Read one ``:AgentInstance`` row, or ``None`` if it does not exist."""
    rows = _authority(engine).query_cypher(
        f"MATCH (a:{_INSTANCE_LABEL} {{id: $id}}) RETURN a.id AS id, "
        "a.agent_name AS agent_name, a.tenant AS tenant, "
        "a.template_def_id AS template_def_id, "
        "a.statechart_instance_id AS statechart_instance_id, "
        "a.lifecycle_state AS lifecycle_state, a.model_id AS model_id, "
        "a.tool_ids AS tool_ids, a.subscriptions AS subscriptions, "
        "a.state_snapshot AS state_snapshot, a.mailbox_depth AS mailbox_depth, "
        "a.origin_principal AS origin_principal",
        {"id": instance_id},
    )
    if not rows:
        return None
    row = dict(rows[0])
    row["tool_ids"] = list(row.get("tool_ids") or [])
    row["subscriptions"] = list(row.get("subscriptions") or [])
    raw_state = row.get("state_snapshot")
    if isinstance(raw_state, str):
        try:
            row["state"] = json.loads(raw_state or "{}")
        except json.JSONDecodeError:
            row["state"] = {}
    else:
        row["state"] = raw_state if isinstance(raw_state, dict) else {}
    return row


def register_agent_instance(
    engine: Any,
    *,
    agent_name: str,
    tenant: str,
    instance_id: str | None = None,
    model_id: str = "",
    tool_ids: Sequence[str] = (),
    subscriptions: Sequence[str] = (),
    origin_principal: str = "",
    state: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    now: float | None = None,
) -> str:
    """Register a DORMANT agent instance and return its id (ADR-6 §1).

    Creates the durable lifecycle authority — an ``eg-statechart`` ``MachineInstance`` of
    :data:`AGENT_LIFECYCLE_DEF`, starting in ``dormant`` — and a ``:AgentInstance`` graph
    node (the placement/mailbox row + a best-effort mirror of the authoritative lifecycle
    state, exactly as a WorkItem mirrors its ``machine_state``). **Dormant cost is these
    two passive rows — no thread, no connection, no heartbeat.** Idempotent by
    ``instance_id``.
    """
    now = now if now is not None else time.time()
    if not str(agent_name or "").strip():
        raise ValueError("register_agent_instance requires a non-empty agent_name")
    if not str(tenant or "").strip():
        raise ValueError("register_agent_instance requires a non-empty tenant")
    instance_id = instance_id or new_instance_id()

    if get_agent_instance(engine, instance_id) is not None:
        return instance_id  # idempotent

    def_id = agent_lifecycle_def_id(engine)
    result = engine.statechart.instantiate(def_id, context={})
    sc_instance_id = result.get("instance_id") if isinstance(result, dict) else None
    if not sc_instance_id:
        raise AgentActivationError(
            f"statechart Instantiate returned no instance id for agent {agent_name!r}"
        )

    _authority(engine).add_node(
        instance_id,
        _INSTANCE_LABEL,
        properties={
            "name": f"AgentInstance: {agent_name}",
            "agent_name": str(agent_name),
            "tenant": str(tenant),
            "template_def_id": def_id,
            "statechart_instance_id": str(sc_instance_id),
            "lifecycle_state": STATE_DORMANT,
            "model_id": str(model_id or ""),
            "tool_ids": list(tool_ids),
            "subscriptions": list(subscriptions),
            "origin_principal": str(origin_principal or ""),
            "state_snapshot": json.dumps(state or {}),
            "mailbox_depth": 0,
            "created_at": now,
            "updated_at": now,
            "metadata": dict(metadata or {}),
        },
    )
    logger.info(
        "[agents-as-data] registered dormant agent instance agent=%s tenant=%s "
        "template=%s",
        agent_name,
        tenant,
        def_id,
    )
    return instance_id


def list_agent_instances(
    engine: Any,
    *,
    tenant: str | None = None,
    lifecycle_state: str | None = None,
    subscription: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """List registered agent instances (a queryable projection over the dormant rows).

    Optional filters: ``tenant``, ``lifecycle_state`` (``dormant``/``active``/
    ``terminated``), and ``subscription`` (an event type the instance reacts to — feeds
    the CDC/broker fan-out). Purely a read over the graph store.
    """
    clauses: list[str] = []
    params: dict[str, Any] = {"limit": int(limit)}
    if tenant:
        clauses.append("a.tenant = $tenant")
        params["tenant"] = tenant
    if lifecycle_state:
        clauses.append("a.lifecycle_state = $lifecycle_state")
        params["lifecycle_state"] = lifecycle_state
    if subscription:
        clauses.append("$subscription IN a.subscriptions")
        params["subscription"] = subscription
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = _authority(engine).query_cypher(
        f"MATCH (a:{_INSTANCE_LABEL}){where} RETURN a.id AS id, "
        "a.agent_name AS agent_name, a.tenant AS tenant, "
        "a.lifecycle_state AS lifecycle_state, a.mailbox_depth AS mailbox_depth, "
        "a.subscriptions AS subscriptions LIMIT $limit",
        params,
    )
    out: list[dict[str, Any]] = []
    for row in rows or []:
        item = dict(row)
        item["subscriptions"] = list(item.get("subscriptions") or [])
        out.append(item)
    return out


def terminate_agent_instance(engine: Any, instance_id: str) -> bool:
    """Retire an instance for good: drive its MachineInstance to ``terminated``.

    Returns ``True`` when the terminal transition fired (or was already terminal). The
    authoritative statechart transition is the source of truth; the graph mirror follows.
    """
    node = get_agent_instance(engine, instance_id)
    if node is None:
        return False
    sc_id = node.get("statechart_instance_id")
    fired = False
    if sc_id:
        out = engine.statechart.send_event(str(sc_id), EV_TERMINATE, {})
        fired = bool(out.get("fired")) if isinstance(out, dict) else bool(out)
    _mirror_lifecycle_state(engine, instance_id, STATE_TERMINATED)
    logger.info(
        "[agents-as-data] terminated agent instance %s (fired=%s)", instance_id, fired
    )
    return True


def _mirror_lifecycle_state(engine: Any, instance_id: str, state: str) -> None:
    """Best-effort mirror of the authoritative statechart state onto the graph node.

    Not lease/lifecycle authority (the ``MachineInstance`` is) — the queryable projection
    :func:`list_agent_instances` reads, exactly like the WorkItem ``machine_state`` mirror.
    """
    try:
        _authority(engine).add_node(
            instance_id,
            _INSTANCE_LABEL,
            properties={"lifecycle_state": state, "updated_at": time.time()},
        )
    except Exception as exc:  # noqa: BLE001 — mirror is never load-bearing
        logger.debug(
            "agents-as-data: lifecycle mirror for %s failed: %s",
            instance_id,
            type(exc).__name__,
        )


# ── the mailbox ──────────────────────────────────────────────────────────────────────


def _message_id(instance_id: str, seq: str) -> str:
    return f"agent-msg:{instance_id.split(':')[-1]}:{seq}"


def _append_mailbox_message(
    engine: Any,
    instance_id: str,
    *,
    message_ref: str,
    source: str,
    correlation_id: str,
    now: float,
) -> str:
    """Append one message to the instance's durable mailbox; return the message id."""
    seq = uuid.uuid4().hex
    msg_id = _message_id(instance_id, seq)
    authority = _authority(engine)
    authority.add_node(
        msg_id,
        _MESSAGE_LABEL,
        properties={
            "name": f"AgentMessage: {source}",
            "instance_id": instance_id,
            "message_ref": message_ref,
            "source": str(source),
            "correlation_id": correlation_id,
            "consumed": False,
            "created_at": now,
        },
    )
    _link(engine, instance_id, msg_id, _HAS_MAILBOX_MESSAGE)
    return msg_id


def _mailbox_messages(
    engine: Any, instance_id: str, *, only_unconsumed: bool = True, limit: int = 256
) -> list[dict[str, Any]]:
    """Read the instance's mailbox messages (oldest-first by creation time)."""
    where = " AND m.consumed = false" if only_unconsumed else ""
    rows = _authority(engine).query_cypher(
        f"MATCH (m:{_MESSAGE_LABEL}) WHERE m.instance_id = $iid{where} "
        "RETURN m.id AS id, m.message_ref AS message_ref, m.source AS source, "
        "m.correlation_id AS correlation_id, m.created_at AS created_at LIMIT $limit",
        {"iid": instance_id, "limit": int(limit)},
    )
    msgs = [dict(r) for r in (rows or [])]
    msgs.sort(key=lambda m: float(m.get("created_at") or 0.0))
    return msgs


def drain_mailbox(engine: Any, instance_id: str) -> list[dict[str, Any]]:
    """Read + mark-consumed every pending message for ``instance_id``.

    Called by the worker once it has activated the instance: one activation drains the
    WHOLE mailbox (so a burst of messages to one instance is processed by a single
    activation, not one WorkItem per message — the coalescing that keeps activation
    O(bursts) not O(messages)).
    """
    msgs = _mailbox_messages(engine, instance_id, only_unconsumed=True)
    authority = _authority(engine)
    now = time.time()
    for m in msgs:
        with contextlib.suppress(Exception):
            authority.add_node(
                m["id"],
                _MESSAGE_LABEL,
                properties={"consumed": True, "consumed_at": now},
            )
    _set_mailbox_depth(engine, instance_id, 0)
    return msgs


def _set_mailbox_depth(engine: Any, instance_id: str, depth: int) -> None:
    with contextlib.suppress(Exception):
        _authority(engine).add_node(
            instance_id,
            _INSTANCE_LABEL,
            properties={"mailbox_depth": int(depth), "updated_at": time.time()},
        )


def _link(engine: Any, source_id: str, target_id: str, rel_type: str) -> None:
    """Best-effort edge write (never blocks an activation on a graph-viz edge)."""
    try:
        linker = getattr(_authority(engine), "link_nodes", None)
        if callable(linker):
            linker(source_id, target_id, rel_type)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "agents-as-data: edge %s -> %s (%s) failed: %s",
            source_id,
            target_id,
            rel_type,
            type(exc).__name__,
        )


# ── the activation (event → mailbox + WorkItem) ──────────────────────────────────────


def deliver_activation(
    engine: Any,
    instance_id: str,
    *,
    message_ref: str,
    source: ActivationSource | str = ActivationSource.DIRECT,
    tenant: str | None = None,
    origin_principal: str = "",
    correlation_id: str = "",
    lease_ttl_s: float = _wi.DEFAULT_LEASE_TTL_S,
    now: float | None = None,
) -> str:
    """Activate a dormant agent instance: mailbox append + submit an activation WorkItem.

    Returns the WorkItem id. The QoS admission class is derived from ``source`` (ADR-6 §5)
    and stamped onto the WorkItem's ``prio_bucket`` (claim ordering) AND its metadata (so
    the worker re-binds the same :class:`PriorityClass` while running it — the engine calls
    then ride the matching W2.4 QoS lane). The originating ``origin_principal`` is carried
    on the WorkItem so the worker can build the ADR-4 delegation chain from the REAL caller.

    Raises :class:`AgentActivationError` if ``instance_id`` is not a registered instance.
    """
    now = now if now is not None else time.time()
    node = get_agent_instance(engine, instance_id)
    if node is None:
        raise AgentActivationError(
            f"cannot activate unknown agent instance {instance_id!r} — register it first"
        )
    if str(node.get("lifecycle_state")) == STATE_TERMINATED:
        raise AgentActivationError(
            f"agent instance {instance_id!r} is terminated and cannot be activated"
        )
    tenant = tenant or str(node.get("tenant") or "")
    origin_principal = origin_principal or str(node.get("origin_principal") or "")
    src = ActivationSource(str(source)) if not isinstance(source, ActivationSource) else source
    priority_class = activation_priority_class(src)

    msg_id = _append_mailbox_message(
        engine,
        instance_id,
        message_ref=message_ref,
        source=str(src),
        correlation_id=correlation_id,
        now=now,
    )
    # Bump the mailbox depth (best-effort; the worker resets it to 0 on drain).
    current_depth = int(node.get("mailbox_depth") or 0)
    _set_mailbox_depth(engine, instance_id, current_depth + 1)

    work_item_id = f"workitem:{WORK_ITEM_KIND}:{instance_id.split(':')[-1]}:{uuid.uuid4().hex}"
    item_id = _wi.submit_work_item(
        engine,
        kind=WORK_ITEM_KIND,
        queue=WORK_ITEM_KIND,
        payload_ref=instance_id,
        tenant=tenant,
        priority=priority_class.rank,
        resource_class=WORK_ITEM_KIND,
        fairness_group=str(node.get("agent_name") or ""),
        correlation_id=correlation_id or None,
        work_item_id=work_item_id,
        idempotency_key=work_item_id,
        metadata={
            "agent_instance_id": instance_id,
            "agent_name": str(node.get("agent_name") or ""),
            "statechart_instance_id": str(node.get("statechart_instance_id") or ""),
            "activation_source": str(src),
            "priority_class": priority_class.value,
            "origin_principal": origin_principal,
            "mailbox_message_id": msg_id,
        },
        now=now,
    )
    _link(engine, item_id, instance_id, _ACTIVATION_OF)
    logger.info(
        "[agents-as-data] activation queued instance=%s source=%s class=%s work_item=%s",
        instance_id,
        src,
        priority_class.value,
        item_id,
    )
    return item_id


# ── the executor seam (Wire-First: a live default + a pluggable full runner) ─────────


@dataclass(slots=True)
class ActivationContext:
    """Everything the executor needs to run one activation.

    Immutable snapshot handed to the executor INSIDE the active identity chain +
    QoS scope. ``messages`` is the drained mailbox; ``delegation`` is the resolved
    on-behalf-of chain (``None`` when delegation is off).
    """

    engine: Any
    instance_id: str
    agent_name: str
    tenant: str
    work_item_id: str
    run_id: str
    messages: tuple[dict[str, Any], ...]
    priority_class: PriorityClass
    delegation: _delegation.SpawnDelegation | None
    model_id: str = ""
    tool_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class ActivationResult:
    """The executor's outcome for one activation.

    ``outcome`` is a WorkItem outcome (``succeeded``/``failed``/``cancelled``);
    ``tool_calls`` is the provenance the worker stamps as ``:ToolCall`` nodes under the
    delegation chain; ``result_ref``/``error_ref`` are opaque references.
    """

    outcome: str = "succeeded"
    tool_calls: tuple[dict[str, Any], ...] = ()
    result_ref: str | None = None
    error_ref: str | None = None
    retryable: bool = True


#: The pluggable full LLM/tool executor. Production wires a real agent runner (the
#: ``execute_agent`` orchestration path); when unbound the worker uses
#: :func:`_default_executor` (a genuine, wired provenance-ack — NOT a stub).
_EXECUTOR: Callable[[ActivationContext], ActivationResult] | None = None


def set_activation_executor(
    executor: Callable[[ActivationContext], ActivationResult] | None,
) -> None:
    """Bind (or clear) the full LLM/tool executor the worker runs for each activation.

    The executor is invoked INSIDE the active delegation + QoS scope (so every engine/LLM
    call it makes carries the caller chain and the QoS priority claim). Binding this is how
    the production agent runner plugs in; the default drains the mailbox and records
    provenance.
    """
    global _EXECUTOR
    _EXECUTOR = executor


def _default_executor(ctx: ActivationContext) -> ActivationResult:
    """The live default: acknowledge each mailbox message as one provenance ``:ToolCall``.

    This is a REAL wired path (not a stub): it processes every drained message and returns
    a tool-call record per message so the worker writes provenance for the run. A
    deployment that wants a full agent turn binds one via :func:`set_activation_executor`;
    until then the activation loop still does real, observable work (mailbox drain +
    provenance) end to end.
    """
    tool_calls = tuple(
        {
            "tool": "activation.acknowledge",
            "args_ref": str(m.get("message_ref") or ""),
            "source": str(m.get("source") or ""),
        }
        for m in ctx.messages
    )
    return ActivationResult(outcome="succeeded", tool_calls=tool_calls)


# ── provenance (the ADR-4 delegation chain recorded on :RunTrace / :ToolCall) ─────────


def _write_provenance(
    engine: Any,
    ctx: ActivationContext,
    result: ActivationResult,
) -> str:
    """Write the ``:RunTrace`` (carrying the full delegation chain) + a ``:ToolCall`` per
    executor tool call, linked to the run trace and the agent instance.

    The delegation ``chain`` array (``[principal, …, agent:<name>:<run_id>]``) is stamped
    on the ``:RunTrace`` so the on-behalf-of identity is visible in provenance (ADR-4
    decision 6 / ADR-6 e2e proof). Returns the run-trace node id.
    """
    authority = _authority(engine)
    trace_id = f"runtrace:{ctx.run_id}"
    chain = list(ctx.delegation.chain) if ctx.delegation is not None else []
    principal = ctx.delegation.principal if ctx.delegation is not None else ""
    agent_instance_id = (
        ctx.delegation.agent_instance_id if ctx.delegation is not None else ctx.agent_name
    )
    authority.add_node(
        trace_id,
        _RUNTRACE_LABEL,
        properties={
            "name": f"RunTrace: {ctx.agent_name}",
            "run_id": ctx.run_id,
            "work_item_id": ctx.work_item_id,
            "agent_instance_id": ctx.instance_id,
            "agent_name": ctx.agent_name,
            "tenant": ctx.tenant,
            "delegation_chain": chain,
            "principal": principal,
            "agent_instance_identity": agent_instance_id,
            "priority_class": ctx.priority_class.value,
            "outcome": result.outcome,
            "created_at": time.time(),
        },
    )
    _link(engine, trace_id, ctx.instance_id, _TRACE_OF)
    for idx, call in enumerate(result.tool_calls):
        call_id = f"toolcall:{ctx.run_id}:{idx}"
        authority.add_node(
            call_id,
            _TOOLCALL_LABEL,
            properties={
                "name": f"ToolCall: {call.get('tool', '')}",
                "run_id": ctx.run_id,
                "tool": str(call.get("tool") or ""),
                "args_ref": str(call.get("args_ref") or ""),
                # The delegation chain rides every ToolCall too, so an engine-side
                # audit of a single tool call shows its full on-behalf-of provenance.
                "delegation_chain": chain,
                "agent_instance_identity": agent_instance_id,
                "sequence": idx,
                "created_at": time.time(),
            },
        )
        _link(engine, call_id, trace_id, _TOOLCALL_OF)
    return trace_id


# ── the worker loop (claim → activate → identity → run → provenance → commit) ─────────


def _build_activation_delegation(
    *,
    agent_name: str,
    run_id: str,
    tenant: str,
    origin_principal: str,
    tool_ids: Sequence[str],
) -> _delegation.SpawnDelegation | None:
    """Resolve the ADR-4 on-behalf-of delegation for one activation, or ``None``.

    Uses the WorkItem-carried ``origin_principal`` (the real submitter) when present, else
    falls back to the ambient caller (:func:`resolve_principal_identity`). Mints a
    per-activation run-scoped token (endpoint allowlist = the instance's tools). Returns
    ``None`` only when delegation is OFF or no principal can be resolved (legacy identity).
    """
    if not _delegation.delegation_enabled():
        return None
    ident = _delegation.resolve_principal_identity()
    principal = origin_principal or ident.principal
    if not principal:
        logger.debug(
            "[agents-as-data] no resolvable principal for %s; running under legacy identity",
            agent_name,
        )
        return None
    ceiling = ident.ceiling
    run_token, expires_at = _delegation.mint_spawn_run_token(
        run_id,
        principal=principal,
        tenant=tenant or ident.tenant,
        allowed_tools=list(tool_ids) or None,
    )
    return _delegation.build_spawn_delegation(
        agent_name=agent_name,
        run_id=run_id,
        principal=principal,
        ceiling=ceiling,
        run_token=run_token,
        oidc_token=None,
        expires_at=expires_at,
    )


def process_one_activation(
    engine: Any,
    claim: dict[str, Any],
    *,
    token: str,
    heartbeat_interval_s: float = 30.0,
    lease_ttl_s: float = _wi.DEFAULT_LEASE_TTL_S,
    now: float | None = None,
) -> str:
    """Run ONE claimed activation to a committed terminal outcome; return the outcome str.

    Composes the whole loop for a WorkItem already claimed by :func:`_wi.claim_next` /
    :func:`_wi.claim_specific`:

    1. Load the agent instance (its ``:AgentInstance`` node + statechart state).
    2. ``statechart activate`` — ``dormant → active`` (OCC = the concurrency guard).
    3. Build the ADR-4 identity chain and ``use_delegation`` it around the run.
    4. Under ``priority_scope(class)`` (engine calls ride the W2.4 QoS lane): drain the
       mailbox, run the executor, and write ``:RunTrace`` / ``:ToolCall`` provenance
       (carrying the delegation chain).
    5. Heartbeat the lease WHILE active (renewing the lease AND revalidating the delegated
       credential — bounded-time revocation).
    6. ``commit_result`` the WorkItem terminally (the lifecycle authority).
    7. ``statechart deactivate`` — ``active → dormant`` — and save state.

    A heartbeat is emitted around the executor so a long run holds its lease. On any
    executor error the WorkItem is committed ``failed`` (retryable) so the ADR-5 retry /
    dead_letter machinery applies.
    """
    now = now if now is not None else time.time()
    work_item_id = str(claim["work_item_id"])
    item = _wi.get_work_item(engine, work_item_id) or {}
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    instance_id = str(metadata.get("agent_instance_id") or item.get("payload_ref") or "")
    node = get_agent_instance(engine, instance_id) if instance_id else None
    if node is None:
        # The activation references an instance that no longer exists — commit failed,
        # non-retryable (retrying cannot conjure the instance back).
        logger.warning(
            "[agents-as-data] activation %s references missing instance %s; failing",
            work_item_id,
            instance_id,
        )
        return _wi.commit_result(
            engine,
            work_item_id,
            claim,
            outcome="failed",
            error_ref=f"agent_activation:{instance_id}:missing_instance",
            retryable=False,
            now=now,
        )

    agent_name = str(node.get("agent_name") or "agent")
    tenant = str(node.get("tenant") or item.get("tenant") or "")
    sc_instance_id = str(node.get("statechart_instance_id") or "")
    priority_class = _priority_class_from(metadata)
    run_id = work_item_id  # the WorkItem id is the stable, traceable run identity

    # (2) authoritative activation: dormant → active. A no-op (already active) means a
    # concurrent activation of the SAME instance holds it — defer this one so it retries
    # after the holder deactivates, rather than double-running the instance.
    if sc_instance_id:
        activated = engine.statechart.send_event(sc_instance_id, EV_ACTIVATE, {})
        fired = bool(activated.get("fired")) if isinstance(activated, dict) else bool(activated)
        if not fired and _statechart_state(activated) == STATE_ACTIVE:
            logger.info(
                "[agents-as-data] instance %s already active (concurrent activation); "
                "deferring %s",
                instance_id,
                work_item_id,
            )
            deferred = _wi.defer_work_item(
                engine,
                work_item_id,
                claim,
                next_retry_at=now + 5.0,
                reason_ref=f"agent_activation:{instance_id}:instance_busy",
                now=now,
            )
            return "deferred" if deferred else "fenced"
    _mirror_lifecycle_state(engine, instance_id, STATE_ACTIVE)

    # (3) the ADR-4 identity chain.
    delegation = _build_activation_delegation(
        agent_name=agent_name,
        run_id=run_id,
        tenant=tenant,
        origin_principal=str(metadata.get("origin_principal") or node.get("origin_principal") or ""),
        tool_ids=node.get("tool_ids") or (),
    )

    outcome = "failed"
    result = ActivationResult(outcome="failed", retryable=True)
    stop_heartbeat = threading.Event()
    heartbeat_thread: threading.Thread | None = None
    try:
        # (5) heartbeat WHILE active: the lease is the liveness signal; renewal also
        # revalidates the delegated credential's expiry (bounded-time revocation).
        heartbeat_thread = _start_heartbeat(
            engine,
            work_item_id,
            claim,
            stop_heartbeat,
            interval_s=heartbeat_interval_s,
            lease_ttl_s=lease_ttl_s,
        )
        with _delegation.use_delegation(delegation), priority_scope(priority_class):
            messages = drain_mailbox(engine, instance_id)
            ctx = ActivationContext(
                engine=engine,
                instance_id=instance_id,
                agent_name=agent_name,
                tenant=tenant,
                work_item_id=work_item_id,
                run_id=run_id,
                messages=tuple(messages),
                priority_class=priority_class,
                delegation=delegation,
                model_id=str(node.get("model_id") or ""),
                tool_ids=tuple(node.get("tool_ids") or ()),
            )
            executor = _EXECUTOR or _default_executor
            result = executor(ctx)
            _write_provenance(engine, ctx, result)
        outcome = result.outcome
    except Exception as exc:  # noqa: BLE001 — record + commit failed so ADR-5 retry applies
        logger.error(
            "[agents-as-data] activation %s for instance %s errored: %s",
            work_item_id,
            instance_id,
            exc,
        )
        result = ActivationResult(
            outcome="failed",
            error_ref=f"agent_activation:{instance_id}:{type(exc).__name__}",
            retryable=True,
        )
        outcome = "failed"
    finally:
        stop_heartbeat.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1.0)

    # (6) commit the WorkItem terminally — the lifecycle authority.
    commit_outcome = _wi.commit_result(
        engine,
        work_item_id,
        claim,
        outcome=outcome,
        result_ref=result.result_ref or (f"outcome:agent_activation:{instance_id}" if outcome == "succeeded" else None),
        error_ref=result.error_ref or (None if outcome == "succeeded" else f"agent_activation:{instance_id}:{outcome}"),
        retryable=result.retryable,
        now=now,
    )

    # (7) release the instance back to dormant (authoritative), save state, mirror.
    if sc_instance_id:
        engine.statechart.send_event(sc_instance_id, EV_DEACTIVATE, {})
    _mirror_lifecycle_state(engine, instance_id, STATE_DORMANT)
    logger.info(
        "[agents-as-data] activation %s instance=%s outcome=%s commit=%s",
        work_item_id,
        instance_id,
        outcome,
        commit_outcome,
    )
    return commit_outcome


def _priority_class_from(metadata: dict[str, Any]) -> PriorityClass:
    raw = str(metadata.get("priority_class") or "")
    try:
        return PriorityClass(raw) if raw else PriorityClass.ORCHESTRATION
    except ValueError:
        return PriorityClass.ORCHESTRATION


def _statechart_state(result: Any) -> str:
    """Extract the single active state from a statechart send_event/get_state result."""
    if not isinstance(result, dict):
        return ""
    inst = result.get("instance") if "instance" in result else result
    if isinstance(inst, dict):
        config = inst.get("configuration")
        if isinstance(config, dict):
            active = config.get("active")
            if isinstance(active, list) and active:
                return str(active[0])
            if isinstance(active, str):
                return active
    return ""


def _start_heartbeat(
    engine: Any,
    work_item_id: str,
    claim: dict[str, Any],
    stop: threading.Event,
    *,
    interval_s: float,
    lease_ttl_s: float,
) -> threading.Thread:
    """Spawn a daemon thread renewing the WorkItem lease every ``interval_s`` while active.

    The heartbeat exists ONLY for the duration of the run (ADR-6 §3: heartbeats only while
    active). It renews the lease AND — via ``work_item.heartbeat`` — revalidates the
    delegated credential, so revoking the caller lapses the lease at the next beat.
    """

    def _beat() -> None:
        # First beat after one interval; the claim's initial lease covers the gap.
        while not stop.wait(interval_s):
            try:
                renewed = _wi.heartbeat(
                    engine, work_item_id, claim, lease_ttl_s=lease_ttl_s
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[agents-as-data] heartbeat error for %s: %s",
                    work_item_id,
                    type(exc).__name__,
                )
                return
            if not renewed:
                logger.warning(
                    "[agents-as-data] lease lost/revoked for %s; stopping heartbeat "
                    "(the activation will be reaped and re-queued)",
                    work_item_id,
                )
                return

    thread = threading.Thread(
        target=_beat, name=f"activation-heartbeat-{work_item_id[-8:]}", daemon=True
    )
    thread.start()
    return thread


def run_activation_worker_loop(
    engine: Any,
    stop_event: threading.Event,
    *,
    worker_id: str | None = None,
    tenants: Sequence[str] | None = None,
    idle_sleep_s: float = 0.5,
    lease_ttl_s: float = _wi.DEFAULT_LEASE_TTL_S,
    heartbeat_interval_s: float = 30.0,
    max_activations: int | None = None,
) -> int:
    """Drain activation WorkItems until ``stop_event`` (one process = one worker).

    The stateless worker: ``claim_next`` an ``agent_activation`` WorkItem (the CAS lease —
    also the liveness signal), run it via :func:`process_one_activation`, repeat. No state
    is held between activations; N processes running this ARE the pool. ``max_activations``
    bounds the loop for tests/one-shot drains. Returns the number of activations processed.

    The engine's native ``ClaimWorkItem`` is tenant-scoped (agent instances live in
    per-tenant graphs, ADR-6 §4). ``tenants`` names the tenants this worker serves and is
    round-robined so one worker drains many tenants fairly; ``None``/empty falls back to
    the worker's ambient bound-session tenant (the single-tenant / session-bound default).
    """
    token = worker_id or _wi._default_token()
    tenant_ring: list[str | None] = list(tenants) if tenants else [None]
    processed = 0
    cursor = 0
    while not stop_event.is_set():
        if max_activations is not None and processed >= max_activations:
            break
        claim = None
        # One full pass over the tenant ring looking for a claimable activation.
        for _ in range(len(tenant_ring)):
            tenant = tenant_ring[cursor % len(tenant_ring)]
            cursor += 1
            try:
                claim = _wi.claim_next(
                    engine,
                    resource_class=WORK_ITEM_KIND,
                    queue=WORK_ITEM_KIND,
                    tenant=tenant,
                    token=token,
                    lease_ttl_s=lease_ttl_s,
                )
            except _wi.NativeWorkItemRequired:
                raise
            except Exception as exc:  # noqa: BLE001 — transient transport error: back off
                logger.warning(
                    "[agents-as-data] claim error (%s); backing off",
                    type(exc).__name__,
                )
                stop_event.wait(2.0)
                claim = None
                break
            if claim is not None:
                break
        if claim is None:
            if max_activations is not None:
                break  # one-shot drain: nothing ready across all tenants, we're done
            stop_event.wait(idle_sleep_s)
            continue
        try:
            _wi.mark_running(engine, claim["work_item_id"], claim)
            process_one_activation(
                engine,
                claim,
                token=token,
                heartbeat_interval_s=heartbeat_interval_s,
                lease_ttl_s=lease_ttl_s,
            )
        except Exception as exc:  # noqa: BLE001 — never let one bad activation wedge the loop
            logger.error(
                "[agents-as-data] worker error on %s (%s)",
                claim.get("work_item_id"),
                type(exc).__name__,
            )
        processed += 1
    return processed


def start_activation_worker_pool(
    engine: Any,
    *,
    worker_count: int = 1,
    tenants: Sequence[str] | None = None,
    stop_event: threading.Event | None = None,
    lease_ttl_s: float = _wi.DEFAULT_LEASE_TTL_S,
) -> tuple[list[threading.Thread], threading.Event]:
    """Start ``worker_count`` stateless activation-worker threads; return (threads, stop).

    Thread-per-worker is the in-process pool; the k8s Deployment shape (W5.1) runs one
    PROCESS per worker across the fleet — the loop is identical either way. ``tenants`` is
    the set of per-tenant graphs the pool serves (ADR-6 §4).
    """
    stop = stop_event or threading.Event()
    threads: list[threading.Thread] = []
    base = _wi._default_token()
    tenant_list = list(tenants) if tenants else None
    for i in range(max(1, worker_count)):

        def _runner(idx: int = i) -> None:
            run_activation_worker_loop(
                engine,
                stop,
                worker_id=f"{base}:{idx}",
                tenants=tenant_list,
                lease_ttl_s=lease_ttl_s,
            )

        t = threading.Thread(
            target=_runner, name=f"AgentActivationWorker-{i}", daemon=True
        )
        t.start()
        threads.append(t)
    logger.info(
        "[agents-as-data] activation worker pool started: %d workers", len(threads)
    )
    return threads, stop


def main(argv: list[str] | None = None) -> int:
    """Entry point: a standalone, host-role-free agents-as-data activation worker.

    Mirrors ``agent_dispatch_worker.main`` — an engine CLIENT (never a KG host): it claims
    ``agent_activation`` WorkItems and runs dormant statechart-agents on demand.
    """
    import argparse
    import os
    import signal

    parser = argparse.ArgumentParser(
        prog="agent-activation-worker",
        description=(
            "Stateless agents-as-data activation worker "
            "(CONCEPT:AU-ORCH.dispatch.agents-as-data-activation, ADR-6/W2.3): claims "
            "agent_activation WorkItems via the CAS lease and runs dormant "
            "statechart-agents under the per-agent identity chain — no KG host role."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Activation-worker threads on this host (default 1; activations are LLM-bound).",
    )
    parser.add_argument(
        "--tenant",
        action="append",
        default=None,
        metavar="TENANT",
        help=(
            "A per-tenant graph this worker serves (repeatable). The native ClaimWorkItem "
            "is tenant-scoped; omit to use the worker's bound-session tenant."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    os.environ.setdefault("KG_DAEMON_ROLE", "client")

    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_or_create()
    if not callable(getattr(engine, "claim_work_item", None)):
        parser.exit(
            2,
            "engine does not expose native ClaimWorkItem — cannot claim activations "
            "(check GRAPH_SERVICE_ENDPOINTS / the packaged-local transport + HMAC secret).\n",
        )
    try:
        engine.query_cypher("MATCH (w:WorkItem) RETURN count(w) AS c")
    except Exception as exc:  # noqa: BLE001
        parser.exit(
            2,
            "cannot reach the epistemic-graph engine as a client: "
            f"error_type={type(exc).__name__}\n",
        )

    threads, stop = start_activation_worker_pool(
        engine, worker_count=args.workers, tenants=args.tenant
    )

    def _handle(_signum: int, _frame: Any) -> None:
        logger.info("[agents-as-data] shutdown signal; draining")
        stop.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)
    while any(t.is_alive() for t in threads) and not stop.is_set():
        stop.wait(1.0)
    stop.set()
    for t in threads:
        t.join(timeout=5.0)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
