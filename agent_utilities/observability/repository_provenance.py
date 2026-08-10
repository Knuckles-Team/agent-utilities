"""CONCEPT:AU-KG.audit.repository-job-provenance-bridge — RMDD-19 repository job provenance.

Generic, repository-domain-agnostic writer and query layer for repository-job
lifecycle provenance (submit, lease/claim, admission, start/heartbeat/
checkpoint, cancel/retry/dead-letter, command result, artifact publication,
validation/certificate, candidate/generation/bisection, concept, landing/
push, GC/reconcile).

This module is deliberately built as a thin additional layer over the
EXISTING canonical ``RunTrace``/``ToolCall`` provenance ontology
(``agent_utilities.observability.trace_ontology`` — RunTrace -[:USED_TOOL]->
ToolCall, KG-2.296). It introduces **no second graph authority and no second
audit store**: one repository-job attempt projects onto one ``RunTrace`` node
(keyed deterministically from ``work_item_id``/``attempt``), and every
lifecycle event of that attempt projects onto a ``ToolCall`` node linked to
it — exactly the shape ``orchestration/agent_runner.py`` already writes for
agent runs, reused rather than forked (RMDD-19 lane brief: "Emit through
existing graph authority; do not introduce another store").

**The one chokepoint.** Every event kind funnels through
:func:`write_repository_event` — the single function that touches the graph.
Callers (repository-manager's domain emitters) never call ``engine.add_node``
directly; this mirrors the H-11 lesson ("wire the control at the shared
function every caller converges on, not at the entrypoint you happen to be
looking at") applied *within this lane's own additive surface*, since this
lane may not edit the actual repository-job call sites yet (RMDD-20 wires
those).

**Idempotency.** Node ids are deterministic functions of
``(work_item_id, attempt, kind, occurrence)`` — never a random uuid4 or a
wall-clock value — and ``engine.add_node``/``engine.link_nodes`` are native
upserts (MERGE + SET). Replaying the same logical event with the same
``occurrence`` therefore upserts the SAME node rather than creating a
duplicate. Callers MUST derive ``occurrence`` deterministically from the
event's own immutable identity (e.g. an index into an already-immutable
tuple of artifacts) — never a runtime/process-local counter, which would
silently break idempotency across a restart.

**Fail loud, never fabricate.** A caller that cannot reach the graph gets an
explicit :class:`RepositoryProvenanceUnavailable` (H-12) — this module never
swallows a write/read failure and reports a false "recorded" or an empty
"nothing happened" in its place. A genuinely empty *read* result (a WorkItem
with no recorded events yet) is a normal, non-degraded outcome and returns
``[]``/``found: False`` — the fail-loud rule is about the boundary being
unreachable, not about the data being absent.

**Fencing.** A terminal effect (``command_result`` succeeded/failed,
``landing_push`` succeeded) whose fence does not match the latest fence
already recorded for that WorkItem is refused with :class:`StaleFenceError`
(``refusal_code = "stale_fence_duplicate_effect"``, matching the C-10 /
``repository_manager.development.enums.RefusalCode`` string value exactly —
this module intentionally does not import that enum, to avoid a reverse
dependency from agent-utilities onto repository-manager; it reuses the same
wire value as a plain string).

**Privacy.** Every free-form payload passes through
``observability.trace_ontology.tool_call_properties`` (which already routes
through ``security.persistence_privacy.PersistencePrivacyGuard``) — raw
content is never persisted, only a content digest + character count + a
redaction report. Correlation references (resource/host/artifact/validation/
generation/landing ids) are minted as opaque HMAC references via
``persistence_reference`` exactly like every other durable identity field in
this codebase.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any, Literal, get_args

from agent_utilities.observability.trace_ontology import (
    TOOL_CALL_NODE_LABEL,
    TRACE_NODE_LABEL,
    TRACE_USED_TOOL_EDGE,
    next_event_sequence,
    tool_call_properties,
    trace_properties,
)
from agent_utilities.observability.trace_ontology import (
    trace_id as canonical_trace_id,
)
from agent_utilities.security.persistence_privacy import (
    PersistencePrivacyGuard,
    persistence_reference,
)

REPOSITORY_PROVENANCE_SCHEMA_VERSION = 1

RepositoryEventKind = Literal[
    "submitted",
    "dependency_ready",
    "lease_claimed",
    "admission_placement",
    "started",
    "heartbeat",
    "checkpoint",
    "cancelled",
    "retried",
    "dead_lettered",
    "command_result",
    "artifact_published",
    "validation_certificate",
    "candidate_event",
    "generation_event",
    "bisection_event",
    "concept_event",
    "landing_push",
    "gc_reconcile",
]

REPOSITORY_EVENT_KINDS: frozenset[str] = frozenset(get_args(RepositoryEventKind))

# Kinds that can carry a *terminal effect* — a claimed success/failure whose
# replay by a superseded (stale-fenced) writer must never be recorded as if
# it were the current outcome.
TERMINAL_EVENT_KINDS: frozenset[str] = frozenset({"command_result", "landing_push"})
TERMINAL_STATUSES: frozenset[str] = frozenset({"succeeded", "failed", "landed"})
# "landed" covers repository_manager.development.enums.LandingOutcome.LANDED --
# landing_push's own success vocabulary differs from command_result's
# (succeeded/failed), so both must be present for the fence guard to protect
# a landing success exactly like a command-result success.

# Matches repository_manager.development.enums.RefusalCode.STALE_FENCE_DUPLICATE_EFFECT
# (C-10) byte-for-byte, without importing repository-manager from agent-utilities.
STALE_FENCE_REFUSAL_CODE = "stale_fence_duplicate_effect"


class RepositoryProvenanceUnavailable(RuntimeError):
    """Raised when the graph authority cannot be reached or a write/read failed.

    Fail-closed boundary (H-12): a component that cannot do its job must
    never return a value its caller reads as "all clear" or "nothing to
    report". This is that explicit failure, distinct from a genuinely empty
    result.
    """


class StaleFenceError(RuntimeError):
    """A terminal event's fence does not match the latest recorded fence.

    Raised instead of recording a possibly-stale effect as success (H-12,
    lane acceptance gate "stale fence event cannot claim effect success").
    """

    refusal_code = STALE_FENCE_REFUSAL_CODE

    def __init__(
        self, message: str, *, expected_fence: str = "", provided_fence: str = ""
    ) -> None:
        super().__init__(message)
        self.expected_fence = expected_fence
        self.provided_fence = provided_fence


def _require_non_blank(value: str, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required")
    return text


def repository_run_id(work_item_id: str, attempt: int) -> str:
    """Deterministic ``RunTrace`` id for one WorkItem attempt.

    Deterministic in ``(work_item_id, attempt)`` so repeated calls for the
    same attempt always resolve to the same node — idempotent by
    construction, never a random id.
    """

    _require_non_blank(work_item_id, "work_item_id")
    if int(attempt) < 1:
        raise ValueError("attempt must be >= 1")
    return canonical_trace_id(f"{work_item_id}:attempt:{int(attempt)}")


def repository_event_id(
    work_item_id: str, attempt: int, kind: str, occurrence: int
) -> str:
    """Deterministic ``ToolCall``-shaped event id.

    Replaying the same ``(work_item_id, attempt, kind, occurrence)`` always
    resolves to the same node id, so a retry/restart upserts rather than
    duplicates.
    """

    if kind not in REPOSITORY_EVENT_KINDS:
        raise ValueError(f"unknown repository event kind: {kind!r}")
    if int(occurrence) < 0:
        raise ValueError("occurrence must be >= 0")
    run_ref = repository_run_id(work_item_id, attempt).removeprefix("trace:")
    return f"toolcall:{run_ref}:{kind}:{int(occurrence)}"


def _stamp_ambient_identity(props: dict[str, Any]) -> None:
    """Best-effort actor/tenant stamp, mirroring orchestration/agent_runner's own idiom.

    Identity is ambient context: absence is a normal, unscoped record, never
    a write failure. Reuses the same primitives
    (``security.brain_context.current_actor`` + ``persistence_reference``)
    every other provenance writer in this codebase already uses instead of
    inventing a second identity mechanism.
    """

    try:
        from agent_utilities.security.brain_context import current_actor

        actor = current_actor()
        if actor.actor_id:
            props.setdefault(
                "actor_ref",
                persistence_reference(
                    "actor", actor.actor_id, namespace="repository-provenance"
                ),
            )
        if actor.tenant_id:
            props.setdefault(
                "tenant_ref",
                persistence_reference(
                    "tenant", actor.tenant_id, namespace="repository-provenance"
                ),
            )
    except Exception:  # noqa: BLE001 — ambient identity is best-effort enrichment
        pass


def repository_event_properties(
    *,
    work_item_id: str,
    attempt: int,
    kind: str,
    occurrence: int,
    status: str,
    timestamp: str,
    payload: Mapping[str, Any] | None = None,
    error: str = "",
    correlations: Mapping[str, str] | None = None,
    fence: str = "",
    event_sequence: int | None = None,
) -> dict[str, Any]:
    """Build the sanitized, versioned properties for one repository event node.

    Reuses ``tool_call_properties`` for the canonical digest/character-count/
    privacy-report shape (raw ``payload``/``error`` content is never
    persisted) and layers repository-specific, opaque correlation references
    on top.
    """

    if kind not in REPOSITORY_EVENT_KINDS:
        raise ValueError(f"unknown repository event kind: {kind!r}")
    run_id = repository_run_id(work_item_id, attempt)
    seq = int(event_sequence or next_event_sequence())
    payload_text = (
        json.dumps(dict(payload), sort_keys=True, default=str) if payload else ""
    )
    props = tool_call_properties(
        run_id=run_id,
        tool_name=f"repository.{kind}",
        args=payload_text,
        result=status,
        error=error,
        status=status,
        sequence=occurrence,
        timestamp=timestamp,
        event_sequence=seq,
    )
    props["repository_provenance_schema_version"] = REPOSITORY_PROVENANCE_SCHEMA_VERSION
    props["event_kind"] = kind
    props["work_item_ref"] = persistence_reference(
        "work_item", work_item_id, namespace="repository-provenance"
    )
    props["attempt"] = int(attempt)
    if fence:
        props["fence_ref"] = persistence_reference(
            "fence", fence, namespace="repository-provenance"
        )
    if correlations:
        guard = PersistencePrivacyGuard()
        for key, value in correlations.items():
            if not value:
                continue
            field = str(key)
            safe_value, _report = guard.sanitize_text(str(value))
            props[f"{field}_ref"] = persistence_reference(
                field, safe_value, namespace="repository-provenance"
            )
    _stamp_ambient_identity(props)
    return props


def _latest_fence_ref(engine: Any, *, work_item_id: str) -> str | None:
    """Best-effort read of the most recently recorded fence for a WorkItem."""

    work_item_ref = persistence_reference(
        "work_item", work_item_id, namespace="repository-provenance"
    )
    try:
        rows = engine.query_cypher(
            f"MATCH (t:{TOOL_CALL_NODE_LABEL} {{work_item_ref: $work_item_ref}}) "
            "WHERE t.fence_ref IS NOT NULL "
            "RETURN t.fence_ref AS fence_ref, t.event_sequence AS event_sequence "
            "ORDER BY t.event_sequence DESC LIMIT 1",
            {"work_item_ref": work_item_ref},
        )
    except Exception as exc:
        raise RepositoryProvenanceUnavailable(
            f"repository provenance fence read failed (error_type={type(exc).__name__})"
        ) from exc
    for row in rows or []:
        if isinstance(row, Mapping) and row.get("fence_ref"):
            return str(row["fence_ref"])
    return None


def _guard_terminal_fence(engine: Any, *, work_item_id: str, fence: str) -> None:
    if not fence:
        # No fence supplied: the caller opted out of fence protection for
        # this write (e.g. a kind that legitimately has no fence concept).
        return
    provided_ref = persistence_reference(
        "fence", fence, namespace="repository-provenance"
    )
    latest = _latest_fence_ref(engine, work_item_id=work_item_id)
    if latest is not None and latest != provided_ref:
        raise StaleFenceError(
            "terminal repository event fence does not match the latest fence "
            "recorded for this WorkItem; refusing to record a possibly-stale "
            "effect as success",
            expected_fence=latest,
            provided_fence=provided_ref,
        )


def write_repository_event(
    engine: Any,
    *,
    work_item_id: str,
    attempt: int,
    kind: str,
    occurrence: int,
    status: str,
    timestamp: str | None = None,
    payload: Mapping[str, Any] | None = None,
    error: str = "",
    correlations: Mapping[str, str] | None = None,
    fence: str = "",
    event_sequence: int | None = None,
) -> dict[str, Any]:
    """Record one versioned, sanitized repository-job lifecycle event.

    The single chokepoint every repository-domain emitter must call — see
    the module docstring. Fails loud (:class:`RepositoryProvenanceUnavailable`)
    when the graph authority is unreachable, and refuses
    (:class:`StaleFenceError`) a terminal effect whose fence has been
    superseded, instead of ever fabricating success.
    """

    if engine is None:
        raise RepositoryProvenanceUnavailable(
            "repository provenance requires graph authority (engine is None)"
        )
    if kind not in REPOSITORY_EVENT_KINDS:
        raise ValueError(f"unknown repository event kind: {kind!r}")
    _require_non_blank(work_item_id, "work_item_id")
    if int(attempt) < 1:
        raise ValueError("attempt must be >= 1")
    ts = timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if kind in TERMINAL_EVENT_KINDS and str(status or "").lower() in TERMINAL_STATUSES:
        _guard_terminal_fence(engine, work_item_id=work_item_id, fence=fence)

    run_id = repository_run_id(work_item_id, attempt)
    event_id = repository_event_id(work_item_id, attempt, kind, occurrence)
    seq = int(event_sequence or next_event_sequence())

    trace_props = trace_properties(
        run_id=run_id,
        agent_name="repository-manager",
        task=f"repository job {kind}",
        status=status,
        timestamp=ts,
        event_sequence=seq,
        execution_mode="repository_job",
    )
    trace_props["repository_provenance_schema_version"] = (
        REPOSITORY_PROVENANCE_SCHEMA_VERSION
    )
    trace_props["work_item_ref"] = persistence_reference(
        "work_item", work_item_id, namespace="repository-provenance"
    )
    trace_props["attempt"] = int(attempt)
    _stamp_ambient_identity(trace_props)

    event_props = repository_event_properties(
        work_item_id=work_item_id,
        attempt=attempt,
        kind=kind,
        occurrence=occurrence,
        status=status,
        timestamp=ts,
        payload=payload,
        error=error,
        correlations=correlations,
        fence=fence,
        event_sequence=seq,
    )

    try:
        engine.add_node(run_id, TRACE_NODE_LABEL, properties=trace_props)
        engine.add_node(event_id, TOOL_CALL_NODE_LABEL, properties=event_props)
        engine.link_nodes(run_id, event_id, TRACE_USED_TOOL_EDGE)
    except (RepositoryProvenanceUnavailable, StaleFenceError):
        raise
    except Exception as exc:
        raise RepositoryProvenanceUnavailable(
            f"repository provenance write failed (error_type={type(exc).__name__})"
        ) from exc

    return {
        "run_id": run_id,
        "event_id": event_id,
        "kind": kind,
        "status": status,
        "event_sequence": seq,
        "occurrence": int(occurrence),
    }


def query_repository_provenance(
    engine: Any,
    *,
    work_item_id: str,
    tenant_ref: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Return the ordered provenance stream for one WorkItem.

    Requires a tenant scope — either ``tenant_ref`` explicitly, or the
    ambient ``current_actor()`` tenant — and matches only events stamped
    with that same tenant reference, so a cross-tenant query returns an
    empty result rather than another tenant's provenance (deny-by-default,
    not merely "narrow by default"). A genuinely empty match is a normal
    result (``[]``); an unreachable graph raises.
    """

    if engine is None:
        raise RepositoryProvenanceUnavailable(
            "repository provenance requires graph authority (engine is None)"
        )
    _require_non_blank(work_item_id, "work_item_id")
    resolved_tenant = tenant_ref
    if not resolved_tenant:
        try:
            from agent_utilities.security.brain_context import current_actor

            actor = current_actor()
            if actor.tenant_id:
                resolved_tenant = persistence_reference(
                    "tenant", actor.tenant_id, namespace="repository-provenance"
                )
        except Exception:  # noqa: BLE001 — ambient identity is best-effort
            resolved_tenant = None
    if not resolved_tenant:
        raise ValueError(
            "repository provenance query requires a tenant scope "
            "(tenant_ref or an ambient authenticated actor)"
        )

    work_item_ref = persistence_reference(
        "work_item", work_item_id, namespace="repository-provenance"
    )
    try:
        rows = engine.query_cypher(
            f"MATCH (t:{TOOL_CALL_NODE_LABEL} "
            "{work_item_ref: $work_item_ref, tenant_ref: $tenant_ref}) "
            "RETURN t ORDER BY t.event_sequence ASC LIMIT $limit",
            {
                "work_item_ref": work_item_ref,
                "tenant_ref": resolved_tenant,
                "limit": max(1, int(limit)),
            },
        )
    except Exception as exc:
        raise RepositoryProvenanceUnavailable(
            f"repository provenance query failed (error_type={type(exc).__name__})"
        ) from exc

    events: list[dict[str, Any]] = []
    for row in rows or []:
        node = row.get("t") if isinstance(row, Mapping) else None
        if isinstance(node, Mapping):
            events.append(dict(node))
    return events


def explain_repository_job(
    engine: Any, *, work_item_id: str, tenant_ref: str | None = None
) -> dict[str, Any]:
    """Aggregate provenance into an operator-facing explanation for one WorkItem.

    Backs the read side of ``rm_jobs status``/``logs``/``reconcile``
    (RMDD-20 owns wiring the MCP/CLI surface; this lane provides the
    projection).
    """

    events = query_repository_provenance(
        engine, work_item_id=work_item_id, tenant_ref=tenant_ref
    )
    if not events:
        return {
            "work_item_id": work_item_id,
            "found": False,
            "event_count": 0,
            "events": [],
            "latest_status": None,
            "latest_kind": None,
            "latest_event_sequence": None,
            "terminal": False,
        }
    ordered = sorted(events, key=lambda e: int(e.get("event_sequence") or 0))
    latest = ordered[-1]
    terminal = (
        str(latest.get("event_kind")) in TERMINAL_EVENT_KINDS
        and str(latest.get("status", "")).lower() in TERMINAL_STATUSES
    )
    return {
        "work_item_id": work_item_id,
        "found": True,
        "event_count": len(ordered),
        "events": ordered,
        "latest_status": latest.get("status"),
        "latest_kind": latest.get("event_kind"),
        "latest_event_sequence": latest.get("event_sequence"),
        "terminal": terminal,
    }


def reconciliation_report(
    engine: Any, *, work_item_id: str, tenant_ref: str | None = None
) -> dict[str, Any]:
    """Link observed provenance facts to a proposed repair, without submitting one.

    This lane owns provenance, not the WorkItem authority that would create
    and dispatch a repair — the report NAMES the repair a reconciler should
    consider; it never mutates state itself.
    """

    explanation = explain_repository_job(
        engine, work_item_id=work_item_id, tenant_ref=tenant_ref
    )
    if not explanation["found"]:
        return {
            **explanation,
            "observed_facts": ["no provenance recorded for this work item"],
            "proposed_repair": None,
        }
    kinds_seen = {event.get("event_kind") for event in explanation["events"]}
    facts = [
        f"{explanation['event_count']} events recorded",
        f"latest_kind={explanation['latest_kind']}",
        f"latest_status={explanation['latest_status']}",
    ]
    proposed_repair: dict[str, str] | None = None
    if not explanation["terminal"]:
        if "lease_claimed" in kinds_seen and "started" not in kinds_seen:
            proposed_repair = {
                "kind": "reclaim_and_relaunch",
                "reason": "leased but never started",
            }
        elif (
            "started" in kinds_seen
            and "heartbeat" not in kinds_seen
            and "checkpoint" not in kinds_seen
            and "command_result" not in kinds_seen
        ):
            proposed_repair = {
                "kind": "reclaim_stale_worker",
                "reason": "started with no heartbeat, checkpoint, or result",
            }
        else:
            proposed_repair = {
                "kind": "manual_review",
                "reason": "no terminal event and no recognizable stall pattern",
            }
    return {
        **explanation,
        "observed_facts": facts,
        "proposed_repair": proposed_repair,
    }


__all__ = [
    "REPOSITORY_PROVENANCE_SCHEMA_VERSION",
    "REPOSITORY_EVENT_KINDS",
    "TERMINAL_EVENT_KINDS",
    "TERMINAL_STATUSES",
    "STALE_FENCE_REFUSAL_CODE",
    "RepositoryEventKind",
    "RepositoryProvenanceUnavailable",
    "StaleFenceError",
    "repository_run_id",
    "repository_event_id",
    "repository_event_properties",
    "write_repository_event",
    "query_repository_provenance",
    "explain_repository_job",
    "reconciliation_report",
]
