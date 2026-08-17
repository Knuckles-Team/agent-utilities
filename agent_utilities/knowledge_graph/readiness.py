#!/usr/bin/python
from __future__ import annotations

"""GraphOS readiness snapshot — ``graphos.readiness.v1`` (GOC-02, CONCEPT:AU-KG.query.readiness-canary-snapshot).

**The defect class this module exists to close.** An independent audit of this
program found that health/readiness can report GREEN before the graph can
actually answer a query: a liveness probe (a raw TCP connect,
``/health``/``/health/ready``'s existing ``is_overall_healthy``) proves a
*listener exists*, not that a real query resolves. ``BUG-004`` recorded exactly
this shape live — ``code_context`` failing with a compiled sparse-index
coverage report of ``0.0%`` while the transport itself was reachable. The
governing rule, repeated eleven times across this program's incident record:
**never trust a signal about state; check state.** A readiness endpoint that is
green while queries fail is worse than one that is red.

**What this module is.** ``collect_readiness_snapshot()`` is the ONE snapshot
authority that joins: engine reachability, the (GOC-15-owned) identity/policy
carrier, a canonical-route catalog digest, source-sync connector coverage,
dense/sparse retrieval-index signal, and — the check that actually proves the
defect class above is closed — a **synthetic query canary** that runs the SAME
``build_code_context`` implementation the live ``graph_code(action=
"code_context")`` MCP/REST route dispatches to
(:mod:`agent_utilities.knowledge_graph.retrieval.code_context`,
:mod:`agent_utilities.mcp.tools.analysis_tools`) — the same live route, never a
substitute for it: a broken/degraded engine flows through the real ``EngineReadDegraded``
path and a genuinely empty index flows through the real "zero citations" path,
and this module reports each honestly (``unavailable`` / ``degraded``) rather
than folding either into a false "ready".

**Non-goals (owned by sibling lanes — see
``plans/graph-os-completion-program/lanes/GOC-02-graphos-readiness-catalog-health.md``).**
This module does NOT implement carrier cryptography (GOC-15) or a new package/
catalog descriptor model (GOC-24); ``identity_policy`` reports only whether a
verified :class:`~agent_utilities.knowledge_graph.core.session.GraphSession`
was actually supplied to the collector — it never infers authority from a bare
``tenant``/``subject`` string.

**State semantics (design contract).** Five states: ``ready``, ``degraded``,
``unavailable``, ``stale``, ``not_configured``. One failed **required** check
(:data:`REQUIRED_CHECKS` — ``engine``, ``catalog``, ``synthetic_query``) makes
the whole snapshot ``unavailable``; a degraded/stale/unavailable **non-required**
check degrades the snapshot to ``degraded`` (never silently ``ready``) — see
:func:`_rollup`. A snapshot is never cached and re-served: every call re-probes.

Every consumer (WebUI, ``deployment/doctor.py``, GOC-31) reads THIS snapshot
rather than re-deriving readiness — see the module's own ``AGENTS.md`` sprawl
rule ("Ontology / daemons — extend before you add"), applied here to readiness
authority: one collector, many thin renderers.
"""

import datetime as _dt
import hashlib
import importlib
import json
import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeoutError
from typing import Any, Literal, TypedDict, cast

SCHEMA_VERSION = "graphos.readiness.v1"

ReadinessState = Literal["ready", "degraded", "unavailable", "stale", "not_configured"]


class ReadinessCheckDict(TypedDict, total=False):
    """One typed readiness check result (mirrors ``observability/runtime_health.py``'s
    ``HealthCheck``/``HealthReport`` convention — a typed public seam instead of a
    bare ``dict[str, Any]``, so a producer/consumer key drift is a type error, not a
    silent ``0.00``). Every check always carries ``state``; the rest are populated
    per check kind (see each ``_check_*`` function above)."""

    state: ReadinessState
    reason: str
    detail: dict[str, Any]
    latency_ms: float
    version: str
    carrier: str
    tenant: str
    policy_epoch: int
    digest: str
    count: int
    missing: list[str]
    expected: int
    covered: int
    stale: list[str]
    coverage_pct: float
    route: str
    evidence_count: int
    capability_id: str


class ReadinessPrincipalDict(TypedDict):
    subject: str | None
    tenant: str | None
    policy_epoch: int | None


class ReadinessRefreshDict(TypedDict):
    mode: str
    cursor: str | None
    next_due_at: str | None


class ReadinessSnapshotDict(TypedDict):
    """The public ``graphos.readiness.v1`` envelope :func:`collect_readiness_snapshot`
    returns — see the module docstring's schema."""

    schema_version: str
    snapshot_id: str
    observed_at: str
    principal: ReadinessPrincipalDict
    overall: ReadinessState
    checks: dict[str, ReadinessCheckDict]
    required_failures: list[str]
    degraded_reasons: list[str]
    refresh: ReadinessRefreshDict


#: Checks whose failure (``state == "unavailable"``) fails the WHOLE snapshot.
#: ``catalog`` and ``synthetic_query`` are the two checks that prove the graph
#: can actually SERVE a query — not merely that a socket is open. This is the
#: production route policy from the lane design: a required canonical route/
#: module missing fails startup readiness with ``required_module_missing``.
REQUIRED_CHECKS: tuple[str, ...] = ("engine", "catalog", "synthetic_query")

#: Worst-state-wins ranking used by :func:`_rollup` and any caller that needs
#: to compare two check states (``ready``/``not_configured`` tie at the
#: bottom — neither should ever read as worse than the other).
_STATE_RANK: dict[str, int] = {
    "ready": 0,
    "not_configured": 0,
    "stale": 1,
    "degraded": 2,
    "unavailable": 3,
}

#: Bounded worker pool for the synthetic-query canary — sized above the check
#: count so a slow probe never queues behind another (mirrors
#: ``observability/runtime_health.py``'s own bounded-probe convention).
_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="au-readiness-probe")

#: The production-profile required route inventory (design §"Required/optional
#: route policy"). Each value is ``module:attr`` — resolved by
#: :func:`_resolve_route`, never executed, so the catalog check stays a cheap
#: import-time proof rather than a live call. A missing entry here is a build
#: break for a genuinely new required capability, not something readiness
#: guesses at.
DEFAULT_REQUIRED_ROUTES: dict[str, str] = {
    "graph_code_context": (
        "agent_utilities.knowledge_graph.retrieval.code_context:build_code_context"
    ),
    "graph_search": (
        "agent_utilities.knowledge_graph.retrieval.hybrid_retriever:HybridRetriever"
    ),
    "connector_coverage": (
        "agent_utilities.knowledge_graph.ingestion.connector_coverage:"
        "assess_connector_coverage"
    ),
}


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


def _check(state: ReadinessState, **fields: Any) -> ReadinessCheckDict:
    out: dict[str, Any] = {"state": state}
    out.update({k: v for k, v in fields.items() if v is not None})
    # Built dynamically from **fields (every caller passes only
    # ReadinessCheckDict-declared keys) — a literal-dict TypedDict construction
    # mypy could check statically is not possible here, so this cast asserts
    # the contract every _check(...) call site upholds by construction.
    return cast(ReadinessCheckDict, out)


# --------------------------------------------------------------------------- #
# engine + dense-index — both reuse the ONE shared runtime-health authority,
# never a second implementation that can drift (AGENTS.md "Universal
# capability" / runtime_health.py's own module docstring).
# --------------------------------------------------------------------------- #
def _collect_health_report() -> dict[str, Any] | None:
    try:
        from ..observability.runtime_health import collect_health

        return dict(collect_health())
    except Exception:  # noqa: BLE001 - a broken collector reports as unavailable, never "ready"
        return None


def _check_engine(report: dict[str, Any] | None) -> ReadinessCheckDict:
    if report is None:
        return _check("unavailable", reason="runtime_health.collect_health() raised")
    by_name = {c.get("name"): c for c in report.get("checks", [])}
    engine = by_name.get("engine")
    if engine is None:
        return _check("unavailable", reason="runtime_health reported no engine check")
    detail = engine.get("detail") or {}
    if engine.get("status") == "ok":
        return _check(
            "ready",
            latency_ms=engine.get("latency_ms"),
            version=detail.get("resolved_mode"),
            detail=detail,
        )
    return _check(
        "unavailable",
        reason=engine.get("reason") or "engine unhealthy",
        detail=detail,
        latency_ms=engine.get("latency_ms"),
    )


def _check_dense_index(report: dict[str, Any] | None) -> ReadinessCheckDict:
    """Semantic/dense retrieval readiness — read-only reuse of the SAME
    embedder-failover breaker snapshot ``runtime_health``'s own
    ``embedding_endpoint`` check already reads; never a second probe.
    """
    if report is None:
        return _check("unavailable", reason="runtime_health.collect_health() raised")
    by_name = {c.get("name"): c for c in report.get("checks", [])}
    check = by_name.get("embedding_endpoint")
    if check is None or check.get("status") == "not_configured":
        return _check(
            "not_configured",
            reason=(check or {}).get("reason") or "no embedding model configured",
        )
    detail = check.get("detail") or {}
    if check.get("status") == "degraded":
        return _check(
            "degraded",
            reason=check.get("reason"),
            coverage_pct=None,
            detail=detail,
        )
    return _check("ready", coverage_pct=100.0, detail=detail)


# --------------------------------------------------------------------------- #
# identity/policy — reports only whether a VERIFIED session/principal was
# supplied; never infers authority from a bare tenant/subject string
# (AGENTS.md "Session currency"). Carrier cryptography itself is GOC-15's.
# --------------------------------------------------------------------------- #
def _check_identity_policy(
    *, session: Any, subject: str, tenant: str, policy_epoch: int
) -> ReadinessCheckDict:
    if session is not None:
        actor = getattr(session, "actor", None)
        if actor is not None and bool(getattr(actor, "authenticated", False)):
            return _check(
                "ready",
                carrier="verified",
                tenant=getattr(session, "tenant", None) or tenant or None,
                policy_epoch=getattr(session, "policy_version", None)
                or policy_epoch
                or None,
            )
        return _check(
            "degraded",
            carrier="unverified",
            reason="session was supplied but its actor is not authenticated",
        )
    if subject or tenant:
        return _check(
            "degraded",
            carrier="unverified",
            reason="subject/tenant supplied without a verified GraphSession",
        )
    return _check(
        "not_configured",
        reason="no verified session/principal was supplied to the readiness probe",
    )


# --------------------------------------------------------------------------- #
# catalog — a cheap, deterministic import-resolution proof of the production
# required-route inventory. Never executes a route; :func:`_check_synthetic_query`
# is what proves a route actually answers.
# --------------------------------------------------------------------------- #
def _resolve_route(dotted: str) -> Any:
    module_name, _, attr = dotted.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _check_catalog(required_routes: dict[str, str]) -> ReadinessCheckDict:
    resolved: list[str] = []
    missing: list[str] = []
    for name in sorted(required_routes):
        try:
            _resolve_route(required_routes[name])
            resolved.append(name)
        except Exception:  # noqa: BLE001 - any resolution failure is a missing route
            missing.append(name)
    digest = hashlib.sha256(",".join(resolved).encode("utf-8")).hexdigest()
    if missing:
        return _check(
            "unavailable",
            reason="required_module_missing",
            digest=f"sha256:{digest}",
            count=len(resolved),
            missing=missing,
        )
    return _check("ready", digest=f"sha256:{digest}", count=len(resolved), missing=[])


# --------------------------------------------------------------------------- #
# source_sync — thin readiness projection over the EXISTING connector-coverage
# helper (``ingestion/connector_coverage.py``); never a second freshness
# authority. ``connector_freshness`` is caller-supplied (the live manifest read
# is a bounded I/O the caller owns, per the design's "bounded reads" budget) —
# absent it, an expected-but-unprobed connector set is reported ``stale``,
# never silently ``ready``.
# --------------------------------------------------------------------------- #
def _check_source_sync(
    connector_freshness: dict[str, str] | None,
) -> ReadinessCheckDict:
    from .ingestion.connector_coverage import (
        assess_connector_coverage,
        enumerate_expected_connectors,
    )

    expected = enumerate_expected_connectors()
    if not expected:
        return _check("not_configured", reason="no connectors are registered")
    if connector_freshness is None:
        return _check(
            "stale",
            expected=len(expected),
            covered=0,
            missing=list(expected),
            reason="connector_freshness_not_supplied",
        )
    report = assess_connector_coverage(expected, connector_freshness)
    missing = report.get("missing") or []
    stale = [s["connector"] for s in report.get("stale") or []]
    covered = int(report.get("covered", 0))
    if not covered:
        return _check(
            "unavailable",
            expected=report["total"],
            covered=0,
            missing=missing,
            reason="no_connector_coverage",
        )
    if missing or stale:
        return _check(
            "degraded",
            expected=report["total"],
            covered=covered,
            missing=missing,
            stale=stale,
            reason="partial_connector_coverage",
        )
    return _check(
        "ready", expected=report["total"], covered=covered, missing=[], stale=[]
    )


# --------------------------------------------------------------------------- #
# synthetic_query — THE KNOWN-BAD-PROOF CHECK. Runs the real
# ``build_code_context`` (the same implementation ``graph_code(action=
# "code_context")`` dispatches to) with a strict deadline, and reports the
# result honestly: engine-degraded and zero-evidence are DIFFERENT failure
# classes and neither is ever folded into "ready".
# --------------------------------------------------------------------------- #
def _run_with_deadline(fn: Any, deadline_s: float) -> tuple[Any, str | None]:
    future: Future = _EXECUTOR.submit(fn)
    try:
        return future.result(timeout=deadline_s), None
    except _FutureTimeoutError:
        return None, "synthetic_query_timeout"
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed into "ready"
        return None, f"synthetic_query_error:{type(exc).__name__}"


def _check_synthetic_query(
    engine: Any, *, query: str, node_id: str, deadline_s: float
) -> ReadinessCheckDict:
    if engine is None:
        return _check(
            "unavailable",
            route="graph_code_context",
            evidence_count=0,
            reason="no_engine_supplied",
        )

    from .retrieval.code_context import build_code_context

    started = time.monotonic()
    result, err = _run_with_deadline(
        lambda: build_code_context(
            engine, query=query, intent="how", node_id=node_id, top_k=3
        ),
        deadline_s,
    )
    latency_ms = round((time.monotonic() - started) * 1000.0, 1)
    if err is not None:
        return _check(
            "unavailable",
            route="graph_code_context",
            evidence_count=0,
            reason=err,
            latency_ms=latency_ms,
        )

    result = result or {}
    status = result.get("status")
    error = result.get("error") or {}
    citations = result.get("citations") or []

    if status == "degraded":
        # BUG-004: the engine itself was unreachable — this is NOT evidence
        # the index is empty, and it must never read as "ready".
        return _check(
            "unavailable",
            route="graph_code_context",
            evidence_count=0,
            reason=str(error.get("code") or "engine_degraded"),
            latency_ms=latency_ms,
        )
    if not citations:
        # A real, non-degraded answer that grounded on nothing. Per the
        # design contract: a query with zero valid evidence is DEGRADED or
        # UNAVAILABLE with reason `evidence_coverage_zero`, never "ok".
        return _check(
            "degraded",
            route="graph_code_context",
            evidence_count=0,
            reason="evidence_coverage_zero",
            latency_ms=latency_ms,
        )
    return _check(
        "ready",
        route="graph_code_context",
        evidence_count=len(citations),
        capability_id=result.get("capability_id") or None,
        latency_ms=latency_ms,
    )


def _check_sparse_index(
    synthetic_query_check: ReadinessCheckDict,
) -> ReadinessCheckDict:
    """Sparse/lexical index signal — a thin projection of the synthetic-query
    canary's own evidence count, not a second index-scanning subsystem
    (no dedicated sparse-index coverage API exists yet in this codebase; see
    the module docstring). ``engine_degraded``/timeout carries no evidence
    about the index at all (BUG-004) and is reported ``not_configured`` here
    rather than a false ``0.0%`` empty-index verdict.
    """
    state = synthetic_query_check.get("state")
    reason = synthetic_query_check.get("reason")
    if state == "unavailable" and reason not in {"evidence_coverage_zero"}:
        return _check(
            "not_configured",
            reason="synthetic query could not run; sparse-index coverage is unknown",
        )
    evidence_count = int(synthetic_query_check.get("evidence_count") or 0)
    if evidence_count <= 0:
        return _check("unavailable", coverage_pct=0.0, reason="compiled_index_empty")
    return _check("ready", coverage_pct=100.0)


# --------------------------------------------------------------------------- #
# rollup + the public collector
# --------------------------------------------------------------------------- #
def _rollup(
    checks: Mapping[str, Mapping[str, Any]], required_failures: list[str]
) -> ReadinessState:
    if required_failures:
        return "unavailable"
    states = {c["state"] for c in checks.values()}
    # A non-required check going "unavailable" still degrades the snapshot —
    # it must never silently read as "ready" (fail-closed on a degraded read,
    # AGENTS.md).
    if "unavailable" in states or "degraded" in states:
        return "degraded"
    if "stale" in states:
        return "stale"
    return "ready"


def collect_readiness_snapshot(
    engine: Any = None,
    *,
    session: Any = None,
    subject: str = "",
    tenant: str = "",
    policy_epoch: int = 0,
    synthetic_query: str = "collect_readiness_snapshot",
    synthetic_node_id: str = "",
    required_routes: dict[str, str] | None = None,
    connector_freshness: dict[str, str] | None = None,
    deadline_s: float = 10.0,
) -> ReadinessSnapshotDict:
    """Collect one truthful ``graphos.readiness.v1`` snapshot.

    Never cached, never re-served: every call re-runs every check, including
    the live synthetic-query canary. ``engine`` is the caller's already-
    resolved KG engine handle (duck-typed — anything with ``query_cypher`` or
    a ``.backend.execute``, matching :mod:`knowledge_graph.retrieval.code_context`'s
    own contract); a caller with no engine handle gets an honest
    ``synthetic_query: unavailable`` rather than a skipped check. ``session``
    should be a verified :class:`~..core.session.GraphSession` when the caller
    has one — this function never accepts ``subject``/``tenant`` as authority,
    only as informational principal fields (see :func:`_check_identity_policy`).

    Returns a plain, JSON-serializable ``dict`` (never a live object) —
    payloads redact source endpoint details, tokens, query text, and document
    bodies by construction (no check below ever stores them).
    """
    observed_at = _now_iso()
    health_report = _collect_health_report()

    checks: dict[str, ReadinessCheckDict] = {}
    checks["engine"] = _check_engine(health_report)
    checks["identity_policy"] = _check_identity_policy(
        session=session, subject=subject, tenant=tenant, policy_epoch=policy_epoch
    )
    checks["catalog"] = _check_catalog(dict(required_routes or DEFAULT_REQUIRED_ROUTES))
    checks["source_sync"] = _check_source_sync(connector_freshness)
    checks["synthetic_query"] = _check_synthetic_query(
        engine, query=synthetic_query, node_id=synthetic_node_id, deadline_s=deadline_s
    )
    checks["dense_index"] = _check_dense_index(health_report)
    checks["sparse_index"] = _check_sparse_index(checks["synthetic_query"])

    required_failures = sorted(
        name for name in REQUIRED_CHECKS if checks[name]["state"] == "unavailable"
    )
    degraded_reasons = sorted(
        f"{name}:{check.get('reason')}"
        for name, check in checks.items()
        if check["state"] in ("degraded", "stale", "unavailable")
        and check.get("reason")
    )
    overall = _rollup(checks, required_failures)

    digest_payload = {
        "checks": checks,
        "overall": overall,
        "required_failures": required_failures,
    }
    snapshot_id = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(digest_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "observed_at": observed_at,
        "principal": {
            "subject": subject or None,
            "tenant": tenant or None,
            "policy_epoch": policy_epoch or None,
        },
        "overall": overall,
        "checks": checks,
        "required_failures": required_failures,
        "degraded_reasons": degraded_reasons,
        "refresh": {
            "mode": "delta" if connector_freshness is not None else "not_probed",
            "cursor": None,
            "next_due_at": None,
        },
    }


def is_snapshot_ready(snapshot: ReadinessSnapshotDict) -> bool:
    """True iff a :func:`collect_readiness_snapshot` result's rollup is ``ready``.

    A ``degraded``/``stale``/``not_configured``/``unavailable`` overall MUST
    NOT grant capability or authorization to a caller (design invariant) —
    this is the one function callers should gate routing/capability decisions
    on, mirroring ``runtime_health.is_overall_healthy``'s role for liveness.
    """
    return snapshot.get("overall") == "ready"
