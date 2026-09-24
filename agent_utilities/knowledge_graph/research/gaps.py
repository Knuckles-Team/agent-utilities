#!/usr/bin/python
from __future__ import annotations

"""The ONE canonical ``:Gap`` — every discovery track folds into it, through EG.

CONCEPT:AU-AHE.harness.canonical-gap-lifecycle — the unified Gap→SDD→Implement→Promote→Close
spine (Wave 6), now owned by the engine (EH-348, graph-driven work market). AU no longer
writes Gap nodes, edges or WorkItems itself: every function here is a thin call to EG's
typed Gap surface (``engine.client.gaps``, generated from the EG contract):

* :func:`submit_gap` → ``GapUpsert``: EG upserts ONE canonical Gap for
  ``(tenant, gap:<source>:<signature>)`` **and** its native WorkItem in one transaction, or
  commits neither. Only evidence the Gap has not seen changes it; new evidence reopens a
  resolved/deferred Gap as a new generation with a new WorkItem; re-sending old evidence is
  ``unchanged`` (a cooldown cannot be bypassed by repetition).
* :func:`get_gap` / :func:`open_gaps` → ``GapGet`` / ``GapList`` (row-key order: a
  listing, never a ranking — ranking legal work is the ``Decide`` layer's).
* :func:`link_gap_to_spec` / :func:`mark_gap_resolved` → ``GapTransition`` (revision CAS);
  :func:`settle_gap` → ``GapSettle``: EG reads the Gap's WorkItem row and records its
  terminal outcome as evidence (the caller never states an outcome).

A missing EG surface raises :class:`GapAuthorityUnavailable`; engine refusals propagate.
Nothing is swallowed and there is no fallback writer.
"""

import hashlib
import logging
import re
from collections.abc import Iterator, Mapping
from typing import Any

logger = logging.getLogger(__name__)

#: The single graph label EG gives a canonical gap row.
GAP_LABEL = "Gap"

#: Lifecycle (EG ``GapStatus``). ``open`` (discovered, schedulable) → ``specified`` (a spec is
#: in flight) → ``resolved``; ``deferred`` parks a gap whose attempt ended without closing it.
STATUS_OPEN = "open"
STATUS_SPECIFIED = "specified"
STATUS_RESOLVED = "resolved"
STATUS_DEFERRED = "deferred"
_TERMINAL = frozenset({STATUS_RESOLVED})

#: The discovery tracks fold into ONE gap type, distinguished only by ``source``.
SOURCE_FAILURE = "failure"  # adaptation/failure_analyzer.file_gap_topic
SOURCE_RESEARCH = "research"  # assimilation/plan_synthesis.synthesize_plan_for_feature
SOURCE_SKILL = "skill"  # adaptation/skill_evolver.SkillGap
SOURCE_AUDIT = "audit"  # harness/audit_gap_detector (Macroscope-class findings)
SOURCE_RUNTIME = "runtime"  # research/runtime_reliability (RUNTIME reliability signals)

#: The WorkItem kind (and queue) EG admits for a Gap.
GAP_WORK_KIND = "gap_remediation"
#: Attempt ceiling of a Gap's WorkItem.
GAP_MAX_ATTEMPTS = 20
#: EG's own page and evidence bounds.
_PAGE_LIMIT = 100
_MAX_PAGES = 64
_MAX_UPSERT_EVIDENCE = 16


class GapAuthorityUnavailable(RuntimeError):
    """The connected engine does not serve the typed Gap surface (or no verified tenant)."""


def gap_client(engine: Any) -> Any:
    """The EG ``gaps`` namespace, or :class:`GapAuthorityUnavailable`. Never a fallback."""
    namespace = getattr(getattr(engine, "client", None), "gaps", None)
    if namespace is None:
        raise GapAuthorityUnavailable("connected engine has no typed Gap surface")
    return namespace


def gap_tenant() -> str:
    """The VERIFIED session tenant every Gap call binds (EG refuses any other)."""
    from agent_utilities.knowledge_graph.core.session import resolve_session

    session = resolve_session(required_scope="kg:write")
    tenant = str(session.tenant or session.graph or "").strip()
    if not tenant:
        raise GapAuthorityUnavailable("the verified session is not bound to a tenant")
    return tenant


def _slug(text: str, *, limit: int = 80) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (s[:limit] or "gap").rstrip("-")


def canonical_gap_id(source: str, signature: str) -> str:
    """The single canonical gap id scheme: ``gap:<source>:<signature>``.

    Replaces the 7+ disjoint id-prefix schemes (``failure_gap:``, ``plan:``, ...) with
    ONE thread every track shares — so "which gap produced which commit" is a traversal.
    """
    return f"gap:{_slug(source, limit=40)}:{_slug(signature, limit=120)}"


def severity_ppm(severity: float) -> int:
    """A 0..1 severity as EG's fixed-point parts-per-million (clamped)."""
    try:
        value = float(severity)
    except (TypeError, ValueError):
        value = 0.5
    return max(0, min(1_000_000, round(value * 1_000_000)))


def evidence_entry(kind: str, reference: str, *parts: str) -> dict[str, str]:
    """One EG evidence entry: ``sha256`` over the kind, reference and any extra parts."""
    digest = hashlib.sha256()
    for part in (kind, reference, *parts):
        encoded = part.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return {
        "digest": f"sha256:{digest.hexdigest()}",
        "kind": kind,
        "reference": reference,
    }


def _evidence(
    source: str, gap_id: str, statement: str, refs: list[str]
) -> list[dict[str, str]]:
    """The upsert's evidence: one entry per cited ref, else the statement itself."""
    cited = [ref for ref in refs if ref][:_MAX_UPSERT_EVIDENCE]
    if not cited:
        return [evidence_entry(source, gap_id, statement)]
    return [evidence_entry(source, ref, gap_id) for ref in cited]


def _upsert_key(gap_id: str, evidence: list[dict[str, str]]) -> str:
    """The retry identity of one upsert: the Gap and EVERY evidence digest it carries."""
    payload = "\0".join([gap_id, *(entry["digest"] for entry in evidence)])
    return f"gap-upsert:{hashlib.sha256(payload.encode()).hexdigest()}"


def _flat(view: Mapping[str, Any]) -> dict[str, Any]:
    """An EG Gap view as the flat dict every downstream stage reads."""
    severity = int(view.get("severity_ppm") or 0) / 1_000_000
    statement = str(view.get("statement") or "")
    return {
        "id": view.get("gap_id"),
        "name": statement[:120],
        "source": view.get("source"),
        "signature": view.get("signature"),
        "statement": statement,
        "gap_statement": statement,
        "domain": view.get("domain"),
        "severity": severity,
        "priority_bucket": view.get("priority_bucket"),
        "status": view.get("status"),
        "concept_ids": list(view.get("concept_ids") or []),
        "evidence_refs": [e.get("reference") for e in view.get("evidence") or []],
        "evidence": list(view.get("evidence") or []),
        "spec_refs": list(view.get("spec_refs") or []),
        "work_item_id": view.get("work_item_id"),
        "generation": view.get("generation"),
        "offer": view.get("offer"),
        "offer_version": view.get("offer_version"),
        "revision": view.get("revision"),
    }


def submit_gap(
    engine: Any,
    *,
    source: str,
    signature: str,
    statement: str,
    domain: str = "",
    severity: float = 0.5,
    concept_ids: list[str] | None = None,
    evidence_refs: list[str] | None = None,
) -> dict[str, Any] | None:
    """Upsert ONE canonical ``:Gap`` and its WorkItem through EG (``GapUpsert``).

    Returns the gap dict downstream stages consume, or ``None`` for an empty statement.
    Idempotent on ``gap:<source>:<signature>`` and on the evidence digests.
    """
    statement = (statement or "").strip()
    if not statement:
        return None
    gap_id = canonical_gap_id(source, signature)
    evidence = _evidence(source, gap_id, statement, list(evidence_refs or []))
    answer = gap_client(engine).upsert(
        tenant=gap_tenant(),
        gap_id=gap_id,
        source=source,
        signature=signature,
        statement=statement[:4096],
        domain=domain,
        severity_ppm=severity_ppm(severity),
        concept_ids=list(concept_ids or [])[:32],
        evidence=evidence,
        work_kind=GAP_WORK_KIND,
        max_attempts=GAP_MAX_ATTEMPTS,
        idempotency_key=_upsert_key(gap_id, evidence),
    )
    return _flat(answer["gap"])


def get_gap(engine: Any, gap_id: str) -> dict[str, Any] | None:
    """Load one canonical ``:Gap`` as a flat dict, or ``None`` when the tenant has none."""
    if engine is None or not gap_id:
        return None
    view = gap_client(engine).get(tenant=gap_tenant(), gap_id=gap_id)
    return None if view is None else _flat(view)


def iter_gaps(engine: Any, *, status: str | None = None) -> Iterator[dict[str, Any]]:
    """Every Gap view of the tenant (optionally of one status), page by bounded page."""
    client, tenant = gap_client(engine), gap_tenant()
    cursor: str | None = None
    for _ in range(_MAX_PAGES):
        page = client.list(
            tenant=tenant, status=status, cursor=cursor, limit=_PAGE_LIMIT
        )
        yield from page["gaps"]
        cursor = page.get("next_cursor")
        if cursor is None:
            return


def open_gaps(engine: Any, *, limit: int = 200) -> list[dict[str, Any]]:
    """Every gap not yet ``resolved``, in EG's listing order (never re-ranked here)."""
    if engine is None:
        return []
    out: list[dict[str, Any]] = []
    for view in iter_gaps(engine):
        if str(view.get("status")) in _TERMINAL:
            continue
        out.append(_flat(view))
        if len(out) >= limit:
            break
    return out


def set_gap_status(
    engine: Any, gap_id: str, status: str, *, reference: str = ""
) -> bool:
    """Move a Gap to ``status`` through ``GapTransition`` (CAS on its current revision).

    ``True`` when EG applied the edge; ``False`` when the Gap is missing or the edge is
    not legal from its current status.
    """
    current = get_gap(engine, gap_id)
    if current is None:
        return False
    answer = gap_client(engine).transition(
        tenant=gap_tenant(),
        gap_id=gap_id,
        expected_revision=int(current["revision"]),
        to=status,
        reference=reference or status,
        idempotency_key=f"gap-transition:{gap_id}:{current['revision']}:{status}",
    )
    return bool(answer["outcome"] == "applied")


def settle_gap(engine: Any, gap_id: str) -> str:
    """Ask EG to record the Gap's current WorkItem outcome (``GapSettle``).

    The engine reads the WorkItem row itself; returns EG's outcome word
    (``resolved``/``deferred``/``recorded``/``pending``/``unchanged``/``not_found``).
    """
    answer = gap_client(engine).settle(
        tenant=gap_tenant(), gap_id=gap_id, idempotency_key=f"gap-settle:{gap_id}"
    )
    return str(answer["outcome"])


def mark_gap_resolved(engine: Any, gap_id: str, *, reference: str = "") -> bool:
    """Close a gap (D5) on ``reference`` (the publication) through ``GapTransition``.

    The Gap's WorkItem is left to the work market: :func:`settle_gap` records its outcome
    when it reaches one, and the reconciliation sweep cancels it if it never runs.
    """
    if not gap_id:
        return False
    ok = set_gap_status(
        engine, gap_id, STATUS_RESOLVED, reference=reference or "published"
    )
    if ok:
        logger.info("[Wave6] gap %s marked resolved", gap_id)
    return ok


def link_gap_to_spec(engine: Any, gap_id: str, spec_id: str) -> bool:
    """Record the spec a Gap is SPECIFIED_BY (chain hop 1, D6) as ``specified``."""
    if engine is None or not gap_id or not spec_id:
        return False
    return set_gap_status(engine, gap_id, STATUS_SPECIFIED, reference=spec_id)


def resolve_gaps_for_loop(engine: Any, loop: Mapping[str, Any]) -> list[str]:
    """Close the origin gap a published develop-Loop carries (D5).

    The develop-Loop bound by ``spec_proposals._bind_develop_loop`` records its origin
    ``gap_id``; there is no RESOLVES edge walk. Returns the resolved gap ids.
    """
    gap_id = str(loop.get("gap_id") or "")
    if engine is None or not gap_id:
        return []
    return (
        [gap_id]
        if mark_gap_resolved(engine, gap_id, reference=str(loop.get("id") or ""))
        else []
    )


__all__ = [
    "GAP_LABEL",
    "GAP_WORK_KIND",
    "STATUS_OPEN",
    "STATUS_SPECIFIED",
    "STATUS_RESOLVED",
    "STATUS_DEFERRED",
    "SOURCE_FAILURE",
    "SOURCE_RESEARCH",
    "SOURCE_SKILL",
    "SOURCE_AUDIT",
    "SOURCE_RUNTIME",
    "GapAuthorityUnavailable",
    "canonical_gap_id",
    "evidence_entry",
    "gap_client",
    "gap_tenant",
    "get_gap",
    "iter_gaps",
    "link_gap_to_spec",
    "mark_gap_resolved",
    "open_gaps",
    "resolve_gaps_for_loop",
    "set_gap_status",
    "settle_gap",
    "severity_ppm",
    "submit_gap",
]
