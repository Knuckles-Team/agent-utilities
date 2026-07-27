#!/usr/bin/python
from __future__ import annotations

"""The ONE canonical ``:Gap`` — a single gap type every discovery track folds into.

CONCEPT:AU-AHE.harness.canonical-gap-lifecycle — the unified Gap→SDD→Implement→Promote→Close
spine (Wave 6). Before this module the platform had **three unrelated notions of "gap"**
(a transient ``failure_gap`` ``:Concept`` dict, a dead-end ``sdd_plan`` node, an in-memory
``SkillGap``) plus a fourth-to-be (code-audit findings), joined by 7+ disjoint id schemes.
The :class:`~agent_utilities.models.knowledge_graph.KnowledgeGapNode` model existed but was
never instantiated. This module repurposes it as the **single** gap representation:

* :func:`submit_gap` persists one canonical ``:Gap`` node (id ``gap:<source>:<signature>``)
  **and gives it a WorkItem/lease** (``ensure_loop_work_item``) so a discovered gap is a
  first-class, leaseable, schedulable work item — not a transient dict. The four discovery
  tracks (production-failure, research/OSS, skill-coverage, code-audit) all call it, so
  every gap flows the SAME lifecycle: ``open → specified → resolved``.
* :func:`link_gap_to_spec` writes the ``(:Gap)-[:SPECIFIED_BY]->(:SpecProposal)`` edge — the
  first hop of the unified provenance chain (D6).
* :func:`mark_gap_resolved` / :func:`resolve_gaps_for_loop` close the loop (D5): when the
  gap's derived develop-Loop publishes, the origin gap's status flips to ``resolved`` — a
  visible END the chain never had before.

All writes are best-effort; reads are backend-agnostic (status recovered from the node's
top-level prop OR its folded ``metadata`` JSON), mirroring ``spec_proposals``' discipline so
it works on strict and schemaless backends alike.
"""

import json
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

#: The single graph label for a canonical gap (acceptance: the ``:Gap`` in the
#: ``:Gap→…→resolved`` traversal). The node carries the ``KnowledgeGapNode`` shape.
GAP_LABEL = "Gap"

#: Lifecycle. ``open`` (discovered, unaddressed) → ``specified`` (a spec was authored) →
#: ``resolved`` (the derived develop-Loop published). ``deferred`` parks a gap.
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

#: Provenance edge names of the unified chain (D6).
REL_SPECIFIED_BY = "SPECIFIED_BY"  # (:Gap)-[:SPECIFIED_BY]->(:SpecProposal)
REL_RESOLVES = "RESOLVES"  # (develop-Loop)-[:RESOLVES]->(:Gap)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _slug(text: str, *, limit: int = 80) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (s[:limit] or "gap").rstrip("-")


def canonical_gap_id(source: str, signature: str) -> str:
    """The single canonical gap id scheme: ``gap:<source>:<signature>``.

    Replaces the 7+ disjoint id-prefix schemes (``failure_gap:``, ``plan:``, ...) with
    ONE thread every track shares — so "which gap produced which commit" is a traversal.
    """
    return f"gap:{_slug(source, limit=40)}:{_slug(signature, limit=120)}"


def severity_to_bucket(severity: float) -> int:
    """Map a 0..1 severity to the shared 0(critical)..3(background) priority bucket.

    High/Critical findings are expedited (bucket 0/1) so ``active_loops`` advances them
    first; nice-to-have gaps sit in the background bucket.
    """
    try:
        s = float(severity)
    except (TypeError, ValueError):
        s = 0.5
    if s >= 0.85:
        return 0
    if s >= 0.6:
        return 1
    if s >= 0.3:
        return 2
    return 3


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
    lease: bool = True,
) -> dict[str, Any] | None:
    """Persist ONE canonical ``:Gap`` (+ a WorkItem/lease) for any discovery track.

    Returns the gap dict downstream stages consume (the same shape ``file_gap_topic``
    fed the remediation cycle), or ``None`` when the node could not be persisted.
    Idempotent on the ``gap:<source>:<signature>`` id.
    """
    statement = (statement or "").strip()
    if not statement:
        return None
    gap_id = canonical_gap_id(source, signature)
    bucket = severity_to_bucket(severity)
    cids = list(concept_ids or [])
    payload = {
        "id": gap_id,
        "source": source,
        "signature": signature,
        "gap_statement": statement,
        "domain": domain,
        "severity": float(severity),
        "priority_bucket": bucket,
        "status": STATUS_OPEN,
        "concept_ids": cids,
        "evidence_refs": list(evidence_refs or []),
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    try:
        engine.add_node(
            gap_id,
            GAP_LABEL,
            properties={
                "name": statement[:120],
                "description": statement[:500],
                "node_type": "knowledge_gap",
                "source": source,
                # Top-level status/severity: a real column on schemaless backends,
                # folded into metadata on strict ones (recovered on read).
                "status": STATUS_OPEN,
                "severity": float(severity),
                "priority_bucket": bucket,
                "timestamp": payload["created_at"],
                "metadata": json.dumps(payload, default=str),
            },
        )
    except Exception as e:  # noqa: BLE001 — best-effort persist
        logger.debug("submit_gap persist failed: %s", e)
        return None
    # D1: give the gap a WorkItem/lease so it is a first-class, schedulable work item
    # (not a transient dict) — the same lease authority a Loop gets.
    if lease:
        try:
            from agent_utilities.orchestration.work_item import ensure_loop_work_item

            ensure_loop_work_item(engine, gap_id, priority=bucket, max_attempts=20)
        except Exception as e:  # noqa: BLE001 — lease is best-effort
            logger.debug("submit_gap lease failed: %s", e)
    # Provenance: the gap DERIVED_FROM each cited concept (transparency, best-effort).
    for cid in cids:
        try:
            engine.add_edge(gap_id, cid, "DERIVED_FROM")
        except Exception as e:  # noqa: BLE001
            logger.debug("gap DERIVED_FROM edge %s->%s failed: %s", gap_id, cid, e)
    return {
        "id": gap_id,
        "name": statement[:120],
        "source": source,
        "signature": signature,
        "statement": statement,
        "severity": float(severity),
        "priority_bucket": bucket,
        "status": STATUS_OPEN,
        "concept_ids": cids,
    }


def _gap_dict(node: dict[str, Any], node_id: str = "") -> dict[str, Any]:
    """Normalize a ``:Gap`` node into a flat dict (backend-agnostic)."""
    out: dict[str, Any] = {}
    meta = node.get("metadata")
    if isinstance(meta, str):
        try:
            parsed = json.loads(meta)
            if isinstance(parsed, dict):
                out.update(parsed)
        except (TypeError, ValueError) as exc:
            logger.debug("gap metadata is not valid JSON, ignoring: %s", exc)
    elif isinstance(meta, dict):
        out.update(meta)
    for k in ("status", "severity", "source", "name", "priority_bucket"):
        if node.get(k) is not None:
            out[k] = node[k]
    out["id"] = node.get("id") or node_id or out.get("id")
    out.setdefault("status", STATUS_OPEN)
    return out


def get_gap(engine: Any, gap_id: str) -> dict[str, Any] | None:
    """Load one canonical ``:Gap`` as a flat dict, or ``None``."""
    if engine is None or not gap_id:
        return None
    try:
        rows = engine.query_cypher(
            "MATCH (n:Gap) WHERE n.id = $id RETURN n LIMIT 1", {"id": gap_id}
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("get_gap query failed: %s", e)
        return None
    for r in rows or []:
        props = r.get("n") if isinstance(r, dict) else None
        if isinstance(props, dict):
            return _gap_dict(props, gap_id)
    return None


def open_gaps(engine: Any, *, limit: int = 200) -> list[dict[str, Any]]:
    """Every gap not yet ``resolved`` — highest priority (lowest bucket) first."""
    if engine is None:
        return []
    try:
        rows = engine.query_cypher("MATCH (n:Gap) RETURN n LIMIT 1000")
    except Exception as e:  # noqa: BLE001
        logger.debug("open_gaps query failed: %s", e)
        return []
    out: list[dict[str, Any]] = []
    for r in rows or []:
        props = r.get("n") if isinstance(r, dict) else None
        if not isinstance(props, dict):
            continue
        d = _gap_dict(props)
        if str(d.get("status")) in _TERMINAL:
            continue
        out.append(d)
    out.sort(key=lambda d: int(d.get("priority_bucket", 2) or 2))
    return out[:limit]


def set_gap_status(engine: Any, gap_id: str, status: str, **extra: Any) -> bool:
    """Upsert a status (+ extra metadata) onto a ``:Gap`` node, best-effort."""
    current = get_gap(engine, gap_id) or {}
    payload = {**current, "status": status, "updated_at": _now_iso(), **extra}
    try:
        engine.add_node(
            gap_id,
            GAP_LABEL,
            properties={
                "status": status,
                "severity": float(payload.get("severity", 0.5) or 0.5),
                "priority_bucket": int(payload.get("priority_bucket", 2) or 2),
                "timestamp": payload["updated_at"],
                "metadata": json.dumps(payload, default=str),
            },
        )
        return True
    except Exception as e:  # noqa: BLE001
        logger.debug("set_gap_status(%s,%s) failed: %s", gap_id, status, e)
        return False


def mark_gap_resolved(engine: Any, gap_id: str) -> bool:
    """Flip a gap's status to ``resolved`` — the loop's visible END (D5)."""
    if not gap_id:
        return False
    ok = set_gap_status(engine, gap_id, STATUS_RESOLVED)
    if ok:
        logger.info("[Wave6] gap %s marked resolved", gap_id)
    return ok


def link_gap_to_spec(engine: Any, gap_id: str, spec_id: str) -> bool:
    """Write ``(:Gap)-[:SPECIFIED_BY]->(:SpecProposal)`` — chain hop 1 (D6)."""
    if engine is None or not gap_id or not spec_id:
        return False
    try:
        engine.add_edge(gap_id, spec_id, REL_SPECIFIED_BY)
    except Exception as e:  # noqa: BLE001
        logger.debug("link_gap_to_spec %s->%s failed: %s", gap_id, spec_id, e)
        return False
    # Mark the gap specified so the backlog reflects it has a spec in flight.
    set_gap_status(engine, gap_id, STATUS_SPECIFIED)
    return True


def resolve_gaps_for_loop(engine: Any, loop_id: str) -> list[str]:
    """Walk ``(loop)-[:RESOLVES]->(:Gap)`` and mark each gap resolved (D5).

    The develop-Loop bound by ``spec_proposals._bind_develop_loop`` carries a RESOLVES
    edge back to its origin gap; on publish this closes those gaps. Best-effort: returns
    the resolved gap ids.
    """
    if engine is None or not loop_id:
        return []
    resolved: list[str] = []
    try:
        rows = engine.query_cypher(
            "MATCH (l)-[:RESOLVES]->(g:Gap) WHERE l.id = $id RETURN g.id AS id",
            {"id": loop_id},
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("resolve_gaps_for_loop query failed: %s", e)
        return []
    for r in rows or []:
        gid = r.get("id") if isinstance(r, dict) else None
        if gid and mark_gap_resolved(engine, str(gid)):
            resolved.append(str(gid))
    return resolved


__all__ = [
    "GAP_LABEL",
    "STATUS_OPEN",
    "STATUS_SPECIFIED",
    "STATUS_RESOLVED",
    "STATUS_DEFERRED",
    "SOURCE_FAILURE",
    "SOURCE_RESEARCH",
    "SOURCE_SKILL",
    "SOURCE_AUDIT",
    "SOURCE_RUNTIME",
    "REL_SPECIFIED_BY",
    "REL_RESOLVES",
    "canonical_gap_id",
    "severity_to_bucket",
    "submit_gap",
    "get_gap",
    "open_gaps",
    "set_gap_status",
    "mark_gap_resolved",
    "link_gap_to_spec",
    "resolve_gaps_for_loop",
]
