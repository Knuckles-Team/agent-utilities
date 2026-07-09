#!/usr/bin/python
from __future__ import annotations

"""The distill → review → develop seam — distilled specs become develop-able work.

CONCEPT:AU-KG.research.close-distill-develop-seam — Fix the distill→develop DEAD-END (north-star GAP 1). Today
``enrichment/distill.py`` writes ``.specify`` SDD markdown, but the develop/code path
(``change_publisher.governed_publish`` → AHE-3.20–3.24) consumes *promoted proposal
nodes*, not those spec files — the two tracks are disjoint. This module connects them:
a distilled :class:`~..enrichment.distill.SpecDraft` is persisted as a first-class,
queryable ``:SpecProposal`` node (``DISTILLED_FROM`` its source concepts for
transparency), and on approval is fed — as the proposal the promotion path already
consumes — into ``governed_publish``. So "distill specs → develop THOSE specs" finally
flows, while the existing capability-ratchet regression gate (AHE-3.24) and the
``merge_promotion`` human-approval gate (OS-5.24) stay on the path (governance intact).

CONCEPT:AU-OS.config.autonomous-spec-develop-off — Spec-level review/veto checkpoint (north-star GAP 5). A distilled
spec is surfaced for optional Claude/human review/edit/veto BEFORE it develops, behind
a new ``spec_promotion`` ActionPolicy tier. The default posture is **propose-and-hold**:
a fresh spec sits in ``pending_review`` and only an explicit approval (or an operator
who relaxed the ``spec_promotion`` tier + enabled ``KG_LOOP_AUTO_DEVELOP``) turns it
into a develop Loop. Review/veto is the early corrigible checkpoint the loop was
missing — steering at the *spec* level, not just loop/schedule level.

All writes are best-effort; reads are backend-agnostic (status is recovered from the
node's top-level prop OR its folded ``metadata`` JSON, so it works on strict and
schemaless backends alike).
"""

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

SPEC_LABEL = "SpecProposal"

#: Lifecycle. ``pending_review`` (default, holds for review) → ``approved`` (a develop
#: Loop is bound) → ``developing`` → ``published`` (a reviewable branch exists) /
#: ``reverted`` (ratchet regression). ``rejected`` is the veto terminal.
STATUS_PENDING = "pending_review"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_DEVELOPING = "developing"
STATUS_PUBLISHED = "published"
STATUS_REVERTED = "reverted"

_TERMINAL = frozenset({STATUS_REJECTED, STATUS_PUBLISHED})
_COUNT_BUCKETS = (
    STATUS_PENDING,
    STATUS_APPROVED,
    STATUS_DEVELOPING,
    STATUS_PUBLISHED,
    STATUS_REVERTED,
    STATUS_REJECTED,
)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _slug(text: str, *, limit: int = 60) -> str:
    import re

    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (s[:limit] or "spec").rstrip("-")


def spec_id_for(title: str) -> str:
    """Stable :SpecProposal id derived from the spec title (idempotent upsert)."""
    return f"spec_proposal:{_slug(title)}"


# ────────────────────────────────────────────────────────────────────────
# Persist a distilled SpecDraft as a develop-able node (KG-2.292)
# ────────────────────────────────────────────────────────────────────────
def persist_spec_proposal(
    engine: Any,
    spec: Any,
    *,
    spec_path: str = "",
    status: str = STATUS_PENDING,
    target_file: str = "",
) -> str | None:
    """Persist a :class:`SpecDraft` as a queryable ``:SpecProposal`` node.

    Stores the full spec in the folded ``metadata`` JSON (backend-safe) and links
    ``DISTILLED_FROM`` to each source concept so the *why* is traversable. Returns
    the node id, or ``None`` on persist failure. Idempotent on the title-derived id.
    """
    title = str(getattr(spec, "title", "") or "").strip()
    if not title:
        return None
    sid = spec_id_for(title)
    concept_ids = list(getattr(spec, "concept_ids", []) or [])
    payload = {
        "title": title,
        "problem": str(getattr(spec, "problem", "") or ""),
        "approach": str(getattr(spec, "approach", "") or ""),
        "value": str(getattr(spec, "value", "") or ""),
        "concept_ids": concept_ids,
        "value_score": float(getattr(spec, "value_score", 0.0) or 0.0),
        "target_codebase": str(getattr(spec, "target_codebase", "") or ""),
        "spec_path": spec_path,
        "target_file": target_file,
        "status": status,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    try:
        engine.add_node(
            sid,
            SPEC_LABEL,
            properties={
                "name": title,
                "description": str(payload["problem"])[:500],
                # Top-level status/value_score: a real column on schemaless backends
                # (queryable), folded into metadata on strict ones (recovered on read).
                "status": status,
                "value_score": payload["value_score"],
                "timestamp": payload["created_at"],
                "metadata": json.dumps(payload, default=str),
            },
        )
    except Exception as e:  # noqa: BLE001 — best-effort persist
        logger.debug("persist_spec_proposal failed: %s", e)
        return None
    # Provenance edges: spec DISTILLED_FROM each source concept (transparency).
    for cid in concept_ids:
        try:
            engine.add_edge(sid, cid, "DISTILLED_FROM")
        except Exception as e:  # noqa: BLE001 — provenance is best-effort
            logger.debug("DISTILLED_FROM edge %s->%s failed: %s", sid, cid, e)
    return sid


def _spec_dict(node: dict[str, Any], node_id: str = "") -> dict[str, Any]:
    """Normalize a :SpecProposal node into a flat dict (backend-agnostic).

    Merges the folded ``metadata`` JSON under the top-level columns so ``status`` and
    the spec body are present whether the backend stored them as columns or folded
    them into ``metadata``.
    """
    out: dict[str, Any] = {}
    meta = node.get("metadata")
    if isinstance(meta, str):
        try:
            parsed = json.loads(meta)
            if isinstance(parsed, dict):
                out.update(parsed)
        except (TypeError, ValueError):
            pass
    elif isinstance(meta, dict):
        out.update(meta)
    # Top-level columns win where present (freshest on schemaless backends).
    for k in ("status", "value_score", "name"):
        if node.get(k) is not None:
            out[k] = node[k]
    out["id"] = node.get("id") or node_id or out.get("id")
    out.setdefault("status", STATUS_PENDING)
    out.setdefault("title", out.get("name") or out["id"])
    return out


def get_spec(engine: Any, spec_id: str) -> dict[str, Any] | None:
    """Load one :SpecProposal as a flat dict, or ``None``."""
    if engine is None or not spec_id:
        return None
    try:
        rows = engine.query_cypher(
            "MATCH (n:SpecProposal) WHERE n.id = $id RETURN n LIMIT 1", {"id": spec_id}
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("get_spec query failed: %s", e)
        return None
    for r in rows or []:
        props = r.get("n") if isinstance(r, dict) else None
        if isinstance(props, dict):
            return _spec_dict(props, spec_id)
    return None


def list_specs(
    engine: Any, *, status: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    """List :SpecProposal nodes (optionally filtered by status), newest-value first.

    Filtering is done in-memory (recovering ``status`` from folded metadata) so it is
    correct on strict backends where custom props are not first-class columns.
    """
    if engine is None:
        return []
    try:
        rows = engine.query_cypher("MATCH (n:SpecProposal) RETURN n LIMIT 500")
    except Exception as e:  # noqa: BLE001
        logger.debug("list_specs query failed: %s", e)
        return []
    out: list[dict[str, Any]] = []
    for r in rows or []:
        props = r.get("n") if isinstance(r, dict) else None
        if not isinstance(props, dict):
            continue
        d = _spec_dict(props)
        if status is not None and str(d.get("status")) != status:
            continue
        out.append(d)
    out.sort(key=lambda d: float(d.get("value_score", 0.0) or 0.0), reverse=True)
    return out[:limit]


def specs_summary(engine: Any, *, sample: int = 10) -> dict[str, Any]:
    """Backlog counts by status + a small sample — for the EvolutionState surface."""
    specs = list_specs(engine, limit=500)
    counts = {b: 0 for b in _COUNT_BUCKETS}
    for s in specs:
        st = str(s.get("status"))
        counts[st] = counts.get(st, 0) + 1
    pending = [s for s in specs if s.get("status") == STATUS_PENDING][:sample]
    return {
        "counts": counts,
        "total": len(specs),
        "pending_review": [
            {
                "id": s.get("id"),
                "title": s.get("title"),
                "value_score": s.get("value_score"),
                "concept_ids": s.get("concept_ids", []),
                "spec_path": s.get("spec_path", ""),
            }
            for s in pending
        ],
    }


def _set_status(engine: Any, spec_id: str, status: str, **extra: Any) -> bool:
    """Upsert a status (+ extra metadata) onto a :SpecProposal node, best-effort."""
    current = get_spec(engine, spec_id) or {}
    payload = {**current, "status": status, "updated_at": _now_iso(), **extra}
    try:
        engine.add_node(
            spec_id,
            SPEC_LABEL,
            properties={
                "status": status,
                "value_score": float(payload.get("value_score", 0.0) or 0.0),
                "timestamp": payload["updated_at"],
                "metadata": json.dumps(payload, default=str),
            },
        )
        return True
    except Exception as e:  # noqa: BLE001
        logger.debug("_set_status(%s,%s) failed: %s", spec_id, status, e)
        return False


# ────────────────────────────────────────────────────────────────────────
# The spec-level review / veto checkpoint (OS-5.73)
# ────────────────────────────────────────────────────────────────────────
def review_spec(
    engine: Any,
    spec_id: str,
    decision: str,
    *,
    reviewer: str = "user",
    edits: dict[str, Any] | None = None,
    submit_develop: bool = True,
) -> dict[str, Any]:
    """Approve / edit / reject a distilled spec BEFORE it develops (the veto point).

    - ``approve`` → status ``approved`` and (default) a ``develop`` Loop is bound to
      the spec so it enters the develop pipeline; steerable via ``graph_loops``.
    - ``reject`` → status ``rejected`` (veto terminal — never develops).
    - ``edit``   → merge ``edits`` (title/problem/approach/value/target_file) and keep
      it in ``pending_review`` for another look.

    Governance: this is the explicit human/Claude decision. ``approve`` records an
    ``ActionDecision`` audit for ``spec_promotion`` so the steer is auditable, but the
    operator's choice stands (a granted approval, not a fresh gate evaluation).
    """
    decision = (decision or "").strip().lower()
    spec = get_spec(engine, spec_id)
    if spec is None:
        return {"status": "not_found", "spec_id": spec_id}
    if str(spec.get("status")) in _TERMINAL:
        return {"status": "terminal", "spec_id": spec_id, "current": spec.get("status")}

    if decision in ("reject", "veto"):
        _set_status(engine, spec_id, STATUS_REJECTED, reviewer=reviewer)
        return {"status": STATUS_REJECTED, "spec_id": spec_id}

    if decision == "edit":
        merged = {**spec, **(edits or {})}
        _set_status(
            engine,
            spec_id,
            STATUS_PENDING,
            reviewer=reviewer,
            **{
                k: merged.get(k)
                for k in ("title", "problem", "approach", "value", "target_file")
                if k in (edits or {})
            },
        )
        return {
            "status": STATUS_PENDING,
            "spec_id": spec_id,
            "edited": list((edits or {}).keys()),
        }

    if decision in ("approve", "accept"):
        _audit_spec_decision(engine, spec_id, reviewer)
        _set_status(engine, spec_id, STATUS_APPROVED, reviewer=reviewer)
        out: dict[str, Any] = {"status": STATUS_APPROVED, "spec_id": spec_id}
        if submit_develop:
            out["develop_loop"] = _bind_develop_loop(engine, spec)
        return out

    return {
        "status": "error",
        "spec_id": spec_id,
        "detail": f"unknown decision {decision!r}",
    }


def _audit_spec_decision(engine: Any, spec_id: str, reviewer: str) -> None:
    """Record a ``spec_promotion`` ActionDecision so the approval is auditable."""
    try:
        from agent_utilities.orchestration.action_policy import (
            ActionRequest,
            get_action_policy,
        )

        get_action_policy(engine).queue_approval(
            ActionRequest(
                kind="spec_promotion",
                target=spec_id,
                source=f"review:{reviewer}",
                reason="spec approved for develop by reviewer",
            ),
            reason="approved at spec-review checkpoint",
        )
    except Exception as e:  # noqa: BLE001 — audit is best-effort
        logger.debug("_audit_spec_decision failed: %s", e)


def _bind_develop_loop(engine: Any, spec: dict[str, Any]) -> dict[str, Any] | None:
    """Submit a ``develop`` Loop bound to the spec (visible + steerable in graph_loops)."""
    from .loops import submit_loop

    spec_id = str(spec.get("id"))
    title = str(spec.get("title") or spec_id)
    loop_id = f"loop:develop:{spec_id}"
    loop = submit_loop(
        engine,
        f"Develop distilled spec: {title}",
        kind="develop",
        loop_id=loop_id,
        source="spec_promotion",
    )
    # Stamp the spec binding so _advance_develop routes this Loop to the spec
    # develop pipeline (governed_publish) instead of a validation_cmd.
    try:
        engine.add_node(loop_id, "Concept", properties={"spec_id": spec_id})
    except Exception as e:  # noqa: BLE001
        logger.debug("_bind_develop_loop stamp failed: %s", e)
    return loop


# ────────────────────────────────────────────────────────────────────────
# The develop step — feed an approved spec into the EXISTING promotion pipeline
# ────────────────────────────────────────────────────────────────────────
def _spec_to_proposal(spec: dict[str, Any]) -> dict[str, Any]:
    """Shape a spec dict into the proposal the promotion path already consumes.

    ``change_synthesis``/``code_synthesis`` read flat ``title``/``problem``/
    ``approach``/``value``/``concept_ids``/``target_file`` fields — so an approved
    spec IS a valid proposal; no new code-gen path is introduced.
    """
    return {
        "id": spec.get("id"),
        "name": spec.get("title"),
        "title": spec.get("title"),
        "problem": spec.get("problem", ""),
        "approach": spec.get("approach", ""),
        "value": spec.get("value", ""),
        "summary": spec.get("problem", "") or spec.get("approach", ""),
        "concept_ids": list(spec.get("concept_ids", []) or []),
        # Only set a target when the spec resolved one — otherwise the pipeline
        # falls through to the SDD spec+tasks skeleton (still a reviewable branch).
        **({"target_file": spec["target_file"]} if spec.get("target_file") else {}),
    }


def develop_spec(
    engine: Any, spec_id: str, *, source: str = "spec_develop"
) -> dict[str, Any]:
    """Feed an approved spec into the develop pipeline via ``governed_publish``.

    Runs the EXISTING ``code_synthesis → change_synthesis → validate_in_sandbox →
    change_publisher → capability_ratchet`` path, which itself applies the
    ``merge_promotion`` human-approval gate (a reviewable branch is queued, never
    auto-merged). The spec node's status is advanced to reflect the outcome.
    """
    spec = get_spec(engine, spec_id)
    if spec is None:
        return {"status": "not_found", "spec_id": spec_id}
    if str(spec.get("status")) not in (STATUS_APPROVED, STATUS_DEVELOPING):
        return {
            "status": "not_approved",
            "spec_id": spec_id,
            "detail": f"spec is {spec.get('status')!r}; approve it first (review checkpoint)",
        }
    _set_status(engine, spec_id, STATUS_DEVELOPING)
    try:
        from .change_publisher import governed_publish

        report = governed_publish(engine, _spec_to_proposal(spec), source=source)
    except Exception as e:  # noqa: BLE001 — never raise into the loop
        _set_status(engine, spec_id, STATUS_APPROVED, develop_error=str(e))
        return {"status": "error", "spec_id": spec_id, "detail": str(e)}

    pub_status = str(report.get("status", ""))
    if pub_status == "published":
        _set_status(engine, spec_id, STATUS_PUBLISHED, develop_result=report)
    elif pub_status == "reverted":
        _set_status(engine, spec_id, STATUS_REVERTED, develop_result=report)
    # approval_queued / other governed outcomes leave it 'developing' (a human grant
    # of the merge_promotion approval triggers the one-shot publish).
    report["spec_id"] = spec_id
    return report


# ────────────────────────────────────────────────────────────────────────
# The 24/7 auto path (opt-in) — gated through spec_promotion
# ────────────────────────────────────────────────────────────────────────
def auto_advance_specs(engine: Any, *, limit: int = 5) -> dict[str, Any]:
    """Auto-advance ``pending_review`` specs through the ``spec_promotion`` gate.

    Only invoked when ``KG_LOOP_AUTO_DEVELOP`` is on (review-first by default). For
    each pending spec it consults the OS-5.24 ActionPolicy under ``spec_promotion``;
    an *allowed* verdict (an operator who relaxed that tier) approves the spec + binds
    a develop Loop, otherwise the spec stays ``pending_review`` and an approval is
    queued for the human. Acquisition is never auto-run; this only governs develop.
    """
    out: dict[str, Any] = {"approved": 0, "queued": 0, "results": []}
    try:
        from agent_utilities.orchestration.action_policy import (
            ActionRequest,
            get_action_policy,
        )

        policy = get_action_policy(engine)
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)
        return out
    for spec in list_specs(engine, status=STATUS_PENDING, limit=limit):
        spec_id = str(spec.get("id"))
        decision = policy.decide(
            ActionRequest(
                kind="spec_promotion",
                target=spec_id,
                source="loop_engine",
                reason="auto-develop distilled spec (KG_LOOP_AUTO_DEVELOP)",
            )
        )
        if decision.allowed:
            review_spec(engine, spec_id, "approve", reviewer="loop_engine")
            out["approved"] += 1
            out["results"].append({"id": spec_id, "decision": "approved"})
        else:
            out["queued"] += 1
            out["results"].append(
                {
                    "id": spec_id,
                    "decision": decision.decision,
                    "approval_id": decision.approval_id,
                }
            )
    return out


__all__ = [
    "SPEC_LABEL",
    "STATUS_PENDING",
    "STATUS_APPROVED",
    "STATUS_REJECTED",
    "STATUS_DEVELOPING",
    "STATUS_PUBLISHED",
    "STATUS_REVERTED",
    "spec_id_for",
    "persist_spec_proposal",
    "get_spec",
    "list_specs",
    "specs_summary",
    "review_spec",
    "develop_spec",
    "auto_advance_specs",
]
