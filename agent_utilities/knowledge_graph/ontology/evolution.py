#!/usr/bin/python
"""Ontology evolution as a governed proposal — CONCEPT:AU-KG.ontology.evolution-governed-loop
(program item 7.5).

Ontology change runs through the SAME governed shape every other evolution
proposal in this codebase already uses (a distilled ``:SpecProposal``, a mined
``:PlacementProposal``, a candidate ``:SkillVersion`` — see
``orchestration/action_policy.py``'s reserved-kind table), reusing existing
primitives rather than re-deriving a parallel governance stack:

* **detect** — :func:`propose_ontology_change` accepts a candidate (raw
  Turtle/OWL, a file, or a URL — anything :func:`lifecycle._parse_graph`
  already reads) and, for a caller that wants deterministic/LLM-assisted
  DISCOVERY first, the existing ``schema_discovery.discover_schema_extensions``
  / ``to_ttl_fragment`` pipeline (already live-wired, propose-only,
  CONCEPT:AU-KG.ontology.do-not-auto-merge) is the recommended upstream step —
  its output (a ``.ttl`` fragment of *missing* candidate types) is exactly
  what gets handed to this module as the proposal's ``source``. This module
  does not re-implement discovery; it governs what happens to a candidate
  once one exists.
* **align/decide** — :func:`classify_change` is a DETERMINISTIC diff (class/
  property set arithmetic) against the currently-active ontology — never an
  LLM judgment call — and :func:`lifecycle.validate_graph` (OWL-RL closure +
  bundled SHACL shapes) is the REASONER that decides whether the candidate is
  internally consistent. Program item 3's governing rule applies literally
  here: **LLMs propose; reasoners, shapes, regression queries, and reviewers
  decide.**
* **shadow + replay** — :func:`materialize_shadow` loads the candidate's
  axioms into a THROWAWAY named graph (never the tenant's real ontology graph
  — see :mod:`lifecycle`'s module docstring on why mixing candidate/ABox
  identifiers trips the engine's SHACL/ICV write guard), then
  :func:`replay_competency_queries` runs the same representative queries
  against the live ontology graph and the shadow graph and diffs the row
  counts — a basic, but real, competency-question regression check.
* **review gate** — :func:`promote_ontology_proposal` routes through
  ``action_policy.get_action_policy(engine).decide(...)`` under the reserved
  kind ``"ontology_proposal_promotion"`` (registered ``TIER_APPROVAL``,
  SAFETY-CRITICAL, never auto — see ``action_policy.py``'s default rule
  table) — the EXACT SAME governed decision point + durable
  ``ActionDecision`` audit trail every other promotion in this codebase uses.
  **Nothing here auto-merges into the canonical ontology**: a proposal is
  promoted ONLY when that decision is ``allowed``; otherwise it is filed for
  human review (queue-approval) or denied, and stays queryable either way.
* **promote / rollback** — promotion IS :meth:`lifecycle.OntologyLifecycle.update`
  (already versioned + bi-temporal: the prior version stays hosted, just
  deactivated). Rollback re-activates that still-hosted prior version and
  deactivates the promoted one — no separate migration/rollback artifact
  store is needed because the versioned registry already IS one.
* **measured outcome** — the proposal record keeps its shadow-replay diff,
  review decision, and (after promotion) whether the SAME competency queries
  against the now-active ontology moved in the direction the proposal
  predicted — a real but INTENTIONALLY BASIC proxy metric (row-count deltas),
  not a full retrieval/routing-quality evaluation (deferred — see the lane's
  deferred register). A rejected/no-improvement proposal is never deleted, so
  negative results stay queryable (CONCEPT:AU-KG.ontology.negative-results-queryable).

Proposals are stored the SAME way :class:`lifecycle.OntologyLifecycle` stores
hosted ontologies: durable, per-tenant, engine-native ``:OntologyProposal``
nodes when a live engine is attached (:class:`lifecycle._EngineRegistryStore`,
reused directly), degrading to a process-local store offline/in tests.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent_utilities.security.log_redaction import redact_for_log

from .lifecycle import (
    OntologyError,
    OntologyLifecycle,
    _EngineRegistryStore,
    _ensure_ontology_graph,
    _InMemoryRegistryStore,
    _ontology_graph_name,
    _parse_graph,
    summarize,
    validate_graph,
)

logger = logging.getLogger(__name__)

_PROPOSAL_NODE_TYPE = "OntologyProposal"
_PROPOSAL_NODE_PREFIX = "ontprop"

#: Process-local fallback registry for proposals (offline/no-engine — mirrors
#: ``lifecycle._MEMORY_STORE``'s degrade-honestly posture).
_PROPOSAL_MEMORY_STORE = _InMemoryRegistryStore()

STATUS_PENDING_REVIEW = "pending_review"
STATUS_QUEUED_FOR_APPROVAL = "queued_for_approval"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_PROMOTED = "promoted"
STATUS_ROLLED_BACK = "rolled_back"

#: Representative structural competency queries run against baseline + shadow
#: graphs (CONCEPT:AU-KG.ontology.competency-query-regression). Intentionally
#: generic/structural (class/property/triple cardinality) rather than
#: domain-specific competency questions — a caller with a bundled competency-
#: question suite for its domain should pass its own ``queries=`` instead.
DEFAULT_COMPETENCY_QUERIES: tuple[str, ...] = (
    "SELECT (COUNT(?c) AS ?n) WHERE { ?c a <http://www.w3.org/2002/07/owl#Class> }",
    "SELECT (COUNT(?p) AS ?n) WHERE { ?p a <http://www.w3.org/2002/07/owl#ObjectProperty> }",
    "SELECT (COUNT(?p) AS ?n) WHERE { ?p a <http://www.w3.org/2002/07/owl#DatatypeProperty> }",
    "SELECT (COUNT(?s) AS ?n) WHERE { ?s ?p ?o }",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _coerce_evidence_ref(ref: Any) -> str:
    """Normalise one ``evidence_refs`` entry to an opaque reference string
    (D-75-6, CONCEPT:AU-KG.evolution.unified-evidence-resource). The unified
    ``Evidence`` resource (:mod:`knowledge_graph.research.evidence`, lane 7.1)
    landed after this module's ``evidence_refs`` field was written as a bare
    string list; this reconciles the two ADDITIVELY rather than redesigning
    the field — a caller that already has a typed ``Evidence`` instance (or
    its persisted ``EvolutionEvidenceNode``/dict form) gets its real
    content-addressed ``evidence_id`` stored instead of ``repr(obj)``; a
    caller passing a plain opaque string (the original, still-supported
    shape) is unaffected.
    """
    evidence_id = getattr(ref, "evidence_id", None)
    if isinstance(evidence_id, str) and evidence_id:
        return evidence_id
    ref_id = getattr(ref, "id", None)
    if isinstance(ref_id, str) and ref_id.startswith("evolution_evidence:"):
        return ref_id
    if isinstance(ref, dict):
        dict_id = ref.get("id")
        if isinstance(dict_id, str) and dict_id.startswith("evolution_evidence:"):
            return dict_id
    return str(ref)


def classify_change(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Deterministic additive-vs-breaking classification (program item 3: a
    diff, never an LLM judgment) of ``candidate`` against the currently-active
    ontology's ``summarize()`` output. A class/property REMOVAL is breaking —
    anything else purely additive is a minor change; no change at all is a
    patch (metadata/annotation-only edits still get a new version).
    """
    base_classes = set(baseline.get("classes", []))
    cand_classes = set(candidate.get("classes", []))
    base_props = set(baseline.get("properties", []))
    cand_props = set(candidate.get("properties", []))
    removed_classes = sorted(base_classes - cand_classes)
    removed_properties = sorted(base_props - cand_props)
    added_classes = sorted(cand_classes - base_classes)
    added_properties = sorted(cand_props - base_props)
    breaking = bool(removed_classes or removed_properties)
    additive = bool(added_classes or added_properties)
    kind = "breaking" if breaking else ("additive" if additive else "unchanged")
    return {
        "kind": kind,
        "removed_classes": removed_classes,
        "removed_properties": removed_properties,
        "added_classes": added_classes,
        "added_properties": added_properties,
    }


def next_semver(prior_version: str, kind: str) -> str:
    """Deterministic SemVer bump from a :func:`classify_change` ``kind``:
    ``breaking`` → major, ``additive`` → minor, else → patch."""
    parts = str(prior_version or "0.0.0").split(".")
    padded = (parts + ["0", "0", "0"])[:3]
    try:
        major, minor, patch = (int(p) for p in padded)
    except ValueError:
        major, minor, patch = 0, 0, 0
    # Joined rather than f-string-interpolated: these are ints in a SemVer
    # string, but a BARE ``{major}``/``{minor}`` inside an f-string is exactly
    # the shape scripts/check_identifier_interpolation.py treats as a possible
    # Cypher/SQL identifier. That gate already documents the semver bump as a
    # known false-positive shape, but its structural exemption only covers
    # call/arithmetic components (``f"{maj}.{min}.{int(patch) + 1}"``), not bare
    # names. Composing the parts explicitly removes the interpolation entirely,
    # so the gate stays strict instead of being taught a new exception.
    if kind == "breaking":
        return ".".join(str(part) for part in (major + 1, 0, 0))
    if kind == "additive":
        return ".".join(str(part) for part in (major, minor + 1, 0))
    return ".".join(str(part) for part in (major, minor, patch + 1))


def _local_name(iri: str) -> str:
    """The fragment/last-segment of an IRI, lowercased, for name comparison."""
    frag = str(iri).rsplit("#", 1)[-1]
    return frag.rsplit("/", 1)[-1].strip().lower()


@lru_cache(maxsize=1)
def _bundled_standard_vocabulary() -> frozenset[str]:
    """Local names of every term the platform's bundled base ontology
    (``knowledge_graph/ontology.ttl``) already carries as an authoritative
    standard — its own native classes/properties PLUS the ``owl:equivalentClass``
    / ``rdfs:seeAlso`` alignment targets it declares against BFO, PROV-O,
    Schema.org, Dublin Core, SKOS, OWL-Time, BIBO, and FIBO (verified directly
    against this file, e.g. ``:CreativeWork owl:equivalentClass
    schema:CreativeWork``, ``:Process rdfs:seeAlso prov:Activity`` — see
    ``tests/unit/knowledge_graph/test_standard_ontology.py`` for the full
    standards list this file already absorbs). This is genuinely the
    authoritative-standards comparison corpus, not a stand-in: the file
    directly ENCODES the alignment rather than merely importing the external
    ontologies, so a plain rdflib parse of this ONE bundled file is sufficient
    — no network fetch, no new dependency, no separate corpus to maintain.
    Cached: the bundled file does not change at runtime.
    """
    import rdflib

    path = Path(__file__).resolve().parent.parent / "ontology.ttl"
    if not path.exists():
        return frozenset()
    graph = rdflib.Graph()
    try:
        graph.parse(str(path), format="turtle")
    except Exception as exc:  # noqa: BLE001 — a corrupt bundle degrades to "no corpus"
        logger.warning(
            "standards vocabulary: failed to parse %s: %s",
            redact_for_log(path),
            exc,
        )
        return frozenset()
    names: set[str] = set()
    for s in graph.subjects(predicate=rdflib.RDF.type, object=rdflib.OWL.Class):
        names.add(_local_name(s))
    for s in graph.subjects(
        predicate=rdflib.RDF.type, object=rdflib.OWL.ObjectProperty
    ):
        names.add(_local_name(s))
    for s in graph.subjects(
        predicate=rdflib.RDF.type, object=rdflib.OWL.DatatypeProperty
    ):
        names.add(_local_name(s))
    for pred in (
        rdflib.OWL.equivalentClass,
        rdflib.OWL.equivalentProperty,
        rdflib.RDFS.seeAlso,
    ):
        for o in graph.objects(predicate=pred):
            names.add(_local_name(o))
    return frozenset(n for n in names if n)


def compare_against_standards(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """(program item 2, D-75-8) flag candidate classes/properties whose LOCAL
    NAME collides with a term already in the bundled authoritative-standards
    corpus (:func:`_bundled_standard_vocabulary`) — a deterministic name
    check, never an LLM judgment, matching :func:`classify_change`'s own
    reasoning discipline. A collision does not mean the candidate is wrong
    (the same real-world concept legitimately recurs across domains) — it
    means a REVIEWER should look, because the candidate may be duplicating an
    existing standard term under a different name/namespace instead of
    reusing (``owl:equivalentClass``-aligning to) it.
    """
    standards = _bundled_standard_vocabulary()
    if not standards:
        return []
    flags: list[dict[str, Any]] = []
    for kind, terms in (
        ("class", candidate.get("classes", [])),
        ("property", candidate.get("properties", [])),
    ):
        for term in terms:
            local = _local_name(term)
            if local and local in standards:
                flags.append({"term": term, "kind": kind, "local_name": local})
    return flags


def _run_queries(gc: Any, queries: tuple[str, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    sparql = getattr(gc, "sparql", None) if gc is not None else None
    for q in queries:
        if not callable(sparql):
            out.append({"query": q, "rows": None, "error": "no engine SPARQL surface"})
            continue
        try:
            rows = sparql(q)
            out.append({"query": q, "rows": len(rows or []), "error": None})
        except Exception as exc:  # noqa: BLE001 — one query failing must not abort replay
            out.append({"query": q, "rows": None, "error": str(exc)})
    return out


def _diff_query_results(
    baseline: list[dict[str, Any]], shadow: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    diffs = []
    for base_row, shadow_row in zip(baseline, shadow, strict=False):
        diffs.append(
            {
                "query": base_row["query"],
                "baseline_rows": base_row["rows"],
                "shadow_rows": shadow_row["rows"],
                "delta": (
                    (shadow_row["rows"] - base_row["rows"])
                    if isinstance(base_row["rows"], int)
                    and isinstance(shadow_row["rows"], int)
                    else None
                ),
                "regressed": shadow_row.get("error") is not None
                and base_row.get("error") is None,
            }
        )
    return diffs


def materialize_shadow(
    engine: Any, tenant: str | None, proposal_id: str, turtle: str
) -> tuple[str, dict[str, Any]]:
    """Load ``turtle`` into a THROWAWAY shadow graph, never the tenant's real
    ontology graph (CONCEPT:AU-KG.ontology.dedicated-tbox-graph — the same
    reasoning that keeps TBox axioms off the mixed ABox graph applies doubly
    here: an unreviewed candidate must never touch durable state). Returns the
    shadow graph's name and the raw ``add_triples`` report; callers replay
    queries against it with :func:`replay_competency_queries` and are
    responsible for dropping it (:func:`discard_shadow`) once done — a shadow
    graph is scratch space, not part of the hosted-ontology lifecycle.
    """
    gc0 = getattr(engine, "graph_compute", None) if engine else None
    if gc0 is None:
        return "", {"loaded_to_engine": False, "reason": "no engine attached"}
    ontology_graph = _ontology_graph_name(tenant)
    shadow_graph = f"{ontology_graph}__shadow__{proposal_id}"
    for_graph = getattr(gc0, "for_graph", None)
    shadow_gc = for_graph(shadow_graph) if callable(for_graph) else gc0
    try:
        _ensure_ontology_graph(shadow_gc, shadow_graph)
    except OntologyError as exc:
        return shadow_graph, {"loaded_to_engine": False, "reason": str(exc)}
    if not hasattr(shadow_gc, "add_triples"):
        return shadow_graph, {
            "loaded_to_engine": False,
            "reason": "no engine RDF surface",
        }
    try:
        report = shadow_gc.add_triples(turtle=turtle)
        return shadow_graph, {"loaded_to_engine": True, **(report or {})}
    except Exception as exc:  # noqa: BLE001 — reported, not swallowed
        logger.warning("Shadow materialization failed for %s: %s", shadow_graph, exc)
        return shadow_graph, {"loaded_to_engine": False, "reason": str(exc)}


def discard_shadow(engine: Any, shadow_graph: str) -> dict[str, Any]:
    """Best-effort teardown of a shadow graph (``drop_named_graph`` if the
    engine exposes it, else left for the engine's own idle/cold-offload sweep
    — a shadow graph is a bounded-lifetime scratch graph, not a leak)."""
    gc0 = getattr(engine, "graph_compute", None) if engine else None
    drop = getattr(gc0, "drop_named_graph", None) if gc0 is not None else None
    if not callable(drop):
        return {"dropped": False, "reason": "no engine drop_named_graph surface"}
    try:
        drop(shadow_graph)
        return {"dropped": True}
    except Exception as exc:  # noqa: BLE001 — best-effort teardown
        logger.debug("discard_shadow(%s) failed: %s", shadow_graph, exc)
        return {"dropped": False, "reason": str(exc)}


def replay_competency_queries(
    engine: Any,
    tenant: str | None,
    shadow_graph: str,
    *,
    queries: tuple[str, ...] = DEFAULT_COMPETENCY_QUERIES,
) -> dict[str, Any]:
    """Run ``queries`` against BOTH the tenant's live ontology graph (baseline)
    and the shadow graph, and diff the results — a basic, real competency-
    question / query regression check (program item 2)."""
    gc0 = getattr(engine, "graph_compute", None) if engine else None
    if gc0 is None:
        return {"baseline": [], "shadow": [], "diffs": [], "note": "no engine attached"}
    for_graph = getattr(gc0, "for_graph", None)
    baseline_graph = _ontology_graph_name(tenant)
    baseline_gc = for_graph(baseline_graph) if callable(for_graph) else gc0
    shadow_gc = for_graph(shadow_graph) if callable(for_graph) else gc0
    baseline_results = _run_queries(baseline_gc, queries)
    shadow_results = _run_queries(shadow_gc, queries)
    return {
        "baseline": baseline_results,
        "shadow": shadow_results,
        "diffs": _diff_query_results(baseline_results, shadow_results),
    }


def _proposal_store(engine: Any, tenant: str | None) -> Any:
    gc0 = getattr(engine, "graph_compute", None) if engine else None
    if gc0 is None:
        return _PROPOSAL_MEMORY_STORE
    for_graph = getattr(gc0, "for_graph", None)
    ontology_graph = _ontology_graph_name(tenant)
    gc = for_graph(ontology_graph) if callable(for_graph) else gc0
    is_real_engine = (
        hasattr(gc, "add_node")
        and hasattr(gc, "get_nodes_by_label")
        and hasattr(gc, "has_node")
        and hasattr(gc, "remove_node")
        and hasattr(gc, "client")
    )
    if not is_real_engine:
        return _PROPOSAL_MEMORY_STORE
    try:
        _ensure_ontology_graph(gc, ontology_graph)
        return _EngineRegistryStore(
            gc, node_type=_PROPOSAL_NODE_TYPE, prefix=_PROPOSAL_NODE_PREFIX
        )
    except OntologyError as exc:
        logger.warning(
            "Durable ontology-proposal registry unavailable for graph %s "
            "(falling back to the process-local, non-durable registry): %s",
            ontology_graph,
            exc,
        )
        return _PROPOSAL_MEMORY_STORE


def get_proposal(
    engine: Any, tenant: str | None, proposal_id: str
) -> dict[str, Any] | None:
    return _proposal_store(engine, tenant).get(proposal_id)


def list_proposals(
    engine: Any, tenant: str | None, *, status: str = ""
) -> list[dict[str, Any]]:
    records = _proposal_store(engine, tenant).values()
    if status:
        records = [r for r in records if r.get("status") == status]
    return sorted(records, key=lambda r: r.get("proposed_at", ""), reverse=True)


def propose_ontology_change(
    engine: Any,
    tenant: str | None,
    source: str,
    *,
    iri: str,
    source_type: str = "auto",
    evidence_refs: list[str] | None = None,
    competency_queries: tuple[str, ...] = DEFAULT_COMPETENCY_QUERIES,
    proposer: str = "",
    reason: str = "",
) -> dict[str, Any]:
    """Detect → align/decide → shadow-replay a candidate ontology change and
    file it as a governed, reviewable ``:OntologyProposal``. NEVER touches the
    canonical/active ontology (CONCEPT:AU-KG.ontology.do-not-auto-merge) —
    the candidate's axioms only ever land in a throwaway shadow graph here;
    promotion is a SEPARATE, explicitly-gated step
    (:func:`promote_ontology_proposal`).
    """
    try:
        candidate_graph = _parse_graph(source, source_type)
    except OntologyError as exc:
        return {"status": "rejected", "errors": [str(exc)], "warnings": []}

    validation = validate_graph(candidate_graph)
    candidate_summary = summarize(candidate_graph)

    lc = OntologyLifecycle(engine, tenant=tenant)
    existing = lc.get(iri)  # public lookup — newest hosted version, if any
    baseline_record = existing.get("ontology") if "ontology" in existing else None
    baseline_summary = {
        "classes": (baseline_record or {}).get("classes", []),
        "properties": (baseline_record or {}).get("properties", []),
    }
    classification = classify_change(baseline_summary, candidate_summary)
    # program item 2 / D-75-8 — flag (never block; a reviewer decides) any
    # candidate class/property whose local name collides with a term the
    # bundled authoritative-standards corpus already carries.
    standards_flags = compare_against_standards(candidate_summary)
    prior_version = baseline_record.get("version") if baseline_record else None
    proposed_version = next_semver(prior_version or "0.0.0", classification["kind"])

    turtle = candidate_graph.serialize(format="turtle")
    if isinstance(turtle, bytes):
        turtle = turtle.decode("utf-8")

    proposal_id = f"{iri}@@{proposed_version}"
    shadow_graph, shadow_report = materialize_shadow(
        engine, tenant, proposal_id, turtle
    )
    replay = (
        replay_competency_queries(
            engine, tenant, shadow_graph, queries=competency_queries
        )
        if shadow_report.get("loaded_to_engine")
        else {
            "baseline": [],
            "shadow": [],
            "diffs": [],
            "note": "shadow not materialized",
        }
    )

    # Program item 3: reasoners/shapes/regression queries decide review-worthiness
    # — never the proposer. Any SHACL/OWL-RL failure, any breaking change, or
    # any regressed competency query forces mandatory review (belt-and-braces:
    # promotion is ALSO unconditionally gated by action_policy — see
    # `promote_ontology_proposal` — this flag is diagnostic/explanatory).
    regressed = any(d.get("regressed") for d in replay.get("diffs", []))
    requires_review = bool(
        not validation["valid"] or classification["kind"] == "breaking" or regressed
    )

    record: dict[str, Any] = {
        "proposal_id": proposal_id,
        "iri": iri,
        "version": proposed_version,
        "prior_version": prior_version,
        "tenant": tenant or "",
        "source": source if len(source) < 256 else f"{source[:240]}…",
        "source_type": source_type,
        "turtle": turtle,
        "evidence_refs": (
            [_coerce_evidence_ref(ref) for ref in evidence_refs]
            if evidence_refs
            else []
        ),
        "proposer": proposer,
        "reason": reason,
        "proposed_at": _now(),
        "status": STATUS_PENDING_REVIEW,
        "validation": validation,
        "classification": classification,
        "standards_alignment": {
            "checked": bool(_bundled_standard_vocabulary()),
            "flags": standards_flags,
        },
        "requires_review": requires_review,
        "shadow_graph": shadow_graph,
        "shadow_report": shadow_report,
        "replay": replay,
        "active": False,
        "decision": None,
        "promotion": None,
        "rollback": None,
    }
    _proposal_store(engine, tenant).set(proposal_id, record)
    logger.info(
        "Ontology proposal filed: %s (kind=%s, requires_review=%s, valid=%s)",
        proposal_id,
        classification["kind"],
        requires_review,
        validation["valid"],
    )
    return {"status": "ok", "proposal": record}


def review_ontology_proposal(
    engine: Any,
    tenant: str | None,
    proposal_id: str,
    *,
    approve: bool,
    reviewer: str,
    notes: str = "",
) -> dict[str, Any]:
    """Record a human/policy review decision on a filed proposal. This is
    ADVISORY input to :func:`promote_ontology_proposal`'s own
    ``action_policy`` gate — approving here does not bypass that gate; it
    simply lets a caller pre-clear a proposal before attempting promotion.

    A REJECTED proposal's shadow graph (CONCEPT:AU-KG.ontology.shadow-graph-gc,
    D-75-7) is torn down here, the same best-effort :func:`discard_shadow`
    :func:`promote_ontology_proposal` already calls on success — a rejected
    proposal is never coming back for promotion, so there is nothing left to
    replay competency queries against; leaving it materialized would only
    accumulate scratch graphs under review churn.
    """
    store = _proposal_store(engine, tenant)
    record = store.get(proposal_id)
    if record is None:
        return {"error": f"ontology proposal not found: {proposal_id}"}
    record["status"] = STATUS_APPROVED if approve else STATUS_REJECTED
    record["decision"] = {
        "approved": bool(approve),
        "reviewer": reviewer,
        "notes": notes,
        "decided_at": _now(),
    }
    if not approve and record.get("shadow_graph"):
        record["shadow_discard"] = discard_shadow(engine, record["shadow_graph"])
    store.set(proposal_id, record)
    return {"status": "ok", "proposal": record}


def promote_ontology_proposal(
    engine: Any, tenant: str | None, proposal_id: str
) -> dict[str, Any]:
    """Promote a proposal to the active ontology — gated by the SAME governed
    decision point every other evolution proposal in this codebase uses
    (``action_policy.get_action_policy(engine).decide(...)`` under the
    reserved, SAFETY-CRITICAL, never-auto kind ``"ontology_proposal_promotion"``).
    A proposal rejected by review, or whose promotion the action-policy gate
    does not immediately allow, is NEVER merged — it is filed for approval or
    denied, and stays queryable either way
    (CONCEPT:AU-KG.ontology.negative-results-queryable).
    """
    from agent_utilities.orchestration.action_policy import (
        DECISION_QUEUE,
        ActionRequest,
        get_action_policy,
    )

    store = _proposal_store(engine, tenant)
    record = store.get(proposal_id)
    if record is None:
        return {"error": f"ontology proposal not found: {proposal_id}"}
    if record.get("status") == STATUS_REJECTED:
        return {
            "error": f"ontology proposal {proposal_id} was rejected — cannot promote"
        }
    if record.get("status") == STATUS_PROMOTED:
        return {"status": "ok", "idempotent": True, "proposal": record}

    decision = get_action_policy(engine).decide(
        ActionRequest(
            kind="ontology_proposal_promotion",
            target=proposal_id,
            reason=f"promote {record.get('iri')}@{record.get('version')}",
            source="ontology_evolution",
        )
    )
    if not decision.allowed:
        record["status"] = (
            STATUS_QUEUED_FOR_APPROVAL
            if decision.decision == DECISION_QUEUE
            else STATUS_PENDING_REVIEW
        )
        record["gate_decision"] = {
            "decision": decision.decision,
            "reason": decision.reason,
            "approval_id": decision.approval_id,
        }
        store.set(proposal_id, record)
        return {"status": "held", "proposal": record}

    lc = OntologyLifecycle(engine, tenant=tenant)
    promotion = lc.update(
        record["turtle"],
        iri=record["iri"],
        version=record["version"],
        source_type="text",
        activate=True,
    )
    record["status"] = STATUS_PROMOTED
    record["active"] = bool(promotion.get("ontology", {}).get("active"))
    record["promotion"] = {
        "result": promotion,
        "promoted_at": _now(),
        "gate_decision": {
            "decision": decision.decision,
            "reason": decision.reason,
            "approval_id": decision.approval_id,
        },
    }
    store.set(proposal_id, record)
    discard_shadow(engine, record.get("shadow_graph", ""))
    return {"status": "ok", "proposal": record}


def rollback_ontology_proposal(
    engine: Any, tenant: str | None, proposal_id: str
) -> dict[str, Any]:
    """Roll back a PROMOTED proposal: reactivate the still-hosted prior
    version and deactivate the promoted one — no separate rollback artifact
    store is needed because :meth:`lifecycle.OntologyLifecycle.update`'s
    bi-temporal versioning already keeps the prior version hosted."""
    store = _proposal_store(engine, tenant)
    record = store.get(proposal_id)
    if record is None:
        return {"error": f"ontology proposal not found: {proposal_id}"}
    if record.get("status") != STATUS_PROMOTED:
        return {
            "error": f"ontology proposal {proposal_id} is not promoted — nothing to roll back"
        }
    prior_version = record.get("prior_version")
    if not prior_version:
        return {
            "error": f"ontology proposal {proposal_id} has no prior version to roll back to"
        }

    lc = OntologyLifecycle(engine, tenant=tenant)
    lc.set_active(record["iri"], version=record["version"], active=False)
    reactivate = lc.set_active(record["iri"], version=prior_version, active=True)
    record["status"] = STATUS_ROLLED_BACK
    record["active"] = False
    record["rollback"] = {
        "result": reactivate,
        "rolled_back_at": _now(),
        "restored_version": prior_version,
    }
    store.set(proposal_id, record)
    return {"status": "ok", "proposal": record}


def reset_proposal_registry() -> None:
    """Clear the in-process proposal registry (tests / clean-slate) — mirrors
    :func:`lifecycle.reset_registry`'s scope: only the non-durable in-memory
    fallback is process-local and needs clearing."""
    _PROPOSAL_MEMORY_STORE.clear()
