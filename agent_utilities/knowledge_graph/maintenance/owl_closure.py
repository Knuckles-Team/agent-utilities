#!/usr/bin/python
"""Background OWL-RL + SHACL closure for recently-ingested knowledge.

CONCEPT:AU-KG.research.research-pipeline-runner — Background semantic closure (L2 of the layered KG)

This is the tail of the universal ingestion funnel::

    reader → structure-router → {open | schema} extraction → ontology grounding
    → **background OWL-RL/SHACL closure**

After the ingest pipeline has written concepts/facts into the LPG, this job
promotes the recently-touched nodes/edges to RDF, runs the OWL-RL reasoner to
**materialize implied edges back into the graph**, and validates the resulting
graph against the governance SHACL shapes. It is the consolidation step that
turns freshly-extracted, possibly-incomplete facts into a closed, governed
sub-graph the rest of the system can query.

The job is **best-effort and never raises**: any missing optional dependency
(``owlready2`` / ``rdflib`` / ``pyshacl``) or runtime error degrades to a clear,
structured no-op summary rather than propagating into the maintenance tick or the
MCP/REST surface that invokes it.

It reuses the existing L2 machinery rather than re-implementing it:

* :class:`~agent_utilities.knowledge_graph.core.owl_bridge.OWLBridge` for the
  promote → reason → downfeed cycle (``run_cycle`` and ``_build_rdf_graph``).
* :class:`~agent_utilities.knowledge_graph.core.shacl_validator.SHACLValidator`
  for governance validation against ``shapes/governance.shapes.ttl``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The hand-authored governance shapes live alongside the knowledge_graph package
# (knowledge_graph/shapes/governance.shapes.ttl). This module is
# knowledge_graph/maintenance/owl_closure.py, so the package root is one parent up.
_GOVERNANCE_SHAPES = (
    Path(__file__).resolve().parent.parent / "shapes" / "governance.shapes.ttl"
)

# CONCEPT:AU-KG.maintenance.canonical-ontology-library — Single canonical ontology library: one root ontology.ttl that
# imports every domain module, no divergent duplicate, no unlinked/orphan modules;
# kept valid + connected by scripts/check_ontology.py (docs/architecture/ontology_library.md).
# The single canonical bundled ontology the OWL backend reasons over
# (knowledge_graph/ontology.ttl). The owlready2 backend's _register_local_imports
# globs every sibling ``ontology*.ttl`` in this file's directory, so pointing at
# the package-root ontology loads the full domain-module set (not the old core/
# subset, which globbed only its own directory and silently dropped every domain
# module). One canonical file, one load path — see docs/architecture/ontology_library.md.
_ONTOLOGY_PATH = Path(__file__).resolve().parent.parent / "ontology.ttl"


def _empty_summary(reason: str) -> dict[str, Any]:
    """A structured no-op summary for the degraded path (deps/engine missing)."""
    return {
        "promoted": 0,
        "inferred_edges": 0,
        "conforms": True,
        "violations": [],
        "status": "skipped",
        "reason": reason,
    }


def run_closure(engine: Any, limit: int = 2000) -> dict[str, Any]:
    """Run one background OWL-RL + SHACL closure pass over the live graph.

    CONCEPT:AU-KG.research.research-pipeline-runner — promote recently-ingested nodes to RDF, materialize the
    OWL-RL closure back into the graph as inferred edges, then validate against
    the governance shapes.

    Args:
        engine: The active ``IntelligenceGraphEngine`` (exposes ``.graph`` — the
            in-memory LPG — and ``.backend`` — the durable tier, possibly ``None``).
        limit: Soft cap on how many recently-touched nodes are considered for
            promotion. Promotion eligibility (recency + importance) is enforced by
            :class:`OWLBridge`; this caps the candidate set so a background tick
            stays bounded on a large graph.

    Returns:
        Summary dict ``{promoted, inferred_edges, conforms, violations, status}``:
            * ``promoted`` — count of nodes+edges promoted to the OWL backend.
            * ``inferred_edges`` — count of implied edges materialized back into
              the graph (``downfed`` from the bridge cycle).
            * ``conforms`` — SHACL conformance against the governance shapes
              (``True`` when validation is skipped / unavailable, so the closure
              never blocks on a missing validator).
            * ``violations`` — structured SHACL violation list (empty when
              conformant or skipped).
            * ``status`` — ``"completed"`` | ``"skipped"`` | ``"error"``.

    Never raises: every failure path returns a structured summary.
    """
    graph = getattr(engine, "graph", None)
    if graph is None:
        return _empty_summary("engine has no graph")

    # CONCEPT:AU-KG.ontology.owl-closure-native — the lightweight closure reasons engine-native
    # (``client.rdf.owl_reason`` over the live graph), so it needs NO owlready2
    # backend: the bridge runs ``run_cycle(lightweight=True)`` with ``owl_backend=None``
    # and the engine materializes the OWL/RDFS+ closure. owlready2 is a true
    # last-resort fallback (used only for the full-DL cycle), kept out of this hot path.
    try:
        from ..core.owl_bridge import OWLBridge
    except Exception as exc:  # pragma: no cover - core import failure
        logger.debug("OWLBridge unavailable, skipping closure: %s", exc)
        return _empty_summary("owl bridge unavailable")

    # owl_backend stays None: the engine reasons over the live graph directly. (The
    # ontology file is no longer loaded into an owlready2 quadstore here -- the engine
    # holds the graph; pack axioms are surfaced by the bridge as Turtle.)
    owl_backend = None

    promoted = 0
    inferred_edges = 0
    try:
        bridge = OWLBridge(
            graph=graph,
            owl_backend=owl_backend,
            backend=getattr(engine, "backend", None),
        )

        # Promote → reason → downfeed. Lightweight RDFS+/OWL-RL closure is fast and
        # materializes transitive/symmetric/inverse/property-chain edges back into
        # the LPG (and, where a durable backend exists, to it). The `limit` keeps a
        # background tick bounded; we apply it by tightening the bridge's candidate
        # window when the graph is large.
        _apply_candidate_limit(bridge, graph, limit)

        stats = bridge.run_cycle(lightweight=True)
        promoted = int(stats.get("promoted_nodes", 0)) + int(
            stats.get("promoted_edges", 0)
        )
        inferred_edges = int(stats.get("downfed", 0))
    except Exception as exc:  # noqa: BLE001 - best-effort closure, never raise
        logger.warning("OWL closure reasoning failed: %s", exc)
        _close_backend(owl_backend)
        return {
            "promoted": promoted,
            "inferred_edges": inferred_edges,
            "conforms": True,
            "violations": [],
            "status": "error",
            "reason": str(exc),
        }

    # SHACL governance validation over the materialized RDF graph. Reuses the
    # bridge's RDF materialization (which now includes the inferred edges) and the
    # governance shapes. A missing pyshacl / shapes file degrades to conforms=True.
    conforms = True
    violations: list[dict[str, Any]] = []
    try:
        conforms, violations = _validate_governance(bridge)
    except Exception as exc:  # noqa: BLE001 - validation must never block closure
        logger.debug("SHACL governance validation skipped: %s", exc)
    finally:
        _close_backend(owl_backend)

    summary = {
        "promoted": promoted,
        "inferred_edges": inferred_edges,
        "conforms": conforms,
        "violations": violations,
        "status": "completed",
    }
    logger.info(
        "OWL closure pass complete: %s", {**summary, "violations": len(violations)}
    )
    return summary


def _apply_candidate_limit(bridge: Any, graph: Any, limit: int) -> None:
    """Bound the promotion candidate set on a large graph.

    :class:`OWLBridge` already filters candidates by recency + importance; this is
    a cheap extra guard so a background tick on a very large graph does not attempt
    to promote the whole thing. When the graph is at or under ``limit`` nodes it is
    a no-op (the bridge's own eligibility filter governs). When larger, we tighten
    the recency window so only the freshest ingest is promoted — matching the
    "recently-ingested" intent. Best-effort; never raises.
    """
    try:
        node_count = graph.number_of_nodes()
    except Exception:
        try:
            node_count = len(graph.nodes)
        except Exception:
            return
    if limit > 0 and node_count > limit:
        # Tighten to the most recent day so a big-graph tick stays bounded.
        try:
            bridge.recency_days = min(getattr(bridge, "recency_days", 7), 1)
        except Exception:
            pass


def _validate_governance(bridge: Any) -> tuple[bool, list[dict[str, Any]]]:
    """Validate the materialized RDF graph against the governance shapes.

    Returns ``(conforms, violations)``. A missing pyshacl / rdflib / shapes file
    yields ``(True, [])`` (the validator itself degrades gracefully).
    """
    from ..core.shacl_validator import SHACLValidator

    if not _GOVERNANCE_SHAPES.exists():
        logger.debug("Governance shapes not found: %s", _GOVERNANCE_SHAPES)
        return True, []

    rdf_graph = bridge._build_rdf_graph()
    report = SHACLValidator().validate(rdf_graph, _GOVERNANCE_SHAPES)
    return bool(report.get("conforms", True)), list(report.get("violations", []))


def _close_backend(owl_backend: Any) -> None:
    """Best-effort close of the OWL backend handle."""
    if owl_backend is None:
        return
    try:
        owl_backend.close()
    except Exception:  # pragma: no cover - best-effort cleanup
        pass
