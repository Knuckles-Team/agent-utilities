#!/usr/bin/python
from __future__ import annotations

"""OntologyReasoningDriver — reasoning AS the research engine (CONCEPT:KG-2.79).

The "magical power" of OWL/RDF is *extrapolating new relationships* via reasoning.
This driver makes that the engine of research: it promotes the Loop's working set
(and the surrounding ecosystem subgraph — agents/services/research are ONE ontology)
into the OWL layer, runs the reasoner (:meth:`OWLBridge.run_cycle`), and **harvests
the newly-inferred relationships** — the relationships that did NOT exist before
reasoning — turning them into research outputs (heuristics/concepts) AND fresh Loop
topics for the next iteration. A closed extrapolation loop over the whole ecosystem.

Until now the research pipeline only ran ``_run_owl_enrichment`` as a one-shot side
effect and **never consumed** the inferences. This driver closes that gap: the Loop
engine calls ``extrapolate`` each cycle (research AND develop/skill kinds), reuses
the existing :class:`OWLBridge` (promote → reason → downfeed; transitive/symmetric/
inverse/domain-range/property-chain + full-DL), and feeds the harvest back. Deps are
injectable so the logic is unit-testable without a live OWL backend.

Concept: ontology-reasoning-driver
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: cap on how many inferred relationships become fresh research topics per cycle.
MAX_HARVEST_TOPICS = 25


@dataclass
class InferenceHarvest:
    """What one reasoning pass extrapolated (CONCEPT:KG-2.79)."""

    stats: dict[str, Any] = field(default_factory=dict)
    #: relationships the reasoner inferred THIS pass (not present before).
    inferred_edges: list[dict[str, Any]] = field(default_factory=list)
    #: fresh research-Loop topics surfaced by cross-ecosystem inferences.
    new_topics: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""


class OntologyReasoningDriver:
    """Run OWL/RDF reasoning over the ecosystem and harvest the extrapolation."""

    def __init__(
        self,
        engine: Any,
        *,
        bridge: Any = None,
        lightweight: bool = True,
        max_topics: int = MAX_HARVEST_TOPICS,
    ) -> None:
        self._engine = engine
        self._bridge = bridge  # injectable for tests
        self._lightweight = lightweight
        self._max_topics = max_topics

    # -- lazy bridge (no OWL backend built until first real use) ------------ #
    def _get_bridge(self) -> Any:
        if self._bridge is not None:
            return self._bridge
        from ...backends.owl import create_owl_backend
        from ...core.owl_bridge import OWLBridge

        self._bridge = OWLBridge(
            graph=getattr(self._engine, "graph", None),
            owl_backend=create_owl_backend(),
            backend=getattr(self._engine, "backend", None),
        )
        return self._bridge

    # -- core ------------------------------------------------------------- #
    def extrapolate(
        self,
        *,
        persist: bool = True,
        topic_filter: Callable[[dict[str, Any]], bool] | None = None,
    ) -> InferenceHarvest:
        """Promote → reason → harvest the newly-inferred relationships.

        Snapshots the inferred-edge set, runs one reasoning cycle, then diffs to
        get exactly the relationships reasoning *extrapolated*. Cross-domain
        inferences (a research Concept newly linked to an ecosystem node — a
        service/agent/capability/code) become fresh research topics. Best-effort:
        a missing/failing OWL backend yields an empty harvest, never an error to
        the Loop. (CONCEPT:KG-2.79)
        """
        graph = getattr(self._engine, "graph", None)
        if graph is None:
            return InferenceHarvest(error="no graph")
        before = self._inferred_keys(graph)
        try:
            stats = self._get_bridge().run_cycle(lightweight=self._lightweight)
        except Exception as e:  # noqa: BLE001 — reasoning never blocks the loop
            logger.debug("reasoning cycle failed: %s", e)
            return InferenceHarvest(error=str(e))

        after_edges = self._inferred_edges(graph)
        new = [e for e in after_edges if self._key(e) not in before]
        topics = self._topics_from(new)
        if topic_filter is not None:
            topics = [t for t in topics if topic_filter(t)]
        topics = topics[: self._max_topics]
        if persist and topics:
            self._persist_topics(topics)
        return InferenceHarvest(
            stats=stats if isinstance(stats, dict) else {},
            inferred_edges=new,
            new_topics=topics,
        )

    # -- helpers ---------------------------------------------------------- #
    @staticmethod
    def _key(e: dict[str, Any]) -> tuple[str, str, str]:
        return (str(e.get("src")), str(e.get("dst")), str(e.get("type")))

    def _inferred_edges(self, graph: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        edges_fn = getattr(graph, "edges", None)
        if not callable(edges_fn):
            return out
        try:
            it = edges_fn(data=True)
        except (TypeError, AttributeError):  # pragma: no cover - non-standard graph
            return out
        for edge in it:
            if not (isinstance(edge, tuple | list) and len(edge) >= 3):
                continue
            src, dst, data = edge[0], edge[1], edge[2]
            if isinstance(data, dict) and data.get("inferred"):
                out.append(
                    {"src": src, "dst": dst, "type": data.get("type", ""), "data": data}
                )
        return out

    def _inferred_keys(self, graph: Any) -> set[tuple[str, str, str]]:
        return {self._key(e) for e in self._inferred_edges(graph)}

    def _topics_from(self, inferred: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Turn cross-domain inferred relationships into fresh research topics.

        A newly-inferred edge whose endpoints span different domains (a research
        node linked to an ecosystem node) is a novel relationship worth a research
        Loop. Within-domain inferences are kept as relationships but not topics.
        """
        topics: list[dict[str, Any]] = []
        seen: set[str] = set()
        for e in inferred:
            src, dst = str(e.get("src")), str(e.get("dst"))
            if self._domain(src) == self._domain(dst):
                continue  # within-domain closure — a fact, not a new research aim
            tid = f"loop:research:rel:{src}>{dst}"
            if tid in seen:
                continue
            seen.add(tid)
            topics.append(
                {
                    "id": tid,
                    "name": f"Cross-domain relationship: {src} -{e.get('type')}-> {dst}",
                    "kind": "research",
                    "source": "owl-inference",
                    "objective": (
                        f"Investigate the inferred relationship {src} "
                        f"{e.get('type')} {dst} surfaced by ecosystem reasoning."
                    ),
                }
            )
        return topics

    @staticmethod
    def _domain(node_id: str) -> str:
        """Coarse domain of a node id (research / ecosystem / code / other)."""
        nid = node_id.lower()
        for prefix, dom in (
            ("article:", "research"),
            ("concept:", "research"),
            ("loop:research", "research"),
            ("paper:", "research"),
            ("ecosystem_package", "ecosystem"),
            ("service:", "ecosystem"),
            ("node:", "ecosystem"),
            ("agent:", "ecosystem"),
            ("skill", "ecosystem"),
            ("capability", "ecosystem"),
            ("code:", "code"),
        ):
            if nid.startswith(prefix) or prefix in nid:
                return dom
        return "other"

    def _persist_topics(self, topics: list[dict[str, Any]]) -> int:
        """Materialize harvested cross-domain relationships as research Loops."""
        from .. import loops as loops_mod

        n = 0
        for t in topics:
            if loops_mod.submit_loop(
                self._engine,
                t["objective"],
                kind="research",
                source="owl-inference",
                loop_id=t["id"],
            ):
                n += 1
        return n


__all__ = ["InferenceHarvest", "OntologyReasoningDriver", "MAX_HARVEST_TOPICS"]
