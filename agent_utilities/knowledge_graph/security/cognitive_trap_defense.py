#!/usr/bin/env python3
"""Zero-Shot Cognitive Trap Defense.

Implements CONCEPT:KG-2.3 (Cognitive Trap Defense)
Uses topological isomorphism via the TopologicalAnalogyEngine to detect and neutralize adversarial subgraphs.
"""

import logging

import networkx as nx

from ..core.analogy_engine import TopologicalAnalogyEngine
from ..core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class CognitiveTrapDefense:
    """Defense mechanism against adversarial cognitive traps in the knowledge graph."""

    def __init__(self, engine: IntelligenceGraphEngine):
        self.engine = engine
        self.analogy_engine = TopologicalAnalogyEngine(engine.graph)

        # Define known trap signatures (malicious subgraphs)
        self.known_traps: list[nx.DiGraph] = self._load_trap_signatures()

    def _load_trap_signatures(self) -> list[nx.DiGraph]:
        """Load predefined topological signatures of known cognitive traps."""
        traps = []
        # Example Trap 1: A cyclic dependency trap
        g1 = nx.DiGraph()
        g1.add_edge("A", "B", type="DEPENDS_ON")
        g1.add_edge("B", "C", type="DEPENDS_ON")
        g1.add_edge("C", "A", type="DEPENDS_ON")
        traps.append(g1)

        # Example Trap 2: A dense sybil cluster trap
        g2 = nx.DiGraph()
        g2.add_edge("Center", "Fake1", type="VALIDATES")
        g2.add_edge("Center", "Fake2", type="VALIDATES")
        g2.add_edge("Center", "Fake3", type="VALIDATES")
        g2.add_edge("Fake1", "Fake2", type="AGREES_WITH")
        g2.add_edge("Fake2", "Fake3", type="AGREES_WITH")
        g2.add_edge("Fake3", "Fake1", type="AGREES_WITH")
        traps.append(g2)

        return traps

    def scan_for_traps(self, target_subgraph: nx.DiGraph | None = None) -> list[dict]:
        """Scan the graph or a specific subgraph for topological isomorphisms matching known traps.

        Returns:
            List of detected trap instances with their node mappings.
        """
        graph_to_scan = (
            target_subgraph if target_subgraph is not None else self.engine.graph
        )

        detected_traps = []
        for i, trap_sig in enumerate(self.known_traps):
            # Use the VF2 matcher from analogy_engine to find isomorphisms
            matcher = nx.algorithms.isomorphism.DiGraphMatcher(
                graph_to_scan,
                trap_sig,
                edge_match=lambda e1, e2: (
                    e1.get("type") == e2.get("type") if "type" in e2 else True
                ),
            )

            for mapping in matcher.subgraph_isomorphisms_iter():
                logger.warning(
                    f"Cognitive Trap Signature {i} detected! Mapping: {mapping}"
                )
                detected_traps.append({"trap_id": i, "mapping": mapping})

        return detected_traps

    def neutralize_traps(self) -> int:
        """Scan for and neutralize all detected traps in the active graph by severing their edges.

        Returns:
            Number of neutralized traps.
        """
        traps = self.scan_for_traps()
        neutralized_count = 0

        for trap in traps:
            mapping = trap["mapping"]
            # To neutralize, we remove the nodes involved in the trap mapping
            # (or sever their edges to quarantine them)
            nodes_to_quarantine = list(mapping.keys())
            for node in nodes_to_quarantine:
                if node in self.engine.graph:
                    self.engine.graph.remove_node(node)
                    # Sync with backend if available
                    if self.engine.backend:
                        self.engine.backend.execute(
                            "MATCH (n {id: $id}) DETACH DELETE n", {"id": node}
                        )
            logger.info(f"Neutralized trap involving nodes: {nodes_to_quarantine}")
            neutralized_count += 1

        return neutralized_count
