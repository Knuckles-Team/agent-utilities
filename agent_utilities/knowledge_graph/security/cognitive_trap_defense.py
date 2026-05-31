#!/usr/bin/env python3
"""Zero-Shot Cognitive Trap Defense.

Implements CONCEPT:KG-2.3 (Cognitive Trap Defense)
Uses topological isomorphism via graph primitives to detect and neutralize adversarial subgraphs.
"""

import logging
from typing import Any

from agent_utilities.knowledge_graph.core import graph_primitives as rx

from ..core.analogy_engine import TopologicalAnalogyEngine
from ..core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class CognitiveTrapDefense:
    """Defense mechanism against adversarial cognitive traps in the knowledge graph."""

    def __init__(self, engine: IntelligenceGraphEngine):
        self.engine = engine
        self.analogy_engine = TopologicalAnalogyEngine(engine.graph)

        # Define known trap signatures (malicious subgraphs)
        self.known_traps: list[rx.PyDiGraph] = self._load_trap_signatures()

    def _load_trap_signatures(self) -> list[rx.PyDiGraph]:
        """Load predefined topological signatures of known cognitive traps."""
        traps: list[rx.PyDiGraph] = []
        # Example Trap 1: A cyclic dependency trap
        g1: rx.PyDiGraph = rx.PyDiGraph()
        a = g1.add_node("A")
        b = g1.add_node("B")
        c = g1.add_node("C")
        g1.add_edge(a, b, {"type": "DEPENDS_ON"})
        g1.add_edge(b, c, {"type": "DEPENDS_ON"})
        g1.add_edge(c, a, {"type": "DEPENDS_ON"})
        traps.append(g1)

        # Example Trap 2: A dense sybil cluster trap
        g2: rx.PyDiGraph = rx.PyDiGraph()
        center = g2.add_node("Center")
        f1 = g2.add_node("Fake1")
        f2 = g2.add_node("Fake2")
        f3 = g2.add_node("Fake3")
        g2.add_edge(center, f1, {"type": "VALIDATES"})
        g2.add_edge(center, f2, {"type": "VALIDATES"})
        g2.add_edge(center, f3, {"type": "VALIDATES"})
        g2.add_edge(f1, f2, {"type": "AGREES_WITH"})
        g2.add_edge(f2, f3, {"type": "AGREES_WITH"})
        g2.add_edge(f3, f1, {"type": "AGREES_WITH"})
        traps.append(g2)

        return traps

    def scan_for_traps(self, target_subgraph: Any | None = None) -> list[dict]:
        """Scan the graph or a specific subgraph for topological isomorphisms matching known traps.

        Returns:
            List of detected trap instances with their node mappings.
        """
        detected_traps: list[dict] = []

        # Build a rustworkx DiGraph from the GCE for isomorphism checking
        scan_graph: rx.PyDiGraph = rx.PyDiGraph()
        node_map: dict[str, int] = {}

        if target_subgraph is not None:
            # Assume target_subgraph is already a rx.PyDiGraph
            scan_graph = target_subgraph
        else:
            # Build from GCE
            for node_id in self.engine.graph.node_ids():
                idx = scan_graph.add_node(node_id)
                node_map[node_id] = idx
            for src, tgt in self.engine.graph._get_all_edges():
                if src in node_map and tgt in node_map:
                    scan_graph.add_edge(node_map[src], node_map[tgt], {})

        for i, trap_sig in enumerate(self.known_traps):
            mappings = rx.vf2_mapping(scan_graph, trap_sig, subgraph=True)
            for mapping in mappings:
                resolved_mapping = {}
                for p_idx, g_idx in mapping.items():
                    resolved_mapping[trap_sig[p_idx]] = scan_graph[g_idx]
                logger.warning(
                    f"Cognitive Trap Signature {i} detected in graph: {resolved_mapping}"
                )
                detected_traps.append({"trap_id": i, "mapping": resolved_mapping})

        return detected_traps

    def neutralize_traps(self) -> int:
        """Scan for and neutralize all detected traps in the active graph by severing their edges.

        Returns:
            Number of neutralized traps.
        """
        traps = self.scan_for_traps()
        neutralized_count = 0

        for trap in traps:
            mapping = trap.get("mapping", {})
            if "Center" in mapping:
                center_node = mapping["Center"]
                if center_node in self.engine.graph:
                    self.engine.graph.remove_node(center_node)
            elif "A" in mapping:
                node_a = mapping["A"]
                if node_a in self.engine.graph:
                    self.engine.graph.remove_node(node_a)
            neutralized_count += 1

        return neutralized_count
