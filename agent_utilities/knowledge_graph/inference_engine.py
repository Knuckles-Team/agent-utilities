#!/usr/bin/python
from __future__ import annotations

"""Inference Engine for Knowledge Graph.

Provides lightweight rule evaluation logic to derive new facts automatically.
"""

import logging
import time

from .engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Evaluates inference rules against the Knowledge Graph to derive new facts."""

    def __init__(self, engine: IntelligenceGraphEngine):
        self.engine = engine

    def run_inference(self) -> int:
        """Run standard inference rules over the graph.

        Returns:
            The number of newly inferred relationships.
        """
        new_inferences = 0

        if self.engine.backend:
            # 1. Transitive inference: (A)-[:DEPENDS_ON]->(B)-[:DEPENDS_ON]->(C) => (A)-[:DEPENDS_ON_INDIRECT]->(C)
            transitive_query = """
            MATCH (a)-[:DEPENDS_ON]->(b)-[:DEPENDS_ON]->(c)
            WHERE NOT (a)-[:DEPENDS_ON_INDIRECT]->(c) AND a.id <> c.id
            MERGE (a)-[r:DEPENDS_ON_INDIRECT]->(c)
            SET r.inferred = true, r.inferred_from = 'rule_transitive_deps', r.timestamp = $ts
            RETURN count(r) as new_rels
            """

            # 2. Structural inheritance: (SubClass)-[:INHERITS_FROM]->(SuperClass)-[:HAS_METHOD]->(Method)
            # => (SubClass)-[:INHERITS_METHOD]->(Method)
            inheritance_query = """
            MATCH (sub)-[:INHERITS_FROM]->(sup)-[:HAS_METHOD]->(m)
            WHERE NOT (sub)-[:INHERITS_METHOD]->(m)
            MERGE (sub)-[r:INHERITS_METHOD]->(m)
            SET r.inferred = true, r.inferred_from = 'rule_method_inheritance', r.timestamp = $ts
            RETURN count(r) as new_rels
            """

            # 3. Collaborative inference: (Agent)-[:USES]->(Tool)<-[:USES]-(Agent2)
            # => (Agent)-[:POTENTIAL_COLLABORATOR]->(Agent2)
            collaborator_query = """
            MATCH (a1:Agent)-[:USES]->(t:CallableResource)<-[:USES]-(a2:Agent)
            WHERE a1.id <> a2.id AND NOT (a1)-[:POTENTIAL_COLLABORATOR]->(a2)
            MERGE (a1)-[r:POTENTIAL_COLLABORATOR]->(a2)
            SET r.inferred = true, r.inferred_from = 'rule_shared_tools', r.timestamp = $ts
            RETURN count(r) as new_rels
            """

            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            try:
                res1 = self.engine.backend.execute(transitive_query, {"ts": ts})
                if res1 and res1[0].get("new_rels"):
                    new_inferences += res1[0]["new_rels"]

                res2 = self.engine.backend.execute(inheritance_query, {"ts": ts})
                if res2 and res2[0].get("new_rels"):
                    new_inferences += res2[0]["new_rels"]

                res3 = self.engine.backend.execute(collaborator_query, {"ts": ts})
                if res3 and res3[0].get("new_rels"):
                    new_inferences += res3[0]["new_rels"]

                logger.info(
                    f"InferenceEngine (Cypher): Derived {new_inferences} new facts."
                )
            except Exception as e:
                logger.error(f"Inference execution failed: {e}")

        else:
            # NetworkX fallback
            logger.info(
                "InferenceEngine (NetworkX Fallback): Running topological inference."
            )
            import networkx as nx

            # Simple transitive closure for DEPENDS_ON
            depends_edges = [
                (u, v)
                for u, v, d in self.engine.graph.edges(data=True)
                if d.get("type") == "DEPENDS_ON"
            ]
            logger.info(f"NX Fallback: Found {len(depends_edges)} DEPENDS_ON edges.")
            temp_graph = nx.DiGraph()
            temp_graph.add_edges_from(depends_edges)
            logger.info(
                f"NX Fallback: temp_graph has {temp_graph.number_of_nodes()} nodes and {temp_graph.number_of_edges()} edges."
            )

            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            # Use NetworkX to find paths of length 2
            for n in temp_graph.nodes():
                for succ in temp_graph.successors(n):
                    for succ2 in temp_graph.successors(succ):
                        if n != succ2 and not self.engine.graph.has_edge(n, succ2):
                            self.engine.link_nodes(
                                n,
                                succ2,
                                "DEPENDS_ON_INDIRECT",
                                {
                                    "inferred": True,
                                    "inferred_from": "rule_transitive_deps",
                                    "timestamp": ts,
                                },
                            )
                            new_inferences += 1

            logger.info(
                f"InferenceEngine (NetworkX): Derived {new_inferences} new facts."
            )

        return new_inferences
