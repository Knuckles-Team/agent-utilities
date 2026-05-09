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

            # --- Standard Ontology Rules (PROV-O, SKOS, DC, Temporal) ---

            # 4. PROV-O Derivation Chain Transitivity
            derivation_query = """
            MATCH (a)-[:WAS_DERIVED_FROM]->(b)-[:WAS_DERIVED_FROM]->(c)
            WHERE NOT (a)-[:WAS_DERIVED_FROM]->(c) AND a.id <> c.id
            MERGE (a)-[r:WAS_DERIVED_FROM]->(c)
            SET r.inferred = true, r.inferred_from = 'rule_prov_derivation_chain', r.timestamp = $ts
            RETURN count(r) as new_rels
            """

            # 5. SKOS Broader Transitivity
            broader_query = """
            MATCH (a)-[:BROADER]->(b)-[:BROADER]->(c)
            WHERE NOT (a)-[:BROADER]->(c) AND a.id <> c.id
            MERGE (a)-[r:BROADER]->(c)
            SET r.inferred = true, r.inferred_from = 'rule_skos_broader_transitive', r.timestamp = $ts
            RETURN count(r) as new_rels
            """

            # 6. Dublin Core Author-Org Linking
            author_org_query = """
            MATCH (d:Document)-[:CREATOR]->(p:Person)-[:HAS_ROLE]->(r:Role)-[:BELONGS_TO_ORGANIZATION]->(o:Organization)
            WHERE NOT (d)-[:WAS_ATTRIBUTED_TO]->(o)
            MERGE (d)-[rel:WAS_ATTRIBUTED_TO]->(o)
            SET rel.inferred = true, rel.inferred_from = 'rule_dc_author_org', rel.timestamp = $ts
            RETURN count(rel) as new_rels
            """

            # 7. Temporal Phase Containment
            phase_containment_query = """
            MATCH (e)-[:OCCURRED_DURING]->(p:Phase)-[:PART_OF]->(pp:Phase)
            WHERE NOT (e)-[:OCCURRED_DURING]->(pp) AND e.id <> pp.id
            MERGE (e)-[r:OCCURRED_DURING]->(pp)
            SET r.inferred = true, r.inferred_from = 'rule_temporal_phase_containment', r.timestamp = $ts
            RETURN count(r) as new_rels
            """

            try:
                for label, query in [
                    ("prov_derivation", derivation_query),
                    ("skos_broader", broader_query),
                    ("dc_author_org", author_org_query),
                    ("phase_containment", phase_containment_query),
                ]:
                    res = self.engine.backend.execute(query, {"ts": ts})
                    if res and res[0].get("new_rels"):
                        count = res[0]["new_rels"]
                        new_inferences += count
                        logger.info(
                            "InferenceEngine (Cypher): %s rule derived %d facts.",
                            label,
                            count,
                        )
            except Exception as e:
                logger.error(f"Standard ontology inference failed: {e}")

        else:
            # NetworkX fallback
            logger.info(
                "InferenceEngine (NetworkX Fallback): Running topological inference."
            )
            import networkx as nx

            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            # Helper for transitive closure on a given edge type
            def _transitive_closure(
                edge_type: str, inferred_type: str, rule_name: str
            ) -> int:
                edges = [
                    (u, v)
                    for u, v, d in self.engine.graph.edges(data=True)
                    if d.get("type") == edge_type
                ]
                if not edges:
                    return 0
                temp = nx.DiGraph()
                temp.add_edges_from(edges)
                count = 0
                for n in temp.nodes():
                    for succ in temp.successors(n):
                        for succ2 in temp.successors(succ):
                            if n != succ2 and not self.engine.graph.has_edge(n, succ2):
                                self.engine.link_nodes(
                                    n,
                                    succ2,
                                    inferred_type,
                                    {
                                        "inferred": True,
                                        "inferred_from": rule_name,
                                        "timestamp": ts,
                                    },
                                )
                                count += 1
                return count

            # 1. DEPENDS_ON transitive closure
            new_inferences += _transitive_closure(
                "DEPENDS_ON", "DEPENDS_ON_INDIRECT", "rule_transitive_deps"
            )

            # 2. PROV-O WAS_DERIVED_FROM transitive closure
            new_inferences += _transitive_closure(
                "was_derived_from", "was_derived_from", "rule_prov_derivation_chain"
            )

            # 3. SKOS BROADER transitive closure
            new_inferences += _transitive_closure(
                "broader", "broader", "rule_skos_broader_transitive"
            )

            # 4. Temporal phase containment
            occurred_edges = [
                (u, v)
                for u, v, d in self.engine.graph.edges(data=True)
                if d.get("type") == "occurred_during"
            ]
            part_of_edges = [
                (u, v)
                for u, v, d in self.engine.graph.edges(data=True)
                if d.get("type") == "part_of"
            ]
            if occurred_edges and part_of_edges:
                for event, phase in occurred_edges:
                    for p, parent in part_of_edges:
                        if p == phase and event != parent:
                            if not self.engine.graph.has_edge(event, parent):
                                self.engine.link_nodes(
                                    event,
                                    parent,
                                    "occurred_during",
                                    {
                                        "inferred": True,
                                        "inferred_from": "rule_temporal_phase_containment",
                                        "timestamp": ts,
                                    },
                                )
                                new_inferences += 1

            logger.info(
                f"InferenceEngine (NetworkX): Derived {new_inferences} new facts."
            )

        return new_inferences
