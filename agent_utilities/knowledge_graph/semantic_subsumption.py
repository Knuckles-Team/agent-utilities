#!/usr/bin/python
"""OWL-Driven Semantic Subsumption.

CONCEPT:KG-2.16 — OWL-Driven Semantic Subsumption
Enables zero-shot ontology alignment. When a new entity is discovered, its topological
embedding is compared against existing OWL class prototypes to automatically inject
it into the correct class hierarchy.
"""

import numpy as np

from agent_utilities.models.knowledge_graph import (
    RegistryNode,
    SubsumptionAlignmentNode,
)


class SemanticSubsumptionEngine:
    """Aligns new nodes into the OWL ontology using topological similarities."""

    def __init__(self, owl_classes: dict[str, list[float]]):
        """Initializes the subsumption engine.

        Args:
            owl_classes: A dictionary mapping OWL class URIs or names to their
                prototype vector embeddings (EncPI).
        """
        self.owl_classes = owl_classes

    def _compute_cosine_similarity(
        self, vec_a: list[float] | None, vec_b: list[float] | None
    ) -> float:
        """Computes cosine similarity between two vectors."""
        if not vec_a or not vec_b:
            return 0.0

        a = np.array(vec_a)
        b = np.array(vec_b)

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def align_node_to_ontology(
        self, node: RegistryNode, threshold: float = 0.85
    ) -> SubsumptionAlignmentNode | None:
        """Determines the most probable parent OWL class for a given node.

        Args:
            node: The newly ingested node to align.
            threshold: The similarity threshold required to make a subsumption claim.

        Returns:
            A SubsumptionAlignmentNode if a highly confident match is found, else None.
        """
        if not node.embedding:
            return None

        best_class = None
        best_score = 0.0

        for owl_class, prototype_embedding in self.owl_classes.items():
            similarity = self._compute_cosine_similarity(
                node.embedding, prototype_embedding
            )

            if similarity > best_score:
                best_score = similarity
                best_class = owl_class

        if best_class and best_score >= threshold:
            alignment_node = SubsumptionAlignmentNode(
                id=f"subsumption_{node.id}_to_{best_class}",
                name=f"Subsumption: {node.name} -> {best_class}",
                source_entity_id=node.id,
                inferred_parent_class=best_class,
                confidence=best_score,
                description=f"Zero-shot alignment of {node.name} into {best_class} based on topological embedding similarity.",
            )
            return alignment_node

        return None
