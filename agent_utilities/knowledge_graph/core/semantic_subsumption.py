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

    def __init__(
        self,
        owl_classes: dict[str, list[float]],
        owl_hierarchy: dict[str, list[str]] | None = None,
    ):
        """Initializes the subsumption engine.

        Args:
            owl_classes: A dictionary mapping OWL class URIs or names to their
                prototype vector embeddings (EncPI).
            owl_hierarchy: Optional dictionary mapping an OWL class to its parent classes,
                used to reconstruct full subsumption lineage.
        """
        self.owl_classes = owl_classes
        self.owl_hierarchy = owl_hierarchy or {}

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

    def _get_lineage(self, class_name: str) -> list[str]:
        """Recursively builds the subsumption lineage for a class."""
        lineage = [class_name]
        current = class_name

        # Simple cyclic-safe traversal for lineage
        visited = {current}
        while current in self.owl_hierarchy:
            parents = self.owl_hierarchy[current]
            if not parents:
                break

            # Follow the primary parent for the lineage
            parent = parents[0]
            if parent in visited:
                break

            lineage.append(parent)
            visited.add(parent)
            current = parent

        return lineage

    def align_node_to_ontology(
        self, node: RegistryNode, threshold: float = 0.85
    ) -> SubsumptionAlignmentNode | None:
        """Determines the most probable parent OWL class and its lineage for a given node.

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
            lineage = self._get_lineage(best_class)

            alignment_node = SubsumptionAlignmentNode(
                id=f"subsumption_{node.id}_to_{best_class}",
                name=f"Subsumption: {node.name} -> {best_class}",
                source_entity_id=node.id,
                inferred_parent_class=best_class,
                inferred_lineage=lineage,
                confidence=best_score,
                description=f"Hierarchy-aware zero-shot alignment of {node.name} into {best_class}. Full lineage: {' -> '.join(lineage)}",
            )
            return alignment_node

        return None
