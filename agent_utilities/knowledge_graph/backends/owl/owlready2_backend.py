#!/usr/bin/python
"""Owlready2 OWL Backend.

Default in-memory + optional SQLite persistence backend using Owlready2
and its bundled HermiT/Pellet reasoner.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .base import OWLBackend

logger = logging.getLogger(__name__)

# Mapping from LPG RegistryNodeType values to OWL class local names
_NODE_TYPE_TO_OWL_CLASS: dict[str, str] = {
    "agent": "Agent",
    "tool": "Tool",
    "skill": "Skill",
    "file": "File",
    "symbol": "Symbol",
    "module": "Module",
    "memory": "Memory",
    "event": "Event",
    "episode": "Episode",
    "phase": "Phase",
    "incident": "Incident",
    "decision": "Decision",
    "observation": "Observation",
    "action": "Action",
    "belief": "Belief",
    "hypothesis": "Hypothesis",
    "fact": "Fact",
    "principle": "Principle",
    "concept": "Concept",
    "evidence": "Evidence",
    "reflection": "Reflection",
    "organization": "Organization",
    "person": "Person",
    "role": "Role",
    "place": "Place",
    "system": "System",
    "team": "Team",
    "reasoning_trace": "ReasoningTrace",
    "outcome_evaluation": "OutcomeEvaluation",
    "critique": "Critique",
    "goal": "Goal",
    "policy": "Policy",
    "server": "Server",
    "code": "Code",
    # Standard Ontology Types (BFO, Schema.org, DC, FIBO)
    "document": "Document",
    "creative_work": "CreativeWork",
    "dataset": "Dataset",
    "software_project": "SoftwareProject",
    "medical_entity": "MedicalEntity",
    "procedure": "Procedure",
    "regulation": "Regulation",
    "financial_instrument": "FinancialInstrument",
    "financial_transaction": "FinancialTransaction",
    "account": "Account",
    # AHE Types (CONCEPT:AU-012)
    "change_manifest": "ChangeManifest",
    "component_edit_record": "ComponentEditRecord",
    "evidence_record": "EvidenceRecord",
    "constraint_state": "ConstraintState",
    # Backfill Gap Types
    "task": "Task",
    "codemap": "Codemap",
    "pattern_template": "PatternTemplate",
    "proposed_skill": "ProposedSkill",
    "system_prompt": "SystemPromptTemplate",
    "prompt": "Prompt",
    "process_flow": "ProcessFlow",
    "process_step": "ProcessStep",
    "knowledge_base": "KnowledgeBase",
    "knowledge_base_topic": "KnowledgeBaseTopic",
    "experiment": "Experiment",
}

# Mapping from LPG edge type values to OWL object property local names
_EDGE_TYPE_TO_OWL_PROP: dict[str, str] = {
    "inherits_from": "inheritsFrom",
    "depends_on": "dependsOn",
    "imports": "imports",
    "provides": "provides",
    "part_of": "partOf",
    "contains": "contains",
    "triggered_by": "triggeredBy",
    "supports_belief": "supportsBeliefProp",
    "contradicts_belief": "contradictsBeliefProp",
    "owns_system": "ownedBy",
    "depends_on_system": "dependsOn",
    "has_role": "hasRole",
    "motivated_by": "motivatedBy",
    "produced_outcome": "producedOutcome",
    "triggered_action": "triggeredAction",
    "observed_by": "observedBy",
    "occurred_during": "occurredDuring",
    "defined_in": "definedIn",
    # Standard Ontology Edges (PROV-O, SKOS, Dublin Core, FIBO)
    "was_generated_by": "wasGeneratedBy",
    "was_derived_from": "wasDerivedFrom",
    "was_attributed_to": "wasAttributedTo",
    "has_temporal_extent": "hasTemporalExtent",
    "broader": "broader",
    "narrower": "narrower",
    "related_concept": "related",
    "exact_match": "exactMatch",
    "close_match": "closeMatch",
    "broad_match": "broadMatch",
    "creator": "creator",
    "cites_source": "citesSource",
    "has_financial_instrument": "hasFinancialInstrument",
    "executed_transaction": "executedTransaction",
    # AHE Edges (CONCEPT:AU-012)
    "edited_in_round": "editedInRound",
    "predicted_fix": "predictedFix",
    "caused_regression": "causedRegression",
    "confirmed_fix": "confirmedFix",
    "verified_by": "verifiedBy",
    "escalated_to": "escalatedTo",
    "applied_edit": "appliedEdit",
    "has_edit_for": "hasEditFor",
}


class Owlready2Backend(OWLBackend):
    """Owlready2 backend with optional SQLite quadstore persistence.

    Environment Variables:
        OWL_DB_PATH: SQLite quadstore file path. If unset, runs in-memory.
    """

    def __init__(
        self,
        ontology_path: str | None = None,
        db_path: str | None = None,
    ):
        try:
            import owlready2  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Owlready2 backend requires the owlready2 package. "
                "Install with: pip install owlready2"
            ) from e

        self._db_path = db_path or os.environ.get("OWL_DB_PATH")
        self._onto: Any = None
        self._world: Any = None
        self._pre_reason_facts: set[tuple[str, str, str]] = set()
        self._inferences: list[dict[str, Any]] = []

        self._init_world()

        if ontology_path:
            self.load_ontology(ontology_path)

    def _init_world(self) -> None:
        """Initialize owlready2 World (isolated store)."""
        import owlready2

        if self._db_path:
            self._world = owlready2.World(filename=self._db_path)
        else:
            self._world = owlready2.World()

    def load_ontology(self, ontology_path: str) -> None:
        """Load an OWL/Turtle ontology into the backend."""
        path = Path(ontology_path)
        if not path.exists():
            raise FileNotFoundError(f"Ontology file not found: {ontology_path}")

        try:
            # Try native load first
            if self._world is not None:
                self._onto = self._world.get_ontology(path.as_uri()).load()
            else:
                raise RuntimeError("owlready2 World not initialized")
        except Exception as e:
            logger.debug(
                "Native owlready2 load failed: %s. Trying rdflib conversion...", e
            )
            try:
                from tempfile import NamedTemporaryFile

                import rdflib

                g = rdflib.Graph()
                g.parse(str(path), format="turtle")

                with NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
                    tmp_path = tmp.name
                    g.serialize(destination=tmp_path, format="pretty-xml")

                try:
                    self._onto = self._world.get_ontology(
                        Path(tmp_path).as_uri()
                    ).load()
                finally:
                    os.unlink(tmp_path)
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load ontology from {ontology_path}: {e2}"
                ) from e2

        logger.info("Loaded ontology from %s", ontology_path)

    def _get_owl_class(self, node_type: str):
        """Resolve a LPG node type string to an owlready2 class."""
        class_name = _NODE_TYPE_TO_OWL_CLASS.get(node_type)
        if not class_name or not self._onto:
            return None
        return getattr(self._onto, class_name, None)

    def _get_owl_property(self, edge_type: str):
        """Resolve a LPG edge type string to an owlready2 object property."""
        prop_name = _EDGE_TYPE_TO_OWL_PROP.get(edge_type)
        if not prop_name or not self._onto:
            return None
        return getattr(self._onto, prop_name, None)

    def _safe_id(self, node_id: str) -> str:
        """Sanitize a node ID for use as an OWL individual name."""
        return node_id.replace(":", "_").replace("/", "_").replace(".", "_")

    def promote(self, stable_nodes: list[dict[str, Any]]) -> int:
        """Create OWL individuals from stable LPG nodes."""
        if not self._onto:
            logger.warning("No ontology loaded; skipping promotion")
            return 0

        count = 0
        for node in stable_nodes:
            node_type = node.get("type", "")
            owl_class = self._get_owl_class(node_type)
            if owl_class is None:
                logger.debug("No OWL class for node type '%s', skipping", node_type)
                continue

            safe_id = self._safe_id(node.get("id", ""))
            if not safe_id:
                continue

            # Create or retrieve individual
            individual = owl_class(safe_id, namespace=self._onto)

            # Map datatype properties
            if hasattr(self._onto, "confidence") and "confidence" in node:
                try:
                    individual.confidence = [float(node["confidence"])]
                except (ValueError, TypeError):
                    pass
            if hasattr(self._onto, "importance") and "importance_score" in node:
                try:
                    individual.importance = [float(node["importance_score"])]
                except (ValueError, TypeError):
                    pass

            count += 1

        logger.info("Promoted %d nodes as OWL individuals", count)
        return count

    def _promote_dc_properties(self, individual: Any, node: dict[str, Any]) -> None:
        """Map Dublin Core datatype properties to OWL individual."""
        dc_mappings = {
            "title": "title",
            "creator": "creator",
            "date": "dateCreated",
            "subject": "subject",
            "identifier": "identifier",
            "language": "language",
            "format": "format",
        }
        for node_key, owl_prop_name in dc_mappings.items():
            if (
                node_key in node
                and node[node_key]
                and hasattr(self._onto, owl_prop_name)
            ):
                try:
                    getattr(individual, owl_prop_name).append(str(node[node_key]))
                except (ValueError, TypeError, AttributeError):
                    pass

    def promote_edges(self, edges: list[dict[str, Any]]) -> int:
        """Create OWL property assertions from stable LPG edges."""
        if not self._onto:
            return 0

        count = 0
        for edge in edges:
            prop = self._get_owl_property(edge.get("type", ""))
            if prop is None:
                continue

            src_id = self._safe_id(edge.get("source", ""))
            tgt_id = self._safe_id(edge.get("target", ""))
            if not src_id or not tgt_id:
                continue

            src_individual = self._onto.search_one(iri=f"*{src_id}")
            tgt_individual = self._onto.search_one(iri=f"*{tgt_id}")

            if src_individual and tgt_individual:
                if tgt_individual not in prop[src_individual]:
                    prop[src_individual].append(tgt_individual)
                    count += 1

        logger.info("Promoted %d edges as OWL property assertions", count)
        return count

    def _snapshot_facts(self) -> set[tuple[str, str, str]]:
        """Take a snapshot of all current triples for diff-based inference."""
        facts: set[tuple[str, str, str]] = set()
        if not self._onto:
            return facts

        for individual in self._onto.individuals():
            for prop in individual.get_properties():
                for value in prop[individual]:
                    obj_str = value.name if hasattr(value, "name") else str(value)
                    facts.add((individual.name, prop.python_name, obj_str))
        return facts

    def reason(self) -> list[dict[str, Any]]:
        """Run OWL DL reasoning and return newly inferred facts."""
        if not self._onto:
            logger.warning("No ontology loaded; skipping reasoning")
            return []

        # Snapshot before reasoning
        pre_facts = self._snapshot_facts()

        # Run reasoner
        try:
            from owlready2 import sync_reasoner_hermit

            with self._onto:
                sync_reasoner_hermit(self._world, infer_property_values=True)
        except Exception as e:
            logger.error("OWL reasoning failed: %s", e)
            return []

        # Snapshot after reasoning
        post_facts = self._snapshot_facts()

        # Diff to find new inferred facts
        new_facts = post_facts - pre_facts
        self._inferences = [
            {
                "subject": s,
                "predicate": p,
                "object": o,
                "inference_type": "owl_hermit",
            }
            for s, p, o in new_facts
        ]

        logger.info("OWL reasoning produced %d new inferences", len(self._inferences))
        return self._inferences

    def get_inferences(self) -> list[dict[str, Any]]:
        """Return cached inferences from the last reasoning run."""
        return self._inferences

    def export_rdf(self, output_path: str, fmt: str = "turtle") -> None:
        """Export ontology + ABox to RDF file."""
        if not self._onto:
            logger.warning("No ontology loaded; nothing to export")
            return

        try:
            import rdflib

            g = rdflib.Graph()
            # Owlready2 can save to RDF/XML natively
            temp_path = output_path + ".tmp.rdf"
            self._onto.save(file=temp_path, format="rdfxml")
            g.parse(temp_path, format="xml")
            Path(temp_path).unlink(missing_ok=True)

            # Serialize in requested format
            fmt_map = {
                "turtle": "turtle",
                "ttl": "turtle",
                "xml": "xml",
                "ntriples": "nt",
                "nt": "nt",
                "json-ld": "json-ld",
            }
            rdflib_fmt = fmt_map.get(fmt.lower(), "turtle")
            g.serialize(destination=output_path, format=rdflib_fmt)
            logger.info("Exported ontology to %s (format=%s)", output_path, rdflib_fmt)
        except ImportError:
            # Fallback: use owlready2 native save
            self._onto.save(file=output_path, format="rdfxml")
            logger.info(
                "Exported ontology to %s (format=rdfxml, rdflib unavailable)",
                output_path,
            )

    def clear(self) -> None:
        """Remove all ABox individuals, preserving the TBox."""
        if not self._onto:
            return

        from owlready2 import destroy_entity

        for individual in list(self._onto.individuals()):
            destroy_entity(individual)

        self._inferences = []
        logger.info("Cleared all OWL individuals (TBox preserved)")

    def close(self) -> None:
        """Release resources."""
        if self._world:
            try:
                self._world.close()
            except Exception as e:
                logger.debug("Failed to close owlready2 world: %s", e)
        self._onto = None
        self._world = None

    def get_stats(self) -> dict[str, int]:
        """Return counts of key ontology elements."""
        if not self._onto:
            return {"individuals": 0, "classes": 0, "properties": 0}

        return {
            "individuals": len(list(self._onto.individuals())),
            "classes": len(list(self._onto.classes())),
            "properties": len(list(self._onto.properties())),
        }
