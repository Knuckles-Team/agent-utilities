#!/usr/bin/python
from __future__ import annotations

"""Owlready2 OWL Backend.

Default in-memory + optional SQLite persistence backend using Owlready2
and its bundled HermiT/Pellet reasoner.
"""


import logging
import os
from pathlib import Path
from typing import Any

from agent_utilities.core.config import setting

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
    # Ontology Action System — governed verbs (CONCEPT:KG-2.25)
    "ontology_action": "OntologyAction",
    "action_invocation": "ActionInvocation",
    "action_parameter": "ActionParameter",
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
    # AHE Types (CONCEPT:AHE-3.0)
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
    # Enterprise OS — canonical ArchiMate concepts & vendor crosswalk (KG-2.9)
    "businessprocess": "BusinessProcess",
    "businesstask": "BusinessTask",
    "businesscapability": "BusinessCapability",
    "applicationevent": "ApplicationEvent",
    "erpnextissue": "ErpNextIssue",
    # Feedback loop — corrections → rules → eval (CONCEPT:KG-2.8)
    "correction": "Correction",
    "governance_rule": "GovernanceRule",
    "voice_rule": "VoiceRule",
    "source_rule": "SourceRule",
    "preference": "Preference",
    "eval_case": "EvalCase",
    # Operating intelligence distilled from calls/docs (CONCEPT:KG-2.8)
    "insight": "Insight",
    "framework": "Framework",
    "playbook": "Playbook",
    "knowledge_base": "KnowledgeBase",
    "knowledge_base_topic": "KnowledgeBaseTopic",
    "experiment": "Experiment",
    # Wellness domain (ontology_wellness.ttl)
    "meal_plan": "MealPlan",
    "meal_entry": "MealEntry",
    "recipe": "Recipe",
    "nutrient_profile": "NutrientProfile",
    "daily_nutrient_summary": "DailyNutrientSummary",
    "shopping_list": "ShoppingList",
    "ingredient": "Ingredient",
    "nutrition_target": "NutritionTarget",
    "workout_routine": "WorkoutRoutine",
    "workout_session": "WorkoutSession",
    "exercise": "Exercise",
    "exercise_set": "ExerciseSet",
    "fitness_goal": "FitnessGoal",
    "body_measurement": "BodyMeasurement",
    "wellness_score": "WellnessScore",
    "calorie_expenditure": "CalorieExpenditure",
    # Social domain (ontology_social.ttl)
    "social_post": "SocialPost",
    "content_draft": "ContentDraft",
    "content_calendar": "ContentCalendar",
    "broadcast_session": "BroadcastSession",
    "stream_highlight": "StreamHighlight",
    "engagement_metric": "EngagementMetric",
    "daily_engagement": "DailyEngagement",
    "aggregated_engagement": "AggregatedEngagement",
    "audience_metric": "AudienceMetric",
    # Personal productivity domain (ontology_personal.ttl)
    "calendar_event": "CalendarEvent",
    "recurring_event": "RecurringEvent",
    "personal_task": "PersonalTask",
    "voice_message": "VoiceMessage",
    "transcript": "Transcript",
    # Media domain (ontology_media.ttl)
    "download_job": "DownloadJob",
    "media_asset": "MediaAsset",
    "media_library": "MediaLibrary",
    "media_collection": "MediaCollection",
    # Deployment/Bootstrap domain (ontology_infrastructure.ttl extensions)
    "deployment_manifest": "DeploymentManifest",
    "deployment_phase": "DeploymentPhase",
    "deployment_prerequisite": "DeploymentPrerequisite",
    "mcp_server_deployment": "MCPServerDeployment",
    "deployment_profile": "DeploymentProfile",
    # Legal Entity & Compliance domain (CONCEPT:LGC-1.0)
    "legal_trust": "LegalTrust",
    "trustee_role": "TrusteeRole",
    "settlor_role": "SettlorRole",
    "beneficiary_role": "BeneficiaryRole",
    "legal_entity": "LegalEntity",
    "company": "Company",
    "ein_application": "EINApplication",
    # Infrastructure additions
    "host": "BladeServer",
    "container": "Container",
    "container_stack": "ContainerStack",
    "platform_service": "PlatformService",
    "gpu_accelerator": "GPUAccelerator",
    "storage_array": "StorageArray",
}

# Mapping from LPG edge type values to OWL object property local names
_EDGE_TYPE_TO_OWL_PROP: dict[str, str] = {
    "inherits_from": "inheritsFrom",
    "depends_on": "dependsOn",
    "realizes": "realizes",
    "corrects": "corrects",
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
    # Ontology Action System — governed verbs (CONCEPT:KG-2.25)
    "acts_on": "actsOn",
    "invokes": "invokes",
    "invoked_by": "invokedBy",
    "acts_on_object": "actsOnObject",
    "may_be_invoked_by": "mayBeInvokedBy",
    "requires_capability": "requiresCapability",
    "provides_capability": "providesCapability",
    # Standard Ontology Edges (PROV-O, SKOS, Dublin Core, FIBO)
    "was_generated_by": "wasGeneratedBy",
    "was_derived_from": "wasDerivedFrom",
    "derived_from": "wasDerivedFrom",
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
    # AHE Edges (CONCEPT:AHE-3.0)
    "edited_in_round": "editedInRound",
    "predicted_fix": "predictedFix",
    "caused_regression": "causedRegression",
    "confirmed_fix": "confirmedFix",
    "verified_by": "verifiedBy",
    "escalated_to": "escalatedTo",
    "applied_edit": "appliedEdit",
    "has_edit_for": "hasEditFor",
    # Cross-domain properties (ontology.ttl extensions)
    "triggered_incident": "triggeredIncident",
    "self_healed_via": "selfHealedVia",
    "resolved_incident": "resolvedIncident",
    "blocks_task": "blocksTask",
    "inspired_change": "inspiredChange",
    # Wellness domain edges
    "prescribed_diet": "prescribedDiet",
    "follows_routine": "followsRoutine",
    "has_nutrient_profile": "hasNutrientProfile",
    "aggregates_to": "aggregatesTo",
    "targets_goal": "targetsGoal",
    "tracked_measurement": "trackedMeasurement",
    "calorie_balance": "calorieBalance",
    "contains_meal": "containsMeal",
    "uses_recipe": "usesRecipe",
    "completed_session": "completedSession",
    "performed_exercise": "performedExercise",
    "estimated_expenditure": "estimatedExpenditure",
    "motivated_by_measurement": "motivatedBy",
    # Social domain edges
    "published_on": "publishedOn",
    "scheduled_in": "scheduledIn",
    "engagement_of": "engagementOf",
    "broadcast_engagement": "broadcastEngagement",
    "broadcasted_via": "broadcastedVia",
    "aggregates_daily": "aggregatesDaily",
    "promotes_research": "promotesResearch",
    "derived_from_content": "derivedFromContent",
    # Personal productivity edges
    "scheduled_for": "scheduledFor",
    "originated_from": "originatedFrom",
    "transcribed_from": "transcribedFrom",
    "spawns_task": "spawnsTask",
    "blocked_by_incident": "blockedByIncident",
    # Media domain edges
    "downloaded_via": "downloadedVia",
    "requested_by": "requestedBy",
    "produced_asset": "producedAsset",
    "managed_by": "managedBy",
    "belongs_to_collection": "belongsToCollection",
    # Deployment/Bootstrap edges
    "bootstrapped_by": "bootstrappedBy",
    "has_phase": "hasPhase",
    "deployed_in_phase": "deployedInPhase",
    "requires_prerequisite": "requiresPrerequisite",
    "serves_endpoint": "servesEndpoint",
    "uses_profile": "usesProfile",
    "transitioned_to": "transitionedTo",
    # Legal Entity & Compliance domain edges (CONCEPT:LGC-1.0)
    "has_trustee": "hasTrustee",
    "has_settlor": "hasSettlor",
    "has_beneficiary": "hasBeneficiary",
    "trust_agreement": "trustAgreement",
    "governed_by_doc": "governedByDoc",
    "filed_by": "filedBy",
    # Infrastructure property additions
    "runs_on": "runsOn",
    "has_accelerator": "hasAccelerator",
    "attached_storage": "attachedStorage",
    "deployed_on": "deployedOn",
    "belongs_to_stack": "belongsToStack",
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

        self._db_path = db_path or setting("OWL_DB_PATH")
        self._onto: Any = None
        self._world: Any = None
        self._pre_reason_facts: set[tuple[str, str, str]] = set()
        self._inferences: list[dict[str, Any]] = []

        self._init_world()

        if ontology_path:
            self.load_ontology(ontology_path)

    def _register_local_imports(self, ontology_path: Path) -> None:
        """Pre-load sibling .ttl files so owl:imports resolve locally.

        owlready2 resolves owl:imports by name-matching files in
        ``onto_path``.  Our files (``ontology_*.ttl``) don't match the
        IRI convention, so we use rdflib (which ignores owl:imports) to
        parse each sibling into the owlready2 world first.

        When ``OWL_ALLOW_REMOTE_IMPORTS=true`` is set, any import IRI
        that can't be resolved locally will still be fetched over HTTP —
        allowing users to inherit external domain-level ontologies and
        extend/override them locally.
        """
        if not self._world:
            return

        parent = ontology_path.parent
        for ttl_file in sorted(parent.glob("ontology*.ttl")):
            if ttl_file == ontology_path:
                continue
            try:
                self._preload_ttl_via_rdflib(ttl_file)
            except Exception as e:
                logger.debug(
                    "Pre-load of %s failed (non-fatal): %s",
                    ttl_file.name,
                    e,
                )

    def _preload_ttl_via_rdflib(self, ttl_file: Path) -> None:
        """Parse a Turtle file via rdflib and inject into the owlready2 world.

        rdflib ignores owl:imports directives, so this avoids recursive
        HTTP fetching while still making the ontology's classes and
        properties available for the main ontology to reference.
        """
        from tempfile import NamedTemporaryFile

        import rdflib

        g = rdflib.Graph()
        g.parse(str(ttl_file), format="turtle")

        # Strip owl:imports so owlready2 doesn't try to resolve them
        OWL = rdflib.Namespace("http://www.w3.org/2002/07/owl#")
        for s, p, o in list(g.triples((None, OWL.imports, None))):
            g.remove((s, p, o))

        with NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
            tmp_path = tmp.name
            g.serialize(destination=tmp_path, format="pretty-xml")

        try:
            self._world.get_ontology(Path(tmp_path).as_uri()).load()
        finally:
            os.unlink(tmp_path)

    def _init_world(self) -> None:
        """Initialize owlready2 World (isolated store)."""
        import owlready2

        if self._db_path:
            self._world = owlready2.World(filename=self._db_path)
        else:
            self._world = owlready2.World()

    def load_ontology(self, ontology_path: str) -> None:
        """Load an OWL/Turtle ontology into the backend.

        Import resolution strategy:

        1. All sibling ``ontology_*.ttl`` files are pre-parsed via rdflib
           (which ignores owl:imports) and loaded into the owlready2 world.
        2. The main ontology is also parsed via rdflib and loaded.
        3. If ``OWL_ALLOW_REMOTE_IMPORTS=true`` is set, any remaining
           unresolved owl:imports will be fetched over HTTP — enabling
           users to inherit external domain ontologies.
        """
        path = Path(ontology_path)
        if not path.exists():
            raise FileNotFoundError(f"Ontology file not found: {ontology_path}")

        allow_remote = setting("OWL_ALLOW_REMOTE_IMPORTS", "false").lower() == "true"

        try:
            # Pre-load sibling ontologies so owl:imports resolve locally
            self._register_local_imports(path)

            # Load the main ontology via rdflib (no remote owl:imports)
            self._preload_ttl_via_rdflib(path)

            # Grab a reference to the loaded ontology from the world
            if self._world is not None:
                for onto in self._world.ontologies.values():
                    if onto.loaded:
                        self._onto = onto
            if self._onto is None:
                raise RuntimeError("Ontology loaded but not found in world")
        except Exception as e:
            if allow_remote:
                # User opted into remote imports — try native owlready2 load
                logger.debug(
                    "Local-only load failed: %s. Retrying with remote "
                    "imports (OWL_ALLOW_REMOTE_IMPORTS=true)...",
                    e,
                )
                if self._world is not None:
                    self._onto = self._world.get_ontology(path.as_uri()).load()
                else:
                    raise RuntimeError("owlready2 World not initialized") from e
            else:
                raise RuntimeError(
                    f"Failed to load ontology from {ontology_path}. "
                    f"Set OWL_ALLOW_REMOTE_IMPORTS=true to allow remote "
                    f"owl:imports resolution: {e}"
                ) from e

        logger.info("Loaded ontology from %s", ontology_path)

    def _get_owl_class(self, node_type: str):
        """Resolve a LPG node type string to an owlready2 class."""
        class_name = _NODE_TYPE_TO_OWL_CLASS.get(node_type)
        if not class_name or not self._world:
            return None
        # We search the world because the class might be in a sibling/imported ontology
        import owlready2

        for res in self._world.search(iri=f"*{class_name}"):
            if isinstance(res, owlready2.ThingClass):
                return res
        return None

    def _get_owl_property(self, edge_type: str):
        """Resolve a LPG edge type string to an owlready2 object property."""
        prop_name = _EDGE_TYPE_TO_OWL_PROP.get(edge_type)
        if not prop_name or not self._world:
            return None
        import owlready2

        for res in self._world.search(iri=f"*{prop_name}"):
            if isinstance(res, owlready2.PropertyClass):
                return res
        return None

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
                and self._world
                and self._world.search_one(iri=f"*{owl_prop_name}")
            ):
                try:
                    getattr(individual, owl_prop_name).append(str(node[node_key]))
                except (ValueError, TypeError, AttributeError):
                    pass

    def promote_edges(self, edges: list[dict[str, Any]]) -> int:
        """Create OWL property assertions from stable LPG edges."""
        if not self._world:
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

            src_individual = self._world.search_one(iri=f"*{src_id}")
            tgt_individual = self._world.search_one(iri=f"*{tgt_id}")

            if src_individual and tgt_individual:
                if tgt_individual not in prop[src_individual]:
                    prop[src_individual].append(tgt_individual)
                    count += 1

        logger.info("Promoted %d edges as OWL property assertions", count)
        return count

    def _snapshot_facts(self) -> set[tuple[str, str, str]]:
        """Take a snapshot of all current triples for diff-based inference."""
        facts: set[tuple[str, str, str]] = set()
        if not self._world:
            return facts

        for individual in self._world.individuals():
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
        if not self._world:
            return {"individuals": 0, "classes": 0, "properties": 0}

        return {
            "individuals": len(list(self._world.individuals())),
            "classes": len(list(self._world.classes())),
            "properties": len(list(self._world.properties())),
        }
