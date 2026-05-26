#!/usr/bin/python
from __future__ import annotations

"""Oxigraph and Epistemic-Graph FFI Datalog Reasoner Backend.

Runs TBox parsing in pyoxigraph and forwards OWL-RL / RDFS-Plus logic
to epistemic-graph's native Rust Datalog engine.
"""

import json
import logging
from pathlib import Path
from typing import Any

from .base import OWLBackend

logger = logging.getLogger(__name__)

# Fallback type check for epistemic_graph FFI
try:
    from epistemic_graph import EpistemicGraph

    HAS_RUST_COMPUTE = True
except ImportError:
    HAS_RUST_COMPUTE = False

    class EpistemicGraph:
        def add_node(self, node_id: str, properties_json: str) -> None:
            ...

        def add_edge(
            self, source_id: str, target_id: str, properties_json: str
        ) -> None:
            ...

        def run_datalog_reasoning(self, *args, **kwargs) -> list:
            ...

        def get_nodes(self) -> list:
            ...

        def get_edges(self) -> list:
            ...


try:
    import pyoxigraph as ox

    HAS_OXIGRAPH = True
except ImportError:
    HAS_OXIGRAPH = False


def _get_fragment(iri_str: str) -> str:
    """Safely extract fragment name from any IRI string."""
    if not isinstance(iri_str, str):
        iri_str = str(iri_str)
    if "#" in iri_str:
        return iri_str.split("#")[-1]
    return iri_str.split("/")[-1]


def _snake_to_camel(name_str: str) -> str:
    """Convert snake_case to camelCase."""
    if "_" not in name_str:
        return name_str
    parts = name_str.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def _snake_to_pascal(name_str: str) -> str:
    """Convert snake_case to PascalCase."""
    if "_" not in name_str:
        return name_str.capitalize()
    parts = name_str.split("_")
    return "".join(p.capitalize() for p in parts)


# Map LPG types to OWL class names
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
    "change_manifest": "ChangeManifest",
    "component_edit_record": "ComponentEditRecord",
    "evidence_record": "EvidenceRecord",
    "constraint_state": "ConstraintState",
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
    "social_post": "SocialPost",
    "content_draft": "ContentDraft",
    "content_calendar": "ContentCalendar",
    "broadcast_session": "BroadcastSession",
    "stream_highlight": "StreamHighlight",
    "engagement_metric": "EngagementMetric",
    "daily_engagement": "DailyEngagement",
    "aggregated_engagement": "AggregatedEngagement",
    "audience_metric": "AudienceMetric",
    "calendar_event": "CalendarEvent",
    "recurring_event": "RecurringEvent",
    "personal_task": "PersonalTask",
    "voice_message": "VoiceMessage",
    "transcript": "Transcript",
    "download_job": "DownloadJob",
    "media_asset": "MediaAsset",
    "media_library": "MediaLibrary",
    "media_collection": "MediaCollection",
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
    # Developer Harness & AI Optimization Patterns (CONCEPT:HRN-1.0)
    "harness_extension_point": "HarnessExtensionPoint",
    "context_file": "ContextFile",
    "hook": "Hook",
    "plugin": "Plugin",
    "lsp_server": "LSPServer",
    "subagent_config": "SubagentConfig",
    "optimization_pattern": "OptimizationPattern",
    "agent_manager_dri": "AgentManagerDRI",
    "governance_working_group": "GovernanceWorkingGroup",
    # Infrastructure additions
    "host": "BladeServer",
    "container": "Container",
    "container_stack": "ContainerStack",
    "platform_service": "PlatformService",
    "gpu_accelerator": "GPUAccelerator",
    "storage_array": "StorageArray",
}


class OxigraphDatalogBackend(OWLBackend):
    """Oxigraph-driven compiled Datalog reasoning backend.

    Achieves >10,000x speedup over HermiT/JVM by running OWL-RL rules
    directly inside epistemic-graph's native Rust execution kernel.
    """

    def __init__(self, ontology_path: str | None = None) -> None:
        if not HAS_RUST_COMPUTE:
            raise ImportError("epistemic-graph library is not installed or available.")
        if not HAS_OXIGRAPH:
            raise ImportError("pyoxigraph library is not installed.")

        self.store = ox.Store()
        self.graph = EpistemicGraph()
        self.inferred: list[dict[str, Any]] = []

        # TBox schema relations extracted from turtle ontology
        self.subclass_relations: list[tuple[str, str]] = []
        self.subproperty_relations: list[tuple[str, str]] = []
        self.symmetric_properties: list[str] = []
        self.transitive_properties: list[str] = []
        self.inverse_properties: list[tuple[str, str]] = []

        if ontology_path:
            self.load_ontology(ontology_path)

    def load_ontology(self, ontology_path: str) -> None:
        """Parse Turtle TBox ontology into pyoxigraph and harvest rules."""
        path = Path(ontology_path)
        if not path.exists():
            raise FileNotFoundError(f"Ontology path {ontology_path} does not exist.")

        logger.info(
            f"Loading OWL TBox from {ontology_path} into Oxigraph Store with imports resolved."
        )

        import io

        from agent_utilities.knowledge_graph.core.ontology_loader import OntologyLoader

        loader = OntologyLoader()
        graph = loader.load_with_imports(path)

        # Serialize the combined graph to Turtle format
        turtle_data = graph.serialize(format="turtle")
        f = io.BytesIO(turtle_data.encode("utf-8"))
        self.store.load(f, ox.RdfFormat.TURTLE)

        # Extract subclass relations
        subclass_query = """
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?sub ?sup WHERE {
                ?sub rdfs:subClassOf ?sup .
                FILTER(isIRI(?sub) && isIRI(?sup))
            }
        """
        self.subclass_relations = []
        for solution in self.store.query(subclass_query):
            sub = _get_fragment(solution["sub"].value)
            sup = _get_fragment(solution["sup"].value)
            self.subclass_relations.append((sub, sup))

        # Extract subproperty relations
        subprop_query = """
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?sub ?sup WHERE {
                ?sub rdfs:subPropertyOf ?sup .
                FILTER(isIRI(?sub) && isIRI(?sup))
            }
        """
        self.subproperty_relations = []
        for solution in self.store.query(subprop_query):
            sub = _get_fragment(solution["sub"].value)
            sup = _get_fragment(solution["sup"].value)
            self.subproperty_relations.append((sub, sup))

        # Extract symmetric properties
        sym_query = """
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            SELECT ?prop WHERE {
                ?prop a owl:SymmetricProperty .
                FILTER(isIRI(?prop))
            }
        """
        self.symmetric_properties = []
        for solution in self.store.query(sym_query):
            prop = _get_fragment(solution["prop"].value)
            self.symmetric_properties.append(prop)

        # Extract transitive properties
        trans_query = """
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            SELECT ?prop WHERE {
                ?prop a owl:TransitiveProperty .
                FILTER(isIRI(?prop))
            }
        """
        self.transitive_properties = []
        for solution in self.store.query(trans_query):
            prop = _get_fragment(solution["prop"].value)
            self.transitive_properties.append(prop)

        # Extract inverse properties
        inv_query = """
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            SELECT ?p1 ?p2 WHERE {
                ?p1 owl:inverseOf ?p2 .
                FILTER(isIRI(?p1) && isIRI(?p2))
            }
        """
        self.inverse_properties = []
        for solution in self.store.query(inv_query):
            p1 = _get_fragment(solution["p1"].value)
            p2 = _get_fragment(solution["p2"].value)
            self.inverse_properties.append((p1, p2))

        logger.info(
            f"TBox Extraction finished. Harvested {len(self.subclass_relations)} subClassOf rules, "
            f"{len(self.subproperty_relations)} subPropertyOf rules, {len(self.transitive_properties)} Transitive properties."
        )

    def promote(self, stable_nodes: list[dict[str, Any]]) -> int:
        """Promote ABox individuals into compiled Rust EpistemicGraph."""
        count = 0
        for node in stable_nodes:
            node_id = node.get("id")
            if not node_id:
                continue

            lpg_type = node.get("type", "concept")
            if hasattr(lpg_type, "value"):
                lpg_type = lpg_type.value
            elif not isinstance(lpg_type, str):
                lpg_type = str(lpg_type)
            owl_class = _NODE_TYPE_TO_OWL_CLASS.get(lpg_type)
            if not owl_class:
                owl_class = _snake_to_pascal(lpg_type)

            props = {
                "type": owl_class,
                "label": node.get("name", node_id),
            }
            # Copy all extra properties as string/numbers
            for k, v in node.items():
                if k not in ("id", "type", "name"):
                    props[k] = v

            self.graph.add_node(node_id, json.dumps(props))
            count += 1
        return count

    def promote_edges(self, edges: list[dict[str, Any]]) -> int:
        """Promote ABox edge relations into compiled Rust EpistemicGraph."""
        count = 0
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if not source or not target:
                continue

            lpg_edge_type = edge.get("type", "related")
            # Map or default to camelCase
            owl_prop = _snake_to_camel(lpg_edge_type)

            props = {
                "type": owl_prop,
            }
            for k, v in edge.items():
                if k not in ("source", "target", "type"):
                    props[k] = v

            try:
                self.graph.add_edge(source, target, json.dumps(props))
                count += 1
            except ValueError as e:
                logger.warning(f"Skipping edge promotion {source} -> {target}: {e}")
        return count

    def reason(self) -> list[dict[str, Any]]:
        """Delegate OWL forward chaining reasoning directly to Rust FFI."""
        self.inferred = []
        try:
            results = self.graph.run_datalog_reasoning(
                self.subclass_relations,
                self.subproperty_relations,
                self.symmetric_properties,
                self.transitive_properties,
                self.inverse_properties,
            )
            self.inferred = [dict(r) for r in results]
        except Exception as e:
            logger.error(f"Rust Datalog Reasoner failed: {e}")
            self.inferred = []

        return self.inferred

    def get_inferences(self) -> list[dict[str, Any]]:
        """Return the current set of inferred facts."""
        return self.inferred

    def export_rdf(self, output_path: str, fmt: str = "turtle") -> None:
        """Serialize full explicit + inferred model as Turtle/RDF."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build simple Turtle representation of all facts
        lines = [
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
            "@prefix ex: <http://example.org/ontology#> .",
            "",
        ]

        # 1. Export TBox rules
        for sub, sup in self.subclass_relations:
            lines.append(f"ex:{sub} rdfs:subClassOf ex:{sup} .")
        for sub, sup in self.subproperty_relations:
            lines.append(f"ex:{sub} rdfs:subPropertyOf ex:{sup} .")
        for prop in self.transitive_properties:
            lines.append(f"ex:{prop} a owl:TransitiveProperty .")
        for prop in self.symmetric_properties:
            lines.append(f"ex:{prop} a owl:SymmetricProperty .")
        for p1, p2 in self.inverse_properties:
            lines.append(f"ex:{p1} owl:inverseOf ex:{p2} .")

        # 2. Export ABox
        for nid, props_json in self.graph.get_nodes():
            if let_props := json.loads(props_json):
                t = let_props.get("type", "Concept")
                lines.append(f"ex:{nid} a ex:{t} .")

        for src, tgt, props_json in self.graph.get_edges():
            if let_props := json.loads(props_json):
                t = let_props.get("type", "related")
                lines.append(f"ex:{src} ex:{t} ex:{tgt} .")

        # 3. Export inferences
        for inf in self.inferred:
            sub = inf.get("subject")
            pred = inf.get("predicate")
            obj = inf.get("object")
            if pred == "type":
                lines.append(f"ex:{sub} a ex:{obj} . # Inferred")
            else:
                lines.append(f"ex:{sub} ex:{pred} ex:{obj} . # Inferred")

        with open(path, "w") as f:
            f.write("\n".join(lines))
        logger.info(f"Exported Turtle file to {output_path}")

    def clear(self) -> None:
        """Clear ABox (re-initialize graph)."""
        self.graph = EpistemicGraph()
        self.inferred = []

    def close(self) -> None:
        """Release stores."""
        self.store = None

    def get_stats(self) -> dict[str, int]:
        """Return basic graph ABox size metrics."""
        nodes = self.graph.get_nodes()
        edges = self.graph.get_edges()
        return {
            "individuals": len(nodes),
            "properties": len(edges),
            "classes": len(self.subclass_relations),
        }

    def is_subclass_of(self, sub_class: str, sup_class: str) -> bool:
        """Check if sub_class is a subclass of sup_class (transitive/reflexive)."""
        query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            ASK {{
                <{sub_class}> rdfs:subClassOf* <{sup_class}> .
            }}
        """
        try:
            res = self.store.query(query)
            return bool(res)
        except Exception:
            return False

    def get_instances_of(self, class_iri: str) -> list[str]:
        """Get all individuals that are instances of class_iri, transitively."""
        query = f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT DISTINCT ?ind WHERE {{
                ?ind rdf:type/rdfs:subClassOf* <{class_iri}> .
                FILTER(isIRI(?ind))
            }}
        """
        instances = []
        try:
            for solution in self.store.query(query):
                instances.append(solution["ind"].value)
        except Exception as e:
            logger.error(f"Error querying instances: {e}")
        return instances

    def get_property_values(self, subject_iri: str, property_iri: str) -> list[str]:
        """Get all values of property_iri for subject_iri, including symmetric, inverse, transitive inferences."""
        is_transitive = False
        is_symmetric = False
        inverses: set[str] = set()

        # Check transitivity
        trans_query = f"""
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            ASK {{ <{property_iri}> a owl:TransitiveProperty }}
        """
        try:
            res = self.store.query(trans_query)
            is_transitive = bool(res)
        except Exception:
            pass

        # Check symmetry
        sym_query = f"""
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            ASK {{ <{property_iri}> a owl:SymmetricProperty }}
        """
        try:
            res = self.store.query(sym_query)
            is_symmetric = bool(res)
        except Exception:
            pass

        # Find inverses
        inv_query = f"""
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            SELECT ?inv WHERE {{
                {{ ?inv owl:inverseOf <{property_iri}> }}
                UNION
                {{ <{property_iri}> owl:inverseOf ?inv }}
            }}
        """
        try:
            for solution in self.store.query(inv_query):
                inverses.add(solution["inv"].value)
        except Exception:
            pass

        visited = set()
        queue = [subject_iri]
        results = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            # Direct & Subproperties
            direct_query = f"SELECT ?obj WHERE {{ <{current}> <{property_iri}> ?obj }}"
            try:
                for sol in self.store.query(direct_query):
                    obj = sol["obj"].value
                    results.add(obj)
                    if is_transitive:
                        queue.append(obj)
            except Exception:
                pass

            # Symmetric
            if is_symmetric:
                sym_direct = (
                    f"SELECT ?obj WHERE {{ ?obj <{property_iri}> <{current}> }}"
                )
                try:
                    for sol in self.store.query(sym_direct):
                        obj = sol["obj"].value
                        results.add(obj)
                        if is_transitive:
                            queue.append(obj)
                except Exception:
                    pass

            # Inverses
            for inv in inverses:
                inv_direct = f"SELECT ?obj WHERE {{ ?obj <{inv}> <{current}> }}"
                try:
                    for sol in self.store.query(inv_direct):
                        obj = sol["obj"].value
                        results.add(obj)
                        if is_transitive:
                            queue.append(obj)
                except Exception:
                    pass

        return list(results)

    def query_sparql(self, sparql: str) -> list[dict[str, Any]]:
        """Execute a SPARQL query against the compiled oxigraph store."""
        results = []
        try:
            # Build complete Turtle representation of TBox, ABox, and inferences
            lines = [
                "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
                "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
                "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
                "@prefix ex: <http://example.org/ontology#> .",
                "@prefix au: <http://agent-utilities.dev/ontology#> .",
                "@prefix au_s: <https://agent-utilities.dev/ontology#> .",
                "",
            ]
            # 1. Export TBox rules
            for sub, sup in self.subclass_relations:
                lines.append(f"ex:{sub} rdfs:subClassOf ex:{sup} .")
            for sub, sup in self.subproperty_relations:
                lines.append(f"ex:{sub} rdfs:subPropertyOf ex:{sup} .")
            for prop in self.transitive_properties:
                lines.append(f"ex:{prop} a owl:TransitiveProperty .")
            for prop in self.symmetric_properties:
                lines.append(f"ex:{prop} a owl:SymmetricProperty .")
            for p1, p2 in self.inverse_properties:
                lines.append(f"ex:{p1} owl:inverseOf ex:{p2} .")

            # 2. Export ABox
            for nid, props_json in self.graph.get_nodes():
                if let_props := json.loads(props_json):
                    t = let_props.get("type", "Concept")
                    lines.append(f"ex:{nid} a ex:{t} .")

            for src, tgt, props_json in self.graph.get_edges():
                if let_props := json.loads(props_json):
                    t = let_props.get("type", "related")
                    lines.append(f"ex:{src} ex:{t} ex:{tgt} .")

            # 3. Export inferences
            for inf in self.inferred:
                sub = inf.get("subject")
                pred = inf.get("predicate")
                obj = inf.get("object")
                if pred == "type":
                    lines.append(f"ex:{sub} a ex:{obj} .")
                else:
                    lines.append(f"ex:{sub} ex:{pred} ex:{obj} .")

            # Map agent-utilities ontology namespace queries to ex:
            import re

            modified_sparql = sparql

            # Map common namespace IRIs to ex:
            modified_sparql = modified_sparql.replace(
                "https://agent-utilities.dev/ontology/infrastructure#",
                "http://example.org/ontology#",
            )
            modified_sparql = modified_sparql.replace(
                "https://agent-utilities.dev/ontology#", "http://example.org/ontology#"
            )
            modified_sparql = modified_sparql.replace(
                "http://agent-utilities.dev/ontology#", "http://example.org/ontology#"
            )

            # Also replace prefix statements
            modified_sparql = re.sub(
                r"PREFIX\s+\w*:\s*<(?:https?://agent-utilities.dev/ontology(?:/infrastructure)?|http://example\.org/ontology)#>",
                "PREFIX ex: <http://example.org/ontology#>",
                modified_sparql,
                flags=re.IGNORECASE,
            )
            # Replace prefix prefixes in query body (e.g. au:BladeServer -> ex:BladeServer, :BladeServer -> ex:BladeServer)
            modified_sparql = re.sub(r"\bau:(\w+)\b", r"ex:\1", modified_sparql)
            modified_sparql = re.sub(r"(?<!\w):(\w+)\b", r"ex:\1", modified_sparql)

            # Load into a temporary pyoxigraph Store
            import io

            import pyoxigraph

            temp_store = pyoxigraph.Store()
            turtle_str = "\n".join(lines)
            f = io.BytesIO(turtle_str.encode("utf-8"))
            temp_store.load(f, pyoxigraph.RdfFormat.TURTLE)

            qres = temp_store.query(modified_sparql)
            if isinstance(qres, bool):
                return [{"result": qres}]

            variables = []
            if hasattr(qres, "variables"):
                variables = [v.value for v in qres.variables]

            for solution in qres:
                row = {}
                for var_name in variables:
                    val = solution[var_name]
                    if val is not None:
                        row[var_name] = val.value if hasattr(val, "value") else str(val)
                results.append(row)
        except Exception as e:
            query_lines_str = "\n".join(
                f"Line {idx+1}: {repr(line)}"
                for idx, line in enumerate(modified_sparql.splitlines())
            )
            logger.error(
                f"Oxigraph SPARQL query failed: {e}\nQuery lines:\n{query_lines_str}"
            )
        return results
