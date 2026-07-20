#!/usr/bin/python
"""ArchiMate EA Governance Layer.

CONCEPT:AU-KG.research.research-pipeline-runner — ArchiMate-to-KG Mapping

Maps Knowledge Graph node types to the ArchiMate 3.2 metamodel, enabling
enterprise-architecture-level governance over the agent ecosystem. Provides
classification, view generation, and ArchiMate Open Exchange Format export.

ArchiMate Layer Mappings:
    Business Layer:   Policy, ProcessFlow, Organization, Role
    Application Layer: Agent, Tool, Skill, SystemPrompt
    Technology Layer:  Server, DataConnector, Pipeline
    Strategy Layer:    Concept, SDDPlan, Experiment, Capability
    Motivation Layer:  Goal, Principle, Regulation

References:
    - ArchiMate 3.2 Specification: https://pubs.opengroup.org/architecture/archimate3-doc/
    - Mendoza ArchiMate-to-RDF: aligns with semantic_subsumption.py patterns
"""

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class ArchiMateLayerType(StrEnum):
    """ArchiMate 3.2 architecture layers."""

    BUSINESS = "business"
    APPLICATION = "application"
    TECHNOLOGY = "technology"
    STRATEGY = "strategy"
    MOTIVATION = "motivation"
    UNCLASSIFIED = "unclassified"


@dataclass
class ArchiMateElement:
    """An ArchiMate 3.2 element classification result."""

    node_type: str
    layer: ArchiMateLayerType
    archimate_type: str
    description: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# KG Node Type → ArchiMate 3.2 Element Mappings
# ═══════════════════════════════════════════════════════════════════════════════

_KG_TO_ARCHIMATE: dict[str, tuple[ArchiMateLayerType, str]] = {
    # Business Layer
    "policy": (ArchiMateLayerType.BUSINESS, "BusinessRule"),
    "process_flow": (ArchiMateLayerType.BUSINESS, "BusinessProcess"),
    "process_step": (ArchiMateLayerType.BUSINESS, "BusinessProcess"),
    "organization": (ArchiMateLayerType.BUSINESS, "BusinessActor"),
    "role": (ArchiMateLayerType.BUSINESS, "BusinessRole"),
    "team": (ArchiMateLayerType.BUSINESS, "BusinessCollaboration"),
    "person": (ArchiMateLayerType.BUSINESS, "BusinessActor"),
    "task": (ArchiMateLayerType.BUSINESS, "BusinessService"),
    "aris_process": (ArchiMateLayerType.BUSINESS, "BusinessProcess"),
    "ea_fact_sheet": (ArchiMateLayerType.STRATEGY, "Resource"),
    "process_model": (ArchiMateLayerType.BUSINESS, "BusinessProcess"),
    "backstage_component": (ArchiMateLayerType.APPLICATION, "ApplicationComponent"),
    "bpmn_process": (ArchiMateLayerType.BUSINESS, "BusinessProcess"),
    "bpmn_step": (ArchiMateLayerType.BUSINESS, "BusinessProcess"),
    "db_schema": (ArchiMateLayerType.TECHNOLOGY, "Node"),
    "db_table": (ArchiMateLayerType.APPLICATION, "DataObject"),
    "db_column": (ArchiMateLayerType.APPLICATION, "DataObject"),
    # Application Layer
    "agent": (ArchiMateLayerType.APPLICATION, "ApplicationComponent"),
    "tool": (ArchiMateLayerType.APPLICATION, "ApplicationService"),
    "skill": (ArchiMateLayerType.APPLICATION, "ApplicationService"),
    "system_prompt": (ArchiMateLayerType.APPLICATION, "ApplicationInterface"),
    "spawned_agent": (ArchiMateLayerType.APPLICATION, "ApplicationComponent"),
    "callable_resource": (ArchiMateLayerType.APPLICATION, "ApplicationFunction"),
    "mcp_server_package": (ArchiMateLayerType.APPLICATION, "ApplicationComponent"),
    "software_component": (ArchiMateLayerType.APPLICATION, "ApplicationComponent"),
    "software_feature": (ArchiMateLayerType.APPLICATION, "ApplicationFunction"),
    "api_contract": (ArchiMateLayerType.APPLICATION, "ApplicationInterface"),
    "test_case": (ArchiMateLayerType.APPLICATION, "ApplicationFunction"),
    # Technology Layer
    "server": (ArchiMateLayerType.TECHNOLOGY, "Node"),
    "data_connector": (ArchiMateLayerType.TECHNOLOGY, "TechnologyService"),
    "pipeline": (ArchiMateLayerType.TECHNOLOGY, "TechnologyProcess"),
    "repository": (ArchiMateLayerType.TECHNOLOGY, "Artifact"),
    "file": (ArchiMateLayerType.TECHNOLOGY, "Artifact"),
    "module": (ArchiMateLayerType.TECHNOLOGY, "Artifact"),
    "ecosystem_package": (ArchiMateLayerType.TECHNOLOGY, "SystemSoftware"),
    # Strategy Layer
    "concept": (ArchiMateLayerType.STRATEGY, "Capability"),
    "capability": (ArchiMateLayerType.STRATEGY, "Capability"),
    "experiment": (ArchiMateLayerType.STRATEGY, "CourseOfAction"),
    "architecture_decision": (ArchiMateLayerType.STRATEGY, "CourseOfAction"),
    "software_project": (ArchiMateLayerType.STRATEGY, "Resource"),
    "research_hypothesis": (ArchiMateLayerType.STRATEGY, "CourseOfAction"),
    "specification": (ArchiMateLayerType.STRATEGY, "Capability"),
    "leanix_fact_sheet": (ArchiMateLayerType.STRATEGY, "Resource"),
    # Motivation Layer
    "goal": (ArchiMateLayerType.MOTIVATION, "Goal"),
    "principle": (ArchiMateLayerType.MOTIVATION, "Principle"),
    "regulation": (ArchiMateLayerType.MOTIVATION, "Constraint"),
    "engineering_rule": (ArchiMateLayerType.MOTIVATION, "Requirement"),
    "optimization_goal": (ArchiMateLayerType.MOTIVATION, "Goal"),
    "requirement": (ArchiMateLayerType.MOTIVATION, "Requirement"),
    "user_story": (ArchiMateLayerType.MOTIVATION, "Requirement"),
    "acceptance_criteria": (ArchiMateLayerType.MOTIVATION, "Requirement"),
    "design_guideline": (ArchiMateLayerType.MOTIVATION, "Principle"),
    "compliance_constraint": (ArchiMateLayerType.MOTIVATION, "Constraint"),
    # Developer Harness & AI Optimization Patterns
    "harness_extension_point": (ArchiMateLayerType.APPLICATION, "ApplicationInterface"),
    "context_file": (ArchiMateLayerType.TECHNOLOGY, "Artifact"),
    "hook": (ArchiMateLayerType.APPLICATION, "ApplicationFunction"),
    "plugin": (ArchiMateLayerType.APPLICATION, "ApplicationComponent"),
    "lsp_server": (ArchiMateLayerType.APPLICATION, "ApplicationComponent"),
    "subagent_config": (ArchiMateLayerType.APPLICATION, "ApplicationComponent"),
    "optimization_pattern": (ArchiMateLayerType.MOTIVATION, "Principle"),
    "agent_manager_dri": (ArchiMateLayerType.BUSINESS, "BusinessRole"),
    "governance_working_group": (ArchiMateLayerType.BUSINESS, "BusinessCollaboration"),
}


class ArchiMateLayer:
    """Maps KG concepts to ArchiMate 3.2 metamodel elements.

    CONCEPT:AU-KG.research.research-pipeline-runner — ArchiMate EA Governance Layer

    Business Layer:   Policy, Goal, Process → BusinessService, BusinessProcess
    Application Layer: Agent, Tool, Skill → ApplicationComponent, ApplicationService
    Technology Layer:  Server, DataConnector → TechnologyService, Node
    Strategy Layer:    Concept, SDDPlan → Capability, CourseOfAction
    Motivation Layer:  Goal, Principle → Goal, Principle
    """

    def __init__(self) -> None:
        self._mappings = dict(_KG_TO_ARCHIMATE)

    def classify(self, node_type: str) -> ArchiMateElement:
        """Map a KG node type to its ArchiMate 3.2 element.

        Args:
            node_type: The KG node type string (e.g. 'agent', 'policy').

        Returns:
            ArchiMateElement with layer and archimate_type populated.
            Returns UNCLASSIFIED layer for unknown types.
        """
        normalized = node_type.lower().strip()
        if normalized in self._mappings:
            layer, am_type = self._mappings[normalized]
            return ArchiMateElement(
                node_type=normalized,
                layer=layer,
                archimate_type=am_type,
                description=f"{normalized} maps to ArchiMate {am_type} in {layer.value} layer",
            )
        return ArchiMateElement(
            node_type=normalized,
            layer=ArchiMateLayerType.UNCLASSIFIED,
            archimate_type="Unknown",
            description=f"No ArchiMate mapping for '{normalized}'",
        )

    def get_layer_members(self, layer: ArchiMateLayerType) -> list[str]:
        """Get all KG node types mapped to a specific ArchiMate layer.

        Args:
            layer: The ArchiMate layer to query.

        Returns:
            List of KG node type strings in that layer.
        """
        return [
            node_type for node_type, (lyr, _) in self._mappings.items() if lyr == layer
        ]

    def get_all_layers(self) -> dict[str, list[str]]:
        """Get complete layer-to-node-type mapping.

        Returns:
            Dict mapping layer names to lists of KG node types.
        """
        result: dict[str, list[str]] = {}
        for layer in ArchiMateLayerType:
            if layer == ArchiMateLayerType.UNCLASSIFIED:
                continue
            result[layer.value] = self.get_layer_members(layer)
        return result

    def generate_archimate_view(
        self, engine: Any, scope: str = "all"
    ) -> dict[str, Any]:
        """Generate a scoped ArchiMate view from KG data.

        Args:
            engine: IntelligenceGraphEngine instance.
            scope: Filter scope — 'all', or a specific layer name.

        Returns:
            Dict with elements and relationships organized by layer.
        """
        view: dict[str, Any] = {
            "view_type": "archimate_3.2",
            "scope": scope,
            "layers": {},
        }

        # Query all node types present in the KG
        try:
            node_counts = engine.query_cypher(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt"
            )
        except Exception as e:
            logger.warning("ArchiMate view generation failed: %s", e)
            node_counts = []

        layer_data: dict[str, list[dict[str, Any]]] = {
            layer.value: []
            for layer in ArchiMateLayerType
            if layer != ArchiMateLayerType.UNCLASSIFIED
        }

        for row in node_counts or []:
            label = (row.get("label") or "").lower()
            element = self.classify(label)
            if element.layer != ArchiMateLayerType.UNCLASSIFIED:
                if scope == "all" or scope == element.layer.value:
                    layer_data[element.layer.value].append(
                        {
                            "kg_type": label,
                            "archimate_type": element.archimate_type,
                            "count": row.get("cnt", 0),
                        }
                    )

        view["layers"] = {k: v for k, v in layer_data.items() if v}
        return view

    @staticmethod
    def generate_mermaid(layer_data: dict[str, list[dict[str, Any]]]) -> str:
        """Generate a Mermaid diagram from ArchiMate layer data.

        Args:
            layer_data: Output from generate_archimate_view()['layers'].

        Returns:
            Mermaid diagram string.
        """
        lines = ["graph TB"]
        for layer_name, elements in layer_data.items():
            safe_layer = layer_name.replace(" ", "_")
            lines.append(f'    subgraph {safe_layer}["{layer_name.title()} Layer"]')
            for elem in elements:
                node_id = f"{safe_layer}_{elem['kg_type']}"
                label = f"{elem['archimate_type']}\\n({elem['kg_type']}: {elem.get('count', '?')})"
                lines.append(f'        {node_id}["{label}"]')
            lines.append("    end")
        return "\n".join(lines)
