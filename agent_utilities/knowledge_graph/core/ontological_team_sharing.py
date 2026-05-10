#!/usr/bin/python
from __future__ import annotations

"""Ontological Team Sharing (CONCEPT:KG-2.52).

Serializes dynamically created TeamCompositions into shareable
semantic formats (OWL/Turtle) for export/import across instances.
"""


import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...models.knowledge_graph import TeamComposition

logger = logging.getLogger(__name__)


class OntologicalTeamExporter:
    """Exports and imports team compositions as OWL/Turtle.

    CONCEPT:KG-2.52
    """

    @staticmethod
    def export_to_turtle(composition: TeamComposition) -> str:
        """Serialize a TeamComposition to OWL/Turtle format."""
        lines = [
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
            "@prefix agent: <http://example.org/agent-ontology#> .",
            "",
            f"agent:{composition.team_id} a agent:TeamComposition ;",
            f'    agent:hasSource "{composition.source}" ;',
            f'    agent:hasTopologyTemplate "{composition.topology_template_id}" ;',
            f'    agent:executionMode "{composition.execution_mode}" ;',
            f"    agent:confidence {composition.confidence} .",
            "",
        ]

        # Add adaptive_agent_router
        for i, s in enumerate(composition.adaptive_agent_router):
            spec_id = f"{composition.team_id}_spec_{i}"
            lines.append(
                f"agent:{composition.team_id} agent:hasSpecialist agent:{spec_id} ."
            )
            lines.append(f"agent:{spec_id} a agent:Specialist ;")
            lines.append(f'    agent:hasRole "{s["role"]}" ;')
            if s.get("agent_id"):
                lines.append(f'    agent:hasAgentId "{s["agent_id"]}" ;')

            # Add tools
            for t in s.get("tools", []):
                lines.append(f'    agent:usesTool "{t}" ;')

            # Close the stanza
            lines[-1] = lines[-1][:-1] + "."
            lines.append("")

        logger.info(
            "[CONCEPT:KG-2.52] Exported team %s to Turtle format", composition.team_id
        )
        return "\n".join(lines)

    @staticmethod
    def import_from_turtle(ttl_content: str) -> dict:
        """Stub for parsing Turtle back into a TeamComposition."""
        # A fully compliant system would use RDFLib to parse the TTL
        logger.debug("Importing TTL content length: %d", len(ttl_content))
        logger.info("[CONCEPT:KG-2.52] Importing team from Turtle format (Stub)")
        return {}
