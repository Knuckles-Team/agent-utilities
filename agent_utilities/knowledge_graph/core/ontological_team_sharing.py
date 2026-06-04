#!/usr/bin/python
from __future__ import annotations

"""Ontological Team Sharing (CONCEPT:KG-2.6).

Serializes dynamically created TeamCompositions into shareable
semantic formats (OWL/Turtle) for export/import across instances.
"""


import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...models.knowledge_graph import TeamComposition

logger = logging.getLogger(__name__)


class OntologicalTeamExporter:
    """Exports and imports team compositions as OWL/Turtle.

    CONCEPT:KG-2.6
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
            "[CONCEPT:KG-2.6] Exported team %s to Turtle format", composition.team_id
        )
        return "\n".join(lines)

    @staticmethod
    def import_from_turtle(ttl_content: str) -> dict:
        """Parse Turtle back into a TeamComposition dict structure."""
        logger.debug("Importing TTL content length: %d", len(ttl_content))
        logger.info("[CONCEPT:KG-2.6] Importing team from Turtle format")

        try:
            import rdflib

            g = rdflib.Graph()
            g.parse(data=ttl_content, format="turtle")

            # Find the TeamComposition node
            team_uri = None
            for s, p, o in g.triples(
                (
                    None,
                    rdflib.RDF.type,
                    rdflib.URIRef("http://example.org/agent-ontology#TeamComposition"),
                )
            ):
                team_uri = s
                break

            if not team_uri:
                # If no explicit type, find any subject that has hasSource or hasTopologyTemplate
                for s, p, o in g.triples(
                    (
                        None,
                        rdflib.URIRef("http://example.org/agent-ontology#hasSource"),
                        None,
                    )
                ):
                    team_uri = s
                    break

            if not team_uri:
                return {}

            team_id = str(team_uri).split("#")[-1]

            # Retrieve team properties
            source = ""
            topology_template_id = ""
            execution_mode = ""
            confidence = 1.0

            for p, o in g.predicate_objects(team_uri):
                pred_str = str(p).split("#")[-1]
                if pred_str == "hasSource":
                    source = str(o)
                elif pred_str == "hasTopologyTemplate":
                    topology_template_id = str(o)
                elif pred_str == "executionMode":
                    execution_mode = str(o)
                elif pred_str == "confidence":
                    try:
                        confidence = float(str(o))
                    except ValueError:
                        confidence = 1.0

            # Retrieve specialists
            specialists = []
            for s, p, o in g.triples(
                (
                    team_uri,
                    rdflib.URIRef("http://example.org/agent-ontology#hasSpecialist"),
                    None,
                )
            ):
                spec_uri = o
                fallback_spec_dict: dict[str, Any] = {
                    "role": "",
                    "agent_id": "",
                    "tools": [],
                }
                for sp, so in g.predicate_objects(spec_uri):
                    spred_str = str(sp).split("#")[-1]
                    if spred_str == "hasRole":
                        fallback_spec_dict["role"] = str(so)
                    elif spred_str == "hasAgentId":
                        fallback_spec_dict["agent_id"] = str(so)
                    elif spred_str == "usesTool":
                        fallback_spec_dict["tools"].append(str(so))
                specialists.append(fallback_spec_dict)

            return {
                "team_id": team_id,
                "source": source,
                "topology_template_id": topology_template_id,
                "execution_mode": execution_mode,
                "confidence": confidence,
                "adaptive_agent_router": specialists,
            }

        except Exception as e:
            logger.warning(
                "rdflib parsing failed or not available, falling back to manual regex parser: %s",
                e,
            )
            import re

            result: dict[str, Any] = {
                "team_id": "",
                "source": "",
                "topology_template_id": "",
                "execution_mode": "",
                "confidence": 1.0,
                "adaptive_agent_router": [],
            }

            # Extract team_id
            m = re.search(r"agent:(\w+)\s+a\s+agent:TeamComposition", ttl_content)
            if m:
                result["team_id"] = m.group(1)
            else:
                m = re.search(r"agent:(\w+)\s+", ttl_content)
                if m:
                    result["team_id"] = m.group(1)

            # Extract team level properties
            m_src = re.search(r'agent:hasSource\s+"([^"]+)"', ttl_content)
            if m_src:
                result["source"] = m_src.group(1)
            m_topo = re.search(r'agent:hasTopologyTemplate\s+"([^"]+)"', ttl_content)
            if m_topo:
                result["topology_template_id"] = m_topo.group(1)
            m_exec = re.search(r'agent:executionMode\s+"([^"]+)"', ttl_content)
            if m_exec:
                result["execution_mode"] = m_exec.group(1)
            m_conf = re.search(r"agent:confidence\s+([\d.]+)", ttl_content)
            if m_conf:
                result["confidence"] = float(m_conf.group(1))

            # Find specialists
            spec_blocks = re.findall(
                r"agent:\w+_spec_\d+\s+a\s+agent:Specialist[\s\S]+?\.", ttl_content
            )
            for block in spec_blocks:
                spec_dict: dict[str, Any] = {"role": "", "agent_id": "", "tools": []}
                r_m = re.search(r'agent:hasRole\s+"([^"]+)"', block)
                if r_m:
                    spec_dict["role"] = r_m.group(1)
                a_m = re.search(r'agent:hasAgentId\s+"([^"]+)"', block)
                if a_m:
                    spec_dict["agent_id"] = a_m.group(1)
                t_m = re.findall(r'agent:usesTool\s+"([^"]+)"', block)
                spec_dict["tools"] = t_m
                result["adaptive_agent_router"].append(spec_dict)

            return result
