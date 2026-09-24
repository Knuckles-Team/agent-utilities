"""Swarm topology through EG's assembly decision (SWARM-TOPOLOGY-DECIDE-DESIGN).

AU supplies the vocabulary and the reference templates; EG decides. The
``swarm-topology`` schema source (:mod:`.schema_source`) is attached through
``GraphSchema`` and reasoned in EG; the reference templates (:mod:`.templates`)
are published agent-graph templates carrying typed topology facts; the
topology question itself (:mod:`agent_utilities.decide.consumers.topology`)
is an ``AgentAssemble`` request whose ``requirements.topology`` EG answers with
a certified plan or a typed abstention.
"""

from __future__ import annotations

from agent_utilities.decide.topology.schema_source import (
    SOURCE_ID,
    SWARM_NS,
    attach_swarm_topology,
    ontology_ttl,
    shapes_ttl,
)
from agent_utilities.decide.topology.templates import (
    REFERENCE_TEMPLATES,
    SlotSpec,
    TemplateSpec,
    publish_reference_templates,
    template_draft,
    topology_facts,
)

__all__ = [
    "REFERENCE_TEMPLATES",
    "SOURCE_ID",
    "SWARM_NS",
    "SlotSpec",
    "TemplateSpec",
    "attach_swarm_topology",
    "ontology_ttl",
    "publish_reference_templates",
    "shapes_ttl",
    "template_draft",
    "topology_facts",
]
