"""Swarm topology through EG's assembly decision (SWARM-TOPOLOGY-DECIDE-DESIGN).

AU supplies the reference templates; EG owns the vocabulary (the core schema
source ``core:swarm-topology@1``, :mod:`.schema_source`) and decides; the reference templates (:mod:`.templates`)
are published agent-graph templates carrying typed topology facts; the
topology question itself (:mod:`agent_utilities.decide.consumers.topology`)
is an ``AgentAssemble`` request whose ``requirements.topology`` EG answers with
a certified plan or a typed abstention.
"""

from __future__ import annotations

from agent_utilities.decide.topology.schema_source import SOURCE_ID, SWARM_NS
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
    "publish_reference_templates",
    "template_draft",
    "topology_facts",
]
