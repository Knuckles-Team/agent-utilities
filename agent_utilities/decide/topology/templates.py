"""The reference swarm topologies as published agent-graph templates.

One data table (:data:`REFERENCE_TEMPLATES`) and one publisher. Each entry is
an agent-graph TEMPLATE: its ``Agent`` nodes are the slots EG's assembly fills
with assembled agents, and its typed topology facts (class IRI, per-slot role,
width and round ranges, declared p95 and tokens, per-agent lease, stop rule)
sit inside the template's definition digest, so pinning a template pins its
topology. Every declared number is a publisher CLAIM an operator may
republish; EG never reads it as an observation.

Slot nodes pin an operator-provided placeholder agent (any published agent):
publish admission resolves every pin, and assembly replaces the pin with the
agent it assembles for that slot.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.decide.topology.schema_source import SWARM_NS

#: What one agent turn declares by default: a claim, never an observation.
DECLARED_P95_MS = 30_000
DECLARED_TOKENS = 4_000


@dataclass(frozen=True, slots=True)
class SlotSpec:
    """One slot: a template ``Agent`` node and its topology facts."""

    node_id: str
    role: str
    widths: tuple[int, int] = (1, 1)
    rounds: int = 1
    lease: tuple[tuple[str, int], ...] = (("llm_generator", 1),)
    p95_ms: int | None = DECLARED_P95_MS
    tokens: int | None = DECLARED_TOKENS
    harness: str | None = None


@dataclass(frozen=True, slots=True)
class TemplateSpec:
    """One reference topology: its class, slots, stop rule and shape."""

    graph_id: str
    class_local: str
    stop: Mapping[str, Any]
    slots: tuple[SlotSpec, ...]
    #: ``(node_id, kind)`` of the non-slot nodes: fanout, join, end.
    control: tuple[tuple[str, str], ...]
    edges: tuple[tuple[str, str], ...]
    entry: str
    max_iterations: int = 16
    depth: int = 1
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def class_iri(self) -> str:
        return SWARM_NS + self.class_local


def _chain(*nodes: str) -> tuple[tuple[str, str], ...]:
    return tuple(zip(nodes, nodes[1:], strict=False))


_END = (("end", "end"),)
_FAN = (("fan", "fanout"), ("join", "join"))

#: The reference topologies covering the single-agent, pipeline, fan-out/join,
#: supervisor-with-workers, critique-loop and council shapes (AU-CONTROL-R010).
REFERENCE_TEMPLATES: tuple[TemplateSpec, ...] = (
    TemplateSpec(
        graph_id="swarm:single",
        class_local="Single",
        stop={"rule": "max_rounds", "n": 1},
        slots=(SlotSpec("agent", "parent"),),
        control=_END,
        edges=_chain("agent", "end"),
        entry="agent",
    ),
    TemplateSpec(
        graph_id="swarm:pipeline",
        class_local="Pipeline",
        stop={"rule": "max_rounds", "n": 1},
        slots=(SlotSpec("draft", "parent"), SlotSpec("refine", "aggregator")),
        control=_END,
        edges=_chain("draft", "refine", "end"),
        entry="draft",
    ),
    TemplateSpec(
        graph_id="swarm:fan-out-join",
        class_local="FanOutJoin",
        stop={"rule": "max_rounds", "n": 1},
        slots=(
            SlotSpec("lead", "parent"),
            SlotSpec("worker", "child", widths=(1, 8)),
            SlotSpec("merge", "aggregator"),
        ),
        control=_FAN + _END,
        edges=_chain("lead", "fan", "worker", "join", "merge", "end"),
        entry="lead",
    ),
    TemplateSpec(
        graph_id="swarm:supervisor-workers",
        class_local="SupervisorWorkers",
        stop={"rule": "budget"},
        slots=(
            SlotSpec("supervisor", "parent"),
            SlotSpec("worker", "child", widths=(1, 8)),
        ),
        control=_FAN + _END,
        edges=_chain("supervisor", "fan", "worker", "join", "end"),
        entry="supervisor",
    ),
    TemplateSpec(
        graph_id="swarm:critique-loop",
        class_local="CritiqueLoop",
        stop={"rule": "verifier_pass", "max_rounds": 4},
        slots=(
            SlotSpec("author", "peer", rounds=4),
            SlotSpec("critic", "verifier", rounds=4),
        ),
        control=_END,
        edges=(*_chain("author", "critic", "end"), ("critic", "author")),
        entry="author",
    ),
    TemplateSpec(
        graph_id="swarm:council",
        class_local="Council",
        stop={"rule": "quorum", "k": 2, "n": 3},
        slots=(
            SlotSpec("member", "peer", widths=(3, 5)),
            SlotSpec("chair", "aggregator"),
        ),
        control=_FAN + _END,
        edges=_chain("fan", "member", "join", "chair", "end"),
        entry="fan",
    ),
)


def _slot_facts(slot: SlotSpec) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "node_id": slot.node_id,
        "role": slot.role,
        "min_width": slot.widths[0],
        "max_width": slot.widths[1],
        "max_rounds": slot.rounds,
        "lease": [{"class": c, "amount": a} for c, a in slot.lease],
    }
    optional = {"p95_ms": slot.p95_ms, "tokens": slot.tokens, "harness": slot.harness}
    facts.update({k: v for k, v in optional.items() if v is not None})
    return facts


def topology_facts(spec: TemplateSpec) -> dict[str, Any]:
    """The template's ``TopologyFacts`` wire body."""
    return {
        "class_iri": spec.class_iri,
        "depth": spec.depth,
        "slots": [_slot_facts(slot) for slot in spec.slots],
        "stop": dict(spec.stop),
    }


def _node(node_id: str, kind: Any) -> dict[str, Any]:
    return {"node_id": node_id, "kind": kind}


def _nodes(spec: TemplateSpec, slot_agent: Mapping[str, str]) -> list[dict[str, Any]]:
    pin = {
        "agent": {
            "agent_id": str(slot_agent["agent_id"]),
            "definition_digest": str(slot_agent["definition_digest"]),
        }
    }
    slots = [_node(slot.node_id, pin) for slot in spec.slots]
    return slots + [_node(node_id, kind) for node_id, kind in spec.control]


def template_draft(
    spec: TemplateSpec, slot_agent: Mapping[str, str], context: Mapping[str, Any]
) -> dict[str, Any]:
    """The ``AgentGraphDraft`` publishing ``spec``; tenant/actor/purpose/policy
    come from the mutation ``context`` (EG rebinds them to the verified caller).
    """
    return {
        "graph_id": spec.graph_id,
        "version": "1",
        "shape": {
            "entry_node": spec.entry,
            "nodes": _nodes(spec, slot_agent),
            "edges": [{"from": a, "to": b} for a, b in spec.edges],
            "max_iterations": spec.max_iterations,
        },
        "tenant_id": str(context["tenant_id"]),
        "actor_scope": str(context["actor_scope"]),
        "purpose_id": str(context["purpose_id"]),
        "policy_digest": str(context["policy_digest"]),
        "topology": topology_facts(spec),
    }


#: Mints the ``AgentLibraryMutationContext`` for publishing one template.
ContextFor = Callable[[str], Awaitable[Mapping[str, Any]]]


async def publish_reference_templates(
    graphs: Any,
    context_for: ContextFor,
    slot_agent: Mapping[str, str],
    *,
    specs: Sequence[TemplateSpec] = REFERENCE_TEMPLATES,
) -> list[Any]:
    """Publish every reference template through ``AgentGraph.publish`` (L3).

    ``context_for`` is the policy owner's context minting (graph-os); AU never
    mints a mutation context itself. EG refuses a template whose topology
    facts fail ``TemplateTopologyShape`` (``TOPOLOGY_SHAPE_INVALID``).
    """
    published = []
    for spec in specs:
        context = await context_for(spec.graph_id)
        draft = template_draft(spec, slot_agent, context)
        published.append(
            await graphs.publish_graph(
                draft, context, idempotency_key=f"swarm-template:{spec.graph_id}"
            )
        )
    return published


__all__ = [
    "DECLARED_P95_MS",
    "DECLARED_TOKENS",
    "REFERENCE_TEMPLATES",
    "ContextFor",
    "SlotSpec",
    "TemplateSpec",
    "publish_reference_templates",
    "template_draft",
    "topology_facts",
]
