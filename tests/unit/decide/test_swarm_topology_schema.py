"""ST-1 / ST-2: the swarm-topology vocabulary, its shapes and the reference templates.

The TBox is the data EG reasons (admissibility entails from it, proofs
included); the shapes validate AU's RDF projection of a template; every
reference template must conform (positive fixtures) and each planted defect
must not (negative fixtures). EG enforces the same rules programmatically at
publish (``TemplateTopologyShape``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pyshacl
import pytest
import rdflib
from rdflib.namespace import OWL, RDFS

from agent_utilities.decide.topology import (
    REFERENCE_TEMPLATES,
    SOURCE_ID,
    SWARM_NS,
    TemplateSpec,
    attach_swarm_topology,
    ontology_ttl,
    publish_reference_templates,
    shapes_ttl,
    template_draft,
    topology_facts,
)

SWARM = rdflib.Namespace(SWARM_NS)
EG = rdflib.Namespace("http://epistemic-graph/owl#")


def _tbox() -> rdflib.Graph:
    return rdflib.Graph().parse(data=ontology_ttl(), format="turtle")


def _projection(spec: TemplateSpec) -> str:
    """A template's RDF projection in the swarm vocabulary."""
    facts = topology_facts(spec)
    lines = [
        f"@prefix swarm: <{SWARM_NS}> .",
        f"<urn:t> a swarm:TopologyTemplate ; swarm:class <{facts['class_iri']}> .",
    ]
    kinds = [kind for _, kind in spec.control]
    lines += [f'<urn:t> swarm:nodeKind "{kind}" .' for kind in kinds]
    for index, slot in enumerate(facts["slots"]):
        role = slot["role"].capitalize()
        lines.append(
            f'<urn:t> swarm:hasSlot <urn:s{index}> . <urn:s{index}> a swarm:Slot ; '
            f'swarm:nodeId "{slot["node_id"]}" ; swarm:role swarm:{role} ; '
            f"swarm:minWidth {slot['min_width']} ; swarm:maxWidth {slot['max_width']} ; "
            f"swarm:maxRounds {slot['max_rounds']} ."
        )
        for n, lease in enumerate(slot["lease"]):
            lines.append(
                f"<urn:s{index}> swarm:lease <urn:l{index}{n}> . <urn:l{index}{n}> a "
                f'swarm:LeaseDemand ; swarm:resourceClass "{lease["class"]}" ; '
                f"swarm:amount {lease['amount']} ."
            )
    lines.append(_stop_ttl(facts["stop"]))
    return "\n".join(lines)


_STOP_CLASS = {
    "max_rounds": "MaxRoundsStop",
    "quorum": "QuorumStop",
    "verifier_pass": "VerifierPassStop",
    "budget": "BudgetStop",
    "deadline": "DeadlineStop",
}


def _stop_ttl(stop: Mapping[str, Any]) -> str:
    numbers = "".join(
        f" ; swarm:{key} {stop[key]}" for key in ("k", "n") if key in stop
    )
    return (
        f"<urn:t> swarm:hasStopRule <urn:stop> . "
        f"<urn:stop> a swarm:{_STOP_CLASS[stop['rule']]}{numbers} ."
    )


def _conforms(data_ttl: str) -> bool:
    data = rdflib.Graph().parse(data=data_ttl, format="turtle")
    shapes = rdflib.Graph().parse(data=shapes_ttl(), format="turtle")
    conforms, _graph, _text = pyshacl.validate(
        data, shacl_graph=shapes, ont_graph=_tbox(), inference="none", advanced=True
    )
    return bool(conforms)


def _spec(graph_id: str) -> TemplateSpec:
    return next(spec for spec in REFERENCE_TEMPLATES if spec.graph_id == graph_id)


def test_the_vocabulary_hangs_off_the_engine_s_two_anchors() -> None:
    tbox = _tbox()
    assert (SWARM.admits, RDFS.subPropertyOf, EG.admitsTopology) in tbox
    assert (SWARM.NeedsIndependentCheck, RDFS.subClassOf, EG.NeedsIndependentCheck) in tbox
    fillers = set(tbox.objects(None, OWL.someValuesFrom))
    declared = set(tbox.subjects(RDFS.subClassOf, SWARM.Topology)) | {
        SWARM.Debate,
        SWARM.Council,
        SWARM.CritiqueLoop,
    }
    assert fillers <= declared, "every admitted class is a declared topology class"
    for spec in REFERENCE_TEMPLATES:
        assert rdflib.URIRef(spec.class_iri) in declared, spec.graph_id


def test_a_sequential_span_never_admits_a_fan_out() -> None:
    tbox = _tbox()
    restrictions = [
        tbox.value(node, OWL.someValuesFrom)
        for node in tbox.objects(SWARM.SequentialDependency, RDFS.subClassOf)
    ]
    assert SWARM.FanOutJoin not in restrictions
    assert (SWARM.SequentialDependency, RDFS.subClassOf, SWARM.IndependentSubtasks) not in tbox


@pytest.mark.parametrize("spec", REFERENCE_TEMPLATES, ids=lambda s: s.graph_id)
def test_every_reference_template_conforms(spec: TemplateSpec) -> None:
    assert _conforms(_projection(spec)), spec.graph_id


def _without_join(spec: TemplateSpec) -> TemplateSpec:
    return replace(spec, control=tuple(c for c in spec.control if c[1] != "join"))


def _inverted_width(spec: TemplateSpec) -> TemplateSpec:
    slots = (replace(spec.slots[0], widths=(3, 2)), *spec.slots[1:])
    return replace(spec, slots=slots)


def _unknown_resource(spec: TemplateSpec) -> TemplateSpec:
    slots = (replace(spec.slots[0], lease=(("quantum", 1),)), *spec.slots[1:])
    return replace(spec, slots=slots)


def _half_quorum(spec: TemplateSpec) -> TemplateSpec:
    return replace(spec, stop={"rule": "quorum", "k": 2})


def _verifier_pass_without_verifier(spec: TemplateSpec) -> TemplateSpec:
    return replace(spec, stop={"rule": "verifier_pass", "max_rounds": 2})


@pytest.mark.parametrize(
    ("plant", "graph_id"),
    [
        (_without_join, "swarm:fan-out-join"),
        (_inverted_width, "swarm:fan-out-join"),
        (_unknown_resource, "swarm:single"),
        (_half_quorum, "swarm:council"),
        (_verifier_pass_without_verifier, "swarm:pipeline"),
    ],
    ids=lambda value: getattr(value, "__name__", str(value)),
)
def test_each_planted_defect_is_flagged(plant: Any, graph_id: str) -> None:
    assert _conforms(_projection(_spec(graph_id)))
    assert not _conforms(_projection(plant(_spec(graph_id))))


class _Compute:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def graph_schema_attach(self, source_id: str, **documents: Any) -> str:
        self.calls.append((source_id, documents))
        return "attached"


def test_the_vocabulary_is_attached_as_one_keyed_schema_source() -> None:
    compute = _Compute()
    assert attach_swarm_topology(compute, if_composed_digest="sha256:ab") == "attached"
    [(source_id, documents)] = compute.calls
    assert source_id == SOURCE_ID
    assert documents["ontology_ttl"] == ontology_ttl()
    assert documents["shapes_ttl"] == shapes_ttl()
    assert documents["if_composed_digest"] == "sha256:ab"


class _Graphs:
    def __init__(self) -> None:
        self.published: list[tuple[dict[str, Any], dict[str, Any], str | None]] = []

    async def publish_graph(
        self, draft: Any, context: Any, *, idempotency_key: str | None = None
    ) -> str:
        self.published.append((draft, context, idempotency_key))
        return str(draft["graph_id"])


_CONTEXT = {
    "tenant_id": "tenant-a",
    "actor_scope": "operator",
    "purpose_id": "agent-graph:publish",
    "policy_digest": "sha256:" + "1" * 64,
}
_SLOT_AGENT = {"agent_id": "agent:slot", "definition_digest": "sha256:" + "2" * 64}


def test_the_publisher_sends_every_template_with_its_topology_facts() -> None:
    graphs = _Graphs()

    async def context_for(graph_id: str) -> dict[str, Any]:
        return dict(_CONTEXT)

    ids = asyncio.run(publish_reference_templates(graphs, context_for, _SLOT_AGENT))
    assert ids == [spec.graph_id for spec in REFERENCE_TEMPLATES]
    for (draft, _context, key), spec in zip(graphs.published, REFERENCE_TEMPLATES, strict=True):
        assert key == f"swarm-template:{spec.graph_id}"
        assert draft["topology"] == topology_facts(spec)
        agents = [n for n in draft["shape"]["nodes"] if isinstance(n["kind"], dict)]
        assert {n["node_id"] for n in agents} == {s.node_id for s in spec.slots}
        assert all(n["kind"]["agent"] == _SLOT_AGENT for n in agents)


def test_a_draft_carries_only_declared_optional_facts() -> None:
    spec = _spec("swarm:single")
    quiet = replace(spec, slots=(replace(spec.slots[0], p95_ms=None, tokens=None),))
    slot = template_draft(quiet, _SLOT_AGENT, _CONTEXT)["topology"]["slots"][0]
    assert "p95_ms" not in slot and "tokens" not in slot and "harness" not in slot
