"""Freeze the ruled ``pack:agent-utilities`` governance-shape body."""

from __future__ import annotations

import hashlib
from pathlib import Path

from rdflib import BNode, Graph, URIRef
from rdflib.compare import to_isomorphic

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "agent_utilities/ontology/shapes/governance.shapes.ttl"
KG = "http://knuckles.team/kg#"
SH = "http://www.w3.org/ns/shacl#"
OWNED_SHAPES = frozenset(
    {
        "AgentShape",
        "HookShape",
        "OntologyActionShape",
        "OptimizationPatternShape",
        "RequirementShape",
        "SoftwareFeatureShape",
        "SpecificationShape",
        "TestCaseShape",
        "ObjectEditShape",
        "ArtifactShape",
        "FragmentShape",
        "ChunkShape",
        "WorkflowDefinitionShape",
        "WorkflowStepShape",
    }
)
PACK_RAW_SHA256 = "6c0a88f7d60b0569c0c88d379f0026d434eb7fdadb8146f0ab4a0dbab5fc39e6"
PACK_SEMANTIC_DIGEST = (
    12032572029067540579955078512830142284492060502898443665828546204068198530876593
)
PACK_ID = "pack:agent-utilities"
PACK_RESOURCE_URI = "shapes://agent-utilities/governance.shapes.ttl"
PACK_MEDIA_TYPE = "text/turtle"


def test_agent_utilities_pack_is_frozen_semantic_authority() -> None:
    pack_bytes = PACK.read_bytes()
    pack = Graph().parse(data=pack_bytes.decode("utf-8"), format="turtle")

    assert len(pack) == 199
    assert (
        len({term for triple in pack for term in triple if isinstance(term, BNode)})
        == 30
    )
    assert hashlib.sha256(pack_bytes).hexdigest() == PACK_RAW_SHA256
    assert to_isomorphic(pack).graph_digest() == PACK_SEMANTIC_DIGEST

    node_shape = URIRef(f"{SH}NodeShape")
    rdf_type = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    actual_roots = {
        str(subject).removeprefix(KG)
        for subject in pack.subjects(rdf_type, node_shape)
        if str(subject).startswith(KG)
    }
    assert actual_roots == OWNED_SHAPES


def test_agent_utilities_pack_has_no_local_runtime_loader() -> None:
    forbidden = (
        "agent_utilities/ontology/shapes/governance.shapes.ttl",
        PACK_RESOURCE_URI,
    )
    consumers: list[str] = []
    for path in (ROOT / "agent_utilities").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(token in text for token in forbidden):
            consumers.append(str(path.relative_to(ROOT)))
    assert not consumers, (
        f"{PACK_ID} ({PACK_MEDIA_TYPE}) must be consumed only through "
        f"ConnectorPack/GraphSchema; local loaders found: {consumers}"
    )
