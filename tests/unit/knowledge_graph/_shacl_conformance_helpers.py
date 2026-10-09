"""Shared SHACL-conformance proof for an emitted LPG graph slice.

``test_process_conformance.py`` and ``test_semantic_event_model.py`` each
prove that their emitted entity/link slice's RDF projection conforms to
``process_intelligence.shapes.ttl``, through the engine's real
``shacl_validate_ad_hoc`` surface (EH-431: AU never validates shapes
itself — see each call site for why a local ``pyshacl`` call would not
prove the same thing). This factors the shared RDF projection and
validation call so the two proofs stay identical only where the contract
actually is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

_SHAPES_PATH = (
    Path(__file__).parents[3]
    / "agent_utilities"
    / "knowledge_graph"
    / "shapes"
    / "process_intelligence.shapes.ttl"
)

__all__ = ["assert_graph_slice_conforms_to_process_intelligence_shapes"]


def _rdf_literal(rdflib: Any, xsd: Any, value: object) -> object | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value:
        return rdflib.Literal(value)
    if isinstance(value, int):
        return rdflib.Literal(value, datatype=xsd.integer)
    if isinstance(value, float):
        return rdflib.Literal(value, datatype=xsd.double)
    return None


def assert_graph_slice_conforms_to_process_intelligence_shapes(
    engine_graph: Any,
    entities: list[dict[str, Any]],
    links: list[dict[str, Any]],
    *,
    extra_type_declarations: dict[str, str] | None = None,
) -> None:
    """Project ``entities``/``links`` to RDF and assert SHACL conformance.

    ``extra_type_declarations`` maps a node id referenced by ``links`` but
    absent from ``entities`` (e.g. a ``ProcessPerspective`` target out of
    this slice's own scope) to the node type the shape contract requires it
    to carry, so the missing node still gets an ``rdf:type`` triple before
    the link edges are added.

    Requires a real engine; skips cleanly when ``rdflib`` is unavailable.
    """
    rdflib = pytest.importorskip("rdflib")
    from rdflib.namespace import RDF, XSD

    graph = rdflib.Graph()
    kg = rdflib.Namespace("http://knuckles.team/kg#")

    for entity in entities:
        subject = kg[f"node/{entity['id']}"]
        graph.add((subject, RDF.type, kg[entity["node_type"]]))
        for key, value in entity.items():
            if key in {"id", "node_type"}:
                continue
            object_value = _rdf_literal(rdflib, XSD, value)
            if object_value is not None:
                graph.add((subject, kg[key], object_value))

    node_iris = {entity["id"]: kg[f"node/{entity['id']}"] for entity in entities}
    for node_id, node_type in (extra_type_declarations or {}).items():
        graph.add((kg[f"node/{node_id}"], RDF.type, kg[node_type]))
        node_iris[node_id] = kg[f"node/{node_id}"]

    for link in links:
        graph.add(
            (
                node_iris[link["source"]],
                kg[link["relationship"]],
                node_iris[link["target"]],
            )
        )

    data_ttl = graph.serialize(format="turtle")
    if isinstance(data_ttl, bytes):
        data_ttl = data_ttl.decode()
    report = engine_graph.shacl_validate_ad_hoc(
        data_ttl, _SHAPES_PATH.read_text(encoding="utf-8")
    )
    assert report.conforms, report.results
