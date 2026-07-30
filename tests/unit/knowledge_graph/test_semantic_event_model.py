"""Canonical OCEL/tEKG/neural graph boundary proofs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_utilities.knowledge_graph.ingestion.semantic_event_model import (
    BusinessObject,
    EntityResolutionProposal,
    EventObjectParticipation,
    NeuralRelationPrediction,
    NeuralRepresentation,
    ObjectCentricGraphSlice,
    ObjectState,
    ProcessEvent,
    ProcessPerspective,
    QualifiedObjectRelationship,
    SemanticEntityRef,
)


def _slice_payload() -> dict:
    return {
        "log_id": "log-1",
        "source_ref": "fixture://ocel/orders",
        "mapping_version": "orders-v1",
        "event_types": [
            {"name": "create order", "attributes": []},
        ],
        "object_types": [
            {
                "name": "Order",
                "attributes": [{"name": "status", "value_type": "string"}],
            },
            {"name": "Item", "attributes": []},
        ],
        "objects": [
            {
                "object_id": "order-1",
                "object_type": "Order",
                "attributes": [
                    {
                        "name": "status",
                        "value": "new",
                        "valid_from": "2026-01-01T00:00:00Z",
                    }
                ],
            },
            {"object_id": "item-1", "object_type": "Item"},
        ],
        "events": [
            {
                "event_id": "event-1",
                "activity": "create order",
                "occurred_at": "2026-01-01T00:00:00Z",
                "source_ref": "fixture://ocel/orders/event-1",
                "objects": [
                    {
                        "object_id": "order-1",
                        "object_type": "Order",
                        "qualifier": "order",
                    },
                    {
                        "object_id": "item-1",
                        "object_type": "Item",
                        "qualifier": "line-item",
                    },
                ],
            }
        ],
        "object_states": [
            {
                "state_id": "state-order-1-v1",
                "object_id": "order-1",
                "object_type": "Order",
                "valid_from": "2026-01-01T00:00:00Z",
                "valid_to": "2026-01-02T00:00:00Z",
                "observed_at": "2026-01-01T00:00:01Z",
                "attributes": [{"name": "status", "value": "new"}],
            }
        ],
        "object_relationships": [
            {
                "relationship_id": "rel-order-item-1",
                "source_object_id": "order-1",
                "target_object_id": "item-1",
                "qualifier": "contains",
            }
        ],
        "perspectives": [
            {
                "perspective_id": "orders",
                "object_types": ["Order", "Item"],
                "qualifiers": ["order", "line-item"],
                "derivation_version": "v1",
            }
        ],
        "neural_representations": [
            {
                "representation_id": "repr-order-1",
                "target": {"kind": "object", "source_id": "order-1"},
                "encoder_id": "fixture-encoder",
                "encoder_version": "1",
                "dimension": 8,
                "artifact_ref": "artifact:fixture-encoder/order-1",
                "source_graph_epoch": 4,
                "content_hash": "a" * 64,
            }
        ],
        "neural_predictions": [
            {
                "prediction_id": "prediction-1",
                "subject": {"kind": "object", "source_id": "order-1"},
                "predicate": "mayContain",
                "object": {"kind": "object", "source_id": "item-1"},
                "score": 0.8,
                "uncertainty": 0.2,
                "model_ref": "model:fixture-relation/1",
                "candidate_set_ref": "candidates:fixture/1",
                "evidence_refs": ["fixture://ocel/orders/event-1"],
            }
        ],
        "entity_resolution_proposals": [
            {
                "proposal_id": "resolution-1",
                "mention_ref": "fixture://document/region/1",
                "candidate": {"kind": "object", "source_id": "order-1"},
                "score": 0.75,
                "blocking_features": ["exact_type", "normalized_identifier"],
                "evidence_refs": ["fixture://document/region/1"],
            }
        ],
    }


def test_compiles_one_lossless_symbolic_and_proposal_graph_slice() -> None:
    model = ObjectCentricGraphSlice.model_validate(_slice_payload())
    entities, links = model.to_graph_slice()

    assert len(model.canonical_digest()) == 64
    assert {item["node_type"] for item in entities} >= {
        "ObjectCentricEventLog",
        "ProcessEventType",
        "BusinessObjectType",
        "BusinessObject",
        "ProcessEvent",
        "EventObjectParticipation",
        "ObjectState",
        "ObjectRelationship",
        "ProcessPerspective",
        "NeuralRepresentation",
        "NeuralRelationPrediction",
        "EntityResolutionProposal",
    }
    assert {item["relationship"] for item in links} >= {
        "HAS_OBJECT",
        "HAS_EVENT",
        "HAS_EVENT_TYPE",
        "HAS_OBJECT_TYPE",
        "INSTANCE_OF_EVENT_TYPE",
        "INSTANCE_OF_OBJECT_TYPE",
        "HAS_PARTICIPATION",
        "PARTICIPATES_AS",
        "STATE_OF",
        "RELATION_SOURCE",
        "RELATION_TARGET",
        "HAS_NEURAL_REPRESENTATION",
        "PREDICTS_SUBJECT",
        "PREDICTS_OBJECT",
        "PROPOSES_RESOLUTION",
    }
    prediction = next(
        item for item in entities if item["node_type"] == "NeuralRelationPrediction"
    )
    assert prediction["decision_status"] == "proposed"
    assert prediction["prediction_score"] == 0.8
    assert "subject" not in prediction
    assert "object" not in prediction
    assert "embedding" not in prediction


def test_digest_and_graph_projection_are_input_order_stable() -> None:
    payload = _slice_payload()
    reordered = deepcopy(payload)
    reordered["objects"].reverse()
    reordered["events"][0]["objects"].reverse()
    reordered["perspectives"][0]["object_types"].reverse()
    first = ObjectCentricGraphSlice.model_validate(payload)
    second = ObjectCentricGraphSlice.model_validate(reordered)

    assert first.canonical_digest() == second.canonical_digest()
    assert first.to_graph_slice() == second.to_graph_slice()


def test_rejects_dangling_or_type_mismatched_symbolic_references() -> None:
    payload = _slice_payload()
    payload["events"][0]["objects"][0]["object_type"] = "Invoice"
    with pytest.raises(ValidationError, match="same object_type"):
        ObjectCentricGraphSlice.model_validate(payload)

    payload = _slice_payload()
    payload["neural_predictions"][0]["object"]["source_id"] = "missing"
    with pytest.raises(ValidationError, match="same validated slice"):
        ObjectCentricGraphSlice.model_validate(payload)


def test_rejects_values_that_violate_ocel_type_declarations() -> None:
    payload = _slice_payload()
    payload["objects"][0]["attributes"][0]["value"] = True
    with pytest.raises(ValidationError, match="declared OCEL names and types"):
        ObjectCentricGraphSlice.model_validate(payload)


def test_rejects_invalid_temporal_windows_and_naive_times() -> None:
    with pytest.raises(ValidationError, match="later than"):
        ObjectState(
            state_id="state-1",
            object_id="object-1",
            object_type="Order",
            valid_from="2026-01-02T00:00:00Z",  # type: ignore[arg-type]
            valid_to="2026-01-01T00:00:00Z",  # type: ignore[arg-type]
            observed_at="2026-01-02T00:00:00Z",  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError, match="timezone"):
        ProcessEvent(
            event_id="event-1",
            activity="create",
            occurred_at="2026-01-01T00:00:00",  # type: ignore[arg-type]
            objects=(
                EventObjectParticipation(object_id="object-1", object_type="Order"),
            ),
            source_ref="fixture://event-1",
        )


def test_neural_outputs_and_resolution_are_proposal_only() -> None:
    ref = SemanticEntityRef(kind="object", source_id="object-1")
    with pytest.raises(ValidationError):
        NeuralRelationPrediction(
            prediction_id="prediction-1",
            subject=ref,
            predicate="relatedTo",
            object=ref,
            score=0.9,
            uncertainty=0.1,
            model_ref="model:1",
            candidate_set_ref="candidates:1",
            evidence_refs=("evidence:1",),
            decision_status="accepted",  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError):
        EntityResolutionProposal(
            proposal_id="proposal-1",
            mention_ref="mention:1",
            candidate=ref,
            score=0.9,
            blocking_features=("exact",),
            evidence_refs=("evidence:1",),
            decision_status="rejected",  # type: ignore[arg-type]
        )


def test_contract_types_remain_directly_constructible_for_owned_consumers() -> None:
    assert BusinessObject(object_id="object-1", object_type="Order").object_type == (
        "Order"
    )
    assert (
        QualifiedObjectRelationship(
            relationship_id="relationship-1",
            source_object_id="object-1",
            target_object_id="object-2",
            qualifier="contains",
        ).qualifier
        == "contains"
    )
    assert ProcessPerspective(
        perspective_id="orders",
        object_types=("Order",),
        derivation_version="1",
    ).object_types == ("Order",)
    assert (
        NeuralRepresentation(
            representation_id="repr-1",
            target=SemanticEntityRef(kind="object", source_id="object-1"),
            encoder_id="encoder",
            encoder_version="1",
            dimension=8,
            artifact_ref="artifact:repr-1",
            source_graph_epoch=1,
            content_hash="b" * 64,
        ).dimension
        == 8
    )


def test_emitted_lpg_vocabulary_conforms_to_process_intelligence_shapes() -> None:
    """Prove the executable ChangeEnvelope names match the semantic contract."""
    rdflib = pytest.importorskip("rdflib")
    pyshacl = pytest.importorskip("pyshacl")
    from rdflib.namespace import RDF, XSD

    model = ObjectCentricGraphSlice.model_validate(_slice_payload())
    entities, links = model.to_graph_slice()
    graph = rdflib.Graph()
    kg = rdflib.Namespace("http://knuckles.team/kg#")

    def literal(value: object) -> object | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, str) and value:
            return rdflib.Literal(value)
        if isinstance(value, int):
            return rdflib.Literal(value, datatype=XSD.integer)
        if isinstance(value, float):
            return rdflib.Literal(value, datatype=XSD.double)
        return None

    for entity in entities:
        subject = kg[f"node/{entity['id']}"]
        graph.add((subject, RDF.type, kg[entity["node_type"]]))
        for key, value in entity.items():
            if key in {"id", "node_type"}:
                continue
            object_value = literal(value)
            if object_value is not None:
                graph.add((subject, kg[key], object_value))
    node_iris = {entity["id"]: kg[f"node/{entity['id']}"] for entity in entities}
    for link in links:
        graph.add(
            (
                node_iris[link["source"]],
                kg[link["relationship"]],
                node_iris[link["target"]],
            )
        )

    shapes_path = (
        Path(__file__).parents[3]
        / "agent_utilities"
        / "knowledge_graph"
        / "shapes"
        / "process_intelligence.shapes.ttl"
    )
    conforms, _, report = pyshacl.validate(
        graph,
        shacl_graph=str(shapes_path),
        inference="none",
    )
    assert conforms, report
