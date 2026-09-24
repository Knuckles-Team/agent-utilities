"""EH-380: connector SHACL admission reads the shapes pack that actually ships.

43197d7c6 moved ``governance.shapes.ttl`` to ``agent_utilities/ontology/shapes``
(the frozen ``pack:agent-utilities`` body, see
``tests/gates/test_agent_utilities_governance_pack.py``) but left
``envelope_ingest`` reading the old ``knowledge_graph/shapes`` path, so every
connector ChangeEnvelope failed closed before reaching the engine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_utilities.knowledge_graph.ingestion import envelope_ingest

_PACK = (
    Path(__file__).resolve().parents[4]
    / "agent_utilities/ontology/shapes/governance.shapes.ttl"
)


def test_admission_shapes_resource_is_the_published_pack() -> None:
    assert envelope_ingest._GOVERNANCE_SHAPES.is_file()
    assert envelope_ingest._GOVERNANCE_SHAPES.read_bytes() == _PACK.read_bytes()


def test_validator_receives_the_pack_and_admits_conforming_rows() -> None:
    seen: list[str] = []

    class _Rdf:
        @staticmethod
        def validate_shacl(shapes: str, _data: str) -> dict[str, object]:
            seen.append(shapes)
            return {"conforms": True, "results": []}

    client = type("_Client", (), {"rdf": _Rdf()})()
    envelope_ingest._shacl_validate_rows(client, [("node:1", {"node_type": "Trace"})])
    assert seen == [_PACK.read_text(encoding="utf-8")]


def test_missing_validator_still_fails_closed() -> None:
    client = type("_Client", (), {"rdf": None})()
    with pytest.raises(envelope_ingest.NativeChangeEnvelopeUnavailable):
        envelope_ingest._shacl_validate_rows(client, [("node:1", {"node_type": "T"})])
