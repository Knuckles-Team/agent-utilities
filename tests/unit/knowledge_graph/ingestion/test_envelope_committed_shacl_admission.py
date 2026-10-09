"""EH-385: connector admission validates against EG's committed GraphSchema.

AU never reads, parses or sends a shapes document on this path. It renders
the data graph and EG validates it against the composed schema it owns. The
engine rejects a report that is not bound to one committed schema snapshot
(``GraphComputeEngine._require_committed_shacl_receipt``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.ingestion import envelope_ingest
from tests.committed_shacl_fakes import (
    CommittedShaclValidator,
    shacl_report,
    shacl_result,
)


def _authority(validator: object) -> SimpleNamespace:
    return SimpleNamespace(shacl_validate_committed=validator)


def test_admission_sends_only_the_data_graph_and_admits_conforming_rows() -> None:
    validator = CommittedShaclValidator()
    envelope_ingest._shacl_validate_rows(
        _authority(validator), [("node:1", {"node_type": "Trace", "name": "t"})]
    )
    [data_graph] = validator.validations
    assert "Trace" in data_graph
    assert "sh:NodeShape" not in data_graph


def test_admission_resolves_the_validator_behind_an_engine_wrapper() -> None:
    validator = CommittedShaclValidator()
    engine = SimpleNamespace(graph_compute=_authority(validator))
    envelope_ingest.validate_rows_against_shacl(engine, [("n", {"node_type": "T"})])
    assert len(validator.validations) == 1


def test_non_conforming_rows_are_rejected_with_the_violation_named() -> None:
    report = shacl_report(
        conforms=False,
        results=(shacl_result(focus_node="node/n", message="name is required"),),
    )
    with pytest.raises(ValueError, match="name is required"):
        envelope_ingest._shacl_validate_rows(
            _authority(CommittedShaclValidator(report)), [("n", {"node_type": "T"})]
        )


def test_missing_validator_fails_closed() -> None:
    with pytest.raises(envelope_ingest.NativeChangeEnvelopeUnavailable):
        envelope_ingest._shacl_validate_rows(
            SimpleNamespace(client=object()), [("node:1", {"node_type": "T"})]
        )


def test_validator_error_and_untyped_report_fail_closed() -> None:
    def _boom(_data_graph: str) -> object:
        raise RuntimeError("SHACL validation was not bound to one committed snapshot")

    with pytest.raises(envelope_ingest.NativeChangeEnvelopeUnavailable):
        envelope_ingest._shacl_validate_rows(_authority(_boom), [("n", {})])
    with pytest.raises(envelope_ingest.NativeChangeEnvelopeUnavailable):
        envelope_ingest._shacl_validate_rows(
            _authority(lambda _d: {"conforms": True}), [("n", {})]
        )
