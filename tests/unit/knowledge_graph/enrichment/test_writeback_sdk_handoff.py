"""AU-SEMANTIC-R017.1 typed model and refusal tests."""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.enrichment.writeback.sdk_handoff import (
    VendorWritebackAuthorityError,
    VendorWritebackRequest,
    execute_local_vendor_writeback,
)


def test_vendor_writeback_request_is_typed() -> None:
    request = VendorWritebackRequest(
        connector="example-connector",
        source_instance="tenant-a",
        record_type="Document",
        payload={"id": "1"},
    )
    assert request.connector == "example-connector"
    assert request.source_instance == "tenant-a"
    assert request.record_type == "Document"
    assert request.payload == {"id": "1"}


def test_local_vendor_writeback_refuses() -> None:
    request = VendorWritebackRequest(
        connector="example-connector",
        source_instance="tenant-a",
        record_type="Document",
        payload={},
    )
    with pytest.raises(VendorWritebackAuthorityError):
        execute_local_vendor_writeback(request)
