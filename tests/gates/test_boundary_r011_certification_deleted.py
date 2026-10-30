"""Connector certification and attestation are deleted from Meristem."""

import pytest
from _census import assert_absent, references


@pytest.mark.spec("MERISTEM-BOUNDARY-R011.4")
def test_certification_modules_absent() -> None:
    assert_absent(
        "knowledge_graph/integrations/connector_certification.py",
        "knowledge_graph/integrations/connector_certification_cli.py",
        "knowledge_graph/integrations/connector_source_attestation.py",
    )
    assert (
        references(
            (
                "connector_certification",
                "connector_source_attestation",
            ),
            include_tests=True,
        )
        == []
    )
