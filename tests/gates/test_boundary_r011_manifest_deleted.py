"""The connector manifest modules and certify script are deleted from Meristem."""

import pytest
from _census import assert_absent, console_scripts, references


@pytest.mark.spec("MERISTEM-BOUNDARY-R011.5")
def test_manifest_modules_and_script_absent() -> None:
    assert_absent(
        "knowledge_graph/ontology/connector_manifest.py",
        "knowledge_graph/ontology/connector_manifest_gate.py",
        "knowledge_graph/ontology/connector_manifests",
    )
    assert "arbora-certify-connector" not in console_scripts()
    assert references(("connector_manifest_gate", "arbora-certify-connector")) == []
