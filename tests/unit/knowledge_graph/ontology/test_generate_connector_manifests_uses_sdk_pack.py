"""AU-BOUNDARY-R030.2: ``scripts/generate_connector_manifests.py`` must no longer
import AU's hand-written Turtle emitter
(``agent_utilities.knowledge_graph.ontology.manifest_compiler``); it compiles
through the SDK's typed, source-agnostic pack compiler instead
(``agent_connector_sdk.manifest.ontology_pack.compile_manifest_ontology``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from agent_connector_sdk.manifest.model import ConnectorManifest as SDKConnectorManifest
from agent_connector_sdk.manifest.ontology_pack import compile_manifest_ontology

from tests.unit.knowledge_graph.ontology._sdk_pack_support import (
    imported_modules,
    parse_script,
)

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[4] / "scripts" / "generate_connector_manifests.py"
)

_SPEC = importlib.util.spec_from_file_location(
    "generate_connector_manifests_r030_2", _SCRIPT_PATH
)
gen = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(gen)


@pytest.mark.spec("AU-BOUNDARY-R030.2")
def test_generate_connector_manifests_uses_sdk_pack() -> None:
    imported = imported_modules(parse_script(_SCRIPT_PATH))

    # No importer of the local emitter module remains in this script.
    assert "agent_utilities.knowledge_graph.ontology.manifest_compiler" not in imported
    assert not hasattr(gen, "compile_manifest")
    assert not hasattr(gen, "export_manifest_ttl")

    # The script is wired onto the SDK's pack compiler instead.
    assert "agent_connector_sdk.manifest.ontology_pack" in imported
    assert "agent_connector_sdk.manifest.model" in imported
    assert gen.compile_manifest_ontology is compile_manifest_ontology
    assert gen.SDKConnectorManifest is SDKConnectorManifest
