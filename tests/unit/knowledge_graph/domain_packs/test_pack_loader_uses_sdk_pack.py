"""AU-BOUNDARY-R030.5: ``pack_loader`` compiles through the SDK's ontology-pack
compiler, not AU's local ``manifest_compiler`` copy.

Mirrors ``connector_manifest_gate``'s own R030 migration
(``_compiled_manifest_graph``): re-validate AU's ``ConnectorManifest`` against
``agent_connector_sdk.manifest.model.ConnectorManifest`` and compile through
``agent_connector_sdk.manifest.ontology_pack.compile_manifest_ontology_spec``
instead of the local ``manifest_compiler.compile_manifest``.
"""

from __future__ import annotations

import sys

import _fixtures
import pytest

from agent_utilities.knowledge_graph.domain_packs import pack_loader
from agent_utilities.knowledge_graph.domain_packs.pack_loader import load_pack


@pytest.mark.spec("AU-BOUNDARY-R030.5")
def test_pack_loader_module_does_not_import_local_manifest_compiler() -> None:
    """The loader's own module no longer names the local compiler at all —
    not as an import, not as a lazily-imported call site."""
    source = sys.modules[pack_loader.__name__].__file__
    assert source is not None
    with open(source, encoding="utf-8") as fh:
        text = fh.read()
    assert "manifest_compiler" not in text
    assert "agent_utilities.knowledge_graph.ontology.manifest_compiler" not in text


@pytest.mark.spec("AU-BOUNDARY-R030.5")
def test_pack_loader_compiles_via_sdk_ontology_pack(tmp_path) -> None:
    """``load_pack`` still produces a correct, usable ``OntologySpec`` — now
    compiled through the SDK's ``compile_manifest_ontology_spec`` instead of
    AU's retired local copy."""
    manifest = _fixtures.build_manifest()
    pack_dir = _fixtures.write_pack(tmp_path, manifest)

    loaded = load_pack(pack_dir)

    class_names = {c.local for c in loaded.ontology_spec.classes}
    assert class_names == {"Runbook", "Document", "Person"}
    # The datatype property declared on the Runbook->Document crosswalk
    # mapping survives the SDK round trip.
    assert any(d.local == "name" for d in loaded.ontology_spec.datatype_properties)
    assert loaded.ontology_spec.type_map["Runbook"] == ("Runbook", "runbook")
