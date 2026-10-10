"""AU-BOUNDARY-R030.3: ``scripts/update_ontology_lock.py`` is wired off AU's
hand-written ``agent_utilities.knowledge_graph.ontology.manifest_compiler``
and onto the SDK's typed, source-agnostic pack compiler
(``agent_connector_sdk.manifest.ontology_pack.compile_manifest_ontology``) —
mirroring the re-validate-then-compile pattern
``connector_manifest_gate._compiled_manifest_graph`` already uses for the
compile-before-sync gate (AU-BOUNDARY-R030.1).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.knowledge_graph.ontology._sdk_pack_support import (
    assert_local_compiler_dropped,
    assert_sdk_pack_wired,
)

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "update_ontology_lock.py"


@pytest.mark.spec("AU-BOUNDARY-R030.3")
def test_update_ontology_lock_no_longer_imports_the_local_compiler():
    assert_local_compiler_dropped(_SCRIPT)


@pytest.mark.spec("AU-BOUNDARY-R030.3")
def test_update_ontology_lock_calls_the_sdk_pack_compiler():
    assert_sdk_pack_wired(_SCRIPT)
