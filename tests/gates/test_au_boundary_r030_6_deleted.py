"""AU-BOUNDARY-R030.6: the local manifest compiler is gone with no importers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.gates._deletion_support import files_importing

_ROOT = Path(__file__).resolve().parents[2]
_MODULE = "agent_utilities/knowledge_graph/ontology/manifest_compiler.py"
_DOTTED = "agent_utilities.knowledge_graph.ontology.manifest_compiler"


@pytest.mark.spec("AU-BOUNDARY-R030.6")
def test_manifest_compiler_deleted_and_unimported() -> None:
    assert not (_ROOT / _MODULE).exists()
    hits = files_importing(
        [_ROOT / "agent_utilities", _ROOT / "scripts"], (_DOTTED,), _ROOT
    )
    assert hits == frozenset()
