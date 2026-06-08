"""Plan 05 — every knowledge_graph/*.ttl must parse, and the deepened
ontologies must carry at least 40 ``owl:Restriction`` axioms in total.

These guard the OWL/SHACL logic added in Plan 05 against syntax regressions.
"""

from __future__ import annotations

import glob
import importlib.util
from pathlib import Path

import pytest

rdflib = pytest.importorskip("rdflib")


def _kg_dir() -> Path:
    """Locate the bundled ``knowledge_graph`` directory holding the *.ttl files."""
    spec = importlib.util.find_spec("agent_utilities.knowledge_graph")
    assert spec is not None and spec.origin is not None
    return Path(spec.origin).parent


KG_DIR = _kg_dir()
TTL_FILES = sorted(glob.glob(str(KG_DIR / "*.ttl")))


def test_expected_ttl_count() -> None:
    """At least the 20 ontology TTL files Plan 05 operates on must be present.

    The domain-ontology library has grown well past the original 20 (banking, legal,
    medical, government, …); every file found is parse-checked by test_each_ttl_parses
    below, so this guards the Plan-05 baseline against accidental removal rather than
    pinning an exact count that legitimately grows.
    """
    assert len(TTL_FILES) >= 20, (
        f"expected >= 20 ttl files, found {len(TTL_FILES)}: {TTL_FILES}"
    )


@pytest.mark.parametrize("ttl_path", TTL_FILES, ids=lambda p: Path(p).name)
def test_each_ttl_parses(ttl_path: str) -> None:
    """Each ontology file parses as valid Turtle with no exception."""
    g = rdflib.Graph()
    g.parse(ttl_path, format="turtle")
    assert len(g) > 0, f"{ttl_path} parsed but produced zero triples"


def test_total_restriction_count_meets_target() -> None:
    """The deepened ontologies expose >= 40 owl:Restriction axioms in total."""
    OWL = rdflib.Namespace("http://www.w3.org/2002/07/owl#")
    total = 0
    per_file: dict[str, int] = {}
    for ttl_path in TTL_FILES:
        g = rdflib.Graph()
        g.parse(ttl_path, format="turtle")
        n = len(list(g.subjects(rdflib.RDF.type, OWL.Restriction)))
        per_file[Path(ttl_path).name] = n
        total += n
    assert total >= 40, (
        f"expected >= 40 owl:Restriction axioms across ttl files, got {total}. "
        f"Per-file: {per_file}"
    )
