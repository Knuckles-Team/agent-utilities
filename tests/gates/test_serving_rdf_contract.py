"""The GraphOS serving image must carry the lightweight RDF parser."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_serving_includes_the_minimal_rdf_ingestion_extra() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]

    assert extras["rdf"] == ["rdflib>=7.6.0"]
    serving = "\n".join(extras["serving"])
    assert ",rdf," in serving
    assert "[owl]" not in serving


def test_unified_image_installs_and_checks_the_rdf_runtime() -> None:
    dockerfile = (ROOT / "docker" / "graphos-unified.Dockerfile").read_text(
        encoding="utf-8"
    )

    # `docker/graphos-unified.Dockerfile`'s own "reconcile runtime stack" commit
    # (5b177b0d) deliberately widened the unified image from the lightweight
    # `rdf` extra to the full `owl` extra (owlready2 + rdflib + pyshacl) — the
    # unified server needs real OWL-DL reasoning + SHACL validation (the
    # hosted-ontology activation-ICV-fallback path), not just RDF parsing, and
    # the build's own smoke-test import (below) was updated in that same
    # commit to match. `owl` is a strict superset of `rdf` (both ship
    # rdflib), so this still satisfies the "carries an RDF parser" contract —
    # just via the fuller extra the image actually needs.
    assert "agent-headless,owl,logfire" in dockerfile
    assert "import owlready2" in dockerfile
    assert "import pyshacl" in dockerfile
    assert "import rdflib" in dockerfile
