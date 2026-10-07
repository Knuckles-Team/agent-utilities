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

    # The unified image carries the same lightweight `rdf` extra as the
    # serving profile. OWL-DL reasoning and SHACL validation are engine
    # authorities (epistemic-graph), so the retired `owl` extra and its
    # owlready2/pyshacl imports must not reappear in the image or its smoke
    # check.
    assert "agent-headless,rdf,logfire" in dockerfile
    assert ",owl," not in dockerfile
    assert "import owlready2" not in dockerfile
    assert "import pyshacl" not in dockerfile
    assert "import rdflib" in dockerfile
