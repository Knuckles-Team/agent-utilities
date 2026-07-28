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
