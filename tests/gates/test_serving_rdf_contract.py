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


def test_unified_image_carries_no_eg_owned_semantic_library() -> None:
    dockerfile = (ROOT / "docker" / "graphos-unified.Dockerfile").read_text(
        encoding="utf-8"
    )

    # RDF/OWL/SHACL semantics are EG-owned (RF-ADR-009 clean cut): the unified
    # image installs neither the deleted `owl` extra nor the `rdf` extra, and
    # its smoke check imports no rdflib/owlready2/pyshacl.
    assert "agent-headless,logfire" in dockerfile
    for library in ("owlready2", "pyshacl", "import rdflib"):
        assert library not in dockerfile, library


def test_no_runtime_module_imports_an_eg_owned_semantic_library() -> None:
    """SHACL/OWL are EG-owned (RF-ADR-009 clean cut): AU declares none of
    pyshacl/owlrl/owlready2 (the schema-authority cutover gate) and no runtime
    module imports one -- AU shapes are validated by the engine."""
    import ast

    owned = {"pyshacl", "owlrl", "owlready2"}
    importers = []
    for module in (ROOT / "agent_utilities").rglob("*.py"):
        for node in ast.walk(ast.parse(module.read_text(encoding="utf-8"))):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            if any(name.split(".")[0] in owned for name in names):
                importers.append(str(module.relative_to(ROOT)))
    assert importers == [], "a runtime module imports an EG-owned semantic library"
