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

    # The `owl` extra (owlready2 + pyshacl) was deleted when semantic
    # authority moved into EG (43197d7c6): OWL reasoning and SHACL validation
    # are engine-native generated contracts. The unified image therefore
    # carries only the RDF parser, and never imports a SHACL/OWL library AU
    # does not declare (SHACL/OWL are EG-owned).
    assert "agent-headless,rdf,logfire" in dockerfile
    assert "owlready2" not in dockerfile
    assert "pyshacl" not in dockerfile
    assert "import rdflib" in dockerfile


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
