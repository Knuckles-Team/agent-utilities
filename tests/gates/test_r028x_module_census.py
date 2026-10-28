"""Retired R028 imports must not grow back into the current harness."""

from pathlib import Path

import pytest

from scripts.check_connector_import_census import inspect_file

ROOT = Path(__file__).resolve().parents[2]
PREFIXES = {
    "infra": ("agent_utilities.knowledge_graph.infra",),
    "observability": (
        "agent_utilities.observability.trace_ontology",
        "agent_utilities.observability.self_ingest",
        "agent_utilities.observability.audit_logger",
    ),
    "governance": ("agent_utilities.governance.relational_authority",),
    "placement": ("agent_utilities.knowledge_graph.research.placement_mining",),
}


def _importers(package_root: Path, prefixes: tuple[str, ...]) -> set[str]:
    found: set[str] = set()
    for source in package_root.rglob("*.py"):
        relative = source.relative_to(package_root)
        package = ".".join(("agent_utilities", *relative.parts[:-1]))
        finding = inspect_file(source, prefixes, package=package)
        if finding.unresolved:
            raise AssertionError(
                f"unresolved import in {relative}: {finding.unresolved}"
            )
        if finding.hits:
            found.add(relative.as_posix())
    return found


def _check(package: str, prefix: str, tmp_path: Path) -> None:
    assert _importers(ROOT / "agent_utilities", PREFIXES[package]) == set()
    source = tmp_path / "caller.py"
    source.write_text(f"import {prefix}\n")
    assert _importers(tmp_path, PREFIXES[package]) == {"caller.py"}


@pytest.mark.spec("AU-BOUNDARY-R028.2.1")
def test_r028_2_1_infra_importer_set_has_not_grown(tmp_path: Path) -> None:
    _check(
        "infra", "agent_utilities.knowledge_graph.infra.placement_optimizer", tmp_path
    )


@pytest.mark.spec("AU-BOUNDARY-R028.6.1")
def test_r028_6_1_observability_triad_importer_set_has_not_grown(
    tmp_path: Path,
) -> None:
    _check("observability", "agent_utilities.observability.trace_ontology", tmp_path)


@pytest.mark.spec("AU-BOUNDARY-R028.7.1")
def test_r028_7_1_relational_authority_importer_set_has_not_grown(
    tmp_path: Path,
) -> None:
    _check("governance", "agent_utilities.governance.relational_authority", tmp_path)


@pytest.mark.spec("AU-BOUNDARY-R028.8.1")
def test_r028_8_1_placement_mining_importer_set_has_not_grown(tmp_path: Path) -> None:
    _check(
        "placement",
        "agent_utilities.knowledge_graph.research.placement_mining",
        tmp_path,
    )


def test_census_resolves_relative_imports_and_refuses_unknown_dynamic_imports(
    tmp_path: Path,
) -> None:
    subdir = tmp_path / "harness"
    subdir.mkdir()
    (subdir / "caller.py").write_text("from ..observability import self_ingest\n")
    assert _importers(tmp_path, PREFIXES["observability"]) == {"harness/caller.py"}
    (subdir / "caller.py").write_text(
        "import importlib\nimportlib.import_module(name)\n"
    )
    with pytest.raises(AssertionError, match="unresolved import"):
        _importers(tmp_path, PREFIXES["observability"])
