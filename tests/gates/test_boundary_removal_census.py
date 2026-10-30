"""Removal proofs for the AU cut: the behavior lives with its owner, not here.

Each test asserts the moved code is gone from Meristem and that nothing in the
package or its tests still names it. The receiving owner of each row is recorded
in ``specs/au-boundary-deconstruction/plan.md`` under "Removal owners".
"""

import re

import pytest
from _census import PACKAGE, ROOT, assert_absent, console_scripts, references

PLAN = (ROOT / "specs/au-boundary-deconstruction/plan.md").read_text(encoding="utf-8")
OWNERS = PLAN.split("\n## Removal owners", 1)[1].split("\n## ", 1)[0]


def _owner_row(row: str) -> str:
    match = re.search(
        rf"^\| `{re.escape(row)}` \| (.+?) \| (https://\S+) \|$", OWNERS, re.M
    )
    assert match is not None, f"{row} has no recorded owner"
    return match.group(2)


@pytest.mark.spec("MERISTEM-BOUNDARY-R002.3.2")
def test_deployment_validation_modules_do_not_import_certification_oidc() -> None:
    assert_absent("deployment")
    assert references(("certification_oidc",)) == []
    assert _owner_row("MERISTEM-BOUNDARY-R002.3.2")


@pytest.mark.spec("MERISTEM-BOUNDARY-R012")
def test_second_graph_projection_and_control_plane_shim_deleted() -> None:
    assert_absent("protocols/epistemic_operations", "control_plane")
    assert references(("epistemic_operations", "control_plane")) == []
    assert _owner_row("MERISTEM-BOUNDARY-R012")


@pytest.mark.spec("MERISTEM-BOUNDARY-R023")
@pytest.mark.spec("MERISTEM-BOUNDARY-R027")
@pytest.mark.spec("MERISTEM-BOUNDARY-R029")
@pytest.mark.spec("MERISTEM-BOUNDARY-R033")
@pytest.mark.spec("MERISTEM-BOUNDARY-R034")
def test_engine_bound_knowledge_graph_tree_is_deleted() -> None:
    """Ingestion commit, derivation, ontology, federation and retrieval left."""
    assert_absent("knowledge_graph", "assimilation", "etl", "streams")
    for row in (
        "MERISTEM-BOUNDARY-R023",
        "MERISTEM-BOUNDARY-R027",
        "MERISTEM-BOUNDARY-R029",
        "MERISTEM-BOUNDARY-R033",
        "MERISTEM-BOUNDARY-R034",
    ):
        assert _owner_row(row)
    assert references(("agent_utilities.knowledge_graph",), include_tests=True) == []


@pytest.mark.spec("MERISTEM-BOUNDARY-R025")
@pytest.mark.spec("MERISTEM-BOUNDARY-R026")
def test_source_and_enrichment_modules_moved_to_the_sdk() -> None:
    assert_absent(
        "ingestion",
        "automation",
        "sdd",
        "ecosystem",
        "enrichment",
        "knowledge_graph/enrichment",
        "knowledge_graph/kb",
        "knowledge_graph/extraction",
    )
    assert _owner_row("MERISTEM-BOUNDARY-R025")
    assert _owner_row("MERISTEM-BOUNDARY-R026")


@pytest.mark.spec("MERISTEM-BOUNDARY-R031")
def test_graph_schema_dtos_replaced_by_engine_types() -> None:
    assert_absent("models", "knowledge_graph/domain_packs")
    assert _owner_row("MERISTEM-BOUNDARY-R031")


@pytest.mark.spec("MERISTEM-BOUNDARY-R018.2")
@pytest.mark.spec("MERISTEM-BOUNDARY-R019.2")
@pytest.mark.spec("MERISTEM-BOUNDARY-R020.2")
@pytest.mark.spec("MERISTEM-BOUNDARY-R021.2")
@pytest.mark.spec("MERISTEM-BOUNDARY-R022.2")
def test_engine_duplicate_modules_and_worker_scripts_absent() -> None:
    from test_boundary_r018_r022_eg_duplicate_inventory import (  # type: ignore[import-not-found]
        INVENTORY,
        _matches,
    )

    for row in ("R018", "R019", "R020", "R021", "R022"):
        assert _matches(PACKAGE, INVENTORY[row]) == set()
        assert _owner_row(f"MERISTEM-BOUNDARY-{row}.2")
    assert set(console_scripts()) == {"agent-utilities"}


@pytest.mark.spec("MERISTEM-BOUNDARY-R028.2")
def test_infra_placement_modules_absent_without_callers() -> None:
    assert_absent("knowledge_graph/infra")
    assert (
        references(("placement_optimizer", "inventory_collector"), include_tests=True)
        == []
    )
    assert _owner_row("MERISTEM-BOUNDARY-R028.2")


@pytest.mark.spec("MERISTEM-BOUNDARY-R030.1")
@pytest.mark.spec("MERISTEM-BOUNDARY-R030.2")
@pytest.mark.spec("MERISTEM-BOUNDARY-R030.3")
@pytest.mark.spec("MERISTEM-BOUNDARY-R030.4")
@pytest.mark.spec("MERISTEM-BOUNDARY-R030.5")
def test_no_local_manifest_compiler_caller_remains() -> None:
    """The four former importers are gone, and none names the local compiler."""
    for script in (
        "generate_connector_manifests.py",
        "update_ontology_lock.py",
        "generate_native_connector_manifest.py",
    ):
        assert not (ROOT / "scripts" / script).exists()
    assert_absent(
        "knowledge_graph/domain_packs/pack_loader.py",
        "knowledge_graph/ontology/manifest_compiler.py",
    )
    assert references(("manifest_compiler",), include_tests=True) == []
    assert _owner_row("MERISTEM-BOUNDARY-R030.2")


DOMAIN_ROWS = {
    "MERISTEM-BOUNDARY-R048.2": "domains/government",
    "MERISTEM-BOUNDARY-R048.3": "domains/hr",
    "MERISTEM-BOUNDARY-R048.4": "domains/law",
    "MERISTEM-BOUNDARY-R048.5": "domains/medical",
    "MERISTEM-BOUNDARY-R048.6": "models/domains",
    "MERISTEM-BOUNDARY-R048.8": "ontology/shapes",
    "MERISTEM-BOUNDARY-R048.9": "data",
    "MERISTEM-BOUNDARY-R048.10": "protocols/voice_supply_chain",
}


@pytest.mark.spec("MERISTEM-BOUNDARY-R048.2")
@pytest.mark.spec("MERISTEM-BOUNDARY-R048.3")
@pytest.mark.spec("MERISTEM-BOUNDARY-R048.4")
@pytest.mark.spec("MERISTEM-BOUNDARY-R048.5")
@pytest.mark.spec("MERISTEM-BOUNDARY-R048.6")
@pytest.mark.spec("MERISTEM-BOUNDARY-R048.8")
@pytest.mark.spec("MERISTEM-BOUNDARY-R048.9")
@pytest.mark.spec("MERISTEM-BOUNDARY-R048.10")
@pytest.mark.parametrize(
    ("row", "directory"),
    [pytest.param(row, directory, id=row) for row, directory in DOMAIN_ROWS.items()],
)
def test_domain_directory_has_one_recorded_disposition_and_is_gone(
    row: str, directory: str
) -> None:
    assert_absent(directory)
    assert references((f"agent_utilities.{directory.replace('/', '.')}",)) == []
    dispositions = re.findall(rf"^\| `{re.escape(directory)}/` \| (\w+) \|", PLAN, re.M)
    assert dispositions == ["deleted"]
    assert _owner_row(row)
