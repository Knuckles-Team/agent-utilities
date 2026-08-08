"""Fail-closed domain-pack loading + the multi-pack registry
(CONCEPT:AU-KG.ingest.domain-pack-framework).

Every test in this file that names a defect asserts the pack is REFUSED
(:class:`DomainPackError`) — never partially loaded, never "loaded with
warnings".
"""

from __future__ import annotations

import dataclasses

import _fixtures
import pytest
import yaml

from agent_utilities.knowledge_graph.domain_packs.domain_pack import (
    ColumnMapping,
    EvaluationCase,
    FrontmatterMapping,
    TableMapping,
)
from agent_utilities.knowledge_graph.domain_packs.pack_loader import (
    DomainPackError,
    DomainPackRegistry,
    canonical_ontology_class_names,
    get_default_registry,
    load_pack,
    reset_default_registry,
)
from agent_utilities.knowledge_graph.ingestion.evidence_spine import Fragment


def test_valid_pack_loads_and_compiles_its_ontology_extension(tmp_path):
    manifest = _fixtures.build_manifest()
    pack_dir = _fixtures.write_pack(tmp_path, manifest)

    loaded = load_pack(pack_dir)

    assert loaded.manifest.pack == "runbooks"
    assert "Runbook" in loaded.own_class_names


def test_canonical_ontology_class_names_includes_document_and_person():
    names = canonical_ontology_class_names()
    assert "Document" in names
    assert "Person" in names


def test_missing_domain_pack_yml_is_refused(tmp_path):
    empty_dir = tmp_path / "ghost-pack"
    empty_dir.mkdir()

    with pytest.raises(DomainPackError, match="no domain_pack.yml"):
        load_pack(empty_dir)


def test_pack_name_must_match_its_directory(tmp_path):
    manifest = _fixtures.build_manifest(pack_name="runbooks")
    pack_dir = _fixtures.write_pack(tmp_path, manifest)
    # Rename the directory so pack != directory name.
    renamed = tmp_path / "wrong-directory-name"
    pack_dir.rename(renamed)

    with pytest.raises(DomainPackError, match="does not match its directory"):
        load_pack(renamed)


def test_hand_edited_pack_after_hashing_is_refused(tmp_path):
    manifest = _fixtures.build_manifest()
    pack_dir = _fixtures.write_pack(tmp_path, manifest)
    manifest_path = pack_dir / "domain_pack.yml"
    data = yaml.safe_load(manifest_path.read_text())
    data["description"] = "tampered after the integrity hash was pinned"
    manifest_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(DomainPackError, match="integrity hash"):
        load_pack(pack_dir)


def test_mapping_referencing_unknown_ontology_class_is_refused(tmp_path):
    manifest = _fixtures.build_manifest(
        mappings=[
            FrontmatterMapping(
                key="status",
                node_type="TotallyMadeUpClassNoOneDeclared",
                produce="property",
                property="status",
            )
        ],
        evaluation_cases=[],
    )
    pack_dir = _fixtures.write_pack(tmp_path, manifest)

    with pytest.raises(DomainPackError, match="unknown ontology class"):
        load_pack(pack_dir)


def test_table_edge_target_referencing_unknown_class_is_refused(tmp_path):
    manifest = _fixtures.build_manifest(
        mappings=[
            TableMapping(
                row_node_type="Runbook",
                row_id_template="{artifact_id}#row:{row_index}",
                columns={
                    "assignee": ColumnMapping(
                        produce="edge",
                        relation="assignedTo",
                        edge_target_type="NotARealClass",
                    )
                },
            )
        ],
        evaluation_cases=[],
    )
    pack_dir = _fixtures.write_pack(tmp_path, manifest)

    with pytest.raises(DomainPackError, match="unknown ontology class"):
        load_pack(pack_dir)


def test_evaluation_case_mismatch_is_refused(tmp_path):
    status_fragment = Fragment.at(
        artifact_id="md:z",
        kind="frontmatter_key",
        label="status",
        text="active",
        attributes={"key": "status", "value": "active"},
    )
    bad_case = EvaluationCase(
        name="wrong-on-purpose",
        artifact={
            "artifact_id": "md:z",
            "connector": "test-fixture",
            "media_type": "text/markdown",
            "content_hash": "sha256:" + "0" * 64,
            "source_object_id": "z.md",
        },
        fragments=[dataclasses.asdict(status_fragment)],
        # Deliberately wrong: the status rule always sets node_type Document.
        expect_entities=[{"id": "md:z", "node_type": "SomethingElseEntirely"}],
        expect_relationships=[],
    )
    manifest = _fixtures.build_manifest(evaluation_cases=[bad_case])
    pack_dir = _fixtures.write_pack(tmp_path, manifest)

    with pytest.raises(DomainPackError, match="does not reproduce"):
        load_pack(pack_dir)


def test_invalid_shacl_shape_file_is_refused(tmp_path):
    manifest = _fixtures.build_manifest(evaluation_cases=[])
    manifest = manifest.model_copy(update={"shacl_shapes": ["shapes.ttl"]})
    # Re-pin the hash since we changed the document after the first hash.
    from agent_utilities.knowledge_graph.domain_packs.pack_loader import (
        pack_integrity_hash,
    )
    from agent_utilities.knowledge_graph.ontology.connector_manifest import (
        IntegrityInfo,
    )

    digest = pack_integrity_hash(manifest)
    manifest = manifest.model_copy(
        update={
            "provenance": manifest.provenance.model_copy(
                update={"integrity": IntegrityInfo(hash=digest)}
            )
        }
    )
    pack_dir = _fixtures.write_pack(tmp_path, manifest)
    (pack_dir / "shapes.ttl").write_text(
        "this is not valid turtle {{{", encoding="utf-8"
    )

    with pytest.raises(DomainPackError, match="not valid Turtle/SHACL"):
        load_pack(pack_dir)


def test_registry_installs_and_lists_multiple_independent_packs(tmp_path):
    runbooks = _fixtures.build_manifest(pack_name="runbooks")
    widgets = _fixtures.build_manifest(pack_name="widgets", version="0.1.0")
    _fixtures.write_pack(tmp_path, runbooks)
    _fixtures.write_pack(tmp_path, widgets)

    registry = DomainPackRegistry(tmp_path)
    loaded = registry.discover_and_install_all()

    assert {p.manifest.pack for p in loaded} == {"runbooks", "widgets"}
    assert registry.get("runbooks") is not None
    assert registry.get("widgets") is not None


def test_removing_one_pack_never_disturbs_a_sibling_pack(tmp_path):
    runbooks = _fixtures.build_manifest(pack_name="runbooks")
    widgets = _fixtures.build_manifest(pack_name="widgets", version="0.1.0")
    _fixtures.write_pack(tmp_path, runbooks)
    _fixtures.write_pack(tmp_path, widgets)
    registry = DomainPackRegistry(tmp_path)
    registry.discover_and_install_all()

    assert registry.remove("runbooks") is True

    assert registry.get("runbooks") is None
    assert registry.get("widgets") is not None
    assert [p.manifest.pack for p in registry.list()] == ["widgets"]


def test_one_invalid_pack_does_not_silently_disable_a_valid_sibling(tmp_path):
    good = _fixtures.build_manifest(pack_name="good-pack")
    _fixtures.write_pack(tmp_path, good)
    bad_dir = tmp_path / "bad-pack"
    bad_dir.mkdir()
    (bad_dir / "domain_pack.yml").write_text("not: [valid, %%%", encoding="utf-8")

    registry = DomainPackRegistry(tmp_path)
    with pytest.raises(DomainPackError):
        registry.discover_and_install_all()
    # The loader fails closed and loudly on the bad pack rather than silently
    # skipping it — but a caller that catches the error can still load the
    # good one directly.
    loaded_good = registry.install(tmp_path / "good-pack")
    assert loaded_good.manifest.pack == "good-pack"


@pytest.fixture(autouse=True)
def _reset_default_registry():
    """The process-wide default registry (D-GP2-3) is cached global state —
    isolate every test in this module from whatever an earlier test (or
    process) left behind."""
    reset_default_registry()
    yield
    reset_default_registry()


def test_default_registry_is_empty_when_no_root_configured(monkeypatch):
    from agent_utilities.core.config import config

    monkeypatch.setattr(config, "domain_packs_root", "")

    registry = get_default_registry()

    assert registry.list() == []
    assert registry.get("runbooks") is None


def test_default_registry_discovers_from_configured_root(tmp_path, monkeypatch):
    from agent_utilities.core.config import config

    manifest = _fixtures.build_manifest(pack_name="runbooks")
    _fixtures.write_pack(tmp_path, manifest)
    monkeypatch.setattr(config, "domain_packs_root", str(tmp_path))

    registry = get_default_registry()

    assert registry.get("runbooks") is not None
    # Cached: a second call returns the SAME registry instance, not a re-scan.
    assert get_default_registry() is registry


def test_default_registry_discovery_failure_does_not_raise(
    tmp_path, monkeypatch, caplog
):
    from agent_utilities.core.config import config

    bad_dir = tmp_path / "bad-pack"
    bad_dir.mkdir()
    (bad_dir / "domain_pack.yml").write_text("not: [valid, %%%", encoding="utf-8")
    monkeypatch.setattr(config, "domain_packs_root", str(tmp_path))

    with caplog.at_level("ERROR"):
        registry = get_default_registry()

    assert registry.get("bad-pack") is None
    assert any("discovery" in record.message for record in caplog.records)


def test_reset_default_registry_forces_rediscovery(tmp_path, monkeypatch):
    from agent_utilities.core.config import config

    monkeypatch.setattr(config, "domain_packs_root", str(tmp_path))
    first = get_default_registry()

    manifest = _fixtures.build_manifest(pack_name="runbooks")
    _fixtures.write_pack(tmp_path, manifest)
    reset_default_registry()
    second = get_default_registry()

    assert first is not second
    assert first.get("runbooks") is None
    assert second.get("runbooks") is not None
