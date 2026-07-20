"""Adversarial current-only provider materialization contracts."""

from __future__ import annotations

import inspect
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from agent_utilities.core import providers, unified_install
from agent_utilities.core.provider_materialization import (
    MANAGED_PROVIDER_MARKER,
    MAX_MARKER_BYTES,
    ProviderAssetError,
    ProviderOwnershipConflict,
    build_asset_manifest,
    inactive_marker,
    marker_for_manifest,
    read_managed_provider_marker,
    resolve_managed_generation,
    write_managed_provider_marker,
)

REGISTRATION = "a" * 64


def _source(root: Path, leg: str, body: str = "current") -> Path:
    source = root / f"source-{leg}-{body}"
    source.mkdir(parents=True)
    if leg == "skills":
        skill = source / "synthetic-skill"
        skill.mkdir()
        (skill / "SKILL.md").write_text(
            f"---\nname: synthetic-skill\n---\n{body}\n", encoding="utf-8"
        )
    elif leg == "prompts":
        (source / "prompt.json").write_text(
            json.dumps({"body": body}), encoding="utf-8"
        )
    else:
        (source / "ontology.ttl").write_text(f"# {body}\n", encoding="utf-8")
    return source


def _materialize(root: Path, source: Path, leg: str, provider: str = "provider-a"):
    manifest = build_asset_manifest(source, leg=leg)
    unified_install._materialize_provider(
        root=root,
        provider=provider,
        leg=leg,
        registration=REGISTRATION,
        source=source,
        manifest=manifest,
    )
    return manifest


def test_marker_v2_is_closed_bounded_path_free_and_atomic(tmp_path: Path) -> None:
    source = _source(tmp_path, "prompts")
    manifest = build_asset_manifest(source, leg="prompts")
    root = tmp_path / "provider"
    marker = marker_for_manifest(
        provider="provider-a",
        leg="prompts",
        registration=REGISTRATION,
        manifest=manifest,
    )

    write_managed_provider_marker(root, marker)

    marker_path = root / MANAGED_PROVIDER_MARKER
    raw = json.loads(marker_path.read_text(encoding="utf-8"))
    assert raw == marker.payload()
    assert raw["schema_version"] == 2
    assert len(marker_path.read_bytes()) <= MAX_MARKER_BYTES
    assert read_managed_provider_marker(
        root, provider="provider-a", leg="prompts"
    ) == marker
    assert list(root.glob(f".{MANAGED_PROVIDER_MARKER}.*")) == []
    assert str(tmp_path) not in marker_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "name",
    ["", ".", "..", "../outside", "nested/provider", "nested\\provider", "CON", "a" * 129],
)
def test_marker_rejects_nonportable_provider_names(name: str) -> None:
    with pytest.raises(ValueError):
        write_managed_provider_marker(
            Path("unused"), inactive_marker(provider=name, leg="skills", registration=REGISTRATION)
        )


def test_marker_rejects_v1_extra_oversized_duplicate_and_symlink(tmp_path: Path) -> None:
    root = tmp_path / "provider"
    root.mkdir()
    marker = root / MANAGED_PROVIDER_MARKER
    marker.write_text('{"schema_version":1,"provider":"provider","leg":"skills"}')
    assert read_managed_provider_marker(root) is None
    extra = inactive_marker(
        provider="provider", leg="skills", registration=REGISTRATION
    ).payload()
    extra["unexpected"] = "value"
    marker.write_text(json.dumps(extra), encoding="utf-8")
    assert read_managed_provider_marker(root) is None
    marker.write_bytes(b"x" * (MAX_MARKER_BYTES + 1))
    assert read_managed_provider_marker(root) is None
    marker.write_text(
        '{"schema_version":2,"provider":"provider","provider":"other",'
        '"leg":"skills","active":false,"registration_digest":"'
        + REGISTRATION
        + '","content_digest":"'
        + ("b" * 64)
        + '","file_count":0,"byte_count":0}'
    )
    assert read_managed_provider_marker(root) is None
    marker.unlink()
    target = tmp_path / "marker-target"
    target.write_text("{}", encoding="utf-8")
    marker.symlink_to(target)
    assert read_managed_provider_marker(root) is None


def test_unmarked_operator_destination_is_never_replaced(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    operator = root / "provider-a"
    operator.mkdir(parents=True)
    original = operator / "SKILL.md"
    original.write_text("operator-owned", encoding="utf-8")
    source = _source(tmp_path, "skills")

    with pytest.raises(ProviderOwnershipConflict):
        _materialize(root, source, "skills")

    assert original.read_text(encoding="utf-8") == "operator-owned"
    assert not (operator / MANAGED_PROVIDER_MARKER).exists()


def test_generation_activation_replaces_view_without_partial_or_deleted_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "prompts"
    root.mkdir()
    old = _source(tmp_path, "prompts", "old")
    old_manifest = _materialize(root, old, "prompts")
    current = _source(tmp_path, "prompts", "new")
    (current / "second.json").write_text("{}", encoding="utf-8")
    current_manifest = _materialize(root, current, "prompts")

    resolved = resolve_managed_generation(
        root / "provider-a",
        provider="provider-a",
        leg="prompts",
        registration=REGISTRATION,
        source_manifest=current_manifest,
    )

    assert resolved is not None
    assert {path.name for path in resolved.glob("*.json")} == {
        "prompt.json",
        "second.json",
    }
    assert old_manifest.content_digest != current_manifest.content_digest


def test_zero_assets_deactivates_old_ontology_instead_of_restamping_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ontologies"
    root.mkdir()
    old = _source(tmp_path, "ontologies")
    old_manifest = _materialize(root, old, "ontologies")
    empty = tmp_path / "empty-ontology"
    empty.mkdir()
    registration = providers.ProviderRegistration(
        name="provider-a",
        group=providers.ONTOLOGY_PROVIDER_GROUP,
        target="provider.ontology",
        owner_name="provider",
        owner_version="1",
        digest=REGISTRATION,
        source_root=empty,
        owned_paths=frozenset(),
    )

    count, active = unified_install._install_registration(
        root=root, leg="ontologies", registration=registration
    )

    marker = read_managed_provider_marker(
        root / "provider-a", provider="provider-a", leg="ontologies"
    )
    assert (count, active) == (0, False)
    assert marker is not None and marker.active is False
    assert (
        resolve_managed_generation(
            root / "provider-a",
            provider="provider-a",
            leg="ontologies",
            registration=REGISTRATION,
            source_manifest=old_manifest,
        )
        is None
    )


def test_source_marker_symlink_and_special_file_are_rejected(tmp_path: Path) -> None:
    source = _source(tmp_path, "skills")
    (source / MANAGED_PROVIDER_MARKER).write_text("{}", encoding="utf-8")
    with pytest.raises(ProviderAssetError, match="reserved marker"):
        build_asset_manifest(source, leg="skills")
    (source / MANAGED_PROVIDER_MARKER).unlink()
    target = source / "target.txt"
    target.write_text("target", encoding="utf-8")
    (source / "linked.txt").symlink_to(target)
    with pytest.raises(ProviderAssetError, match="linked or special"):
        build_asset_manifest(source, leg="skills")
    (source / "linked.txt").unlink()
    if hasattr(os, "mkfifo"):
        os.mkfifo(source / "pipe")
        with pytest.raises(ProviderAssetError, match="linked or special"):
            build_asset_manifest(source, leg="skills")


def test_unselected_files_cannot_bypass_the_tree_entry_bound(
    tmp_path: Path, monkeypatch
) -> None:
    source = _source(tmp_path, "prompts")
    (source / "ignored-a.txt").write_text("a", encoding="utf-8")
    (source / "ignored-b.txt").write_text("b", encoding="utf-8")
    monkeypatch.setattr(
        "agent_utilities.core.provider_materialization.MAX_PROVIDER_FILES", 2
    )

    with pytest.raises(ProviderAssetError, match="tree-entry bound"):
        build_asset_manifest(source, leg="prompts")


def test_managed_tree_removal_validation_is_bounded_and_link_safe(
    tmp_path: Path, monkeypatch
) -> None:
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "one").write_text("1", encoding="utf-8")
    (managed / "two").write_text("2", encoding="utf-8")
    monkeypatch.setattr(unified_install, "_MAX_MANAGED_TREE_ENTRIES", 1)
    assert unified_install._safe_generated_tree(managed) is False

    (managed / "two").unlink()
    target = tmp_path / "outside"
    target.write_text("outside", encoding="utf-8")
    (managed / "linked").symlink_to(target)
    monkeypatch.setattr(unified_install, "_MAX_MANAGED_TREE_ENTRIES", 10)
    assert unified_install._safe_generated_tree(managed) is False


def test_source_destination_overlap_is_rejected_before_deletion(tmp_path: Path) -> None:
    source = _source(tmp_path, "skills")
    manifest = build_asset_manifest(source, leg="skills")
    with pytest.raises(ProviderOwnershipConflict, match="overlaps"):
        unified_install._materialize_provider(
            root=source,
            provider="synthetic-skill",
            leg="skills",
            registration=REGISTRATION,
            source=source,
            manifest=manifest,
        )
    assert (source / "synthetic-skill" / "SKILL.md").is_file()


def test_failed_stage_leaves_previous_complete_generation_active(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "prompts"
    root.mkdir()
    old = _source(tmp_path, "prompts", "old")
    old_manifest = _materialize(root, old, "prompts")
    current = _source(tmp_path, "prompts", "new")
    current_manifest = build_asset_manifest(current, leg="prompts")
    monkeypatch.setattr(
        unified_install,
        "copy_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("private path")),
    )

    with pytest.raises(OSError):
        unified_install._materialize_provider(
            root=root,
            provider="provider-a",
            leg="prompts",
            registration=REGISTRATION,
            source=current,
            manifest=current_manifest,
        )

    assert (
        resolve_managed_generation(
            root / "provider-a",
            provider="provider-a",
            leg="prompts",
            registration=REGISTRATION,
            source_manifest=old_manifest,
        )
        is not None
    )
    assert list((root / "provider-a" / ".generations").glob(".stage-*")) == []


def test_serialized_concurrent_activation_leaves_one_complete_generation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "prompts"
    root.mkdir()
    sources = [_source(tmp_path, "prompts", value) for value in ("one", "two")]

    def install(source: Path) -> None:
        manifest = build_asset_manifest(source, leg="prompts")
        with unified_install._materialization_lock(root):
            unified_install._materialize_provider(
                root=root,
                provider="provider-a",
                leg="prompts",
                registration=REGISTRATION,
                source=source,
                manifest=manifest,
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(install, sources))

    marker = read_managed_provider_marker(
        root / "provider-a", provider="provider-a", leg="prompts"
    )
    assert marker is not None and marker.active
    generation = root / "provider-a" / ".generations" / marker.content_digest
    actual = build_asset_manifest(generation, leg="prompts")
    assert actual.content_digest == marker.content_digest


class _Distribution:
    def __init__(self, root: Path, name: str = "provider-dist", version: str = "1"):
        self.root = root
        self.metadata = {"Name": name}
        self.version = version
        self.files = tuple(
            PurePosixPath(path.relative_to(root).as_posix())
            for path in sorted(root.rglob("*"))
            if path.is_file()
        )

    def locate_file(self, path):
        return self.root / str(path)


def _entry(root: Path, *, name: str = "provider-a", target: str = "pkg.prompts"):
    distribution = _Distribution(root)
    return SimpleNamespace(
        name=name,
        value=target,
        group=providers.PROMPT_PROVIDER_GROUP,
        dist=distribution,
    )


def test_registration_is_distribution_owned_and_never_imports_provider(
    tmp_path: Path, monkeypatch
) -> None:
    owned = tmp_path / "pkg" / "prompts"
    owned.mkdir(parents=True)
    (owned / "prompt.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(providers, "entry_points", lambda **_kwargs: [_entry(tmp_path)])
    monkeypatch.setattr(
        "importlib.import_module",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not import")),
    )

    registrations = providers.provider_registrations(providers.PROMPT_PROVIDER_GROUP)

    assert len(registrations) == 1
    assert registrations[0].source_root == owned
    assert len(registrations[0].digest) == 64


def test_nonmaterialized_data_provider_group_remains_bounded_and_nonimporting(
    tmp_path: Path, monkeypatch
) -> None:
    owned = tmp_path / "pkg" / "connectors"
    owned.mkdir(parents=True)
    (owned / "mcp_source_presets.json").write_text("{}", encoding="utf-8")
    entry = _entry(tmp_path, target="pkg.connectors")
    monkeypatch.setattr(providers, "entry_points", lambda **_kwargs: [entry])

    assert providers.iter_provider_dirs(
        "agent_utilities.source_connector_providers"
    ) == [("provider-a", owned)]


def test_unowned_target_is_unresolved_and_not_served(tmp_path: Path, monkeypatch) -> None:
    owned = tmp_path / "different" / "prompts"
    owned.mkdir(parents=True)
    (owned / "prompt.json").write_text("{}", encoding="utf-8")
    entry = _entry(tmp_path, target="pkg.prompts")
    monkeypatch.setattr(providers, "entry_points", lambda **_kwargs: [entry])

    registration = providers.provider_registrations(
        providers.PROMPT_PROVIDER_GROUP
    )[0]

    assert registration.source_root is None
    assert providers.current_provider_assets(providers.PROMPT_PROVIDER_GROUP) == ()


def test_unrecorded_provider_asset_is_not_served(tmp_path: Path, monkeypatch) -> None:
    owned = tmp_path / "pkg" / "prompts"
    owned.mkdir(parents=True)
    (owned / "prompt.json").write_text("{}", encoding="utf-8")
    entry = _entry(tmp_path)
    # Snapshot distribution ownership first, then simulate an injected file in
    # the otherwise-valid data package.
    (owned / "injected.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(providers, "entry_points", lambda **_kwargs: [entry])

    registrations = providers.provider_registrations(providers.PROMPT_PROVIDER_GROUP)

    assert registrations[0].source_root == owned
    assert registrations[0].owned_paths == frozenset({"prompt.json"})
    assert providers.current_provider_assets(providers.PROMPT_PROVIDER_GROUP) == ()


def test_duplicate_and_casefold_provider_owners_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    owned = tmp_path / "pkg" / "prompts"
    owned.mkdir(parents=True)
    (owned / "prompt.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        providers,
        "entry_points",
        lambda **_kwargs: [
            _entry(tmp_path, name="Provider-A"),
            _entry(tmp_path, name="provider-a"),
        ],
    )

    with pytest.raises(providers.ProviderRegistrationConflict):
        providers.provider_registrations(providers.PROMPT_PROVIDER_GROUP)


def test_empty_or_tampered_generation_falls_back_to_current_prompt_source(
    tmp_path: Path, monkeypatch
) -> None:
    source = _source(tmp_path, "prompts")
    manifest = build_asset_manifest(source, leg="prompts")
    registration = providers.ProviderRegistration(
        name="provider-a",
        group=providers.PROMPT_PROVIDER_GROUP,
        target="provider.prompts",
        owner_name="provider",
        owner_version="1",
        digest=REGISTRATION,
        source_root=source,
        owned_paths=frozenset({"prompt.json"}),
    )
    assets = providers.ProviderAssets(registration, source, manifest)
    root = tmp_path / "xdg-prompts"
    root.mkdir()
    _materialize(root, source, "prompts")
    marker = read_managed_provider_marker(root / "provider-a")
    assert marker is not None
    generation = root / "provider-a" / ".generations" / marker.content_digest
    (generation / "prompt.json").write_text("tampered", encoding="utf-8")
    monkeypatch.setattr(providers, "current_provider_assets", lambda _group: (assets,))
    monkeypatch.setattr(
        "agent_utilities.core.paths.unified_prompts_dir", lambda: root
    )

    assert providers.resolve_prompt_provider_dirs() == [("provider-a", source)]


def test_base_prompt_reader_uses_the_exact_validated_generation(
    tmp_path: Path, monkeypatch
) -> None:
    source = _source(tmp_path, "prompts")
    manifest = build_asset_manifest(source, leg="prompts")
    root = tmp_path / "xdg-prompts"
    root.mkdir()
    _materialize(root, source, "prompts", provider="agent-utilities")
    monkeypatch.setattr(
        unified_install,
        "own_provider_asset",
        lambda _leg: (source, REGISTRATION, manifest),
    )
    monkeypatch.setattr("agent_utilities.core.paths.unified_prompts_dir", lambda: root)

    resolved = providers.resolve_base_prompt_dir()

    assert resolved.is_relative_to(root)
    assert resolved.name == manifest.content_digest


def test_every_ontology_runtime_consumer_uses_the_validated_resolver(
    tmp_path: Path, monkeypatch
) -> None:
    from agent_utilities.knowledge_graph.core import ontology_federation
    from agent_utilities.knowledge_graph.core.ontology_loader import OntologyLoader
    from agent_utilities.mcp.tools.ontology_tools import _sync_package_ontologies

    ttl = tmp_path / "ontology_demo.ttl"
    ttl.write_text("# current", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        ontology_federation,
        "resolve_provider_ontologies",
        lambda: [("provider-a", ttl)],
    )

    assert OntologyLoader._federated_path_for(
        "http://knuckles.team/kg/demo", "demo"
    ) == ttl

    lifecycle = SimpleNamespace(
        load=lambda path, source_type: calls.append((path, source_type))
        or {"idempotent": False}
    )
    result = _sync_package_ontologies(lifecycle)

    assert result["artifacts_loaded"] == 1
    assert calls == [(str(ttl), "file")]


def test_retired_marked_root_is_never_reinterpreted_as_flat_operator_skill(
    tmp_path: Path, monkeypatch
) -> None:
    retired = tmp_path / "retired-provider"
    retired.mkdir()
    (retired / "SKILL.md").write_text(
        "---\nname: retired-skill\n---\nstale", encoding="utf-8"
    )
    write_managed_provider_marker(
        retired,
        inactive_marker(
            provider="retired-provider", leg="skills", registration=REGISTRATION
        ),
    )
    monkeypatch.setattr("agent_utilities.core.paths.skills_dir", lambda: tmp_path)
    monkeypatch.setattr(providers, "current_provider_assets", lambda _group: ())

    roots = providers.resolve_skill_provider_dirs()

    assert retired not in [root for _provider, root in roots]


def test_duplicate_flat_skill_identity_fails_deterministically(
    tmp_path: Path, monkeypatch
) -> None:
    for directory, identity in (("one", "duplicate-local"), ("two", "DUPLICATE-LOCAL")):
        root = tmp_path / directory
        root.mkdir()
        (root / "SKILL.md").write_text(
            f"---\nname: {identity}\n---\nbody", encoding="utf-8"
        )
    monkeypatch.setattr("agent_utilities.core.paths.skills_dir", lambda: tmp_path)
    monkeypatch.setattr(providers, "current_provider_assets", lambda _group: ())

    with pytest.raises(providers.DuplicateSkillIdentity):
        providers.resolve_skill_provider_dirs()


def test_skill_coverage_uses_only_the_unified_validated_reader(
    tmp_path: Path, monkeypatch
) -> None:
    from agent_utilities.mcp import skill_coverage

    current = tmp_path / "current"
    current.mkdir()
    monkeypatch.setattr(
        providers,
        "resolve_skill_provider_dirs",
        lambda: [("provider-a", current)],
    )

    assert skill_coverage._provider_dirs() == [current]


def test_install_api_has_no_force_and_result_is_path_free(tmp_path: Path, monkeypatch) -> None:
    roots = {leg: tmp_path / leg for leg in ("skills", "prompts", "ontologies")}
    sources = {
        leg: _source(tmp_path / "own", leg) for leg in roots
    }
    monkeypatch.setattr(unified_install, "unified_skills_dir", lambda: roots["skills"])
    monkeypatch.setattr(unified_install, "unified_prompts_dir", lambda: roots["prompts"])
    monkeypatch.setattr(
        unified_install, "unified_ontologies_dir", lambda: roots["ontologies"]
    )
    monkeypatch.setattr(unified_install, "provider_registrations", lambda _group: ())
    monkeypatch.setattr(
        unified_install,
        "_own_source",
        lambda leg: (sources[leg], REGISTRATION, build_asset_manifest(sources[leg], leg=leg)),
    )

    result = unified_install.install_unified()
    rendered = json.dumps(result, sort_keys=True)

    assert tuple(inspect.signature(unified_install.install_unified).parameters) == ()
    assert result["path_free"] is True
    assert str(tmp_path) not in rendered
    assert result["pruned"] == {"skills": 0, "prompts": 0, "ontologies": 0}
