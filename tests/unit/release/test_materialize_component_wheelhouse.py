"""Focused contracts for deterministic exact-component wheelhouses."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.utils import canonicalize_name

from scripts.release import materialize_component_wheelhouse as materializer
from scripts.release import promote_local_release as promoter


def _artifact(
    root: Path,
    name: str,
    *,
    requires: tuple[str, ...] = (),
    extras: tuple[str, ...] = (),
    extra_metadata_paths: tuple[str, ...] = (),
) -> tuple[promoter.LockedRequirement, promoter.WheelArtifact]:
    canonical = canonicalize_name(name)
    filename = canonical.replace("-", "_") + "-1.0.0-py3-none-any.whl"
    path = root / filename
    headers = [f"Name: {name}", "Version: 1.0.0"]
    headers.extend(f"Provides-Extra: {extra}" for extra in extras)
    headers.extend(f"Requires-Dist: {requirement}" for requirement in requires)
    metadata = ("\n".join(headers) + "\n\n").encode()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{canonical.replace('-', '_')}-1.0.0.dist-info/METADATA",
            metadata,
        )
        for metadata_path in extra_metadata_paths:
            archive.writestr(metadata_path, metadata)
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    locked = promoter.LockedRequirement(
        name=canonical,
        version="1.0.0",
        extras=frozenset(),
        digest=digest,
    )
    artifact = promoter.WheelArtifact(
        name=canonical,
        version="1.0.0",
        filename=filename,
        digest=digest,
        path=path,
        record_entries={},
        member_count=1,
        uncompressed_bytes=len(metadata),
        generated_scripts=frozenset(),
    )
    return locked, artifact


@pytest.fixture
def component_wheel_with_vendored_metadata(
    tmp_path: Path,
) -> promoter.WheelArtifact:
    _locked, artifact = _artifact(
        tmp_path,
        "epistemic-graph",
        extra_metadata_paths=(
            "epistemic_graph/_vendor/dependency-2.0.dist-info/METADATA",
        ),
    )
    return artifact


@pytest.fixture
def component_wheel_with_duplicate_top_level_metadata(
    tmp_path: Path,
) -> promoter.WheelArtifact:
    _locked, artifact = _artifact(
        tmp_path,
        "epistemic-graph",
        extra_metadata_paths=("duplicate-1.0.0.dist-info/METADATA",),
    )
    return artifact


def test_metadata_reader_ignores_nested_vendored_dist_info(
    component_wheel_with_vendored_metadata: promoter.WheelArtifact,
) -> None:
    metadata = materializer._metadata_for_wheel(component_wheel_with_vendored_metadata)

    assert metadata["Name"] == "epistemic-graph"
    assert metadata["Version"] == "1.0.0"


def test_metadata_reader_rejects_duplicate_top_level_metadata(
    component_wheel_with_duplicate_top_level_metadata: promoter.WheelArtifact,
) -> None:
    with pytest.raises(promoter.ReleaseError, match="component-wheel-metadata-invalid"):
        materializer._metadata_for_wheel(
            component_wheel_with_duplicate_top_level_metadata
        )


def _ecosystem(
    tmp_path: Path,
) -> tuple[
    dict[str, promoter.LockedRequirement],
    dict[str, promoter.WheelArtifact],
]:
    declarations = (
        (
            "epistemic-graph",
            ("eg-base>=1", 'numeric-core==1.0.0; extra == "full"'),
            ("full",),
        ),
        (
            "agent-utilities",
            (
                "epistemic-graph[full]>=1",
                'langfuse-agent[mcp]>=1; extra == "serving"',
                'serving-only>=1; extra == "serving"',
                'mcp-only>=1; extra == "mcp"',
            ),
            ("mcp", "serving"),
        ),
        (
            "langfuse-agent",
            (
                "agent-utilities[mcp]>=1",
                'lf-mcp-only>=1; extra == "mcp"',
            ),
            ("mcp",),
        ),
        ("eg-base", (), ()),
        ("numeric-core", (), ()),
        ("serving-only", (), ()),
        ("mcp-only", (), ()),
        ("lf-mcp-only", (), ()),
        ("unrelated", (), ()),
    )
    locked: dict[str, promoter.LockedRequirement] = {}
    wheels: dict[str, promoter.WheelArtifact] = {}
    for name, requires, extras in declarations:
        requirement, artifact = _artifact(
            tmp_path,
            name,
            requires=requires,
            extras=extras,
        )
        locked[requirement.name] = requirement
        wheels[artifact.name] = artifact
    return locked, wheels


def test_profiles_select_minimal_marker_and_extra_closures(tmp_path: Path) -> None:
    locked, wheels = _ecosystem(tmp_path)

    engine = materializer.select_component_closure("epistemic-graph", locked, wheels)
    serving = materializer.select_component_closure("agent-utilities", locked, wheels)
    langfuse = materializer.select_component_closure("langfuse-agent", locked, wheels)

    assert engine == {
        "eg-base": frozenset(),
        "epistemic-graph": frozenset({"full"}),
        "numeric-core": frozenset(),
    }
    assert serving["agent-utilities"] == frozenset({"mcp", "serving"})
    assert serving["epistemic-graph"] == frozenset({"full"})
    assert serving["langfuse-agent"] == frozenset({"mcp"})
    assert "serving-only" in serving
    assert "mcp-only" in serving
    assert "unrelated" not in serving
    assert langfuse["agent-utilities"] == frozenset({"mcp"})
    assert langfuse["epistemic-graph"] == frozenset({"full"})
    assert langfuse["langfuse-agent"] == frozenset({"mcp"})
    assert "serving-only" not in langfuse
    assert "mcp-only" in langfuse
    assert "unrelated" not in langfuse


def test_engine_resolution_reads_only_its_reachable_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    locked, wheels = _ecosystem(tmp_path)
    observed: list[str] = []
    original = materializer._metadata_for_wheel

    def tracking(artifact: promoter.WheelArtifact):
        observed.append(artifact.name)
        return original(artifact)

    monkeypatch.setattr(materializer, "_metadata_for_wheel", tracking)

    materializer.select_component_closure("epistemic-graph", locked, wheels)

    assert set(observed) == {"epistemic-graph", "eg-base", "numeric-core"}


def test_missing_dependency_and_unavailable_extra_fail_closed(tmp_path: Path) -> None:
    requirement, artifact = _artifact(
        tmp_path,
        "epistemic-graph",
        requires=("absent-package>=1",),
        extras=("full",),
    )
    with pytest.raises(promoter.ReleaseError, match="component-dependency-missing"):
        materializer.select_component_closure(
            "epistemic-graph",
            {requirement.name: requirement},
            {artifact.name: artifact},
        )

    requirement, artifact = _artifact(tmp_path, "epistemic-graph")
    with pytest.raises(promoter.ReleaseError, match="component-extra-unavailable"):
        materializer.select_component_closure(
            "epistemic-graph",
            {requirement.name: requirement},
            {artifact.name: artifact},
        )


def test_lock_rendering_is_canonical_and_hash_locked(tmp_path: Path) -> None:
    locked, wheels = _ecosystem(tmp_path)
    closure = materializer.select_component_closure("epistemic-graph", locked, wheels)

    first = materializer.render_component_lock(closure, locked)
    second = materializer.render_component_lock(dict(reversed(closure.items())), locked)

    assert first == second
    assert first.decode().splitlines() == [
        f"eg-base==1.0.0 --hash={locked['eg-base'].digest}",
        (f"epistemic-graph[full]==1.0.0 --hash={locked['epistemic-graph'].digest}"),
        f"numeric-core==1.0.0 --hash={locked['numeric-core'].digest}",
    ]
    assert materializer._sha256(first).startswith("sha256:")


def test_materialization_publishes_private_minimal_output_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    locked, wheels = _ecosystem(tmp_path)
    source = tmp_path / "source-wheelhouse"
    source.mkdir(mode=0o700)
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    private.chmod(0o700)
    output = private / "engine-wheelhouse"
    spec_digest = "sha256:" + "a" * 64
    monkeypatch.setattr(
        materializer,
        "load_spec",
        lambda _path, *, release_id: SimpleNamespace(digest=spec_digest),
    )
    monkeypatch.setattr(
        materializer,
        "validate_wheelhouse",
        lambda _source, _spec: (locked, wheels, b"source-lock"),
    )

    result = materializer.materialize_component_wheelhouse(
        component="epistemic-graph",
        release_id="release-test",
        spec_path=tmp_path / "spec.json",
        source=source,
        output=output.absolute(),
    )

    assert result == {
        "apiVersion": "graphos.io/v1",
        "kind": "ComponentWheelhouseMaterialization",
        "component": "epistemic-graph",
        "packageCount": 3,
        "requirementsFile": "epistemic-graph-requirements.txt",
        "requirementsSha256": materializer._sha256(
            (output / "epistemic-graph-requirements.txt").read_bytes()
        ),
        "sourceSpecSha256": spec_digest,
    }
    assert {path.name for path in output.iterdir()} == {
        "epistemic-graph-requirements.txt",
        wheels["epistemic-graph"].filename,
        wheels["eg-base"].filename,
        wheels["numeric-core"].filename,
    }
    assert output.stat().st_mode & 0o077 == 0
    assert (output / "epistemic-graph-requirements.txt").stat().st_mode & 0o077 == 0
    assert all(str(tmp_path) not in str(value) for value in result.values())

    with pytest.raises(promoter.ReleaseError, match="component-output-exists"):
        materializer.materialize_component_wheelhouse(
            component="epistemic-graph",
            release_id="release-test",
            spec_path=tmp_path / "spec.json",
            source=source,
            output=output.absolute(),
        )


def test_source_output_overlap_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(promoter.ReleaseError, match="component-input-output-overlap"):
        materializer._reject_source_output_overlap(source, source / "derived")
