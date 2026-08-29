"""Focused contracts for deterministic connector and skill release catalogs."""

from __future__ import annotations

import copy
import json
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, ValidationError

from agent_utilities.release_catalogs import (
    ReleaseCatalogError,
    canonical_value_digest,
    content_digest,
    prebundled_skill_catalog,
    prebundled_skill_catalog_digest,
)
from agent_utilities.skills import BUNDLED_SKILLS
from scripts.release import check_release_catalogs as release_gate
from scripts.release import generate_connector_bundle_catalog as connector_catalog
from scripts.release.generate_prebundled_skill_catalog import render_catalog

ROOT = Path(__file__).resolve().parents[3]
RELEASE_ROOT = ROOT / "deploy" / "release"
MATRIX = RELEASE_ROOT / "compatibility-matrix.yml"
SKILLS_ROOT = ROOT / "agent_utilities" / "skills"
_DIGEST = re.compile(r"^sha256:(?!0{64}$)[a-f0-9]{64}$")
_PRIVATE = (
    re.compile(r"(?:^|[\s\"'])/(?:home|Users|mnt|root|tmp|var|srv|opt|etc)/"),
    re.compile(r"\b[A-Za-z]:\\"),
    re.compile(r"\\\\[^\s\\]+\\[^\s\\]+"),
    re.compile(r"~[/\\]"),
    re.compile(r"://"),
    re.compile(r"@"),
)


def test_release_catalogs_are_declared_wheel_and_sdist_members() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = pyproject["tool"]["setuptools"]["package-data"]["deploy.release"]
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()

    assert {"*.catalog.json", "*.schema.json"} <= set(package_data)
    assert "recursive-include deploy/release *.json *.yml" in manifest
    assert (RELEASE_ROOT / "__init__.py").is_file()
    assert {
        path.name for path in RELEASE_ROOT.glob("*.catalog.json") if path.is_file()
    } >= {
        "connector-bundles.catalog.json",
        "index-migrations.catalog.json",
        "prebundled-skills.catalog.json",
    }


def _write_dependency_metadata(
    root: Path, *, project_extra: str = "test", lock_extra: str = "test"
) -> tuple[Path, Path]:
    pyproject = root / "pyproject.toml"
    pyproject.write_text(
        "[project]\n"
        "name = 'agent-utilities'\n"
        "[project.optional-dependencies]\n"
        f"{project_extra} = []\n",
        encoding="utf-8",
    )
    lock = root / "uv.lock"
    lock.write_text(
        "[[package]]\n"
        "name = 'agent-utilities'\n"
        "source = { editable = '.' }\n"
        "[package.optional-dependencies]\n"
        f"{lock_extra} = []\n"
        "[package.metadata]\n"
        f"provides-extras = ['{lock_extra}']\n",
        encoding="utf-8",
    )
    return pyproject, lock


def test_dependency_extra_gate_allows_stable_test_extras(tmp_path: Path) -> None:
    pyproject, lock = _write_dependency_metadata(tmp_path)

    release_gate._validate_dependency_extras(
        pyproject_path=pyproject,
        lock_path=lock,
    )


@pytest.mark.parametrize("extra", ["throwaway-dep-test", "fixture-only", "test-only"])
def test_dependency_extra_gate_rejects_ephemeral_manifest_extra(
    tmp_path: Path, extra: str
) -> None:
    pyproject, lock = _write_dependency_metadata(tmp_path)
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace("test = []", f"{extra} = []"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ephemeral dependency extras"):
        release_gate._validate_dependency_extras(
            pyproject_path=pyproject,
            lock_path=lock,
        )


def test_dependency_extra_gate_rejects_stale_ephemeral_lock_extra(
    tmp_path: Path,
) -> None:
    pyproject, lock = _write_dependency_metadata(
        tmp_path, project_extra="test", lock_extra="throwaway-dep-test"
    )

    with pytest.raises(ValueError, match="in uv.lock"):
        release_gate._validate_dependency_extras(
            pyproject_path=pyproject,
            lock_path=lock,
        )


def _load(name: str) -> dict[str, Any]:
    value = json.loads((RELEASE_ROOT / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _validator(name: str) -> Draft202012Validator:
    schema = _load(name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _fixture_skills(root: Path) -> None:
    for skill in BUNDLED_SKILLS:
        directory = root / skill
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            f"---\nname: {skill}\ndescription: Synthetic contract.\n---\n",
            encoding="utf-8",
        )


def _synthetic_entry(name: str) -> dict[str, Any]:
    return {
        "connector": name,
        "manifestDigest": canonical_value_digest([name, "manifest"]),
        "certificationDigest": canonical_value_digest([name, "certification"]),
        "bundleDigest": canonical_value_digest([name, "bundle"]),
        "artifactCount": 1,
        "sourcePresetCount": 0,
    }


def test_retained_prebundled_skill_catalog_is_exact_and_deterministic() -> None:
    retained = RELEASE_ROOT / "prebundled-skills.catalog.json"
    payload = render_catalog(skills_root=SKILLS_ROOT, matrix_path=MATRIX)

    assert payload == retained.read_bytes()
    document = json.loads(payload)
    _validator("prebundled-skill-catalog.schema.json").validate(document)
    assert [entry["skill"] for entry in document["entries"]] == sorted(BUNDLED_SKILLS)
    assert document["membershipDigest"] == canonical_value_digest(
        sorted(BUNDLED_SKILLS)
    )
    assert prebundled_skill_catalog_digest(SKILLS_ROOT) == content_digest(payload)
    assert all(_DIGEST.fullmatch(entry["treeDigest"]) for entry in document["entries"])


def test_retained_connector_catalog_is_exact_workspace_membership() -> None:
    document = _load("connector-bundles.catalog.json")
    _validator("connector-bundle-catalog.schema.json").validate(document)
    configured = sorted(
        connector_catalog._configured_provider_names(
            connector_catalog.DEFAULT_WORKSPACE
        )
    )

    assert [entry["connector"] for entry in document["entries"]] == configured
    assert len(configured) == len(set(configured)) == 71
    assert document["membershipDigest"] == canonical_value_digest(configured)
    assert all(
        _DIGEST.fullmatch(entry[field])
        for entry in document["entries"]
        for field in ("manifestDigest", "certificationDigest", "bundleDigest")
    )


def test_catalogs_retain_no_endpoint_identity_or_local_location() -> None:
    for name in (
        "connector-bundles.catalog.json",
        "prebundled-skills.catalog.json",
    ):
        text = (RELEASE_ROOT / name).read_text(encoding="utf-8")
        assert all(pattern.search(text) is None for pattern in _PRIVATE)
        assert all(
            field not in text
            for field in (
                '"endpoint"',
                '"host"',
                '"identity"',
                '"path"',
                '"signer"',
                '"url"',
                '"user"',
            )
        )


def test_prebundled_skill_catalog_rejects_extra_membership(tmp_path: Path) -> None:
    _fixture_skills(tmp_path)
    extra = tmp_path / "extra-skill"
    extra.mkdir()
    (extra / "SKILL.md").write_text("synthetic\n", encoding="utf-8")

    with pytest.raises(
        ReleaseCatalogError, match="prebundled_skill_membership_not_exact"
    ):
        prebundled_skill_catalog(tmp_path)


def test_prebundled_skill_catalog_rejects_symlinks(tmp_path: Path) -> None:
    _fixture_skills(tmp_path)
    target = tmp_path / "target.txt"
    target.write_text("synthetic\n", encoding="utf-8")
    link = tmp_path / BUNDLED_SKILLS[0] / "linked.txt"
    link.symlink_to(target)

    with pytest.raises(ReleaseCatalogError, match="prebundled_skill_symlink_rejected"):
        prebundled_skill_catalog(tmp_path)


def test_connector_catalog_is_order_stable_after_bundle_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("provider-b", "provider-a"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(connector_catalog, "_matrix_expected_entries", lambda _path: 2)
    monkeypatch.setattr(
        connector_catalog,
        "_configured_provider_names",
        lambda _path: ("provider-b", "provider-a"),
    )
    monkeypatch.setattr(
        connector_catalog,
        "_provider_owned_names",
        lambda _path: ("provider-a", "provider-b"),
    )
    checked: list[str] = []

    def _check(repo: Path, **_kwargs: object) -> list[str]:
        checked.append(repo.name)
        return []

    monkeypatch.setattr(connector_catalog, "check_one", _check)
    monkeypatch.setattr(
        connector_catalog, "_entry", lambda repo: _synthetic_entry(repo.name)
    )
    arguments = {
        "agents_root": tmp_path,
        "workspace_path": tmp_path / "workspace.yml",
        "bundled_root": tmp_path / "bundled",
        "lock_path": tmp_path / "lock",
        "matrix_path": tmp_path / "matrix.yml",
    }

    first = connector_catalog.connector_bundle_catalog(**arguments)
    second = connector_catalog.connector_bundle_catalog(**arguments)

    assert first == second
    assert [entry["connector"] for entry in first["entries"]] == [
        "provider-a",
        "provider-b",
    ]
    assert first["membershipDigest"] == canonical_value_digest(
        ["provider-a", "provider-b"]
    )
    assert checked == ["provider-a", "provider-b"] * 2


def test_connector_catalog_rejects_duplicate_or_missing_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(connector_catalog, "_matrix_expected_entries", lambda _path: 2)
    monkeypatch.setattr(
        connector_catalog,
        "_configured_provider_names",
        lambda _path: ("provider-a", "provider-a"),
    )
    monkeypatch.setattr(
        connector_catalog, "_provider_owned_names", lambda _path: ("provider-a",)
    )

    with pytest.raises(
        ReleaseCatalogError, match="connector_catalog_membership_not_exact"
    ):
        connector_catalog.connector_bundle_catalog(
            agents_root=tmp_path,
            workspace_path=tmp_path / "workspace.yml",
            bundled_root=tmp_path / "bundled",
            lock_path=tmp_path / "lock",
            matrix_path=tmp_path / "matrix.yml",
        )


def test_connector_catalog_rejects_any_bundle_gate_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("provider-a", "provider-b"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(connector_catalog, "_matrix_expected_entries", lambda _path: 2)
    monkeypatch.setattr(
        connector_catalog,
        "_configured_provider_names",
        lambda _path: ("provider-a", "provider-b"),
    )
    monkeypatch.setattr(
        connector_catalog,
        "_provider_owned_names",
        lambda _path: ("provider-a", "provider-b"),
    )
    monkeypatch.setattr(
        connector_catalog,
        "check_one",
        lambda repo, **_kwargs: (
            ["certified artifact hash differs"] if repo.name == "provider-b" else []
        ),
    )
    monkeypatch.setattr(
        connector_catalog, "_entry", lambda repo: _synthetic_entry(repo.name)
    )

    with pytest.raises(
        ReleaseCatalogError, match="connector_catalog_bundle_validation_failed"
    ):
        connector_catalog.connector_bundle_catalog(
            agents_root=tmp_path,
            workspace_path=tmp_path / "workspace.yml",
            bundled_root=tmp_path / "bundled",
            lock_path=tmp_path / "lock",
            matrix_path=tmp_path / "matrix.yml",
        )


@pytest.mark.parametrize(
    ("schema_name", "catalog_name"),
    [
        ("connector-bundle-catalog.schema.json", "connector-bundles.catalog.json"),
        ("prebundled-skill-catalog.schema.json", "prebundled-skills.catalog.json"),
    ],
)
def test_catalog_schemas_reject_environment_fields(
    schema_name: str, catalog_name: str
) -> None:
    document = copy.deepcopy(_load(catalog_name))
    document["endpoint"] = "https://example.invalid"

    with pytest.raises(ValidationError):
        _validator(schema_name).validate(document)


# The four `test_release_resource_catalog_*` cases that lived here are gone with
# the machinery they covered. They exercised `_write_resource_catalog` -- ~100
# lines of O_NOFOLLOW/dir_fd/temp-and-rename atomic writing whose only job was
# to maintain `deploy/release/release-contract-resources.catalog.json`, a list
# of this repository's own files and their sha256.
#
# Inside one git repository that ledger re-proves what the commit already
# proves, and the copy that DID matter -- the one shipped inside the wheel --
# is now compared byte-for-byte against these files directly by
# `scripts/release/check_release_wheel.py`. No derived artifact, no `--write`,
# nothing to keep atomic.
#
# It also removed a deadlock: `compatibility-matrix.yml` is one of the
# catalogued files, so every version bump staled the catalog, and the
# default-stage pre-commit gate that refused the stale catalog blocked the very
# bump commit that would have refreshed it.
