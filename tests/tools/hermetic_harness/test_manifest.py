"""GOC-38-W02 manifest contract tests: property tests reject missing fields,
and the interpreter-identity helper never relies on realpath()."""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import jsonschema
import pytest

from scripts.hermetic_harness import manifest as manifest_mod
from scripts.hermetic_harness.manifest import (
    build_manifest,
    filtered_env,
    validate_manifest,
    venv_identity_digest,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _fake_venv(tmp_path: Path, *, with_pyvenv_cfg: bool = True) -> Path:
    venv = tmp_path / "venv"
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    py = bin_dir / "python3"
    py.symlink_to(sys.executable)
    if with_pyvenv_cfg:
        (venv / "pyvenv.cfg").write_text("home = /usr\nversion = 3.14.4\n")
    site_packages = venv / f"lib/python3.14/site-packages"
    site_packages.mkdir(parents=True)
    (site_packages / "foo-1.0.dist-info").mkdir()
    (site_packages / "bar-2.0.dist-info").mkdir()
    return venv


def test_build_manifest_round_trips_and_validates(tmp_path):
    venv = _fake_venv(tmp_path)
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("fake lock content\n")

    m = build_manifest(
        repo="agent-utilities",
        repo_path=REPO_ROOT,
        test_paths=["tests/tools/hermetic_harness"],
        lockfile_path=lockfile,
        venv_path=venv,
        timeout_seconds=300,
    )
    validate_manifest(m.to_dict())  # must not raise
    assert m.temp_root.startswith("/var/tmp/")
    assert m.lock_digest.startswith("sha256:")
    assert m.interpreter_digest.startswith("sha256:")
    # Deterministic digest for identical content.
    assert m.digest() == m.digest()


def test_build_manifest_refuses_venv_without_pyvenv_cfg(tmp_path):
    venv = _fake_venv(tmp_path, with_pyvenv_cfg=False)
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("x")
    with pytest.raises(FileNotFoundError):
        build_manifest(
            repo="agent-utilities",
            repo_path=REPO_ROOT,
            test_paths=["tests/tools/hermetic_harness"],
            lockfile_path=lockfile,
            venv_path=venv,
            timeout_seconds=300,
        )


@pytest.mark.parametrize(
    "missing_field",
    [
        "candidate_sha",
        "lock_digest",
        "interpreter_digest",
        "resource_limits",
        "timeout_seconds",
        "temp_root",
    ],
)
def test_manifest_schema_rejects_missing_required_field(tmp_path, missing_field):
    venv = _fake_venv(tmp_path)
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("x")
    m = build_manifest(
        repo="agent-utilities",
        repo_path=REPO_ROOT,
        test_paths=["tests/tools/hermetic_harness"],
        lockfile_path=lockfile,
        venv_path=venv,
        timeout_seconds=300,
    )
    body = m.to_dict()
    del body[missing_field]
    with pytest.raises(jsonschema.ValidationError):
        validate_manifest(body)


def test_manifest_schema_rejects_temp_root_under_tmp(tmp_path):
    venv = _fake_venv(tmp_path)
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("x")
    # /tmp is RAM-backed tmpfs on this host and previously ENOSPC'd a
    # session under bulk test temp; the schema pins the root to /var/tmp.
    # build_manifest validates before returning, so the rejection surfaces
    # directly from construction, not from a separate post-hoc check.
    with pytest.raises(jsonschema.ValidationError):
        build_manifest(
            repo="agent-utilities",
            repo_path=REPO_ROOT,
            test_paths=["tests/tools/hermetic_harness"],
            lockfile_path=lockfile,
            venv_path=venv,
            timeout_seconds=300,
            temp_root="/tmp/would-be-ram-backed",
        )


def test_venv_identity_digest_returns_none_without_pyvenv_cfg(tmp_path):
    empty = tmp_path / "not-a-venv"
    empty.mkdir()
    assert venv_identity_digest(empty) is None


def test_venv_identity_digest_never_calls_realpath():
    """realpath() on a uv .venv/bin/python3 always resolves OUT of the venv
    (it's typically a symlink to a base interpreter) -- a guard that used it
    to decide interpreter identity previously os.execve()'d out of the venv
    mid-run and killed a whole pytest session silently (exit 75). Assert the
    source of venv_identity_digest textually never references realpath, so
    this defect class can't be reintroduced without this test noticing."""
    # co_names covers every name the function's BYTECODE actually looks up
    # (attribute access, calls, module references) -- unlike a substring
    # match on raw source text, it can't be fooled by the docstring's own
    # prose discussion of realpath(), and it can't miss a real call either.
    assert "realpath" not in venv_identity_digest.__code__.co_names


def test_filtered_env_strips_inherited_uv_project_environment():
    """A test previously inherited UV_PROJECT_ENVIRONMENT and rebuilt the
    lane's real venv as its own fixture (349 packages -> 1 'alpha'). The
    default allowlist must not include it, so a child process launched
    through this harness cannot repeat that."""
    source_env = {
        "PATH": "/usr/bin",
        "HOME": "/home/x",
        "UV_PROJECT_ENVIRONMENT": "/home/x/some/other/lane/.venv",
        "SOME_RANDOM_SECRET": "shh",
    }
    child_env, rejected = filtered_env(list(manifest_mod.DEFAULT_ENV_ALLOWLIST), source_env)
    assert "UV_PROJECT_ENVIRONMENT" not in child_env
    assert "UV_PROJECT_ENVIRONMENT" in rejected
    assert "SOME_RANDOM_SECRET" not in child_env
    assert child_env["PATH"] == "/usr/bin"
