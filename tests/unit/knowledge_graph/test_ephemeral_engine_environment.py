"""Security contract for the real ephemeral test engine."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import tests._test_engine as engine_helper
from tests._test_engine import (
    TEST_AGENT_ID,
    TEST_AUDIENCE,
    TEST_POLICY_VERSION,
    TEST_SIGNER_KEY,
    TEST_SIGNER_REGISTRY,
    TEST_TENANT,
    EngineUnavailable,
    strict_server_env,
)


def test_strict_server_env_retains_auth_and_only_opts_out_of_oidc() -> None:
    """The local fixture must not weaken transport or request authentication."""

    auth_secret = "synthetic-engine-auth-secret"  # nosec B105 - test-only  # sanitizer:ignore — synthetic fixture, never a real credential
    state_dir = "/synthetic/security-state"

    env = strict_server_env(state_dir, auth_secret=auth_secret)

    assert env == {
        "GRAPH_SERVICE_AUTH_SECRET": auth_secret,
        "EPISTEMIC_GRAPH_AUDIENCE": TEST_AUDIENCE,
        "EPISTEMIC_GRAPH_TENANT": TEST_TENANT,
        "EPISTEMIC_GRAPH_POLICY_VERSION": TEST_POLICY_VERSION,
        "EPISTEMIC_GRAPH_REQUIRE_OIDC": "false",
        "EPISTEMIC_GRAPH_SECURITY_STATE_DIR": state_dir,
        "EPISTEMIC_GRAPH_SIGNER_KEYS_JSON": json.dumps(
            TEST_SIGNER_REGISTRY, sort_keys=True
        ),
    }
    assert json.loads(env["EPISTEMIC_GRAPH_SIGNER_KEYS_JSON"]) == {
        TEST_AGENT_ID: {
            "key": TEST_SIGNER_KEY,
            "allowed_roles": [],
            "may_grant_system": True,
        }
    }
    assert "EPISTEMIC_GRAPH_ALLOW_INSECURE" not in env


def test_fixture_signer_scope_is_genesis_only_and_unknown_signers_are_untrusted() -> (
    None
):
    """The fixture signer cannot authorize another identity or RBAC role."""

    registry = json.loads(
        strict_server_env(
            "/synthetic/security-state",
            auth_secret="synthetic-engine-auth-secret",  # sanitizer:ignore - synthetic fixture, never a real credential
        )["EPISTEMIC_GRAPH_SIGNER_KEYS_JSON"]
    )

    assert set(registry) == {TEST_AGENT_ID}
    assert registry[TEST_AGENT_ID]["allowed_roles"] == []
    assert registry[TEST_AGENT_ID]["may_grant_system"] is True
    assert "service:unauthorized-fixture-signer" not in registry


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o700)
    return path


_PATH_ENV_NAMES = (
    "EPISTEMIC_GRAPH_SERVER_BIN",
    "EPISTEMIC_GRAPH_TEST_BINARY",
)
_METADATA_ENV_NAMES = (
    "EPISTEMIC_GRAPH_SERVER_BIN_SHA256",
    "EPISTEMIC_GRAPH_TEST_BINARY_SHA256",
    "EPISTEMIC_GRAPH_SERVER_BIN_SOURCE_REVISION",
    "EPISTEMIC_GRAPH_TEST_BINARY_SOURCE_REVISION",
)


def _clear_locator_env(monkeypatch) -> None:
    for name in (*_PATH_ENV_NAMES, *_METADATA_ENV_NAMES):
        monkeypatch.delenv(name, raising=False)


def _configure_explicit(
    monkeypatch, candidate: Path, *, env_name: str = "EPISTEMIC_GRAPH_SERVER_BIN"
) -> str:
    _clear_locator_env(monkeypatch)
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    monkeypatch.setenv(env_name, str(candidate))
    monkeypatch.setenv(f"{env_name}_SHA256", digest)
    monkeypatch.setenv(f"{env_name}_SOURCE_REVISION", "a" * 40)
    return digest


def test_engine_locator_accepts_only_an_explicit_regular_executable(
    tmp_path, monkeypatch
) -> None:
    candidate = _executable(tmp_path / "epistemic-graph-server")
    _configure_explicit(monkeypatch, candidate)

    assert engine_helper.resolve_engine_binary() == candidate


def test_existing_exact_test_binary_digest_contract_is_bound(
    tmp_path, monkeypatch
) -> None:
    candidate = _executable(tmp_path / "epistemic-graph-server")
    digest = _configure_explicit(
        monkeypatch, candidate, env_name="EPISTEMIC_GRAPH_TEST_BINARY"
    )

    identity = engine_helper.resolve_engine_binary_identity()

    assert identity.path == candidate
    assert identity.source_env == "EPISTEMIC_GRAPH_TEST_BINARY"
    assert identity.expected_sha256 == digest
    assert identity.source_revision == "a" * 40


def test_explicit_engine_identity_binds_complete_artifact_digest(
    tmp_path, monkeypatch
) -> None:
    candidate = _executable(tmp_path / "epistemic-graph-server")
    digest = _configure_explicit(monkeypatch, candidate)

    identity = engine_helper.resolve_engine_binary_identity()

    assert identity.path == candidate
    assert identity.selection == "explicit"
    assert identity.artifact_size == candidate.stat().st_size
    assert len(identity.artifact_sha256) == 64
    assert identity.artifact_sha256 == digest
    assert identity.expected_sha256 == digest
    assert identity.source_env == "EPISTEMIC_GRAPH_SERVER_BIN"
    assert identity.source_revision == "a" * 40
    assert identity.distribution_version is None
    assert identity.distribution_record is None

    candidate.write_text("#!/bin/sh\nchanged\n", encoding="utf-8")
    candidate.chmod(0o700)
    with pytest.raises(EngineUnavailable, match="changed|does not match"):
        identity.verify_for_launch()
    with pytest.raises(EngineUnavailable, match="does not match"):
        engine_helper.resolve_engine_binary_identity()


@pytest.mark.parametrize("kind", ["directory", "non-executable", "symlink"])
def test_engine_locator_rejects_non_regular_or_non_executable_explicit_paths(
    tmp_path, monkeypatch, kind: str
) -> None:
    target = _executable(tmp_path / "target")
    candidate = tmp_path / "epistemic-graph-server"
    if kind == "directory":
        candidate.mkdir()
    elif kind == "non-executable":
        candidate.write_text("not executable", encoding="utf-8")
    else:
        if not hasattr(os, "symlink"):
            pytest.skip("symlinks are unavailable")
        candidate.symlink_to(target)
    _clear_locator_env(monkeypatch)
    monkeypatch.setenv("EPISTEMIC_GRAPH_SERVER_BIN", str(candidate))
    monkeypatch.setenv("EPISTEMIC_GRAPH_SERVER_BIN_SHA256", "0" * 64)
    monkeypatch.setenv("EPISTEMIC_GRAPH_SERVER_BIN_SOURCE_REVISION", "a" * 40)

    with pytest.raises(EngineUnavailable):
        engine_helper.resolve_engine_binary()


def test_engine_locator_never_uses_a_stale_path_binary(tmp_path, monkeypatch) -> None:
    _executable(tmp_path / "epistemic-graph-server")
    monkeypatch.setenv("PATH", str(tmp_path))
    _clear_locator_env(monkeypatch)

    def missing_distribution(_name: str):
        raise engine_helper.importlib_metadata.PackageNotFoundError(_name)

    monkeypatch.setattr(
        engine_helper.importlib_metadata, "distribution", missing_distribution
    )

    with pytest.raises(EngineUnavailable, match="distribution"):
        engine_helper.resolve_engine_binary()


@pytest.mark.parametrize(
    "missing",
    [
        "EPISTEMIC_GRAPH_SERVER_BIN_SHA256",
        "EPISTEMIC_GRAPH_SERVER_BIN_SOURCE_REVISION",
    ],
)
def test_explicit_engine_requires_digest_and_source_revision(
    tmp_path, monkeypatch, missing: str
) -> None:
    candidate = _executable(tmp_path / "epistemic-graph-server")
    _configure_explicit(monkeypatch, candidate)
    monkeypatch.delenv(missing, raising=False)

    with pytest.raises(EngineUnavailable, match="requires expected SHA-256"):
        engine_helper.resolve_engine_binary_identity()


@pytest.mark.parametrize(
    ("digest", "revision", "message"),
    [
        ("not-a-digest", "a" * 40, "SHA-256"),
        ("a" * 63, "a" * 40, "SHA-256"),
        ("a" * 64, "main", "source revision"),
        ("a" * 64, "feature/test", "source revision"),
    ],
)
def test_explicit_engine_rejects_malformed_digest_or_revision(
    tmp_path, monkeypatch, digest: str, revision: str, message: str
) -> None:
    candidate = _executable(tmp_path / "epistemic-graph-server")
    _clear_locator_env(monkeypatch)
    monkeypatch.setenv("EPISTEMIC_GRAPH_SERVER_BIN", str(candidate))
    monkeypatch.setenv("EPISTEMIC_GRAPH_SERVER_BIN_SHA256", digest)
    monkeypatch.setenv("EPISTEMIC_GRAPH_SERVER_BIN_SOURCE_REVISION", revision)

    with pytest.raises(EngineUnavailable, match=message):
        engine_helper.resolve_engine_binary_identity()


def test_explicit_engine_rejects_conflicting_digest_authorities(
    tmp_path, monkeypatch
) -> None:
    candidate = _executable(tmp_path / "epistemic-graph-server")
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    _clear_locator_env(monkeypatch)
    monkeypatch.setenv("EPISTEMIC_GRAPH_SERVER_BIN", str(candidate))
    monkeypatch.setenv("EPISTEMIC_GRAPH_TEST_BINARY", str(candidate))
    monkeypatch.setenv("EPISTEMIC_GRAPH_SERVER_BIN_SHA256", digest)
    monkeypatch.setenv("EPISTEMIC_GRAPH_TEST_BINARY_SHA256", "0" * 64)
    monkeypatch.setenv("EPISTEMIC_GRAPH_SERVER_BIN_SOURCE_REVISION", "a" * 40)
    monkeypatch.setenv("EPISTEMIC_GRAPH_TEST_BINARY_SOURCE_REVISION", "a" * 40)

    with pytest.raises(EngineUnavailable, match="conflicting"):
        engine_helper.resolve_engine_binary_identity()


def test_explicit_metadata_without_a_path_fails_closed(tmp_path, monkeypatch) -> None:
    _clear_locator_env(monkeypatch)
    monkeypatch.setenv("EPISTEMIC_GRAPH_TEST_BINARY_SHA256", "a" * 64)
    monkeypatch.setenv("EPISTEMIC_GRAPH_TEST_BINARY_SOURCE_REVISION", "a" * 40)

    with pytest.raises(EngineUnavailable, match="requires an explicit engine path"):
        engine_helper.resolve_engine_binary_identity()


def test_engine_locator_rejects_conflicting_explicit_artifacts(
    tmp_path, monkeypatch
) -> None:
    first = _executable(tmp_path / "first" / "epistemic-graph-server")
    second = _executable(tmp_path / "second" / "epistemic-graph-server")
    monkeypatch.setenv("EPISTEMIC_GRAPH_SERVER_BIN", str(first))
    monkeypatch.setenv("EPISTEMIC_GRAPH_TEST_BINARY", str(second))

    with pytest.raises(EngineUnavailable, match="conflicting"):
        engine_helper.resolve_engine_binary()


def test_engine_locator_uses_the_active_distribution_record_only(
    tmp_path, monkeypatch
) -> None:
    candidate = _executable(tmp_path / "epistemic-graph-server")

    class _Distribution:
        files = (Path("epistemic-graph-1.0.0.data/scripts/epistemic-graph-server"),)

        def locate_file(self, _record: Path) -> Path:
            return candidate

    _clear_locator_env(monkeypatch)
    monkeypatch.setattr(
        engine_helper.importlib_metadata,
        "distribution",
        lambda name: _Distribution(),
    )

    assert engine_helper.resolve_engine_binary() == candidate


def test_distribution_engine_identity_records_the_selected_script(
    tmp_path, monkeypatch
) -> None:
    candidate = _executable(tmp_path / "epistemic-graph-server")

    class _Distribution:
        version = "1.0.0"
        files = (Path("epistemic-graph-1.0.0.data/scripts/epistemic-graph-server"),)

        def locate_file(self, _record: Path) -> Path:
            return candidate

    _clear_locator_env(monkeypatch)
    monkeypatch.setattr(
        engine_helper.importlib_metadata,
        "distribution",
        lambda name: _Distribution(),
    )

    identity = engine_helper.resolve_engine_binary_identity()

    assert identity.path == candidate
    assert identity.selection == "distribution"
    assert identity.distribution_version == "1.0.0"
    assert identity.distribution_record == str(_Distribution.files[0])
    assert identity.distribution_record_sha256 is None


def test_engine_locator_rejects_ambiguous_or_mismatched_distribution_records(
    tmp_path, monkeypatch
) -> None:
    first = _executable(tmp_path / "first" / "epistemic-graph-server")
    second = _executable(tmp_path / "second" / "epistemic-graph-server")

    class _Distribution:
        files = (
            Path("one.data/scripts/epistemic-graph-server"),
            Path("two.data/scripts/epistemic-graph-server"),
        )

        def locate_file(self, record: Path) -> Path:
            return first if record.parts[0] == "one.data" else second

    _clear_locator_env(monkeypatch)
    monkeypatch.setattr(
        engine_helper.importlib_metadata,
        "distribution",
        lambda name: _Distribution(),
    )

    with pytest.raises(EngineUnavailable, match="exactly one"):
        engine_helper.resolve_engine_binary()


def test_ephemeral_engine_rechecks_identity_before_spawn(tmp_path, monkeypatch) -> None:
    candidate = _executable(tmp_path / "epistemic-graph-server")
    _configure_explicit(monkeypatch, candidate)
    identity = engine_helper.resolve_engine_binary_identity()
    candidate.write_text("#!/bin/sh\nmutated\n", encoding="utf-8")
    candidate.chmod(0o700)

    def unexpected_spawn(*args, **kwargs):
        raise AssertionError("identity failure must happen before Popen")

    monkeypatch.setattr(engine_helper.subprocess, "Popen", unexpected_spawn)
    with pytest.raises(EngineUnavailable, match="changed|does not match"):
        engine_helper.EphemeralEngine(identity).start()
