"""Security and reload contract for the implicit XDG runtime-secret source."""

from __future__ import annotations

import json
import os
import stat
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_utilities.core import config


@pytest.fixture(autouse=True)
def _isolated_runtime_secret_projection():
    environment = dict(os.environ)
    loaded = config._env_loaded
    xdg_projection = dict(config._xdg_injected_environment)
    secret_projection = dict(config._xdg_injected_runtime_secrets)
    source_state = config.runtime_secret_source_status()
    # This module's tests call the REAL ``config.load_config(reload=True)``
    # against a temp XDG root, which mutates the process-wide typed config
    # singleton (``_LAZY_CACHE``/``_CONFIG_PROXY._target``) — a side effect
    # this fixture did not previously restore, so a test that reloads onto
    # e.g. a synthetic ``embedding_models`` entry silently leaked that
    # snapshot into every later test in the suite (across files) that reads
    # ``config``.
    previous_lazy_cache = config._LAZY_CACHE
    previous_proxy_target = config._CONFIG_PROXY._current()
    config._env_loaded = False
    config._xdg_injected_environment.clear()
    config._xdg_injected_runtime_secrets.clear()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(environment)
        config._env_loaded = loaded
        config._xdg_injected_environment.clear()
        config._xdg_injected_environment.update(xdg_projection)
        config._xdg_injected_runtime_secrets.clear()
        config._xdg_injected_runtime_secrets.update(secret_projection)
        config._runtime_secret_source_state.clear()
        config._runtime_secret_source_state.update(source_state)
        config._LAZY_CACHE = previous_lazy_cache
        config._CONFIG_PROXY._swap(previous_proxy_target)


def _write_config(root: Path, value: object) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)
    return path


def _write_secrets(root: Path, value: object, *, mode: int = 0o600) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "runtime-secrets.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(mode)
    return path


def _select_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(root))
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "false")
    monkeypatch.setenv("APP_PROFILE", "dev")


def test_agent_config_projects_only_nested_referenced_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "isolated-config"
    _write_config(
        root,
        {
            "LANGFUSE_PUBLIC_KEY_REF": "env://TEST_XDG_PUBLIC",
            "CHAT_MODELS": [
                {
                    "id": "synthetic-model",
                    "provider": "openai",
                    "headers_ref": "env://TEST_XDG_NESTED",
                }
            ],
        },
    )
    _write_secrets(
        root,
        {
            "TEST_XDG_PUBLIC": "synthetic-public-material",
            "TEST_XDG_NESTED": '{"X-Client":"synthetic-nested-material"}',
            "TEST_XDG_UNUSED": "must-not-be-projected",
        },
    )
    _select_root(monkeypatch, root)
    for key in ("TEST_XDG_PUBLIC", "TEST_XDG_NESTED", "TEST_XDG_UNUSED"):
        monkeypatch.delenv(key, raising=False)

    parsed = config.AgentConfig()

    assert parsed.langfuse_public_key_ref == "env://TEST_XDG_PUBLIC"
    assert os.environ["TEST_XDG_PUBLIC"] == "synthetic-public-material"
    assert os.environ["TEST_XDG_NESTED"] == ('{"X-Client":"synthetic-nested-material"}')
    assert "TEST_XDG_UNUSED" not in os.environ
    assert all(
        isinstance(fingerprint, bytes) and len(fingerprint) == 32
        for fingerprint in config._xdg_injected_runtime_secrets.values()
    )
    tracking = repr(config._xdg_injected_runtime_secrets)
    assert "synthetic-public-material" not in tracking
    assert "synthetic-nested-material" not in tracking
    assert config.runtime_secret_source_status() == {
        "state": "ready",
        "present": True,
        "valid": True,
        "referenced_count": 2,
        "matched_count": 2,
        "projected_count": 2,
        "overridden_count": 0,
    }


def test_explicit_process_value_wins_without_becoming_loader_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "isolated-config"
    _write_config(root, {"LANGFUSE_SECRET_KEY_REF": "env://TEST_XDG_OVERRIDE"})
    _write_secrets(root, {"TEST_XDG_OVERRIDE": "file-material"})
    _select_root(monkeypatch, root)
    monkeypatch.setenv("TEST_XDG_OVERRIDE", "operator-material")

    config.load_config(reload=True)

    assert os.environ["TEST_XDG_OVERRIDE"] == "operator-material"
    assert "TEST_XDG_OVERRIDE" not in config._xdg_injected_runtime_secrets
    status = config.runtime_secret_source_status()
    assert status["matched_count"] == 1
    assert status["projected_count"] == 0
    assert status["overridden_count"] == 1


def test_concurrent_initial_load_never_exposes_an_in_progress_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "isolated-config"
    _write_config(
        root,
        {
            "LANGFUSE_PUBLIC_KEY_REF": "env://TEST_CONCURRENT_PUBLIC",
            "LANGFUSE_SECRET_KEY_REF": "env://TEST_CONCURRENT_SECRET",
        },
    )
    _write_secrets(
        root,
        {
            "TEST_CONCURRENT_PUBLIC": "public-material",
            "TEST_CONCURRENT_SECRET": "secret-material",
        },
    )
    _select_root(monkeypatch, root)
    monkeypatch.delenv("TEST_CONCURRENT_PUBLIC", raising=False)
    monkeypatch.delenv("TEST_CONCURRENT_SECRET", raising=False)

    reader_entered = threading.Event()
    allow_reader = threading.Event()
    second_started = threading.Event()
    second_finished = threading.Event()
    failures: list[BaseException] = []
    observed: dict[str, str | None] = {}
    real_reader = config._read_runtime_secret_source

    def blocking_reader(*args, **kwargs):
        result = real_reader(*args, **kwargs)
        reader_entered.set()
        if not allow_reader.wait(timeout=5):
            raise TimeoutError("test did not release staged source read")
        return result

    monkeypatch.setattr(config, "_read_runtime_secret_source", blocking_reader)

    def initial_loader() -> None:
        try:
            config.load_config(reload=True)
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    def parallel_loader() -> None:
        second_started.set()
        try:
            config.load_config()
            observed["public"] = os.environ.get("TEST_CONCURRENT_PUBLIC")
            observed["secret"] = os.environ.get("TEST_CONCURRENT_SECRET")
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)
        finally:
            second_finished.set()

    first = threading.Thread(target=initial_loader)
    second = threading.Thread(target=parallel_loader)
    first.start()
    assert reader_entered.wait(timeout=5)
    second.start()
    assert second_started.wait(timeout=5)
    assert not second_finished.wait(timeout=0.1)
    allow_reader.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert failures == []
    assert observed == {
        "public": "public-material",
        "secret": "secret-material",
    }


def test_config_save_serializes_file_write_and_projection_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "isolated-config"
    _write_config(
        root,
        {"LANGFUSE_SECRET_KEY_REF": "env://TEST_SAVE_INITIAL"},
    )
    _write_secrets(
        root,
        {
            "TEST_SAVE_INITIAL": "initial-material",
            "TEST_SAVE_UPDATED": "updated-material",
        },
    )
    _select_root(monkeypatch, root)
    monkeypatch.setattr(config, "_LAZY_CACHE", {})
    monkeypatch.delenv("TEST_SAVE_INITIAL", raising=False)
    monkeypatch.delenv("TEST_SAVE_UPDATED", raising=False)
    config.load_config(reload=True)

    writer_entered = threading.Event()
    allow_writer = threading.Event()
    parallel_finished = threading.Event()
    failures: list[BaseException] = []
    real_writer = config._write_private_configuration_mapping

    def blocking_writer(*args, **kwargs):
        writer_entered.set()
        if not allow_writer.wait(timeout=5):
            raise TimeoutError("test did not release staged config write")
        return real_writer(*args, **kwargs)

    monkeypatch.setattr(config, "_write_private_configuration_mapping", blocking_writer)

    def save() -> None:
        try:
            config.save_config_item(
                "LANGFUSE_SECRET_KEY_REF", "env://TEST_SAVE_UPDATED"
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    def parallel_reload() -> None:
        try:
            config.load_config(reload=True)
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)
        finally:
            parallel_finished.set()

    first = threading.Thread(target=save)
    second = threading.Thread(target=parallel_reload)
    first.start()
    assert writer_entered.wait(timeout=5)
    second.start()
    assert not parallel_finished.wait(timeout=0.1)
    allow_writer.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert failures == []
    assert os.environ.get("TEST_SAVE_INITIAL") is None
    assert os.environ["TEST_SAVE_UPDATED"] == "updated-material"


def test_reload_rotates_removes_and_preserves_late_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "isolated-config"
    config_path = _write_config(
        root, {"LANGFUSE_SECRET_KEY_REF": "env://TEST_XDG_ROTATE_A"}
    )
    secret_path = _write_secrets(
        root,
        {
            "TEST_XDG_ROTATE_A": "first-material",
            "TEST_XDG_ROTATE_B": "second-material",
        },
    )
    _select_root(monkeypatch, root)
    monkeypatch.delenv("TEST_XDG_ROTATE_A", raising=False)
    monkeypatch.delenv("TEST_XDG_ROTATE_B", raising=False)

    config.load_config(reload=True)
    assert os.environ["TEST_XDG_ROTATE_A"] == "first-material"

    config_path.write_text(
        json.dumps({"LANGFUSE_SECRET_KEY_REF": "env://TEST_XDG_ROTATE_B"}),
        encoding="utf-8",
    )
    secret_path.write_text(
        json.dumps(
            {
                "TEST_XDG_ROTATE_A": "changed-unused-material",
                "TEST_XDG_ROTATE_B": "second-material",
            }
        ),
        encoding="utf-8",
    )
    config.load_config(reload=True)
    assert "TEST_XDG_ROTATE_A" not in os.environ
    assert os.environ["TEST_XDG_ROTATE_B"] == "second-material"

    os.environ["TEST_XDG_ROTATE_B"] = "late-operator-material"
    secret_path.write_text(
        json.dumps({"TEST_XDG_ROTATE_B": "third-material"}),
        encoding="utf-8",
    )
    config.load_config(reload=True)
    assert os.environ["TEST_XDG_ROTATE_B"] == "late-operator-material"
    assert "TEST_XDG_ROTATE_B" not in config._xdg_injected_runtime_secrets


def test_failed_reload_preserves_previous_projection_without_disclosure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    root = tmp_path / "private-location"
    _write_config(root, {"LANGFUSE_SECRET_KEY_REF": "env://PRIVATE_TARGET_NAME"})
    secret_path = _write_secrets(
        root, {"PRIVATE_TARGET_NAME": "private-secret-material"}
    )
    _select_root(monkeypatch, root)
    monkeypatch.delenv("PRIVATE_TARGET_NAME", raising=False)
    config.load_config(reload=True)
    prior_fingerprint = config._xdg_injected_runtime_secrets["PRIVATE_TARGET_NAME"]

    secret_path.write_text("{malformed", encoding="utf-8")
    with pytest.raises(config.ConfigurationSourceError) as caught:
        config.load_config(reload=True)

    assert os.environ["PRIVATE_TARGET_NAME"] == "private-secret-material"
    assert (
        config._xdg_injected_runtime_secrets["PRIVATE_TARGET_NAME"] == prior_fingerprint
    )
    rendered = f"{caught.value}\n{caplog.text}"
    assert str(root) not in rendered
    assert "PRIVATE_TARGET_NAME" not in rendered
    assert "private-secret-material" not in rendered


def test_schema_invalid_reload_preserves_dynamic_and_typed_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "isolated-config"
    config_path = _write_config(root, {"ENABLE_OTEL": False})
    _select_root(monkeypatch, root)
    monkeypatch.setattr(config, "_LAZY_CACHE", config.BoundedLRUCache(max_size=64))
    monkeypatch.delenv("ENABLE_OTEL", raising=False)
    # ``_reload_typed_singleton_locked`` only re-populates ``_LAZY_CACHE`` when
    # a singleton ALREADY exists there (``singleton is not None``) — it is a
    # *reload*, not a first-time init. A fresh, empty cache has no existing
    # ``"_config"`` entry, so a bare ``load_config(reload=True)`` on it is a
    # silent no-op for population — deterministic only because an EARLIER
    # test in the full suite happens to have already populated the (later
    # monkeypatched-away) real cache. Force one real population explicitly so
    # this test's outcome does not depend on suite ordering.
    config._init_lazy_config(force=True)
    config.load_config(reload=True)
    stable = config.config
    previous = config._LAZY_CACHE["_config"]

    config_path.write_text(
        json.dumps({"ENABLE_OTEL": "not-a-boolean"}),
        encoding="utf-8",
    )
    with pytest.raises(config.ConfigurationSourceError):
        config.load_config(reload=True)

    assert config.setting("ENABLE_OTEL", True) is False
    assert config.config is stable
    assert config._LAZY_CACHE["_config"] is previous
    assert previous.enable_otel is False


def test_invalid_save_preserves_prior_file_and_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "isolated-config"
    config_path = _write_config(root, {"ENABLE_OTEL": False})
    _select_root(monkeypatch, root)
    monkeypatch.setattr(config, "_LAZY_CACHE", {})
    monkeypatch.delenv("ENABLE_OTEL", raising=False)
    config.load_config(reload=True)
    prior_document = json.loads(config_path.read_text(encoding="utf-8"))

    with pytest.raises(config.ConfigurationSourceError):
        config.save_config_item("DEPLOYMENT_PROFILE", "invalid-profile")

    assert json.loads(config_path.read_text(encoding="utf-8")) == prior_document
    assert config.setting("ENABLE_OTEL", True) is False


def test_reader_observes_only_complete_snapshot_during_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "isolated-config"
    config_path = _write_config(root, {"ENABLE_OTEL": False})
    _select_root(monkeypatch, root)
    monkeypatch.setattr(config, "_LAZY_CACHE", config.BoundedLRUCache(max_size=64))
    monkeypatch.delenv("ENABLE_OTEL", raising=False)
    # ``_reload_typed_singleton_locked`` only re-populates ``_LAZY_CACHE`` when
    # a singleton ALREADY exists there (``singleton is not None``) — it is a
    # *reload*, not a first-time init. A fresh, empty cache has no existing
    # ``"_config"`` entry, so a bare ``load_config(reload=True)`` on it is a
    # silent no-op for population — deterministic only because an EARLIER
    # test in the full suite happens to have already populated the (later
    # monkeypatched-away) real cache. Force one real population explicitly so
    # this test's outcome does not depend on suite ordering.
    config._init_lazy_config(force=True)
    config.load_config(reload=True)
    stable = config.config
    previous = config._LAZY_CACHE["_config"]
    config_path.write_text(json.dumps({"ENABLE_OTEL": True}), encoding="utf-8")

    reader_entered = threading.Event()
    allow_reader = threading.Event()
    observed = threading.Event()
    failures: list[BaseException] = []
    result: dict[str, object] = {}
    real_reader = config._read_runtime_secret_source

    def blocking_source(*args, **kwargs):
        value = real_reader(*args, **kwargs)
        reader_entered.set()
        if not allow_reader.wait(timeout=5):
            raise TimeoutError("test did not release staged source read")
        return value

    monkeypatch.setattr(config, "_read_runtime_secret_source", blocking_source)

    def reload() -> None:
        try:
            config.load_config(reload=True)
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    def read() -> None:
        try:
            result["dynamic"] = config.setting("ENABLE_OTEL", False)
            result["typed"] = config.config.enable_otel
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)
        finally:
            observed.set()

    writer = threading.Thread(target=reload)
    reader = threading.Thread(target=read)
    writer.start()
    assert reader_entered.wait(timeout=5)
    reader.start()
    assert not observed.wait(timeout=0.1)
    assert previous.enable_otel is False
    allow_reader.set()
    writer.join(timeout=5)
    reader.join(timeout=5)

    assert failures == []
    assert result == {"dynamic": True, "typed": True}
    assert config.config is stable
    assert config._LAZY_CACHE["_config"] is not previous
    assert previous.enable_otel is False


@pytest.mark.parametrize(
    ("raw", "error_class"),
    [
        ("[]", "TypeError"),
        ('{"INVALID-NAME":"value"}', "ValueError"),
        ('{"VALID_NAME":42}', "TypeError"),
        ('{"VALID_NAME":""}', "ValueError"),
        ('{"VALID_NAME":"bad\\u0000value"}', "ValueError"),
        ('{"VALID_NAME":"one","VALID_NAME":"two"}', "ValueError"),
        ('{"VALID_NAME":"one","valid_name":"two"}', "ValueError"),
    ],
)
def test_present_invalid_document_fails_closed(
    tmp_path: Path,
    raw: str,
    error_class: str,
) -> None:
    path = tmp_path / "runtime-secrets.json"
    path.write_text(raw, encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(config.ConfigurationSourceError) as caught:
        config._read_runtime_secret_source(
            path,
            targets=frozenset({"VALID_NAME"}),
        )

    assert caught.value.source_type == "runtime-secrets"
    assert caught.value.error_class == error_class
    assert str(tmp_path) not in str(caught.value)
    assert "VALID_NAME" not in str(caught.value)


@pytest.mark.skipif(os.name != "posix", reason="POSIX private-mode contract")
@pytest.mark.parametrize("mode", [0o000, 0o200, 0o440, 0o640, 0o700])
def test_posix_mode_must_be_exactly_read_only_or_read_write_for_owner(
    tmp_path: Path,
    mode: int,
) -> None:
    path = _write_secrets(tmp_path, {"TEST_MODE_KEY": "material"}, mode=mode)

    with pytest.raises(config.ConfigurationSourceError, match="PermissionError"):
        config._read_runtime_secret_source(
            path,
            targets=frozenset({"TEST_MODE_KEY"}),
        )


@pytest.mark.skipif(os.name != "posix", reason="POSIX private-mode contract")
@pytest.mark.parametrize("mode", [0o400, 0o600])
def test_posix_private_modes_are_accepted(tmp_path: Path, mode: int) -> None:
    path = _write_secrets(tmp_path, {"TEST_MODE_KEY": "material"}, mode=mode)

    present, selected = config._read_runtime_secret_source(
        path,
        targets=frozenset({"TEST_MODE_KEY"}),
    )

    assert present is True
    assert selected == {"TEST_MODE_KEY": "material"}


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership contract")
def test_untrusted_owner_metadata_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = SimpleNamespace(
        st_mode=stat.S_IFREG | 0o600,
        st_size=1,
        st_uid=1001,
    )
    monkeypatch.setattr(config.os, "geteuid", lambda: 1002)

    with pytest.raises(PermissionError):
        config._validate_runtime_secret_metadata(metadata)


def test_non_posix_private_file_posture_fails_closed(monkeypatch) -> None:
    metadata = SimpleNamespace(
        st_mode=stat.S_IFREG | 0o600,
        st_size=1,
        st_uid=0,
    )
    monkeypatch.setattr(config.os, "name", "nt")

    with pytest.raises(PermissionError, match="unsupported"):
        config._validate_runtime_secret_metadata(metadata)
    with pytest.raises(PermissionError, match="unsupported"):
        config._validate_configuration_metadata(metadata, strict=True)


@pytest.mark.skipif(os.name != "posix", reason="POSIX no-follow contract")
def test_symbolic_link_is_rejected(tmp_path: Path) -> None:
    target = _write_secrets(tmp_path / "target", {"TEST_LINK_KEY": "material"})
    link = tmp_path / "runtime-secrets.json"
    link.symlink_to(target)

    with pytest.raises(config.ConfigurationSourceError, match="PermissionError"):
        config._read_runtime_secret_source(
            link,
            targets=frozenset({"TEST_LINK_KEY"}),
        )


def test_oversized_source_and_value_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "runtime-secrets.json"
    source.write_bytes(b"x" * (config._MAX_RUNTIME_SECRET_SOURCE_BYTES + 1))
    source.chmod(0o600)
    with pytest.raises(config.ConfigurationSourceError, match="ValueError"):
        config._read_runtime_secret_source(source, targets=frozenset())

    source.write_text(
        json.dumps(
            {"TEST_LARGE_VALUE": "x" * (config._MAX_RUNTIME_SECRET_VALUE_BYTES + 1)}
        ),
        encoding="utf-8",
    )
    source.chmod(0o600)
    with pytest.raises(config.ConfigurationSourceError, match="ValueError"):
        config._read_runtime_secret_source(
            source,
            targets=frozenset({"TEST_LARGE_VALUE"}),
        )


def test_invalid_utf8_and_nonregular_source_are_rejected(tmp_path: Path) -> None:
    invalid_utf8 = tmp_path / "runtime-secrets.json"
    invalid_utf8.write_bytes(b'{"TEST_UTF8":"\xff"}')
    invalid_utf8.chmod(0o600)
    with pytest.raises(config.ConfigurationSourceError, match="UnicodeDecodeError"):
        config._read_runtime_secret_source(
            invalid_utf8,
            targets=frozenset({"TEST_UTF8"}),
        )

    invalid_utf8.unlink()
    invalid_utf8.mkdir()
    with pytest.raises(config.ConfigurationSourceError, match="PermissionError"):
        config._read_runtime_secret_source(
            invalid_utf8,
            targets=frozenset({"TEST_UTF8"}),
        )


def test_source_mutation_during_read_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_secrets(tmp_path, {"TEST_MUTATION": "material"})
    real_fstat = os.fstat
    calls = 0

    def changed_fstat(descriptor: int):
        nonlocal calls
        calls += 1
        metadata = real_fstat(descriptor)
        if calls != 2:
            return metadata
        return SimpleNamespace(
            st_mode=metadata.st_mode,
            st_size=metadata.st_size,
            st_uid=metadata.st_uid,
            st_dev=metadata.st_dev,
            st_ino=metadata.st_ino,
            st_mtime_ns=metadata.st_mtime_ns + 1,
        )

    monkeypatch.setattr(config.os, "fstat", changed_fstat)

    with pytest.raises(config.ConfigurationSourceError, match="PermissionError"):
        config._read_runtime_secret_source(
            source,
            targets=frozenset({"TEST_MUTATION"}),
        )


def test_optional_missing_source_is_valid_and_projects_nothing(
    tmp_path: Path,
) -> None:
    present, selected = config._read_runtime_secret_source(
        tmp_path / "runtime-secrets.json",
        targets=frozenset({"TEST_MISSING_KEY"}),
    )

    assert present is False
    assert selected == {}


def test_durable_secret_target_collision_is_rejected_without_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "isolated-config"
    _write_config(
        root,
        {
            "LANGFUSE_SECRET_KEY_REF": "env://MCP_TOOL_MODE",
            "MCP_TOOL_MODE": "intent",
        },
    )
    _write_secrets(root, {"MCP_TOOL_MODE": "runtime-material"})
    _select_root(monkeypatch, root)

    with pytest.raises(config.ConfigurationSourceError) as caught:
        config.load_config(reload=True)

    assert caught.value.error_class == "SecretTargetCollisionError"
    assert "MCP_TOOL_MODE" not in str(caught.value)
    assert str(root) not in str(caught.value)


@pytest.mark.parametrize(
    "durable_value",
    [
        {"OPENAI_API_KEY": "literal-material"},
        {"OPENAI_API_KEY": "env://RUNTIME_PROVIDER_KEY"},
        {"MESSAGING_SLACK_TOKEN": "literal-material"},
        {"MCP_FLEET_SECRET_REFS": {"CHILD_TOKEN": "literal-material"}},
        {"EXTRA_HEADERS": {"X-Client": "literal-material"}},
        {
            "CHAT_MODELS": [
                {
                    "id": "synthetic-model",
                    "provider": "openai",
                    "api_key": "literal-material",
                }
            ]
        },
        {
            "EMBEDDING_MODELS": [
                {
                    "id": "synthetic-embedder",
                    "provider": "openai",
                    "headers": {"X-Client": "literal-material"},
                }
            ]
        },
    ],
)
def test_durable_xdg_rejects_raw_credentials_and_headers_without_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    durable_value: dict[str, object],
) -> None:
    root = tmp_path / "isolated-config"
    _write_config(root, durable_value)
    _select_root(monkeypatch, root)

    with pytest.raises(config.ConfigurationSourceError) as caught:
        config.load_config(reload=True)

    assert caught.value.error_class == "DurableSecretError"
    rendered = str(caught.value)
    assert "literal-material" not in rendered
    assert "OPENAI_API_KEY" not in rendered
    assert "MESSAGING_SLACK_TOKEN" not in rendered
    assert "MCP_FLEET_SECRET_REFS" not in rendered
    assert "EXTRA_HEADERS" not in rendered
    assert "CHAT_MODELS" not in rendered
    assert "EMBEDDING_MODELS" not in rendered
    assert str(root) not in rendered


def test_durable_xdg_accepts_only_supported_secret_reference_forms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "isolated-config"
    _write_config(
        root,
        {
            "LANGFUSE_SECRET_KEY_REF": "env://TEST_XDG_LANGFUSE_SECRET",
            "PERMISSIONS_SIGNING_KEY_REF": "env://TEST_XDG_PERMISSION_KEY",
            "MCP_FLEET_SECRET_REFS": {"CHILD_TOKEN": "env://CHILD_TOKEN"},
            "PROVIDER_CONFIGS": {
                "synthetic-provider": {
                    "enabled": True,
                    "endpoint_ref": "env://TEST_XDG_PROVIDER_ENDPOINT",
                    "credential_refs": {
                        "PROVIDER_TOKEN": "env://TEST_XDG_PROVIDER_TOKEN"
                    },
                    "selector_refs": {
                        "PROVIDER_SCOPE": "env://TEST_XDG_PROVIDER_SCOPE"
                    },
                    "tls_profile_ref": "env://TEST_XDG_PROVIDER_TLS",
                }
            },
            "CHAT_MODELS": [
                {
                    "id": "synthetic-model",
                    "provider": "openai",
                    "oauth2": {
                        "token_url": "https://identity.example.test/token",
                        "client_id": "synthetic-client",
                        "client_secret": "env://TEST_XDG_MODEL_SECRET",
                    },
                }
            ],
            "EMBEDDING_MODELS": [
                {
                    "id": "synthetic-embedder",
                    "provider": "openai",
                    "api_key_ref": "env://TEST_XDG_EMBEDDING_KEY",
                    "headers_ref": "env://TEST_XDG_EMBEDDING_HEADERS",
                }
            ],
        },
    )
    _write_secrets(
        root,
        {
            "TEST_XDG_LANGFUSE_SECRET": "runtime-langfuse-material",
            "TEST_XDG_PERMISSION_KEY": "runtime-permission-authority-key-32b",
            "TEST_XDG_MODEL_SECRET": "runtime-model-material",
            "TEST_XDG_EMBEDDING_KEY": "runtime-embedding-material",
            "TEST_XDG_EMBEDDING_HEADERS": '{"X-Client":"runtime-client"}',
            "CHILD_TOKEN": "runtime-child-material",
            "TEST_XDG_PROVIDER_ENDPOINT": "https://provider.example.test/api",
            "TEST_XDG_PROVIDER_TOKEN": "runtime-provider-material",
            "TEST_XDG_PROVIDER_SCOPE": "read-only",
            "TEST_XDG_PROVIDER_TLS": '{"system_trust":true}',
        },
    )
    _select_root(monkeypatch, root)

    config.load_config(reload=True)

    assert os.environ["TEST_XDG_LANGFUSE_SECRET"] == "runtime-langfuse-material"
    assert os.environ["TEST_XDG_PERMISSION_KEY"] == (
        "runtime-permission-authority-key-32b"
    )
    assert os.environ["TEST_XDG_MODEL_SECRET"] == "runtime-model-material"
    assert os.environ["TEST_XDG_EMBEDDING_KEY"] == "runtime-embedding-material"
    assert os.environ["TEST_XDG_EMBEDDING_HEADERS"] == ('{"X-Client":"runtime-client"}')
    assert os.environ["CHILD_TOKEN"] == "runtime-child-material"
    assert config.AgentConfig().mcp_fleet_secret_refs == {
        "CHILD_TOKEN": "env://CHILD_TOKEN"
    }
    provider = config.AgentConfig().provider_configs["synthetic-provider"]
    assert provider.endpoint_ref == "env://TEST_XDG_PROVIDER_ENDPOINT"
    assert provider.credential_refs == {
        "PROVIDER_TOKEN": "env://TEST_XDG_PROVIDER_TOKEN"
    }


def test_agent_config_never_reads_repository_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".env").write_text(
        "LANGFUSE_SECRET_KEY_REF=env://SHOULD_NOT_LOAD\n",
        encoding="utf-8",
    )
    root = tmp_path / "isolated-config"
    root.mkdir()
    _select_root(monkeypatch, root)
    monkeypatch.chdir(checkout)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY_REF", raising=False)

    parsed = config.AgentConfig()

    assert parsed.langfuse_secret_key_ref is None
    assert config.AgentConfig.model_config.get("env_file") is None


def test_agent_config_does_not_accept_an_arbitrary_secret_file_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arbitrary = tmp_path / "arbitrary-source.json"
    arbitrary.write_text(
        json.dumps({"LANGFUSE_SECRET_KEY_REF": "env://SHOULD_NOT_LOAD"}),
        encoding="utf-8",
    )
    arbitrary.chmod(0o600)
    root = tmp_path / "isolated-config"
    root.mkdir()
    _select_root(monkeypatch, root)
    monkeypatch.setenv("AGENT_SECRETS_FILE", str(arbitrary))
    monkeypatch.delenv("LANGFUSE_SECRET_KEY_REF", raising=False)

    parsed = config.AgentConfig()

    assert parsed.langfuse_secret_key_ref is None


@pytest.mark.skipif(os.name != "posix", reason="POSIX private-mode contract")
def test_doctor_rejects_invalid_source_without_reporting_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.deployment.doctor import run_doctor

    root = tmp_path / "private-doctor-location"
    _write_config(
        root,
        {"LANGFUSE_SECRET_KEY_REF": "env://PRIVATE_DOCTOR_TARGET"},
    )
    source = _write_secrets(
        root,
        {"PRIVATE_DOCTOR_TARGET": "private-doctor-material"},
    )
    source.chmod(0o640)
    _select_root(monkeypatch, root)
    config._env_loaded = False

    result = run_doctor(only=["secrets"])
    rendered = json.dumps(result, sort_keys=True)

    assert result["status"] == "unhealthy"
    assert result["checks"][0]["status"] == "error"
    assert result["checks"][0]["data"] == {"redacted": True}
    assert str(root) not in rendered
    assert "PRIVATE_DOCTOR_TARGET" not in rendered
    assert "private-doctor-material" not in rendered
