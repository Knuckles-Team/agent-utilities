#!/usr/bin/python
"""Configuration Management Module.

This module handles the loading and validation of agent settings from the XDG
configuration document and process-bound runtime injection. It defines a
centralized AgentConfig class and exports default configuration constants used
throughout the agent-utilities package.
"""

import ipaddress
import os
import pathlib
import re
import threading
from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

import platformdirs
from pydantic import Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# Re-export the dependency-free env accessor (config discipline). Kept in its own
# module so it stays importable while this module is still initializing — see
# agent_utilities/core/_env.py. Modules use `from agent_utilities.core.config
# import setting`.
from agent_utilities.core._env import setting  # noqa: F401

_LANGFUSE_DEFAULT_HOST = "https://cloud.langfuse.com"


def resolve_langfuse_host(
    default: str = _LANGFUSE_DEFAULT_HOST,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve canonical ``LANGFUSE_HOST`` without exposing it in logs."""

    def read(name: str) -> str:
        if environ is not None:
            return str(environ.get(name, "") or "")
        return str(setting(name, "") or "")

    host = read("LANGFUSE_HOST")
    candidate = str(host) if host else default
    if not candidate:
        return ""
    return _validated_langfuse_host(candidate)


DEFAULT_DB_PATH = str(
    platformdirs.user_data_path("agent-utilities", "knuckles-team") / "graph_state"
)

# CONCEPT:AU-ORCH.execution.reserved-inference-slots — local-inference slots always kept free for the interactive path
# (the messaging responder + graph-os-spawned pydantic-ai agents, which share the default
# model). Background KG work is bounded to (capacity − this). A constant, not a knob: 1 is
# the correct universal default (config discipline — no flag for a one-correct-value).
RESERVED_INTERACTIVE_INSTANCES = 1

from agent_utilities.base_utilities import (
    to_boolean,
    to_dict,
    to_list,
)

try:
    import logfire  # noqa: F401

    HAS_LOGFIRE = True
except ImportError:
    HAS_LOGFIRE = False

os.environ.setdefault("LOGFIRE_SEND_TO_LOGFIRE", "false")


def _apply_otel_sdk_policy(enabled: bool) -> None:
    """Project the single typed OTel toggle into the SDK's disable switch."""

    if enabled:
        os.environ.pop("OTEL_SDK_DISABLED", None)
    else:
        os.environ["OTEL_SDK_DISABLED"] = "true"


meta = {"name": "Agent", "description": "AI Agent"}


_env_loaded = False
# Values copied from the XDG document into ``os.environ`` are tracked separately
# from real process-environment overrides.  Without this distinction a reload
# cannot replace (or remove) a value that the previous XDG load injected because
# it appears indistinguishable from an operator-supplied environment variable.
_xdg_injected_environment: dict[str, str] = {}
# Secret projections have a separate ownership ledger so reload can replace or
# remove only values that this loader still owns.  Neither mapping is logged or
# exposed through doctor output.
_xdg_injected_runtime_secrets: dict[str, bytes] = {}
_runtime_secret_fingerprint_key = os.urandom(32)
_xdg_projection_lock = threading.RLock()

_MAX_CONFIGURATION_SOURCE_BYTES = 1_048_576
_MAX_RUNTIME_SECRET_SOURCE_BYTES = 1_048_576
_MAX_RUNTIME_SECRET_VALUE_BYTES = 262_144
_MAX_RUNTIME_SECRET_ENTRIES = 1_024
_MAX_RUNTIME_REFERENCE_SCAN_NODES = 100_000
_RUNTIME_SECRET_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_RUNTIME_SECRET_REFERENCE_RE = re.compile(r"^env://([A-Za-z_][A-Za-z0-9_]{0,127})$")
_runtime_secret_source_state: dict[str, Any] = {
    "state": "not_loaded",
    "present": False,
    "valid": True,
    "referenced_count": 0,
    "matched_count": 0,
    "projected_count": 0,
    "overridden_count": 0,
}
_PRODUCTION_PROFILE_VALUES = frozenset({"prod", "production"})
PRODUCTION_CERTIFICATION_SCENARIOS: tuple[str, ...] = (
    "identity-tls-policy-trace",
    "kill-commit-phases",
    "worker-process-loss",
    "raft-leader-loss",
    "broker-leader-loss",
    "node-loss",
    "zone-isolation",
    "broker-rebalance",
    "online-reshard",
    "atomic-exact-release-cutover",
    "one-time-index-migration",
    "one-time-ontology-migration",
    "backup-restore",
    "regional-recovery",
    "policy-and-deletion-propagation",
)
_RETIRED_CERTIFICATION_HOOK_KEYS = frozenset(
    {
        "CERT_HOOK_IDENTITY_TLS_TRACE",
        "CERT_HOOK_KILL_COMMIT_PHASE",
        "CERT_HOOK_WORKER_LOSS",
        "CERT_HOOK_RAFT_LEADER_LOSS",
        "CERT_HOOK_BROKER_LEADER_LOSS",
        "CERT_HOOK_NODE_LOSS",
        "CERT_HOOK_ZONE_ISOLATION",
        "CERT_HOOK_BROKER_REBALANCE",
        "CERT_HOOK_ONLINE_RESHARD",
        "CERT_HOOK_ATOMIC_RELEASE_CUTOVER",
        "CERT_HOOK_INDEX_MIGRATION",
        "CERT_HOOK_ONTOLOGY_MIGRATION",
        "CERT_HOOK_BACKUP_RESTORE",
        "CERT_HOOK_REGIONAL_RECOVERY",
        "CERT_HOOK_POLICY_DELETION",
    }
)
_RETIRED_CERTIFICATION_FAULT_KEYS = frozenset(
    f"CERT_{operation}_{scenario.upper().replace('-', '_')}"
    for operation in ("ACTION", "PROBE")
    for scenario in PRODUCTION_CERTIFICATION_SCENARIOS
)
_RETIRED_CONFIGURATION_KEYS = (
    frozenset(
        {
            "ENGINE_" + "MODE",
            "ENGINE_" + "ENDPOINT",
            "EPISTEMIC_GRAPH_" + "AUTOSTART",
            "EPISTEMIC_GRAPH_" + "ENCRYPTION_KEY",
            "GRAPH_SERVICE_" + "SOCKET",
            "GRAPH_SERVICE_TCP_" + "ADDR",
            "GRAPH_DIRECT_" + "EXECUTION",
            "GRAPH_" + "BACKEND",
            "GRAPH_" + "AUTHORITY",
            "MESSAGING_" + "REACTIONS",
            "A2A_BROKER_" + "URL",
            "A2A_STORAGE_" + "URL",
            "PERMISSIONS_SIGNING_" + "KEY",
            "CERT_PROMETHEUS_BEARER_TOKEN_FILE",
        }
    )
    | _RETIRED_CERTIFICATION_HOOK_KEYS
    | _RETIRED_CERTIFICATION_FAULT_KEYS
)
_RETIRED_DURABLE_OTLP_CONFIGURATION_KEYS = frozenset(
    {
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_EXPORTER_OTLP_PUBLIC_KEY",
        "OTEL_EXPORTER_OTLP_SECRET_KEY",
    }
)
_RETIRED_DURABLE_OUTBOUND_SECRET_KEYS = frozenset(
    {
        "OIDC_CLIENT_SECRET",
        "MCP_BASIC_AUTH_PASSWORD",
    }
)


def _require_current_configuration_keys(keys: Any, *, durable: bool = True) -> None:
    """Reject removed durable configuration keys without inspecting values."""
    retired_keys = _RETIRED_CONFIGURATION_KEYS
    if durable:
        retired_keys = (
            retired_keys
            | _RETIRED_DURABLE_OTLP_CONFIGURATION_KEYS
            | _RETIRED_DURABLE_OUTBOUND_SECRET_KEYS
        )
    retired = sorted(
        {
            str(key).strip().upper()
            for key in keys
            if str(key).strip().upper() in retired_keys
        }
    )
    if retired:
        raise ValueError(
            "retired durable configuration key(s) are not accepted: "
            f"{', '.join(retired)}"
        )


def retired_configuration_keys(*, durable: bool = True) -> frozenset[str]:
    """The full set of retired (removed) configuration keys, upper-cased.

    Exposed so the deployment doctor can detect + migrate a stale ``config.json``
    that still carries keys this build no longer accepts (which otherwise fail the
    load with a :class:`ValueError`).
    """
    keys = _RETIRED_CONFIGURATION_KEYS
    if durable:
        keys = (
            keys
            | _RETIRED_DURABLE_OTLP_CONFIGURATION_KEYS
            | _RETIRED_DURABLE_OUTBOUND_SECRET_KEYS
        )
    return frozenset(str(key).strip().upper() for key in keys)


def strip_retired_configuration_keys(
    mapping: Any, *, durable: bool = True
) -> tuple[dict[str, Any], list[str]]:
    """Return a copy of ``mapping`` with every retired key removed, plus the
    sorted list of original-cased keys that were dropped.

    Operates on a raw mapping so it can run BEFORE ``AgentConfig`` validation
    (which rejects retired keys outright). Retired keys are top-level env-style
    keys, so this does not recurse.
    """
    retired = retired_configuration_keys(durable=durable)
    cleaned: dict[str, Any] = {}
    removed: list[str] = []
    for key, value in dict(mapping).items():
        if str(key).strip().upper() in retired:
            removed.append(str(key))
        else:
            cleaned[key] = value
    return cleaned, sorted(removed)


class ConfigurationSourceError(RuntimeError):
    """A configuration source failed validation without exposing its location."""

    def __init__(self, source_type: str, error_class: str) -> None:
        self.source_type = source_type
        self.error_class = error_class
        super().__init__(f"{source_type} configuration source rejected ({error_class})")


def runtime_secret_source_status() -> dict[str, Any]:
    """Return aggregate runtime-secret readiness without source-specific data."""
    with _xdg_projection_lock:
        return dict(_runtime_secret_source_state)


def _set_runtime_secret_source_status(
    *,
    state: str,
    present: bool,
    valid: bool,
    referenced_count: int = 0,
    matched_count: int = 0,
    projected_count: int = 0,
    overridden_count: int = 0,
) -> None:
    with _xdg_projection_lock:
        _runtime_secret_source_state.clear()
        _runtime_secret_source_state.update(
            {
                "state": state,
                "present": present,
                "valid": valid,
                "referenced_count": referenced_count,
                "matched_count": matched_count,
                "projected_count": projected_count,
                "overridden_count": overridden_count,
            }
        )


def _runtime_secret_fingerprint(value: str) -> bytes:
    """Return a process-local fingerprint used only for reload ownership."""
    import hmac

    return hmac.digest(
        _runtime_secret_fingerprint_key,
        value.encode("utf-8"),
        "sha256",
    )


def _collect_env_reference_targets(value: Any) -> frozenset[str]:
    """Collect exact ``env://`` targets from a bounded nested JSON value."""
    targets: set[str] = set()
    pending: list[Any] = [value]
    visited = 0
    while pending:
        current = pending.pop()
        visited += 1
        if visited > _MAX_RUNTIME_REFERENCE_SCAN_NODES:
            raise ConfigurationSourceError("xdg", "ReferenceLimitError")
        if isinstance(current, Mapping):
            pending.extend(current.values())
        elif isinstance(current, list | tuple):
            pending.extend(current)
        elif isinstance(current, str):
            match = _RUNTIME_SECRET_REFERENCE_RE.fullmatch(current.strip())
            if match is not None:
                targets.add(match.group(1))
    return frozenset(targets)


def _validate_runtime_secret_metadata(metadata: Any) -> None:
    """Enforce the file-type, ownership, mode, and size contract."""
    import stat

    if not stat.S_ISREG(metadata.st_mode):
        raise PermissionError("runtime secret source is not a regular file")
    if not 0 <= metadata.st_size <= _MAX_RUNTIME_SECRET_SOURCE_BYTES:
        raise ValueError("runtime secret source exceeds the size limit")
    if os.name == "posix":
        if metadata.st_uid not in {0, os.geteuid()}:
            raise PermissionError("runtime secret source owner is not trusted")
        if stat.S_IMODE(metadata.st_mode) not in {0o400, 0o600}:
            raise PermissionError("runtime secret source mode is not accepted")
    else:
        # A native Windows file source needs descriptor-level ACL validation to
        # provide the same guarantee. Use explicit process injection meanwhile.
        raise PermissionError("runtime secret source posture is unsupported")


def _read_runtime_secret_source(
    path: "os.PathLike[str] | str",
    *,
    targets: frozenset[str],
    update_status: bool = True,
) -> tuple[bool, dict[str, str]]:
    """Read and filter the implicit XDG runtime-secret document.

    The path and all document data remain inside this boundary.  A missing file
    is an optional, valid state; any present file that fails validation is
    rejected with category-only diagnostics.
    """
    import json
    import stat

    descriptor = -1
    observed = False
    try:
        before_open = os.lstat(path)
        observed = True
        if stat.S_ISLNK(before_open.st_mode):
            raise PermissionError("runtime secret source link is not accepted")
        flags = os.O_RDONLY
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_BINARY", 0)
        descriptor = os.open(path, flags)
        before_read = os.fstat(descriptor)
        _validate_runtime_secret_metadata(before_read)
        if (
            before_open.st_dev,
            before_open.st_ino,
        ) != (
            before_read.st_dev,
            before_read.st_ino,
        ):
            raise PermissionError("runtime secret source changed during open")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read(_MAX_RUNTIME_SECRET_SOURCE_BYTES + 1)
            after_read = os.fstat(handle.fileno())
        _validate_runtime_secret_metadata(after_read)
        if len(payload) > _MAX_RUNTIME_SECRET_SOURCE_BYTES:
            raise ValueError("runtime secret source exceeds the size limit")
        if len(payload) != before_read.st_size or (
            before_read.st_dev,
            before_read.st_ino,
            before_read.st_size,
            before_read.st_mtime_ns,
        ) != (
            after_read.st_dev,
            after_read.st_ino,
            after_read.st_size,
            after_read.st_mtime_ns,
        ):
            raise PermissionError("runtime secret source changed during read")

        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError("runtime secret source has duplicate keys")
                result[key] = item
            return result

        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
        )
        if not isinstance(document, dict):
            raise TypeError("runtime secret source must contain a JSON object")
        if len(document) > _MAX_RUNTIME_SECRET_ENTRIES:
            raise ValueError("runtime secret source has too many entries")

        selected: dict[str, str] = {}
        casefolded_names: set[str] = set()
        for key, value in document.items():
            if _RUNTIME_SECRET_ENV_NAME_RE.fullmatch(key) is None:
                raise ValueError("runtime secret source contains an invalid key")
            folded = key.casefold()
            if folded in casefolded_names:
                raise ValueError("runtime secret source has ambiguous keys")
            casefolded_names.add(folded)
            if not isinstance(value, str):
                raise TypeError("runtime secret source values must be strings")
            encoded = value.encode("utf-8")
            if (
                not encoded
                or len(encoded) > _MAX_RUNTIME_SECRET_VALUE_BYTES
                or "\x00" in value
            ):
                raise ValueError("runtime secret source contains an invalid value")
            if key in targets:
                selected[key] = value
        return True, selected
    except FileNotFoundError as exc:
        if not observed:
            return False, {}
        if update_status:
            _set_runtime_secret_source_status(
                state="invalid", present=True, valid=False
            )
        raise ConfigurationSourceError("runtime-secrets", type(exc).__name__) from None
    except Exception as exc:
        if update_status:
            _set_runtime_secret_source_status(
                state="invalid", present=observed, valid=False
            )
        raise ConfigurationSourceError("runtime-secrets", type(exc).__name__) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _production_configuration_is_strict() -> bool:
    """Return whether explicitly configured files must fail closed."""
    return (
        str(os.environ.get("APP_PROFILE") or "").strip().casefold()
        in _PRODUCTION_PROFILE_VALUES
    )


def _validate_configuration_metadata(metadata: Any, *, strict: bool) -> None:
    """Validate a config descriptor; strict non-POSIX posture fails closed."""
    import stat

    if not stat.S_ISREG(metadata.st_mode):
        raise PermissionError("configuration source is not a regular file")
    if not 0 <= metadata.st_size <= _MAX_CONFIGURATION_SOURCE_BYTES:
        raise ValueError("configuration source exceeds the size limit")
    if not strict:
        return
    if os.name != "posix":
        raise PermissionError("private configuration posture is unsupported")
    if metadata.st_uid not in {0, os.geteuid()}:
        raise PermissionError("configuration source owner is not trusted")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise PermissionError("configuration source permissions are not private")


def _read_configuration_mapping(
    path: "os.PathLike[str] | str",
    *,
    source_type: str,
    strict: bool,
) -> dict[str, Any]:
    """Read one bounded JSON object, enforcing private-file posture in production.

    Diagnostics intentionally expose only the source category and exception class.
    Paths, file content, owners, and permission bits are never logged or returned.
    """
    import json
    import stat
    from pathlib import Path

    candidate = Path(path)
    descriptor = -1
    try:
        before_open = os.lstat(candidate)
        if stat.S_ISLNK(before_open.st_mode) or (
            getattr(before_open, "st_file_attributes", 0) & 0x400
        ):
            raise PermissionError("configuration source link is not accepted")
        flags = os.O_RDONLY
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_BINARY", 0)
        descriptor = os.open(candidate, flags)
        before_read = os.fstat(descriptor)
        _validate_configuration_metadata(before_read, strict=strict)
        if (before_open.st_dev, before_open.st_ino) != (
            before_read.st_dev,
            before_read.st_ino,
        ):
            raise PermissionError("configuration source changed during open")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read(_MAX_CONFIGURATION_SOURCE_BYTES + 1)
            after_read = os.fstat(handle.fileno())
        if len(payload) > _MAX_CONFIGURATION_SOURCE_BYTES:
            raise ValueError("configuration source exceeds the size limit")
        _validate_configuration_metadata(after_read, strict=strict)
        if (
            before_read.st_dev,
            before_read.st_ino,
            before_read.st_size,
            before_read.st_mtime_ns,
        ) != (
            after_read.st_dev,
            after_read.st_ino,
            after_read.st_size,
            after_read.st_mtime_ns,
        ) or len(payload) != before_read.st_size:
            raise PermissionError("configuration source changed during read")

        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError("configuration source has duplicate keys")
                result[key] = item
            return result

        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
        )
        if not isinstance(value, dict):
            raise TypeError("configuration source must contain a JSON object")
        return value
    except Exception as exc:
        if descriptor >= 0:
            os.close(descriptor)
        error_class = type(exc).__name__
        raise ConfigurationSourceError(source_type, error_class) from None


def _write_private_configuration_mapping(
    path: "os.PathLike[str] | str", value: Mapping[str, Any]
) -> None:
    """Atomically write a private JSON configuration file."""
    import json
    import tempfile
    from pathlib import Path

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=".config-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            os.chmod(temporary_name, 0o600)
            json.dump(value, handle, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = ""
    finally:
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass


def _ensure_env_loaded():
    global _env_loaded
    with _xdg_projection_lock:
        if _env_loaded:
            return

        # Hermetic tests never inherit a host's XDG deployment. Tests set explicit
        # process values or opt into a temporary XDG root. Repository dotenv files
        # are not an AgentConfig source in any profile.
        if to_boolean(os.environ.get("AGENT_UTILITIES_TESTING", "false")):
            _env_loaded = True
            return

        try:
            _load_xdg_json_config_locked()
        except Exception:
            _env_loaded = False
            raise
        _env_loaded = True


def load_config(*, reload: bool = False) -> None:
    """Load the agent-utilities configuration into the process environment.

    Public, idempotent entry point for XDG ``config.json`` and its optional
    implicit ``runtime-secrets.json``. Config keys are projected into the
    process; only secret keys targeted by exact ``env://`` references in the
    configuration are projected, and an explicit process value always wins.

    Agent MCP entry points call this as the sole deployment loader, so the whole
    fleet resolves settings through one shared configuration boundary. It is
    idempotent and safe to call repeatedly;
    pass ``reload=True`` to stage and re-read after either XDG file changes.
    Under the test harness it is a deliberate no-op unless a test calls the
    private loader against an explicitly isolated root.

    CONCEPT:AU-OS.config.fleet-xdg-standardization — fleet XDG config standardization
    """
    global _env_loaded
    with _xdg_projection_lock:
        snapshot = _capture_projection_state_locked() if reload else None
        try:
            if reload:
                _env_loaded = False
            _ensure_env_loaded()
            if reload:
                _reload_typed_singleton_locked()
        except Exception:
            if snapshot is not None:
                _restore_projection_state_locked(snapshot)
            raise


def _capture_projection_state_locked() -> tuple[
    bool,
    dict[str, str],
    dict[str, bytes],
    dict[str, Any],
    dict[str, str | None],
]:
    """Capture only loader-owned state for scoped reload rollback."""
    owned_keys = set(_xdg_injected_environment).union(_xdg_injected_runtime_secrets)
    return (
        _env_loaded,
        dict(_xdg_injected_environment),
        dict(_xdg_injected_runtime_secrets),
        dict(_runtime_secret_source_state),
        {key: os.environ.get(key) for key in owned_keys},
    )


def _restore_projection_state_locked(
    snapshot: tuple[
        bool,
        dict[str, str],
        dict[str, bytes],
        dict[str, Any],
        dict[str, str | None],
    ],
) -> None:
    """Restore the previous loader-owned generation after a failed reload."""
    global _env_loaded
    loaded, config_ledger, secret_ledger, status, values = snapshot
    _commit_xdg_environment_projection({}, {})
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    _xdg_injected_environment.update(config_ledger)
    _xdg_injected_runtime_secrets.update(secret_ledger)
    _runtime_secret_source_state.clear()
    _runtime_secret_source_state.update(status)
    _env_loaded = loaded


def _reload_typed_singleton_locked() -> None:
    """Validate a generation, then atomically replace any materialized snapshot."""
    singleton = globals().get("_LAZY_CACHE", {}).get("_config")
    config_type = singleton.__class__ if singleton is not None else AgentConfig
    candidate = config_type()
    candidate.assert_production_safe(profile=candidate.app_profile)
    if singleton is not None:
        _init_lazy_config(existing=candidate, force=True)


def _under_pytest() -> bool:
    """True during a pytest session.

    Used to keep the developer's XDG-default deployment ``config.json`` out of
    the test environment even if a test flips ``AGENT_UTILITIES_TESTING`` off (a
    few tests do, to exercise real-validation branches). ``PYTEST_CURRENT_TEST``
    is set by pytest while a test runs — exactly when such a config reload would
    leak host-specific values into ``os.environ`` and pollute later tests.
    """
    import sys

    return "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules


def _load_xdg_json_config() -> None:
    """Serialize one staged XDG projection refresh within this process."""
    with _xdg_projection_lock:
        _load_xdg_json_config_locked()


def _canonicalize_xdg_configuration(
    data: Mapping[str, Any],
) -> dict[str, Any]:
    """Return canonical aliases after rejecting unknown or ambiguous keys."""
    aliases = {
        str(field.alias or name.upper()).upper(): str(field.alias or name.upper())
        for name, field in AgentConfig.model_fields.items()
    }
    observed: set[str] = set()
    canonical: dict[str, Any] = {}
    for key in data:
        normalized = str(key).strip().upper()
        if normalized in observed:
            raise ConfigurationSourceError("xdg", "AmbiguousKeyError")
        observed.add(normalized)
        if normalized in aliases:
            canonical[aliases[normalized]] = data[key]
        else:
            # A dynamic ``config.setting()`` key (connector/service config), not a
            # modelled ``AgentConfig`` field. ``config.json`` is the documented
            # source for these, so preserve it verbatim rather than rejecting it.
            # Retired keys are still rejected by
            # ``_require_current_configuration_keys``; genuine typos surface
            # (non-blocking) via the doctor's ``unknown_configuration_keys`` report.
            canonical[str(key)] = data[key]
    return canonical


_DURABLE_CREDENTIAL_SUFFIXES = (
    "_API_KEY",
    "_ENCRYPTION_KEY",
    "_HMAC_KEY",
    "_PASSWORD",
    "_PRIVATE_KEY",
    "_SECRET",
    "_SECRET_KEY",
    "_SIGNING_KEY",
    "_TOKEN",
)
_DURABLE_HEADER_CONTAINER_KEYS = frozenset(
    {"HEADERS", "EXTRA_HEADERS", "CUSTOM_HEADERS"}
)
_DURABLE_SENSITIVE_MAPPING_KEYS = frozenset(
    {
        "API_KEY",
        "AUTHORIZATION",
        "COOKIE",
        "PASSWORD",
        "PROXY_AUTHORIZATION",
        "SECRET",
        "SET_COOKIE",
        "TOKEN",
        "X_API_KEY",
    }
)


def _durable_value_is_empty(value: Any) -> bool:
    """Return whether a generated durable placeholder carries no material."""
    return value is None or value == "" or value == [] or value == {}


def _validate_durable_xdg_secret_policy(data: Mapping[str, Any]) -> None:
    """Reject credential and header material from durable XDG configuration.

    AgentConfig still accepts runtime credentials from the explicit process
    environment. Durable ``config.json`` may declare only dedicated ``*_REF``
    fields (and the validated OAuth2 ``client_secret`` reference). Keeping this
    policy at the XDG boundary prevents a permissive nested model or free-form
    mapping from becoming an accidental secret store.

    Errors deliberately identify only the policy class: key names, nesting,
    values, and the configuration path can themselves disclose deployment
    details and must not cross the doctor/MCP boundary.
    """

    def visit(value: Any, *, parent: str = "") -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key).strip().upper().replace("-", "_")
                if _durable_value_is_empty(child):
                    continue
                if parent in {
                    "MCP_FLEET_SECRET_REFS",
                    "CREDENTIAL_REFS",
                    "SELECTOR_REFS",
                }:
                    if not (
                        isinstance(child, str)
                        and _RUNTIME_SECRET_REF_RE.fullmatch(child.strip())
                    ):
                        raise ConfigurationSourceError("xdg", "DurableSecretError")
                    continue
                if key in _DURABLE_HEADER_CONTAINER_KEYS:
                    raise ConfigurationSourceError("xdg", "DurableSecretError")
                if key == "CLIENT_SECRET" and parent == "OAUTH2":
                    # The strict OAuth2 submodel validates the URI. This is the
                    # one intentionally nested reference form in model config.
                    if not (
                        isinstance(child, str)
                        and _RUNTIME_SECRET_REF_RE.fullmatch(child.strip())
                    ):
                        raise ConfigurationSourceError("xdg", "DurableSecretError")
                elif key in _DURABLE_SENSITIVE_MAPPING_KEYS or (
                    key.endswith(_DURABLE_CREDENTIAL_SUFFIXES)
                    and not key.endswith("_REF")
                ):
                    raise ConfigurationSourceError("xdg", "DurableSecretError")
                visit(child, parent=key)
        elif isinstance(value, list):
            for child in value:
                visit(child, parent=parent)

    visit(data)


def plaintext_secret_keys(data: Mapping[str, Any]) -> list[str]:
    """Return the config key *names* that hold an inline plaintext secret.

    Mirrors :func:`_validate_durable_xdg_secret_policy` exactly, but collects the
    offending key names instead of raising a value-free ``DurableSecretError``, so
    the doctor and migration reporter can tell an operator *which* keys to relocate
    to a durable reference (``<KEY>_REF`` → OpenBao/Vault). Only key names cross
    this boundary — never a value, nesting path, or the configuration path.

    A key is flagged when it is a sensitive mapping key or ends with a credential
    suffix (``_TOKEN``/``_SECRET``/``_PASSWORD``/``_API_KEY``/…) and is not already
    a ``_REF``; when a ``*_REFS`` container or ``OAUTH2.CLIENT_SECRET`` holds a
    non-reference value; or when it is an inline header container — the exact
    conditions the durable-secret policy rejects at load.
    """

    offenders: list[str] = []

    def visit(value: Any, *, parent: str = "") -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key).strip().upper().replace("-", "_")
                if _durable_value_is_empty(child):
                    continue
                if parent in {
                    "MCP_FLEET_SECRET_REFS",
                    "CREDENTIAL_REFS",
                    "SELECTOR_REFS",
                }:
                    if not (
                        isinstance(child, str)
                        and _RUNTIME_SECRET_REF_RE.fullmatch(child.strip())
                    ):
                        offenders.append(key)
                    continue
                if key in _DURABLE_HEADER_CONTAINER_KEYS:
                    offenders.append(key)
                elif key == "CLIENT_SECRET" and parent == "OAUTH2":
                    if not (
                        isinstance(child, str)
                        and _RUNTIME_SECRET_REF_RE.fullmatch(child.strip())
                    ):
                        offenders.append(key)
                elif key in _DURABLE_SENSITIVE_MAPPING_KEYS or (
                    key.endswith(_DURABLE_CREDENTIAL_SUFFIXES)
                    and not key.endswith("_REF")
                ):
                    offenders.append(key)
                visit(child, parent=key)
        elif isinstance(value, list):
            for child in value:
                visit(child, parent=parent)

    visit(data)
    return sorted(set(offenders))


def _validate_xdg_configuration_schema(data: Mapping[str, Any]) -> None:
    """Validate a staged durable document without invoking settings sources."""
    _validate_durable_xdg_secret_policy(data)
    try:
        candidate = _validate_agent_config_without_settings(data)
        candidate.assert_production_safe(profile=candidate.app_profile)
    except Exception as exc:
        raise ConfigurationSourceError("xdg", type(exc).__name__) from None


def _validate_agent_config_without_settings(
    data: Mapping[str, Any],
) -> "AgentConfig":
    """Validate explicit values without consulting environment/file sources."""
    candidate = AgentConfig.__new__(AgentConfig)
    return AgentConfig.__pydantic_validator__.validate_python(
        dict(data),
        self_instance=candidate,
    )


def _mapping_selects_production(data: Mapping[str, Any]) -> bool:
    """Return whether a validated staged document selects production."""
    for key, value in data.items():
        if str(key).strip().upper() == "APP_PROFILE":
            return str(value).strip().casefold() in _PRODUCTION_PROFILE_VALUES
    return False


def _load_xdg_json_config_locked() -> None:
    import json
    from pathlib import Path

    import platformdirs

    from agent_utilities.core.paths import runtime_secrets_path

    APP_NAME = "agent-utilities"
    APP_AUTHOR = "knuckles-team"

    override = os.environ.get("AGENT_UTILITIES_CONFIG_DIR")
    # Hermetic tests never read the developer's XDG-default deployment
    # ``config.json``. Config loading injects those into ``os.environ`` and would
    # override the unit suite's defaults — making tests fail on a dev box while
    # staying green in CI (which has no such file). An explicit config root used
    # by integration fixtures is still honored.
    if not override and (
        _under_pytest()
        or to_boolean(os.environ.get("AGENT_UTILITIES_TESTING", "false"))
    ):
        _commit_xdg_environment_projection({}, {})
        _set_runtime_secret_source_status(state="hermetic", present=False, valid=True)
        return
    if override:
        cfg_dir = Path(override).expanduser()
    else:
        cfg_dir = Path(platformdirs.user_config_path(APP_NAME, APP_AUTHOR))

    cfg_file = cfg_dir / "config.json"
    strict = _production_configuration_is_strict()
    data: dict[str, Any] = {}
    if not cfg_file.exists():
        if strict and override:
            raise ConfigurationSourceError("xdg", "FileNotFoundError")
    else:
        data = _read_configuration_mapping(
            cfg_file,
            source_type="xdg",
            strict=strict,
        )
    _require_current_configuration_keys(data)
    data = _canonicalize_xdg_configuration(data)
    if not strict and _mapping_selects_production(data):
        # Re-open through the production posture after the staged document has
        # selected it. The second bounded, stable read is the one projected.
        data = _read_configuration_mapping(
            cfg_file,
            source_type="xdg",
            strict=True,
        )
        _require_current_configuration_keys(data)
        data = _canonicalize_xdg_configuration(data)
    _validate_xdg_configuration_schema(data)
    targets = _collect_env_reference_targets(data)
    durable_env_keys = {str(key).upper() for key in data}
    if any(target.upper() in durable_env_keys for target in targets):
        raise ConfigurationSourceError("xdg", "SecretTargetCollisionError")

    present, available = _read_runtime_secret_source(
        runtime_secrets_path(),
        targets=targets,
    )
    explicit_environment = _environment_without_xdg_projections()
    config_projection: dict[str, str] = {}
    for k, v in data.items():
        env_key = k.upper()
        if env_key not in explicit_environment:
            if isinstance(v, list | dict):
                rendered = json.dumps(v)
            elif v is None:
                rendered = ""
            else:
                rendered = str(v)
            try:
                rendered.encode("utf-8")
            except UnicodeError:
                raise ConfigurationSourceError(
                    "xdg", "EnvironmentProjectionError"
                ) from None
            if (
                _RUNTIME_SECRET_ENV_NAME_RE.fullmatch(env_key) is None
                or "\x00" in rendered
            ):
                raise ConfigurationSourceError("xdg", "EnvironmentProjectionError")
            config_projection[env_key] = rendered

    runtime_projection = {
        key: value
        for key, value in available.items()
        if key not in explicit_environment
    }
    overridden = len(set(available).intersection(explicit_environment))
    _commit_xdg_environment_projection(config_projection, runtime_projection)
    _set_runtime_secret_source_status(
        state="ready" if present else "absent",
        present=present,
        valid=True,
        referenced_count=len(targets),
        matched_count=len(available),
        projected_count=len(runtime_projection),
        overridden_count=overridden,
    )


def _environment_without_xdg_projections() -> dict[str, str]:
    """Return process environment with still-owned projections omitted."""
    explicit = dict(os.environ)
    for env_key, injected_value in _xdg_injected_environment.items():
        if explicit.get(env_key) == injected_value:
            explicit.pop(env_key, None)
    for env_key, fingerprint in _xdg_injected_runtime_secrets.items():
        current = explicit.get(env_key)
        if current is not None and _runtime_secret_fingerprint(current) == fingerprint:
            explicit.pop(env_key, None)
    return explicit


def _commit_xdg_environment_projection(
    config_projection: Mapping[str, str],
    runtime_projection: Mapping[str, str],
) -> None:
    """Activate fully validated projections while preserving later overrides."""
    owned_config = dict(_xdg_injected_environment)
    owned_secrets = dict(_xdg_injected_runtime_secrets)
    for env_key in set(owned_config).union(owned_secrets):
        current = os.environ.get(env_key)
        config_owned = env_key in owned_config and current == owned_config[env_key]
        secret_owned = (
            env_key in owned_secrets
            and current is not None
            and _runtime_secret_fingerprint(current) == owned_secrets[env_key]
        )
        if config_owned or secret_owned:
            os.environ.pop(env_key, None)
    _xdg_injected_environment.clear()
    _xdg_injected_runtime_secrets.clear()

    for env_key, value in config_projection.items():
        os.environ[env_key] = value
        _xdg_injected_environment[env_key] = value
    for env_key, value in runtime_projection.items():
        os.environ[env_key] = value
        _xdg_injected_runtime_secrets[env_key] = _runtime_secret_fingerprint(value)


def _xdg_config_file():
    """Path to the XDG ``config.json`` (honors ``AGENT_UTILITIES_CONFIG_DIR``)."""
    from pathlib import Path

    import platformdirs

    override = os.environ.get("AGENT_UTILITIES_CONFIG_DIR")
    cfg_dir = (
        Path(override).expanduser()
        if override
        else Path(platformdirs.user_config_path("agent-utilities", "knuckles-team"))
    )
    return cfg_dir / "config.json"


def save_config_item(key: str, value) -> str:
    """Persist one config item to ``config.json`` AND live ``os.environ``, then reload.

    CONCEPT:AU-KG.storage.config-writeback — the write-back companion to the read-only XDG loader, so a
    config change made via the MCP/REST surfaces survives restart and applies live
    for settings read at call time (``config.setting`` / re-parsed fields). Returns
    the resolved env key. Engine-rebuild settings update the value but need a
    restart to take effect — see the restart classifier.
    """
    from pathlib import Path

    _require_current_configuration_keys((key,))

    if key.lower() == "kg_connections":
        if not isinstance(value, list):
            raise ValueError("kg_connections must be a list")
        # Durable connection declarations are reportable configuration, not a
        # secret store. Fail before touching disk or the process environment if
        # endpoint, identity, credential, database, or local-path material is
        # present as a literal.
        from agent_utilities.knowledge_graph.core.connection_registry import (
            validate_persistable_connection_spec,
        )

        for entry in value:
            if not isinstance(entry, dict):
                raise ValueError("kg_connections entries must be objects")
            validate_persistable_connection_spec(entry)

    env_key = key.upper()
    with _xdg_projection_lock:
        from agent_utilities.core.paths import runtime_secrets_path

        cfg_file = _xdg_config_file()
        Path(cfg_file).parent.mkdir(parents=True, exist_ok=True)
        existed = cfg_file.exists()
        prior_data: dict[str, Any] = {}
        if existed:
            prior_data = _read_configuration_mapping(
                cfg_file,
                source_type="xdg",
                strict=_production_configuration_is_strict(),
            )
        staged = _canonicalize_xdg_configuration(prior_data)
        # A dynamic ``config.setting()`` key (connector/service config) is a valid
        # thing to persist into config.json — only retired keys are rejected, by
        # ``_require_current_configuration_keys`` below.
        staged[env_key] = value
        _require_current_configuration_keys(staged)
        staged = _canonicalize_xdg_configuration(staged)
        _validate_xdg_configuration_schema(staged)
        targets = _collect_env_reference_targets(staged)
        if any(target.upper() in staged for target in targets):
            raise ConfigurationSourceError("xdg", "SecretTargetCollisionError")
        _read_runtime_secret_source(
            runtime_secrets_path(), targets=targets, update_status=False
        )

        _write_private_configuration_mapping(cfg_file, staged)

        # The file write, environment projection, typed singleton, and derived
        # cache transition share one lock, so a parallel loader cannot observe
        # an in-progress save.
        try:
            load_config(reload=True)
        except Exception:
            if existed:
                _write_private_configuration_mapping(cfg_file, prior_data)
            else:
                cfg_file.unlink(missing_ok=True)
            raise

    if env_key.startswith(("LANGFUSE_", "TRACE_EXPORT_", "TLS_")):
        from agent_utilities.observability.langfuse_exporter import (
            reset_langfuse_exporter,
        )

        reset_langfuse_exporter()

    # Native/declared child availability depends on several live settings (most
    # notably Langfuse and TLS references).  Invalidate every in-process fleet
    # catalog; the next discovery operation reparses it without spawning a child.
    import sys

    multiplexer_module = sys.modules.get("agent_utilities.mcp.multiplexer")
    if multiplexer_module is not None:
        multiplexer_module.invalidate_live_catalogs()
    return env_key


from pydantic import BaseModel, ConfigDict


def _total_model_capacity(parallel_instances: int, max_parallel_calls: int) -> int:
    """Resolve a model's total parallel-call capacity (CONCEPT:AU-KG.compute.concurrency-controller-sizing).

    ``total_capacity = parallel_instances * max_parallel_calls`` — the number of
    in-flight LLM/embedding calls the model can serve at once: ``N`` vLLM
    instances behind one endpoint, each serving ``max_parallel_calls`` concurrent
    requests. Always at least ``1`` (unknown/misconfigured collapses to safe
    sequential behaviour, never zero-capacity).
    """
    return max(1, int(parallel_instances or 1) * int(max_parallel_calls or 1))


def _validate_oauth2_block(oauth2: dict[str, Any], owner_label: str) -> dict[str, Any]:
    """Validate + normalize a raw ``oauth2`` dict via the strict submodel (CONCEPT:AU-OS.identity.oauth2-client-credentials-lifecycle).

    Imported lazily (function-body, not module-top) to avoid a core↔security import cycle:
    ``agent_utilities.security`` (via its package ``__init__``) imports several submodules that
    themselves import ``agent_utilities.core.config`` — safe once this module has finished
    defining ``setting`` (near the top), but not safe to import eagerly at THIS module's own
    top-level. This import only actually fires when a config carries a non-empty ``oauth2``
    block, which is never true during this module's own class-definition phase.

    Raises ``ValueError`` (via pydantic) if the shape is wrong or ``client_secret`` is a
    plaintext value rather than a secret-reference URI.
    """
    from agent_utilities.security.oauth_client_credentials import (
        OAuth2ClientCredentialsConfig,
    )

    try:
        return OAuth2ClientCredentialsConfig.model_validate(oauth2).model_dump()
    except (
        Exception
    ) as exc:  # re-raise with the owning model/id for a diagnosable error
        raise ValueError(f"{owner_label}: invalid oauth2 block: {exc}") from exc


class ChatModelConfig(BaseModel):
    id: str
    provider: str
    intelligence_level: str = "normal"
    base_url: str | None = None
    api_key_ref: str | None = None
    """Runtime reference for this model's API key. Resolved only while the
    provider client is built; literal API keys are not model configuration."""
    oauth2: dict[str, Any] | None = None
    """OAuth2 ``client_credentials`` block (CONCEPT:AU-OS.identity.oauth2-client-credentials-lifecycle) — machine-to-
    machine auth for enterprise OpenAI-compatible/Azure endpoints that require a short-lived
    minted bearer instead of a static key. Mutually exclusive with
    :pyattr:`api_key_ref` (validated below). Shape:
    ``agent_utilities.security.oauth_client_credentials.OAuth2ClientCredentialsConfig``
    (``token_url``/``client_id``/``client_secret``[secret-ref]/``scope``/``audience``/``extra_params``).
    Stored as a plain dict here (not the strict submodel) to avoid a core↔security import cycle;
    validated lazily below and by every consumer via ``httpx_auth_from_config``."""
    headers_ref: str | None = None
    """Runtime reference resolving to a bounded JSON header object. Resolved
    headers sit under per-call ``custom_headers`` and are never serialized back
    into AgentConfig."""
    reasoning_effort: str | None = "inherit"
    """Per-model reasoning/thinking-effort override for the OpenAI-compatible request.
    The default sentinel ``"inherit"`` means *don't override* — use the caller's
    ``reasoning_effort`` (``create_model`` defaults to ``"none"`` = thinking OFF, so utility
    calls return content directly). Pin an explicit level (``"none"``/``"low"``/``"medium"``/
    ``"high"``/``"xhigh"``) to fix it for THIS model, or set it to ``null`` (None) to opt this
    model back into its NATIVE reasoning — no ``reasoning_effort`` is sent, so the model uses
    its own default. Honored by :func:`agent_utilities.core.model_factory.create_model`."""
    supports_json: bool = False
    vision: bool = False
    reasoning: bool = False
    tools_enabled: bool = False
    parallel_instances: int = 1
    """Number of parallel vLLM instances behind this model's ``base_url``. The
    per-instance concurrency is ``max_parallel_calls``; total parallel-call
    capacity is the product (see :pyattr:`total_capacity`)."""
    max_parallel_calls: int = 1
    """How many concurrent requests ONE vLLM instance of this model can serve
    (its per-instance concurrency, e.g. vLLM ``--max-num-seqs``). Default ``1``
    keeps callers sequential and is always safe. CONCEPT:AU-KG.compute.concurrency-controller-sizing."""
    max_concurrent_requests: int | None = None
    """Hard ceiling on the **aggregate** in-flight requests this model's *server*
    can serve safely — its real vLLM ``--max-num-seqs`` / KV-cache + (for a
    unified-memory accelerator host) the shared-memory headroom (CONCEPT:AU-KG.compute.same-semantics-as).

    This is the ONE number the model SERVER's capacity dictates, NOT the local
    host's CPU count. It varies across edge hosts, accelerator workstations, and
    clusters, and
    cannot be auto-derived from the local box, so it is a legitimate explicit
    config (Configuration-discipline). When set it caps the SUM of every demand
    source hitting this endpoint (embeds + enrichment + orchestration) via the
    shared priority gate, so the client can never pile hundreds of concurrent
    requests onto the server and OOM it. When unset it falls back to
    ``max(total_capacity, MODEL_MAX_CONCURRENT_REQUESTS)`` (a conservative
    default). Set it to your server's ``--max-num-seqs`` (or just below)."""
    gpu_group: str | None = None
    """Optional shared-GPU group tag (CONCEPT:AU-KG.compute.pure-config-enumeration-fail). Models that physically
    share one GPU are grouped under one concurrency budget so their fan-out cannot
    jointly oversubscribe the device. Explicit tag wins; when unset the group
    defaults to the ``base_url`` host (see :meth:`Config.gpu_group`)."""
    can_route: bool = False
    can_kg: bool = False
    context_window: int | None = Field(default=None, ge=1)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_auth_mode(self) -> "ChatModelConfig":
        """Reference-backed API keys and OAuth2 are mutually exclusive."""
        if self.api_key_ref and self.oauth2:
            raise ValueError(
                f"ChatModelConfig {self.id!r}: 'api_key_ref' and 'oauth2' are mutually exclusive — "
                "configure exactly one authentication mode."
            )
        for attribute in ("api_key_ref", "headers_ref"):
            reference = getattr(self, attribute)
            if reference is not None and not _RUNTIME_SECRET_REF_RE.fullmatch(
                reference.strip()
            ):
                raise ValueError("model runtime material must use a secret reference")
            if reference is not None:
                setattr(self, attribute, reference.strip())
        if self.oauth2:
            self.oauth2 = _validate_oauth2_block(
                self.oauth2, f"ChatModelConfig {self.id!r}"
            )
        return self

    @property
    def total_capacity(self) -> int:
        """Total in-flight calls this model can serve = instances × per-instance.

        CONCEPT:AU-KG.compute.concurrency-controller-sizing — used by the shared concurrency controller to size the
        fan-out gate for this model.
        """
        return _total_model_capacity(self.parallel_instances, self.max_parallel_calls)


class EmbeddingModelConfig(BaseModel):
    id: str
    provider: str
    base_url: str | None = None
    api_key_ref: str | None = None
    """Runtime reference for this embedder's API key."""
    oauth2: dict[str, Any] | None = None
    """OAuth2 ``client_credentials`` block — same shape/semantics as :pyattr:`ChatModelConfig.oauth2`
    (CONCEPT:AU-OS.identity.oauth2-client-credentials-lifecycle); mutually exclusive with
    :pyattr:`api_key_ref`."""
    headers_ref: str | None = None
    """Runtime reference resolving to this embedder's bounded JSON header object."""
    parallel_instances: int = 1
    """Number of parallel vLLM instances behind this embedding model's
    ``base_url``. Total parallel-call capacity is ``parallel_instances *
    max_parallel_calls`` (see :pyattr:`total_capacity`)."""
    max_parallel_calls: int = 1
    """How many concurrent embedding requests ONE vLLM instance of this model can
    serve (its per-instance concurrency). Default ``1`` keeps batch embedding
    sequential and is always safe. CONCEPT:AU-KG.compute.concurrency-controller-sizing."""
    max_concurrent_requests: int | None = None
    """Hard ceiling on the aggregate in-flight embedding requests this model's
    *server* can serve safely (CONCEPT:AU-KG.compute.same-semantics-as). Same semantics as
    :pyattr:`ChatModelConfig.max_concurrent_requests`: the SERVER's real capacity,
    capping the embedding fan-out so bulk embedding can never oversubscribe the
    endpoint. On a unified-memory host the embedder shares accelerator memory with
    the generator — keep this conservative. Unset → ``max(total_capacity,
    MODEL_MAX_CONCURRENT_REQUESTS)``."""
    gpu_group: str | None = None
    """Optional shared-GPU group tag (CONCEPT:AU-KG.compute.pure-config-enumeration-fail). Tag this with the same
    value as a chat model that shares the physical GPU (for example, both
    ``"accelerator-shared"``) so
    bulk embedding yields its concurrency to latency-sensitive chat under
    contention. Explicit tag wins; else defaults to the ``base_url`` host."""
    chunk_size: int = 768
    context_window: int | None = Field(default=None, ge=1)
    fallback: "EmbeddingModelConfig | None" = None
    """Optional automatic-failover endpoint (CONCEPT:AU-KG.enrichment.each-call-resolves-active). When the PRIMARY
    embedder (this config) is unreachable — its circuit breaker (CONCEPT:AU-ORCH.routing.load-shedding-backoff)
    is OPEN — embedding traffic is transparently re-routed to this fallback
    endpoint, and routed back automatically once the primary recovers. The fallback
    is a full ``EmbeddingModelConfig`` with its OWN ``base_url``, ``gpu_group``, and
    ``max_concurrent_requests``, so the capacity guard applies the FALLBACK's GPU
    budget while failed-over: point the primary at a dedicated embedder (for example,
    ``gpu_group="accelerator-primary"``) and the fallback at a shared backend (for
    example, ``gpu_group="accelerator-shared"``) so fallback embeds share that
    backend's joint budget with the generator and can never OOM it. A nested
    ``fallback`` here is ignored (single-level failover)."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_auth_mode(self) -> "EmbeddingModelConfig":
        """Reference-backed API keys and OAuth2 are mutually exclusive."""
        if self.api_key_ref and self.oauth2:
            raise ValueError(
                f"EmbeddingModelConfig {self.id!r}: 'api_key_ref' and 'oauth2' are mutually "
                "exclusive — configure exactly one authentication mode."
            )
        for attribute in ("api_key_ref", "headers_ref"):
            reference = getattr(self, attribute)
            if reference is not None and not _RUNTIME_SECRET_REF_RE.fullmatch(
                reference.strip()
            ):
                raise ValueError("model runtime material must use a secret reference")
            if reference is not None:
                setattr(self, attribute, reference.strip())
        if self.oauth2:
            self.oauth2 = _validate_oauth2_block(
                self.oauth2, f"EmbeddingModelConfig {self.id!r}"
            )
        return self

    @property
    def total_capacity(self) -> int:
        """Total in-flight embedding calls this model can serve.

        CONCEPT:AU-KG.compute.concurrency-controller-sizing — ``parallel_instances × max_parallel_calls``; used by
        the concurrency controller to fan out embedding batches.
        """
        return _total_model_capacity(self.parallel_instances, self.max_parallel_calls)


# Self-referential ``fallback`` field (CONCEPT:AU-KG.enrichment.each-call-resolves-active): config.py does not use
# ``from __future__ import annotations``, so rebuild the model to resolve the
# forward reference to ``EmbeddingModelConfig`` after the class is defined.
EmbeddingModelConfig.model_rebuild()


_RUNTIME_SECRET_REF_RE = re.compile(
    r"^(?:"
    r"env://[A-Za-z_][A-Za-z0-9_]{0,127}"
    r"|(?:vault|secret)://[A-Za-z0-9][A-Za-z0-9_./#-]{0,511}"
    r")$"
)
_MCP_FLEET_SECRET_ALIAS_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
_NEUTRAL_ALIAS_RE = re.compile(r"^[a-z][a-z0-9-]{1,62}$")


def _validated_runtime_http_url(
    value: Any,
    *,
    require_server_placeholder: bool = False,
) -> str | None:
    """Normalize one runtime-only HTTP base URL without resolving or fetching it."""
    if value in (None, ""):
        return None
    rendered = str(value).strip()
    if not rendered:
        return None
    if len(rendered) > 2_048 or any(char.isspace() for char in rendered):
        raise ValueError(
            "runtime HTTP endpoints must be bounded URLs without whitespace"
        )

    placeholders = re.findall(r"\{([^{}]+)\}", rendered)
    if require_server_placeholder:
        if not placeholders or any(item != "server" for item in placeholders):
            raise ValueError(
                "FLEET_MCP_URL_TEMPLATE must contain only the '{server}' placeholder"
            )
        if "{" in rendered.replace("{server}", "") or "}" in rendered.replace(
            "{server}", ""
        ):
            raise ValueError("FLEET_MCP_URL_TEMPLATE placeholders are malformed")
    elif "{" in rendered or "}" in rendered:
        raise ValueError("runtime HTTP endpoints cannot contain placeholders")
    if rendered.count("{") != rendered.count("}"):
        raise ValueError("runtime HTTP endpoint placeholders are malformed")

    from urllib.parse import urlsplit

    try:
        parsed = urlsplit(rendered)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError("runtime HTTP endpoint is malformed") from exc
    if scheme not in {"http", "https"} or not parsed.netloc or not hostname:
        raise ValueError("runtime HTTP endpoints must use http:// or https://")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("runtime HTTP endpoints cannot contain inline credentials")
    if parsed.query or parsed.fragment:
        raise ValueError(
            "runtime HTTP base URLs cannot contain query strings or fragments"
        )
    if port is not None and not 1 <= port <= 65_535:
        raise ValueError("runtime HTTP endpoint port is out of range")
    return f"{scheme}{rendered[len(parsed.scheme) :]}".rstrip("/")


def _validated_langfuse_host(value: Any) -> str:
    """Return a canonical secure Langfuse base URL.

    Project transport policy permits cleartext HTTP only for the three exact
    loopback spellings used by local integration fixtures. Private-network and
    arbitrary hostnames still require HTTPS.
    """
    rendered = _validated_runtime_http_url(value)
    if rendered is None:
        raise ValueError("LANGFUSE_HOST must be an absolute HTTP(S) URL")

    from urllib.parse import urlsplit

    parsed = urlsplit(rendered)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme == "http" and hostname not in {
        "127.0.0.1",
        "::1",
        "localhost",
    }:
        raise ValueError("LANGFUSE_HOST requires HTTPS outside loopback")
    return rendered


class ExternalGraphConnectorConfig(BaseModel):
    """Secret-ref-only declaration for an external graph or GraphQL source.

    Runtime endpoint, database, credentials, auth, TLS, queries, mappings, and
    variables live behind refs. Only neutral aliases, bounded discovery/ingest
    policy, and booleans are serializable in AgentConfig. GraphQL sources may
    use a mapping-policy ref or explicitly permit introspection-driven proposal
    generation; they always remain role=read adapters. Resolved GraphQL
    connection, mapping, and auth documents use the exact current versioned
    formats documented by the universal connector contract.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    source_alias: str
    backend: Literal[
        "neo4j",
        "opencypher",
        "age",
        "ladybug",
        "epistemic_graph",
        "graphql",
    ]
    connection_profile_ref: str
    mapping_policy_ref: str | None = None
    tls_profile_ref: str | None = None
    auth_profile_ref: str | None = None
    variables_ref: str | None = None
    discovery_max_types: int = 200
    discovery_max_depth: int = 6
    ingest_operation: str | None = None
    ingest_max_records: int = 1_000
    ingest_page_size: int = 500
    ingest_max_pages: int = 100
    ingest_max_row_bytes: int = 1_048_576
    ingest_max_total_bytes: int = 16_777_216
    ingest_max_nesting_depth: int = 16
    ingest_max_collection_items: int = 10_000
    sync_mode: Literal["auto", "cdc", "snapshot"] = "auto"
    reconcile_deletions: bool = True
    allow_empty_snapshot: bool = False
    contextual: bool = True
    semantic_mapping: bool = False
    allow_introspection: bool = False
    schema_drift_policy: Literal["fail_closed"] = "fail_closed"
    require_approval: Literal[True] = True

    @field_validator("name", "source_alias")
    @classmethod
    def _validate_alias(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if not _NEUTRAL_ALIAS_RE.fullmatch(normalized):
            raise ValueError("external graph aliases must be neutral lowercase names")
        return normalized

    @field_validator(
        "connection_profile_ref",
        "mapping_policy_ref",
        "tls_profile_ref",
        "auth_profile_ref",
        "variables_ref",
    )
    @classmethod
    def _validate_runtime_ref(cls, value: str | None) -> str | None:
        if value is None:
            return None
        rendered = str(value).strip()
        if not _RUNTIME_SECRET_REF_RE.fullmatch(rendered):
            raise ValueError("external graph profiles must use runtime secret refs")
        return rendered

    @field_validator("discovery_max_types")
    @classmethod
    def _bound_discovery_types(cls, value: int) -> int:
        parsed = int(value)
        if not 1 <= parsed <= 500:
            raise ValueError("discovery_max_types must be between 1 and 500")
        return parsed

    @field_validator("discovery_max_depth")
    @classmethod
    def _bound_discovery_depth(cls, value: int) -> int:
        parsed = int(value)
        if not 1 <= parsed <= 12:
            raise ValueError("discovery_max_depth must be between 1 and 12")
        return parsed

    @field_validator("ingest_max_records", mode="before")
    @classmethod
    def _bound_ingest_records(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("ingest_max_records must be an integer")
        parsed = int(value)
        if not 1 <= parsed <= 10_000:
            raise ValueError("ingest_max_records must be between 1 and 10000")
        return parsed

    @field_validator("ingest_page_size", "ingest_max_pages", mode="before")
    @classmethod
    def _bound_property_graph_pages(cls, value: int, info: Any) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{info.field_name} must be an integer")
        parsed = int(value)
        if not 1 <= parsed <= 1_000:
            raise ValueError(f"{info.field_name} must be between 1 and 1000")
        return parsed

    @field_validator("reconcile_deletions", "allow_empty_snapshot", mode="before")
    @classmethod
    def _validate_property_graph_sync_booleans(cls, value: Any, info: Any) -> bool:
        if not isinstance(value, bool):
            raise ValueError(f"{info.field_name} must be boolean")
        return value

    @field_validator("ingest_max_row_bytes", mode="before")
    @classmethod
    def _bound_ingest_row_bytes(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("ingest_max_row_bytes must be an integer")
        parsed = int(value)
        if not 256 <= parsed <= 8_388_608:
            raise ValueError("ingest_max_row_bytes must be between 256 and 8388608")
        return parsed

    @field_validator("ingest_max_total_bytes", mode="before")
    @classmethod
    def _bound_ingest_total_bytes(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("ingest_max_total_bytes must be an integer")
        parsed = int(value)
        if not 256 <= parsed <= 67_108_864:
            raise ValueError("ingest_max_total_bytes must be between 256 and 67108864")
        return parsed

    @field_validator("ingest_max_nesting_depth", mode="before")
    @classmethod
    def _bound_ingest_nesting_depth(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("ingest_max_nesting_depth must be an integer")
        parsed = int(value)
        if not 1 <= parsed <= 64:
            raise ValueError("ingest_max_nesting_depth must be between 1 and 64")
        return parsed

    @field_validator("ingest_max_collection_items", mode="before")
    @classmethod
    def _bound_ingest_collection_items(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("ingest_max_collection_items must be an integer")
        parsed = int(value)
        if not 1 <= parsed <= 100_000:
            raise ValueError("ingest_max_collection_items must be between 1 and 100000")
        return parsed

    @field_validator("ingest_operation")
    @classmethod
    def _validate_operation(cls, value: str | None) -> str | None:
        if value is None:
            return None
        rendered = str(value).strip().lower()
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", rendered):
            raise ValueError("ingest_operation must be a neutral operation alias")
        return rendered

    @model_validator(mode="after")
    def _validate_graphql_bootstrap(self) -> "ExternalGraphConnectorConfig":
        if self.ingest_max_total_bytes < self.ingest_max_row_bytes:
            raise ValueError("ingest_max_total_bytes must cover one bounded row")
        if self.backend == "graphql" and self.semantic_mapping:
            raise ValueError(
                "semantic_mapping is available only for property graph sources"
            )
        if (
            self.backend == "graphql"
            and not self.mapping_policy_ref
            and not self.allow_introspection
        ):
            raise ValueError(
                "GraphQL sources require mapping_policy_ref or explicit "
                "allow_introspection"
            )
        return self


class ProviderRuntimeProfile(BaseModel):
    """Reference-only runtime configuration for one external provider.

    Durable configuration carries only neutral profile identity, runtime
    references, and an explicit named TLS selector. Endpoint and credential
    material are resolved in memory at the provider boundary and are never
    projected into the configuration model, doctor output, logs, or reports.

    A provider with multiple independently trusted endpoints declares one
    profile per connection. This keeps each endpoint, trust policy, and set of
    credentials inseparable while remaining independent of provider-specific
    schemas or deployment topology.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    endpoint_ref: str | None = None
    credential_refs: dict[str, str] = Field(default_factory=dict)
    selector_refs: dict[str, str] = Field(default_factory=dict)
    tls_profile: str | None = None
    tls_profile_ref: str | None = None

    @field_validator("endpoint_ref", "tls_profile_ref")
    @classmethod
    def _validate_profile_reference(cls, value: str | None) -> str | None:
        if value is None:
            return None
        rendered = str(value).strip()
        if _RUNTIME_SECRET_REF_RE.fullmatch(rendered) is None:
            raise ValueError("provider runtime values must use runtime references")
        return rendered

    @field_validator("credential_refs", "selector_refs", mode="before")
    @classmethod
    def _validate_reference_mapping(cls, value: Any) -> dict[str, str]:
        if value in (None, ""):
            return {}
        if not isinstance(value, Mapping) or len(value) > 16:
            raise ValueError("provider runtime reference mappings must be bounded")
        validated: dict[str, str] = {}
        for raw_alias, raw_reference in value.items():
            if not isinstance(raw_alias, str) or not isinstance(raw_reference, str):
                raise ValueError("provider runtime reference mappings are invalid")
            alias = raw_alias.strip()
            reference = raw_reference.strip()
            if (
                alias != raw_alias
                or reference != raw_reference
                or _MCP_FLEET_SECRET_ALIAS_RE.fullmatch(alias) is None
                or _RUNTIME_SECRET_REF_RE.fullmatch(reference) is None
            ):
                raise ValueError("provider runtime reference mappings are invalid")
            validated[alias] = reference
        return validated

    @field_validator("tls_profile")
    @classmethod
    def _validate_tls_profile_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        rendered = str(value).strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,127}", rendered):
            raise ValueError("provider TLS profile name is invalid")
        return rendered

    @model_validator(mode="after")
    def _validate_runtime_contract(self) -> "ProviderRuntimeProfile":
        if self.tls_profile and self.tls_profile_ref:
            raise ValueError("provider runtime profile has ambiguous TLS selectors")
        if self.endpoint_ref and not (self.tls_profile or self.tls_profile_ref):
            raise ValueError("provider endpoints require an explicit TLS profile")
        if set(self.credential_refs).intersection(self.selector_refs):
            raise ValueError(
                "provider credential and selector aliases must be distinct"
            )
        if self.enabled and not (
            self.endpoint_ref or self.credential_refs or self.selector_refs
        ):
            raise ValueError("enabled provider runtime profiles cannot be empty")
        return self


# _load_xdg_json_config() is called dynamically via _ensure_env_loaded().


class AgentConfig(BaseSettings):
    """Configuration schema for the AI Agent server.

    Uses Pydantic BaseSettings to validate the XDG configuration projection and
    process-bound runtime values via canonical aliases. Covers LLM settings,
    server networking, observability (OTEL), A2A communication, and safety guards.
    """

    model_config = SettingsConfigDict(
        env_file=None,
        env_ignore_empty=True,
        extra="ignore",
        secrets_dir=None,
    )

    def __init__(self, **values: Any) -> None:
        with _xdg_projection_lock:
            snapshot = _capture_projection_state_locked() if not _env_loaded else None
            try:
                _ensure_env_loaded()
                super().__init__(**values)
            except Exception:
                if snapshot is not None:
                    _restore_projection_state_locked(snapshot)
                raise

    @model_validator(mode="before")
    @classmethod
    def _reject_retired_configuration(cls, value: Any) -> Any:
        supplied = value.keys() if isinstance(value, Mapping) else ()
        environment = (key for key in os.environ if key in _RETIRED_CONFIGURATION_KEYS)
        _require_current_configuration_keys(supplied)
        _require_current_configuration_keys(environment, durable=False)
        return value

    app_profile: str = Field(default="dev", alias="APP_PROFILE")
    """Deployment posture forwarded to locally autostarted components."""

    deployment_profile: Literal["tiny", "single-node-prod", "enterprise"] = Field(
        default="tiny", alias="DEPLOYMENT_PROFILE"
    )
    """Explicit Agent Utilities deployment shape; independent of APP_PROFILE."""

    # --- Exact skill certification deployment references ---

    skill_cert_runtime_configuration: str | None = Field(
        default=None, alias="SKILL_CERT_RUNTIME_CONFIGURATION"
    )
    skill_cert_runtime_profile: str | None = Field(
        default=None, alias="SKILL_CERT_RUNTIME_PROFILE"
    )
    skill_cert_release_spec: str | None = Field(
        default=None, alias="SKILL_CERT_RELEASE_SPEC"
    )
    skill_cert_promotion_evidence: str | None = Field(
        default=None, alias="SKILL_CERT_PROMOTION_EVIDENCE"
    )
    skill_cert_graphos_endpoint: str | None = Field(
        default=None, alias="SKILL_CERT_GRAPHOS_ENDPOINT"
    )
    skill_cert_graphos_command: list[str] = Field(
        default_factory=list, alias="SKILL_CERT_GRAPHOS_COMMAND"
    )
    skill_validation_evidence_signer_command: list[str] = Field(
        default_factory=list, alias="SKILL_VALIDATION_EVIDENCE_SIGNER_COMMAND"
    )
    skill_validation_evidence_verifier_command: list[str] = Field(
        default_factory=list, alias="SKILL_VALIDATION_EVIDENCE_VERIFIER_COMMAND"
    )
    skill_cert_identity_authority_mode: Literal["ephemeral-https-loopback"] = Field(
        default="ephemeral-https-loopback",
        alias="SKILL_CERT_IDENTITY_AUTHORITY_MODE",
    )
    """Current exact certification owns one verified HTTPS loopback authority."""

    skill_cert_identity_token_ttl_seconds: int = Field(
        default=300,
        ge=180,
        le=3_600,
        alias="SKILL_CERT_IDENTITY_TOKEN_TTL_SECONDS",
    )
    """Lifetime of renewable credentials minted only within one campaign."""

    @field_validator(
        "skill_cert_runtime_configuration",
        "skill_cert_runtime_profile",
        "skill_cert_release_spec",
        "skill_cert_promotion_evidence",
    )
    @classmethod
    def _validate_skill_certification_path(cls, value: str | None) -> str | None:
        if value in (None, ""):
            return None
        if not isinstance(value, str):
            raise ValueError("skill certification path must be a string")
        rendered = value.strip()
        if (
            not rendered
            or len(rendered.encode("utf-8")) > 4_096
            or any(character in rendered for character in "\x00\r\n")
        ):
            raise ValueError("skill certification path is invalid")
        path = pathlib.Path(rendered)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("skill certification path must be absolute")
        return rendered

    @field_validator("skill_cert_graphos_endpoint")
    @classmethod
    def _validate_skill_certification_endpoint(cls, value: str | None) -> str | None:
        if value in (None, ""):
            return None
        if not isinstance(value, str):
            raise ValueError("skill certification endpoint must be a string")
        rendered = value.strip()
        if (
            not rendered
            or len(rendered.encode("utf-8")) > 4_096
            or any(character in rendered for character in "\x00\r\n")
        ):
            raise ValueError("skill certification endpoint is invalid")
        try:
            parsed = urlsplit(rendered)
            host = str(parsed.hostname or "").casefold().rstrip(".")
            port = parsed.port
        except ValueError as exc:
            raise ValueError("skill certification endpoint is invalid") from exc
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host in {"localhost", "localhost.localdomain"}
        if (
            parsed.scheme.casefold() not in {"http", "https"}
            or not loopback
            or (port is not None and not 1 <= port <= 65_535)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("skill certification endpoint must be loopback HTTP(S)")
        return rendered

    @field_validator(
        "skill_cert_graphos_command",
        "skill_validation_evidence_signer_command",
        "skill_validation_evidence_verifier_command",
    )
    @classmethod
    def _validate_skill_certification_command(cls, value: list[str]) -> list[str]:
        if value == []:
            return value
        if (
            not isinstance(value, list)
            or not 1 <= len(value) <= 32
            or any(
                not isinstance(item, str)
                or not 1 <= len(item) <= 4_096
                or any(character in item for character in "\x00\r\n")
                for item in value
            )
        ):
            raise ValueError("skill certification command must be bounded JSON argv")
        executable = pathlib.Path(value[0])
        if not executable.is_absolute() or ".." in executable.parts:
            raise ValueError("skill certification executable must be absolute")
        return value

    # --- Production certification runtime ---

    certification_mode: Literal["disabled", "production"] = Field(
        default="disabled", alias="CERTIFICATION_MODE"
    )
    cert_release_manifest: str | None = Field(
        default=None, alias="CERT_RELEASE_MANIFEST"
    )
    cert_artifacts_dir: str | None = Field(default=None, alias="CERT_ARTIFACTS_DIR")
    cert_hardware_class: str | None = Field(default=None, alias="CERT_HARDWARE_CLASS")
    cert_load_command: list[str] = Field(
        default_factory=list, alias="CERT_LOAD_COMMAND"
    )
    cert_metrics_command: list[str] = Field(
        default_factory=list, alias="CERT_METRICS_COMMAND"
    )
    cert_hook_commands: dict[str, list[str]] = Field(
        default_factory=dict, alias="CERT_HOOK_COMMANDS"
    )
    cert_fault_action_commands: dict[str, list[str]] = Field(
        default_factory=dict, alias="CERT_FAULT_ACTION_COMMANDS"
    )
    cert_fault_probe_commands: dict[str, list[str]] = Field(
        default_factory=dict, alias="CERT_FAULT_PROBE_COMMANDS"
    )
    cert_evidence_signer_command: list[str] = Field(
        default_factory=list, alias="CERT_EVIDENCE_SIGNER_COMMAND"
    )
    cert_evidence_verifier_command: list[str] = Field(
        default_factory=list, alias="CERT_EVIDENCE_VERIFIER_COMMAND"
    )
    cert_prometheus_url: str | None = Field(default=None, alias="CERT_PROMETHEUS_URL")
    cert_prometheus_bearer_token_ref: str | None = Field(
        default=None, alias="CERT_PROMETHEUS_BEARER_TOKEN_REF"
    )
    cert_prometheus_tls_profile: str | None = Field(
        default=None, alias="CERT_PROMETHEUS_TLS_PROFILE"
    )
    cert_prometheus_tls_profile_ref: str | None = Field(
        default=None, alias="CERT_PROMETHEUS_TLS_PROFILE_REF"
    )

    @field_validator("cert_release_manifest", "cert_artifacts_dir")
    @classmethod
    def _validate_production_certification_path(cls, value: str | None) -> str | None:
        if value in (None, ""):
            return None
        if not isinstance(value, str):
            raise ValueError("production certification path must be a string")
        rendered = value.strip()
        if (
            not rendered
            or len(rendered.encode("utf-8")) > 4_096
            or any(character in rendered for character in "\x00\r\n")
        ):
            raise ValueError("production certification path is invalid")
        path = pathlib.Path(rendered)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("production certification path must be absolute")
        return rendered

    @field_validator("cert_hardware_class")
    @classmethod
    def _validate_certification_hardware_class(cls, value: str | None) -> str | None:
        if value in (None, ""):
            return None
        rendered = str(value).strip()
        if not re.fullmatch(r"(?:capacity|tier)-[a-z0-9][a-z0-9._-]{1,63}", rendered):
            raise ValueError(
                "certification hardware class must be a non-identifying capacity-* "
                "or tier-* label"
            )
        return rendered

    @staticmethod
    def _production_certification_argv(value: Any) -> list[str]:
        if (
            not isinstance(value, list)
            or not 1 <= len(value) <= 64
            or any(
                not isinstance(item, str)
                or not 1 <= len(item) <= 4_096
                or any(character in item for character in "\x00\r\n")
                for item in value
            )
        ):
            raise ValueError(
                "production certification command must be bounded JSON argv"
            )
        executable = pathlib.Path(value[0])
        if not executable.is_absolute() or ".." in executable.parts:
            raise ValueError(
                "production certification executable must be an absolute path"
            )
        return value

    @field_validator(
        "cert_load_command",
        "cert_metrics_command",
        "cert_evidence_signer_command",
        "cert_evidence_verifier_command",
    )
    @classmethod
    def _validate_production_certification_command(cls, value: list[str]) -> list[str]:
        if value == []:
            return value
        return cls._production_certification_argv(value)

    @field_validator(
        "cert_hook_commands",
        "cert_fault_action_commands",
        "cert_fault_probe_commands",
    )
    @classmethod
    def _validate_production_certification_command_map(
        cls, value: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        if value == {}:
            return value
        if not isinstance(value, dict) or not set(value).issubset(
            PRODUCTION_CERTIFICATION_SCENARIOS
        ):
            raise ValueError(
                "production certification command map has an invalid scenario set"
            )
        return {
            scenario: cls._production_certification_argv(command)
            for scenario, command in value.items()
        }

    @field_validator("cert_prometheus_url")
    @classmethod
    def _validate_certification_prometheus_url(cls, value: str | None) -> str | None:
        rendered = _validated_runtime_http_url(value)
        if rendered is not None and urlsplit(rendered).scheme.casefold() != "https":
            raise ValueError("CERT_PROMETHEUS_URL must use HTTPS")
        return rendered

    @field_validator(
        "cert_prometheus_bearer_token_ref", "cert_prometheus_tls_profile_ref"
    )
    @classmethod
    def _validate_certification_runtime_ref(cls, value: str | None) -> str | None:
        if value in (None, ""):
            return None
        rendered = str(value).strip()
        if not _RUNTIME_SECRET_REF_RE.fullmatch(rendered):
            raise ValueError(
                "production certification secret material must use a runtime reference"
            )
        return rendered

    @field_validator("cert_prometheus_tls_profile")
    @classmethod
    def _validate_certification_tls_profile(cls, value: str | None) -> str | None:
        if value in (None, ""):
            return None
        rendered = str(value).strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", rendered):
            raise ValueError("production certification TLS profile is invalid")
        return rendered

    # --- General ---

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
        )

    chat_models: list[ChatModelConfig] = Field(
        default_factory=list, alias="CHAT_MODELS"
    )
    embedding_models: list[EmbeddingModelConfig] = Field(
        default_factory=list, alias="EMBEDDING_MODELS"
    )

    # Runtime-only transport profiles. Durable configuration contains refs,
    # never endpoints, CA subjects/material, client keys, or machine paths.
    tls_profile: str | None = Field(default=None, alias="TLS_PROFILE")
    tls_profile_ref: str | None = Field(default=None, alias="TLS_PROFILE_REF")
    tls_profiles_ref: str | None = Field(default=None, alias="TLS_PROFILES_REF")
    tls_ca_bundle_ref: str | None = Field(default=None, alias="TLS_CA_BUNDLE_REF")
    tls_client_cert_ref: str | None = Field(default=None, alias="TLS_CLIENT_CERT_REF")
    tls_client_key_ref: str | None = Field(default=None, alias="TLS_CLIENT_KEY_REF")
    tls_client_key_password_ref: str | None = Field(
        default=None, alias="TLS_CLIENT_KEY_PASSWORD_REF"
    )
    tls_proxy_url_ref: str | None = Field(default=None, alias="TLS_PROXY_URL_REF")
    tls_system_trust: bool = Field(default=True, alias="TLS_SYSTEM_TRUST")
    tls_trust_env: bool = Field(default=True, alias="TLS_TRUST_ENV")

    # Shared outbound-source boundary. Private/reserved destinations and
    # cross-host redirects are denied unless an operator names the exact host;
    # response and redirect limits apply to every source fetch path.
    source_http_allowed_private_hosts: list[str] = Field(
        default_factory=list, alias="SOURCE_HTTP_ALLOWED_PRIVATE_HOSTS"
    )
    source_http_allowed_redirect_hosts: list[str] = Field(
        default_factory=list, alias="SOURCE_HTTP_ALLOWED_REDIRECT_HOSTS"
    )
    source_http_max_response_bytes: int = Field(
        default=10 * 1024 * 1024,
        ge=1_024,
        le=64 * 1024 * 1024,
        alias="SOURCE_HTTP_MAX_RESPONSE_BYTES",
    )
    source_http_max_redirects: int = Field(
        default=3, ge=0, le=10, alias="SOURCE_HTTP_MAX_REDIRECTS"
    )
    # Headless browsers have a materially larger egress and sandbox surface
    # than the bounded HTTP client. Keep browser-backed hydration explicit.
    source_http_allow_browser_fetch: bool = Field(
        default=False, alias="SOURCE_HTTP_ALLOW_BROWSER_FETCH"
    )

    # Optional MCP policy decision point. The endpoint, policy path, trust
    # profile, private-host exceptions, and credential reference are runtime
    # deployment data; the package contains only the native connection surface.
    eunomia_type: Literal["none", "embedded", "remote"] = Field(
        default="none", alias="EUNOMIA_TYPE"
    )
    eunomia_policy_file: str | None = Field(default=None, alias="EUNOMIA_POLICY_FILE")
    eunomia_remote_url: str | None = Field(default=None, alias="EUNOMIA_REMOTE_URL")
    eunomia_api_key_ref: str | None = Field(default=None, alias="EUNOMIA_API_KEY_REF")
    eunomia_tls_profile: str | None = Field(default=None, alias="EUNOMIA_TLS_PROFILE")
    eunomia_tls_profile_ref: str | None = Field(
        default=None, alias="EUNOMIA_TLS_PROFILE_REF"
    )
    eunomia_allowed_private_hosts: list[str] = Field(
        default_factory=list, alias="EUNOMIA_ALLOWED_PRIVATE_HOSTS"
    )
    eunomia_timeout_seconds: float = Field(
        default=10.0, gt=0.0, le=300.0, alias="EUNOMIA_TIMEOUT_SECONDS"
    )
    eunomia_max_response_bytes: int = Field(
        default=1024 * 1024,
        ge=1_024,
        le=64 * 1024 * 1024,
        alias="EUNOMIA_MAX_RESPONSE_BYTES",
    )
    eunomia_bulk_check_max: int = Field(
        default=100, ge=1, le=100, alias="EUNOMIA_BULK_CHECK_MAX"
    )
    # External vector stores are runtime deployment connections.  The codebase
    # carries only the native connection surface; endpoints, local roots, and
    # secret references live in AgentConfig/XDG or the launcher environment.
    vector_database_type: Literal[
        "epistemic_graph", "postgres", "qdrant", "mongodb"
    ] = Field(default="epistemic_graph", alias="DATABASE_TYPE")
    vector_db_host: str | None = Field(default=None, alias="DB_HOST")
    vector_db_port: int | None = Field(default=None, ge=1, le=65_535, alias="DB_PORT")
    vector_db_name: str | None = Field(default=None, alias="DBNAME")
    vector_db_username_ref: str | None = Field(default=None, alias="DB_USERNAME_REF")
    vector_db_password_ref: str | None = Field(default=None, alias="DB_PASSWORD_REF")
    vector_document_directory: str | None = Field(
        default=None, alias="DOCUMENT_DIRECTORY"
    )

    postgres_tls_profile: str | None = Field(default=None, alias="POSTGRES_TLS_PROFILE")
    postgres_tls_profile_ref: str | None = Field(
        default=None, alias="POSTGRES_TLS_PROFILE_REF"
    )
    postgres_request_timeout: int = Field(
        default=30, ge=1, le=300, alias="POSTGRES_REQUEST_TIMEOUT"
    )
    postgres_max_pool_size: int = Field(
        default=20, ge=1, le=100, alias="POSTGRES_MAX_POOL_SIZE"
    )

    qdrant_api_key_ref: str | None = Field(default=None, alias="QDRANT_API_KEY_REF")
    qdrant_tls_profile: str | None = Field(default=None, alias="QDRANT_TLS_PROFILE")
    qdrant_tls_profile_ref: str | None = Field(
        default=None, alias="QDRANT_TLS_PROFILE_REF"
    )
    qdrant_http_allowed_private_hosts: list[str] = Field(
        default_factory=list, alias="QDRANT_HTTP_ALLOWED_PRIVATE_HOSTS"
    )
    qdrant_request_timeout: int = Field(
        default=30, ge=1, le=300, alias="QDRANT_REQUEST_TIMEOUT"
    )
    mongodb_uri_ref: str | None = Field(default=None, alias="MONGODB_URI_REF")
    mongodb_tls_profile: str | None = Field(default=None, alias="MONGODB_TLS_PROFILE")
    mongodb_tls_profile_ref: str | None = Field(
        default=None, alias="MONGODB_TLS_PROFILE_REF"
    )
    mongodb_request_timeout_ms: int = Field(
        default=30_000,
        ge=1_000,
        le=300_000,
        alias="MONGODB_REQUEST_TIMEOUT_MS",
    )
    mongodb_max_pool_size: int = Field(
        default=20, ge=1, le=100, alias="MONGODB_MAX_POOL_SIZE"
    )
    redis_connection_profile_ref: str | None = Field(
        default=None,
        alias="REDIS_CONNECTION_PROFILE_REF",
    )
    redis_tls_profile: str | None = Field(default=None, alias="REDIS_TLS_PROFILE")
    redis_tls_profile_ref: str | None = Field(
        default=None,
        alias="REDIS_TLS_PROFILE_REF",
    )

    model_http_allowed_private_hosts: list[str] = Field(
        default_factory=list, alias="MODEL_HTTP_ALLOWED_PRIVATE_HOSTS"
    )
    """Exact runtime-configured private/loopback hosts that model clients may
    reach through the DNS-pinned transport. Empty denies private model egress;
    public HTTPS providers remain available. Values are never emitted by
    doctor, logs, traces, graph records, or public APIs."""

    model_tls_profile: str | None = Field(default=None, alias="MODEL_TLS_PROFILE")
    model_tls_profile_ref: str | None = Field(
        default=None, alias="MODEL_TLS_PROFILE_REF"
    )
    embedding_tls_profile: str | None = Field(
        default=None, alias="EMBEDDING_TLS_PROFILE"
    )
    embedding_tls_profile_ref: str | None = Field(
        default=None, alias="EMBEDDING_TLS_PROFILE_REF"
    )

    # OAuth2 client-credentials token minting has its own trust boundary.  It
    # may select a profile from the shared runtime catalog (or a profile held
    # entirely behind a secret reference) without persisting CA material,
    # endpoint identity, or workstation paths in model configuration.
    oauth2_token_tls_profile: str | None = Field(
        default=None, alias="OAUTH2_TOKEN_TLS_PROFILE"
    )
    oauth2_token_tls_profile_ref: str | None = Field(
        default=None, alias="OAUTH2_TOKEN_TLS_PROFILE_REF"
    )

    external_graph_connectors: list[ExternalGraphConnectorConfig] = Field(
        default_factory=list, alias="EXTERNAL_GRAPH_CONNECTORS"
    )

    provider_configs: dict[str, ProviderRuntimeProfile] = Field(
        default_factory=dict,
        alias="PROVIDER_CONFIGS",
    )
    """Deployment-owned, reference-only external provider profiles.

    Provider packages select a neutral profile and resolve it at their runtime
    boundary. The durable document never contains endpoint values, credentials,
    certificate material, personalized paths, or customized provider schemas.
    """

    # --- Provider API Keys (global fallbacks for ad-hoc model creation) ---
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    mistral_api_key: str | None = Field(default=None, alias="MISTRAL_API_KEY")
    hugging_face_api_key: str | None = Field(default=None, alias="HUGGING_FACE_API_KEY")
    deepseek_api_key: str | None = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str | None = Field(default=None, alias="DEEPSEEK_BASE_URL")

    # --- Messaging reach + agent KG layer (CONCEPT:AU-ECO.messaging.messaging-reach-service-governed–4.61) ---
    # Outbound/inbound messaging (Telegram/Slack/Teams/Mattermost/…). Tokens per backend
    # (e.g. TELEGRAM_BOT_TOKEN) auto-enable that backend; these tune routing + the agent.
    telegram_bot_token: str | None = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    messaging_default_platform: str = Field(
        default="telegram", alias="MESSAGING_DEFAULT_PLATFORM"
    )
    messaging_default_channel: str = Field(
        default="", alias="MESSAGING_DEFAULT_CHANNEL"
    )
    messaging_alert_intake_port: int | None = Field(
        default=None, ge=1, le=65535, alias="MESSAGING_ALERT_INTAKE_PORT"
    )
    messaging_alert_intake_host: str = Field(
        default="127.0.0.1", alias="MESSAGING_ALERT_INTAKE_HOST"
    )
    messaging_alert_intake_token_ref: str | None = Field(
        default=None, alias="MESSAGING_ALERT_INTAKE_TOKEN_REF"
    )
    messaging_alert_intake_allow_remote: bool = Field(
        default=False, alias="MESSAGING_ALERT_INTAKE_ALLOW_REMOTE"
    )
    # CONCEPT:AU-ECO.messaging.universal-graph-agent — the universal graph agent a chat turn routes to. Defaults to the
    # dedicated "messaging-assistant" identity in code; set to route a chat turn to a
    # different named agent. Unresolved names still go through the full orchestration graph.
    messaging_agent: str = Field(default="", alias="MESSAGING_AGENT")
    messaging_claude_trigger: str = Field(
        default="/claude", alias="MESSAGING_CLAUDE_TRIGGER"
    )
    messaging_claude_model: str = Field(
        default="claude-sonnet-4-6", alias="MESSAGING_CLAUDE_MODEL"
    )
    messaging_local_model: str = Field(default="", alias="MESSAGING_LOCAL_MODEL")
    reactions: str = Field(default="1", alias="REACTIONS")
    # Burst coalescing (CONCEPT:AU-ECO.messaging.burst-mode-coalescing): collapse a rapid run of messages into ONE reply.
    messaging_burst_window_s: str = Field(
        default="2.5", alias="MESSAGING_BURST_WINDOW_S"
    )
    messaging_burst_max_s: str = Field(default="12", alias="MESSAGING_BURST_MAX_S")
    # Post-conversation enrichment (CONCEPT:AU-ECO.messaging.post-conversation-enrichment): mine chats → KG concepts (opt-out).
    messaging_enrich: str = Field(default="1", alias="MESSAGING_ENRICH")
    # Surface goals / SDD specs from chats (CONCEPT:AU-ECO.messaging.surfaced, opt-out).
    messaging_goals: str = Field(default="1", alias="MESSAGING_GOALS")
    # Webhook push (CONCEPT:AU-ECO.messaging.telegram-webhook-receiver-started): set the PUBLIC base URL (served via tunnel/edge to a
    # LOCAL port) to switch from polling to instant webhook delivery; empty = polling.
    messaging_webhook_base_url: str = Field(
        default="", alias="MESSAGING_WEBHOOK_BASE_URL"
    )
    messaging_webhook_port: str = Field(default="8443", alias="MESSAGING_WEBHOOK_PORT")
    messaging_webhook_secret: str = Field(default="", alias="MESSAGING_WEBHOOK_SECRET")
    # Voice input (CONCEPT:AU-ECO.messaging.telegram-voice-note): transcribe voice notes to text via Whisper (opt-out).
    messaging_voice: str = Field(default="1", alias="MESSAGING_VOICE")
    messaging_voice_model: str = Field(default="base", alias="MESSAGING_VOICE_MODEL")
    # KG as a first-class default tool layer for every agent (opt-out).
    agent_kg_tools: str = Field(default="True", alias="AGENT_KG_TOOLS")

    # --- Ingestion sources (CONCEPT:AU-KG.query.vendor-agnostic-traversal web-fetch) ---
    # When set, ArchiveBox (a deployed web-archiving instance reached via the
    # archivebox-api MCP server) is preferred over a live crawl: the unified
    # web-fetch resolver serves the preserved snapshot (fast, no re-crawl,
    # archive-on-miss). Unset → crawl4ai (if installed) → requests+markitdown.
    # The presence of a URL is the on-signal; the credential lives with the MCP
    # server, so only this toggle is needed here.
    archivebox_url: str | None = Field(default=None, alias="ARCHIVEBOX_URL")

    # Runtime-only infrastructure inventory. The path may identify an operator
    # or machine and therefore must never be written to graph metadata, traces,
    # or doctor output. Empty keeps inventory ingestion disabled.
    infra_inventory_path: str | None = Field(default=None, alias="INFRA_INVENTORY_PATH")

    @field_validator("infra_inventory_path", mode="before")
    @classmethod
    def _validate_infra_inventory_path(cls, value: Any) -> str | None:
        if value in (None, ""):
            return None
        rendered = str(value).strip()
        if not rendered:
            return None
        if (
            len(rendered) > 4_096
            or "\x00" in rendered
            or "\n" in rendered
            or "\r" in rendered
        ):
            raise ValueError("INFRA_INVENTORY_PATH is malformed")
        return rendered

    # --- Media service endpoints ---
    # These are runtime-only base URLs for interchangeable self-hosted or
    # managed media services. They are deliberately unset by default and are
    # omitted from persistence, logs, traces, and doctor output.
    comfyui_url: str | None = Field(default=None, alias="COMFYUI_URL")
    xtts_url: str | None = Field(default=None, alias="XTTS_URL")
    openai_tts_url: str | None = Field(default=None, alias="OPENAI_TTS_URL")
    whisper_url: str | None = Field(default=None, alias="WHISPER_URL")
    faster_whisper_url: str | None = Field(default=None, alias="FASTER_WHISPER_URL")
    flux_url: str | None = Field(default=None, alias="FLUX_URL")
    sd35_url: str | None = Field(default=None, alias="SD35_URL")
    hunyuan_url: str | None = Field(default=None, alias="HUNYUAN_URL")
    svd_url: str | None = Field(default=None, alias="SVD_URL")

    @field_validator(
        "comfyui_url",
        "xtts_url",
        "openai_tts_url",
        "whisper_url",
        "faster_whisper_url",
        "flux_url",
        "sd35_url",
        "hunyuan_url",
        "svd_url",
        mode="before",
    )
    @classmethod
    def _validate_media_service_url(cls, value: Any) -> str | None:
        return _validated_runtime_http_url(value)

    # --- Graph / KG tuning knobs ---
    # Whole-workflow orchestration budget (ms). Lowered 20min→10min: engine RPC
    # hangs are now caught in seconds by the client's per-RPC timeout, so this is a
    # backstop for a wedged multi-agent run, not the primary hang detector. Kept
    # generous enough for long legitimate multi-step workflows; override per deploy.
    graph_timeout: str | None = Field(default="600000", alias="GRAPH_TIMEOUT")
    max_recursion_depth: str | None = Field(default="2", alias="MAX_RECURSION_DEPTH")
    routing_percentile: str | None = Field(default="50.0", alias="ROUTING_PERCENTILE")
    # Must match the embedding model's output dimension (768). The schema vector
    # column size is derived from this, so a mismatch breaks node inserts.
    kg_embedding_dim: str | None = Field(default="768", alias="KG_EMBEDDING_DIM")

    # Single dev switch that disables ALL KG background daemons (maintenance
    # scheduler: enrichment/reconcile/file-watch/hygiene/task-reaper + the
    # embedding backfill). Production keeps them all on; this replaces the old
    # per-daemon KG_*_DAEMON env toggles (CONCEPT:EG-KG.storage.nonblocking-checkpoint, config discipline).
    kg_dev_mode: bool = Field(default=False, alias="KG_DEV_MODE")

    # --- Observability / usage analytics (CONCEPT:AU-OS.observability.usage-analytics-store / ECO-4.40 / OS-5.31) ---
    # Backend for the usage/cost/session fact store. Zero-config default is a
    # per-host SQLite+FTS5 file (no external deps); "postgres" / "duckdb" promote
    # to enterprise-scale shared backends via the same UsageBackend interface.
    usage_db_backend: str = Field(default="sqlite", alias="USAGE_DB_BACKEND")
    # Optional explicit path/URI for the usage store. Empty = derive from the
    # state_store seam (per-host data dir for sqlite, STATE_DB_URI for postgres).
    usage_db_uri: str | None = Field(default=None, alias="USAGE_DB_URI")
    # Master switch for runtime usage instrumentation (plane B). Default-on but
    # best-effort: a recorder failure never breaks a graph run.
    usage_tracking_enabled: bool = Field(default=True, alias="USAGE_TRACKING_ENABLED")
    # Usage is an analytics fact store, not a transcript archive. Metadata-only
    # is the privacy-safe default. ``sanitized`` is an explicit opt-in for a
    # separately governed non-production store; raw retention is unsupported.
    usage_content_retention: str = Field(
        default="metadata", alias="USAGE_CONTENT_RETENTION"
    )
    # LiteLLM pricing source. Empty keeps the bundled offline fallback only
    # (fully functional with no network); the daemon refreshes from this URL.
    pricing_litellm_url: str = Field(
        default=(
            "https://raw.githubusercontent.com/BerriAI/litellm/main/"
            "model_prices_and_context_window.json"
        ),
        alias="PRICING_LITELLM_URL",
    )

    # --- Model registry helpers (derive from chat_models / embedding_models) ---

    def _chat_model_by_level(self, level: str) -> ChatModelConfig | None:
        """Return the first chat model matching the given intelligence_level."""
        for m in self.chat_models:
            if m.intelligence_level == level:
                return m
        return None

    @property
    def default_chat_model(self) -> ChatModelConfig | None:
        """Primary chat model (intelligence_level='normal', fallback to first)."""
        return self._chat_model_by_level("normal") or (
            self.chat_models[0] if self.chat_models else None
        )

    @property
    def lite_chat_model(self) -> ChatModelConfig | None:
        """Lightweight chat model (intelligence_level='light')."""
        return self._chat_model_by_level("light") or self.default_chat_model

    @property
    def super_chat_model(self) -> ChatModelConfig | None:
        """Super/heavy chat model (intelligence_level='super')."""
        return self._chat_model_by_level("super") or self.default_chat_model

    @property
    def default_embedding_model(self) -> EmbeddingModelConfig | None:
        """Primary embedding model (first in list)."""
        return self.embedding_models[0] if self.embedding_models else None

    # --- Parallel-call capacity resolution (CONCEPT:AU-KG.compute.concurrency-controller-sizing) ---

    def _resolve_model_config(
        self, model: str | None = None
    ) -> "ChatModelConfig | EmbeddingModelConfig | None":
        """Resolve a model id/role to its config object (CONCEPT:AU-KG.compute.concurrency-controller-sizing).

        ``model`` may be a model id (matched against both chat and embedding
        registries), a role (``"chat"``/``"default"``, ``"lite"``, ``"super"``,
        ``"embedding"``/``"embed"``), or ``None`` (→ default chat model). Returns
        ``None`` when nothing matches.
        """
        cfg: ChatModelConfig | EmbeddingModelConfig | None = None
        key = (model or "").strip().lower()
        if key in ("", "chat", "default"):
            cfg = self.default_chat_model
        elif key == "lite":
            cfg = self.lite_chat_model
        elif key == "super":
            cfg = self.super_chat_model
        elif key in ("embedding", "embed"):
            cfg = self.default_embedding_model
        elif key in ("embedding:fallback", "embed:fallback", "embedding-fallback"):
            # The automatic-failover endpoint (CONCEPT:AU-KG.enrichment.each-call-resolves-active): resolve it as a
            # first-class model key so the WHOLE capacity guard — server_ceiling,
            # adaptive capacity, gpu_group budget (CONCEPT:AU-KG.ingest.keys-off) — keys off the
            # FALLBACK endpoint's config (its own gpu_group / max_concurrent_requests)
            # while failed-over, so fallback embeds inherit the shared GPU's joint
            # budget and can't OOM it.
            primary = self.default_embedding_model
            cfg = primary.fallback if primary is not None else None
        else:
            for m in self.chat_models:
                if m.id == model:
                    cfg = m
                    break
            if cfg is None:
                for em in self.embedding_models:
                    if em.id == model:
                        cfg = em
                        break
        return cfg

    def resolve_chat_model_config(
        self, model: str | None = None
    ) -> ChatModelConfig | None:
        """Resolve only a configured chat model id or tier.

        Provider-serving paths use this typed public seam so a chat request can
        never accidentally select an embedding model endpoint through the more
        general capacity resolver.
        """

        resolved = self._resolve_model_config(model)
        return resolved if isinstance(resolved, ChatModelConfig) else None

    def model_capacity(self, model: str | None = None) -> int:
        """Resolve a model's total parallel-call capacity by id/role.

        CONCEPT:AU-KG.compute.concurrency-controller-sizing. ``model`` may be a model id (matched against both chat
        and embedding registries), one of the roles ``"chat"``/``"default"``,
        ``"lite"``, ``"super"``, ``"embedding"``/``"embed"``, or ``None`` (→
        default chat model). Unknown/unconfigured models resolve to ``1`` — safe
        sequential behaviour, never zero.
        """
        cfg = self._resolve_model_config(model)
        return cfg.total_capacity if cfg is not None else 1

    def model_max_concurrent_requests(self, model: str | None = None) -> int | None:
        """Resolve a model's explicit server-capacity ceiling, if configured.

        CONCEPT:AU-KG.compute.same-semantics-as. Returns the model's ``max_concurrent_requests`` (the
        server's real ``--max-num-seqs`` / safe in-flight budget) by id or role,
        or ``None`` when unset/unknown so the caller applies the conservative
        default. A non-positive/garbage value resolves to ``None`` (no hard cap
        from config — fall back to the default), never zero.
        """
        cfg = self._resolve_model_config(model)
        if cfg is None:
            return None
        val = getattr(cfg, "max_concurrent_requests", None)
        if val is None:
            return None
        try:
            v = int(val)
        except (TypeError, ValueError):
            return None
        return v if v > 0 else None

    def model_endpoint(self, model: str | None = None) -> tuple[str | None, str | None]:
        """Resolve a model id/role to its ``(model_id, base_url)`` (CONCEPT:AU-KG.compute.surfaces-universal-latency-signal).

        Used by the adaptive concurrency controller to derive a model's vLLM
        ``/metrics`` URL and the ``model_name`` label its Prometheus gauges carry.
        Returns ``(None, None)`` when the model is unknown/unconfigured.
        """
        cfg = self._resolve_model_config(model)
        if cfg is None:
            return (None, None)
        return (cfg.id, cfg.base_url)

    def gpu_group(self, model: str | None = None) -> str | None:
        """Resolve a model's shared-GPU group key (CONCEPT:AU-KG.compute.pure-config-enumeration-fail).

        Models that share one physical GPU are grouped so a per-GPU budget can cap
        their *joint* concurrency (e.g. embedding must leave headroom for chat on a
        shared unified-memory device). Resolution order:

        1. The model's explicit ``gpu_group`` tag, if set — this is the only way to
           group models served from **different** endpoints onto one GPU (for example,
           tag both an embedding model and a chat model with
           ``gpu_group="accelerator-shared"``).
        2. Otherwise the ``base_url`` host (netloc), so same-endpoint models group
           automatically with zero config.
        3. ``None`` when the model is unknown or has neither a tag nor a base_url —
           the caller then applies no budget (per-model behaviour, no regression).
        """
        cfg = self._resolve_model_config(model)
        if cfg is None:
            return None
        tag = getattr(cfg, "gpu_group", None)
        if tag:
            return str(tag).strip().lower() or None
        base_url = getattr(cfg, "base_url", None)
        if not base_url:
            return None
        from urllib.parse import urlsplit

        netloc = urlsplit(str(base_url)).netloc.strip().lower()
        return netloc or None

    def embedding_capacity(self) -> int:
        """Total parallel-call capacity of the default embedding model.

        CONCEPT:AU-KG.compute.concurrency-controller-sizing — convenience for the embedding fan-out path.
        """
        em = self.default_embedding_model
        return em.total_capacity if em is not None else 1

    def reload(self) -> "AgentConfig":
        """Reload and return the new immutable typed configuration snapshot."""
        with _xdg_projection_lock:
            load_config(reload=True)
            current = _LAZY_CACHE.get("_config")
            if current is not None:
                return current
            candidate = self.__class__()
            candidate.assert_production_safe(profile=candidate.app_profile)
            return candidate

    default_agent_name: str = Field(default=meta["name"], alias="DEFAULT_AGENT_NAME")
    agent_description: str = Field(
        default=meta["description"], alias="AGENT_DESCRIPTION"
    )
    agent_system_prompt: str | None = Field(default=None, alias="AGENT_SYSTEM_PROMPT")

    workspace_path: str | None = Field(default=None, alias="WORKSPACE_PATH")
    evolution_staging_root: str | None = Field(
        default=None, alias="EVOLUTION_STAGING_ROOT"
    )
    """Runtime-only root for reviewable evolution artifacts. The value is never
    emitted by doctor, logs, graph metadata, traces, or public API responses."""
    agent_utilities_config_dir: str | None = Field(
        default=None, alias="AGENT_UTILITIES_CONFIG_DIR"
    )

    @field_validator("evolution_staging_root", mode="before")
    @classmethod
    def _validate_evolution_staging_root(cls, value: Any) -> str | None:
        if value in (None, ""):
            return None
        rendered = str(value).strip()
        if not rendered:
            return None
        if (
            len(rendered) > 4_096
            or "\x00" in rendered
            or "\n" in rendered
            or "\r" in rendered
        ):
            raise ValueError("EVOLUTION_STAGING_ROOT is malformed")
        return rendered

    host: str = Field(default="127.0.0.1", alias="HOST")
    port: int = Field(default=9000, alias="PORT")
    debug: bool = Field(default=False, alias="DEBUG")
    # CONCEPT:AU-OS.deployment.airgap-mode — sovereign/self-hosted gate
    # (reports/surpass-6mo/04-five-intersections.md §3). The ONE flag: when
    # set, the canonical outbound HTTP factory (agent_utilities/core/http_client.py)
    # and the LLM client constructor (core/model_factory.py) refuse any request
    # whose target host is not loopback/private/link-local, fail-closed with
    # AirgapViolation instead of silently phoning out. Off by default — flipping
    # it on is an explicit sovereign-deployment decision, not a background
    # behavior change (config discipline: no second knob for the allowlist —
    # point air-gapped endpoints at their private/loopback IP).
    airgap_mode: bool = Field(default=False, alias="AIRGAP_MODE")
    """When true, block outbound HTTP requests to non-local hosts (see
    :mod:`agent_utilities.core.http_client`'s ``airgap_guard_transport``)."""
    enable_web_ui: bool = Field(default=False, alias="ENABLE_WEB_UI")
    enable_terminal_ui: bool = Field(default=False, alias="ENABLE_TERMINAL_UI")
    enable_web_logs: bool = Field(default=False, alias="ENABLE_WEB_LOGS")
    enable_acp: bool = Field(default=False, alias="ENABLE_ACP")
    acp_port: int = Field(default=8001, alias="ACP_PORT")
    acp_session_root: str = Field(default=".acp-sessions", alias="ACP_SESSION_ROOT")

    mcp_url: str | None = Field(default=None, alias="MCP_URL")
    mcp_config: str | None = Field(default=None, alias="MCP_CONFIG")
    mcp_fleet_secret_refs: dict[str, str] = Field(
        default_factory=dict,
        alias="MCP_FLEET_SECRET_REFS",
    )
    """Neutral MCP fleet alias to runtime secret-reference mapping.

    Persistent child catalogs retain ``env://ALIAS`` values. At the child
    boundary, a directly projected alias wins; this mapping is consulted only
    when that alias is unavailable. Values must be ``env://``, ``vault://``, or
    ``secret://`` references and never resolved material.
    """

    @field_validator("mcp_fleet_secret_refs", mode="before")
    @classmethod
    def _validate_mcp_fleet_secret_refs(cls, value: Any) -> dict[str, str]:
        if value in (None, ""):
            return {}
        if isinstance(value, str):
            import json as _json

            def reject_duplicates(
                pairs: list[tuple[str, Any]],
            ) -> dict[str, Any]:
                parsed: dict[str, Any] = {}
                for key, item in pairs:
                    if key in parsed:
                        raise ValueError(
                            "MCP_FLEET_SECRET_REFS contains duplicate aliases"
                        )
                    parsed[key] = item
                return parsed

            try:
                value = _json.loads(value, object_pairs_hook=reject_duplicates)
            except (TypeError, ValueError):
                raise ValueError(
                    "MCP_FLEET_SECRET_REFS must be a JSON object of runtime references"
                ) from None
        if not isinstance(value, Mapping) or len(value) > 512:
            raise ValueError("MCP_FLEET_SECRET_REFS must be a bounded mapping")
        validated: dict[str, str] = {}
        for raw_alias, raw_reference in value.items():
            if not isinstance(raw_alias, str) or not isinstance(raw_reference, str):
                raise ValueError(
                    "MCP_FLEET_SECRET_REFS aliases and references must be strings"
                )
            alias = raw_alias.strip()
            reference = raw_reference.strip()
            if (
                alias != raw_alias
                or reference != raw_reference
                or _MCP_FLEET_SECRET_ALIAS_RE.fullmatch(alias) is None
            ):
                raise ValueError("MCP_FLEET_SECRET_REFS contains an invalid alias")
            if _RUNTIME_SECRET_REF_RE.fullmatch(reference) is None:
                raise ValueError(
                    "MCP_FLEET_SECRET_REFS values must be runtime secret references"
                )
            scheme, _separator, target = reference.partition("://")
            if (
                scheme == "env" and _MCP_FLEET_SECRET_ALIAS_RE.fullmatch(target) is None
            ) or (scheme in {"vault", "secret"} and ".." in target.split("/")):
                raise ValueError(
                    "MCP_FLEET_SECRET_REFS contains an invalid runtime reference"
                )
            validated[alias] = reference
        return validated

    mcp_tool_mode: Literal["intent", "condensed", "verbose", "both"] = Field(
        default="intent", alias="MCP_TOOL_MODE"
    )
    mcp_http_allowed_private_hosts: list[str] = Field(
        default_factory=list, alias="MCP_HTTP_ALLOWED_PRIVATE_HOSTS"
    )
    """Exact private hostnames permitted for DNS-pinned remote MCP children."""
    mcp_static_tokens_ref: str | None = Field(
        default=None, alias="FASTMCP_SERVER_AUTH_STATIC_TOKENS_REF"
    )
    """Secret reference containing the JSON token map for FastMCP ``static``
    authentication. Token values are never accepted inline in configuration."""

    mcp_auth_type: Literal[
        "none", "static", "jwt", "oauth-proxy", "oidc-proxy", "remote-oauth"
    ] = Field(default="none", alias="AUTH_TYPE")
    mcp_jwt_jwks_uri: str | None = Field(
        default=None, alias="FASTMCP_SERVER_AUTH_JWT_JWKS_URI"
    )
    mcp_jwt_issuer: str | None = Field(
        default=None, alias="FASTMCP_SERVER_AUTH_JWT_ISSUER"
    )
    mcp_jwt_audience: str | None = Field(
        default=None, alias="FASTMCP_SERVER_AUTH_JWT_AUDIENCE"
    )
    mcp_jwt_algorithm: str | None = Field(
        default=None, alias="FASTMCP_SERVER_AUTH_JWT_ALGORITHM"
    )
    mcp_jwt_required_scopes: str | None = Field(
        default=None, alias="FASTMCP_SERVER_AUTH_JWT_REQUIRED_SCOPES"
    )
    mcp_jwt_secret_ref: str | None = Field(
        default=None, alias="FASTMCP_SERVER_AUTH_JWT_SECRET_REF"
    )
    mcp_tls_certfile: str | None = Field(default=None, alias="MCP_TLS_CERTFILE")
    mcp_tls_keyfile: str | None = Field(default=None, alias="MCP_TLS_KEYFILE")
    mcp_tls_terminated: bool = Field(default=False, alias="MCP_TLS_TERMINATED")
    mcp_trusted_proxy_cidrs: str | None = Field(
        default=None, alias="MCP_TRUSTED_PROXY_CIDRS"
    )
    mcp_allowed_hosts: str | None = Field(default=None, alias="MCP_ALLOWED_HOSTS")
    mcp_allowed_origins: str | None = Field(default=None, alias="MCP_ALLOWED_ORIGINS")
    mcp_max_request_bytes: int = Field(
        default=4 * 1024 * 1024,
        ge=1_024,
        le=256 * 1024 * 1024,
        alias="MCP_MAX_REQUEST_BYTES",
    )
    mcp_max_connections: int = Field(
        default=128, ge=1, le=10_000, alias="MCP_MAX_CONNECTIONS"
    )
    mcp_listen_backlog: int = Field(
        default=256, ge=1, le=65_535, alias="MCP_LISTEN_BACKLOG"
    )
    mcp_metrics_token_ref: str | None = Field(
        default=None, alias="MCP_METRICS_TOKEN_REF"
    )

    max_upload_size: int = Field(default=10 * 1024 * 1024, alias="MAX_UPLOAD_SIZE")

    auth_jwt_jwks_uri: str | None = Field(default=None, alias="AUTH_JWT_JWKS_URI")
    """JWKS URI for JWT Bearer token verification (e.g. Azure AD, Okta)."""

    auth_jwt_issuer: str | None = Field(default=None, alias="AUTH_JWT_ISSUER")
    """Expected JWT issuer claim for validation."""

    auth_jwt_audience: str | None = Field(default=None, alias="AUTH_JWT_AUDIENCE")
    """Expected JWT audience claim for validation."""

    # --- Knowledge Graph identity enforcement (CONCEPT:AU-OS.identity.authenticated-identity-enforcement) ---
    kg_policy_version: str | None = Field(default=None, alias="KG_POLICY_VERSION")
    """Required immutable authorization-policy revision stamped into every
    server-minted graph session."""

    auth_jwt_algorithms: list[str] = Field(
        default_factory=lambda: [
            "RS256",
            "RS384",
            "RS512",
            "PS256",
            "PS384",
            "PS512",
            "ES256",
            "ES384",
            "ES512",
            "EdDSA",
        ],
        alias="AUTH_JWT_ALGORITHMS",
    )
    """Explicit asymmetric JWT signature algorithms accepted by REST auth."""

    identity_group_capability_map: dict[str, list[str]] | None = Field(
        default=None, alias="IDENTITY_GROUP_CAPABILITY_MAP"
    )
    """Optional IdP group/role → capability mapping for provider-agnostic role
    inheritance (CONCEPT:AU-OS.identity.idp-agnostic-role-inheritance). A JSON
    object, e.g. ``{"okta-group-id-0oa...": ["kg-admin"], "engineering": ["kg-write"]}``.
    Lets an opaque Okta group id (or a differently-named Keycloak group) map to
    the same base capability a Keycloak role would grant, so both providers are
    interchangeable. Unset (default) means a group name IS its capability
    (identity mapping) — zero-config when group/role names already match."""

    # --- Knowledge Graph process identity ---

    kg_auth_token_ref: str | None = Field(default=None, alias="KG_AUTH_TOKEN_REF")
    """Runtime secret reference resolving to the graph process JWT.

    It scopes engine bootstrap and stdio tool calls and is mutually exclusive
    with ``KG_IDENTITY_OAUTH2``. Raw tokens are never valid AgentConfig material.
    """

    kg_identity_oauth2: dict[str, Any] | None = Field(
        default=None, alias="KG_IDENTITY_OAUTH2"
    )
    """OAuth2 client-credentials block used to acquire a short-lived graph
    process JWT. Client secrets must be runtime secret references."""

    # --- Fleet events webhook ingress (CONCEPT:AU-OS.config.fleet-event-ingress) ---

    fleet_events_token_ref: str | None = Field(
        default=None, alias="FLEET_EVENTS_TOKEN_REF"
    )
    """Secret-provider reference for the fleet-events webhook credential."""

    # --- Gateway middle-tier hardening (CONCEPT:AU-OS.observability.no-op-without-metrics) ---

    gateway_metrics: bool = Field(default=True, alias="GATEWAY_METRICS")
    """Expose Python-tier Prometheus metrics on the gateway: a pure-ASGI
    middleware recording ``agent_utilities_gateway_*`` series (request totals,
    duration histogram, in-flight gauge, rate-limit/breaker counters) plus a
    ``GET /metrics`` endpoint (exempt from auth — scrapers cannot mint JWTs).
    Requires the optional ``metrics`` extra (``prometheus-client``); without it
    the middleware degrades to a no-op and ``/metrics`` returns a placeholder."""

    gateway_rate_limit: float = Field(default=0.0, alias="GATEWAY_RATE_LIMIT")
    """Per-tenant sustained request rate (requests/second) enforced by the
    gateway token-bucket middleware. ``0`` (default) disables rate limiting.
    Bucket key: ActorContext tenant → authenticated actor id → client IP.
    Buckets are in-memory and PER-PROCESS: with N workers/replicas the
    effective limit is N× this value (see docs/architecture/gateway_scaling.md)."""

    gateway_rate_burst: float = Field(default=0.0, alias="GATEWAY_RATE_BURST")
    """Token-bucket burst capacity (max requests served instantly from a full
    bucket). ``0`` (default) derives 2× ``GATEWAY_RATE_LIMIT``."""

    gateway_workers: int = Field(default=1, alias="GATEWAY_WORKERS")
    """Number of gateway worker processes. Default ``1`` preserves the
    single-process behaviour (in-process KG daemon, one event loop). With
    ``>1`` the server pre-forks workers sharing one listen socket; exactly ONE
    worker wins the KG host flock and runs the consolidated daemon/ticks while
    the rest self-heal to clients. Metrics and rate-limit buckets are
    per-worker. Ignored when the terminal UI is enabled or under pytest."""

    engine_breaker_threshold: int = Field(default=5, alias="ENGINE_BREAKER_THRESHOLD")
    """Consecutive engine connect/timeout failures before the epistemic-graph
    client circuit breaker opens (fast, typed ``EngineCircuitOpenError``
    instead of hammering a dead engine). ``0`` disables the breaker."""

    engine_breaker_cooldown: float = Field(
        default=15.0, alias="ENGINE_BREAKER_COOLDOWN"
    )
    """Seconds an open engine circuit breaker waits before allowing a single
    half-open probe; the probe's outcome closes or re-opens the circuit."""

    # --- MCP multiplexer child resilience (CONCEPT:AU-ECO.mcp.profile-differences-from-client) ---

    mcp_child_max_concurrency: int = Field(
        default=8, ge=1, le=128, alias="MCP_CHILD_MAX_CONCURRENCY"
    )
    """Maximum in-flight tool calls per multiplexer child server. Excess calls
    queue (bounded by ``MCP_CHILD_QUEUE_TIMEOUT``) instead of piling onto the
    child unbounded. Per-server override: the ``max_concurrency`` key on the
    server's ``mcp_config.json`` entry. The limit cannot be disabled."""

    mcp_child_queue_timeout: float = Field(
        default=30.0, ge=0.001, le=300.0, alias="MCP_CHILD_QUEUE_TIMEOUT"
    )
    """Seconds a tool call may wait for a free per-child concurrency slot
    before failing with the typed ``MCPChildBusyError`` (no silent hangs).
    Per-server override: the ``queue_timeout`` key on the server entry."""

    mcp_child_pool_size: int = Field(
        default=1, ge=1, le=64, alias="MCP_CHILD_POOL_SIZE"
    )
    """Session-pool size for remote (streamable-http/SSE) multiplexer
    children: N independent connections per child, round-robin dispatched,
    enabling parallel in-flight calls. Default 1 preserves the historical
    one-connection resource profile. Stdio children are single-pipe and
    always keep exactly one session. Per-server override: the ``pool_size``
    key on the server entry."""

    mcp_child_max_restarts: int = Field(
        default=5, ge=0, le=100, alias="MCP_CHILD_MAX_RESTARTS"
    )
    """How many automatic restarts a crashed multiplexer child may consume
    within ``MCP_CHILD_RESTART_WINDOW`` before it is marked ``failed`` (calls
    then fail fast with the typed ``MCPChildUnavailableError`` instead of
    retry-looping forever). ``0`` disables auto-restart entirely."""

    mcp_child_restart_window: float = Field(
        default=300.0, ge=0.001, le=86_400.0, alias="MCP_CHILD_RESTART_WINDOW"
    )
    """Sliding window (seconds) over which ``MCP_CHILD_MAX_RESTARTS`` is
    counted. Restarts older than the window are forgiven, so a child that
    crashes rarely keeps restarting indefinitely while a crash-looping child
    is parked as ``failed``."""

    mcp_child_breaker_threshold: int = Field(
        default=5, ge=0, le=100, alias="MCP_CHILD_BREAKER_THRESHOLD"
    )
    """Consecutive transport failures/timeouts on one multiplexer child
    before its circuit breaker opens (fast, typed
    ``MCPChildCircuitOpenError`` instead of hammering a dead child). ``0``
    disables the breaker. Per-server override: ``breaker_threshold``."""

    mcp_child_breaker_cooldown: float = Field(
        default=15.0, ge=0.001, le=3_600.0, alias="MCP_CHILD_BREAKER_COOLDOWN"
    )
    """Seconds an open per-child circuit breaker waits before allowing a
    single half-open probe call; the probe's outcome closes or re-opens the
    circuit. Per-server override: ``breaker_cooldown``."""

    # --- Embedded MCP fleet discovery (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog) ---

    # GraphOS has one strict-current fleet posture: its own tools and the fleet
    # meta-tools are registered at boot, while child servers are mounted lazily.
    # There is deliberately no alternate eager/standalone fleet posture.

    mcp_dynamic_top_k: int = Field(default=8, alias="MCP_DYNAMIC_TOP_K")
    """Default number of ranked tool candidates ``find_tools`` returns when the
    caller does not specify ``top_k``. Kept small so the discovery result is
    itself cheap to read; callers can request more explicitly."""

    # --- OIDC / OAuth 2.0 Delegation (CONCEPT:AU-ECO.messaging.native-backend-abstraction) ---

    mcp_client_auth: Literal["none", "oidc-client-credentials", "basic"] = Field(
        default="none", alias="MCP_CLIENT_AUTH"
    )
    """Outbound MCP child-auth mode. Enabled modes fail closed when incomplete."""

    oidc_config_url: str | None = Field(default=None, alias="OIDC_CONFIG_URL")
    """OIDC discovery URL (e.g. https://idp.example.com/.well-known/openid-configuration).
    Works with any OIDC-compliant Identity Provider."""

    oidc_client_id: str | None = Field(default=None, alias="OIDC_CLIENT_ID")
    """OAuth 2.0 client ID registered with the Identity Provider."""

    oidc_client_secret_ref: str | None = Field(
        default=None, alias="OIDC_CLIENT_SECRET_REF"
    )
    """Runtime secret reference for the OAuth 2.0 client secret."""

    oidc_audience: str | None = Field(default=None, alias="OIDC_AUDIENCE")
    """Exact audience requested for outbound MCP child credentials."""

    oidc_issuer: str | None = Field(default=None, alias="OIDC_ISSUER")
    """Explicit issuer used for provider-neutral token-endpoint discovery."""

    oidc_token_url: str | None = Field(default=None, alias="OIDC_TOKEN_URL")
    """Explicit token endpoint; when absent, discovery uses ``OIDC_ISSUER``."""

    oidc_scope: str | None = Field(default=None, alias="OIDC_SCOPE")
    """Optional space-separated scopes for outbound MCP child credentials."""

    mcp_basic_auth_username: str | None = Field(
        default=None, alias="MCP_BASIC_AUTH_USERNAME"
    )
    """Runtime username for outbound MCP HTTP Basic authentication."""

    mcp_basic_auth_password_ref: str | None = Field(
        default=None, alias="MCP_BASIC_AUTH_PASSWORD_REF"
    )
    """Runtime secret reference for outbound MCP HTTP Basic authentication."""

    oidc_tls_profile: str | None = Field(default=None, alias="OIDC_TLS_PROFILE")
    """Named runtime TLS profile used for discovery, JWKS, and token calls."""

    oidc_tls_profile_ref: str | None = Field(default=None, alias="OIDC_TLS_PROFILE_REF")
    """Secret reference containing an OIDC-specific runtime TLS profile."""

    oidc_http_allowed_private_hosts: list[str] = Field(
        default_factory=list, alias="OIDC_HTTP_ALLOWED_PRIVATE_HOSTS"
    )
    """Exact private hostnames permitted for DNS-pinned OIDC/JWKS egress."""

    enable_delegation: bool = Field(default=False, alias="ENABLE_DELEGATION")
    """Enable OIDC token delegation (RFC 8693 Token Exchange) for downstream API calls."""

    delegation_audience: str | None = Field(default=None, alias="AUDIENCE")
    """Target audience for delegated tokens (e.g. the downstream API base URL)."""

    delegated_scopes: str = Field(default="api", alias="DELEGATED_SCOPES")
    """Space-separated scopes requested during token delegation."""

    # --- Vault Secrets Backend (CONCEPT:AU-OS.config.secrets-authentication) ---

    vault_url: str | None = Field(default=None, alias="SECRETS_VAULT_URL")
    """HashiCorp Vault URL for the secrets backend."""

    vault_mount: str = Field(default="secret", alias="SECRETS_VAULT_MOUNT")
    """Vault KV v2 mount point."""

    vault_auth_method: str = Field(default="auto", alias="VAULT_AUTH_METHOD")
    """Vault auth method: 'oidc', 'approle', 'token', 'kubernetes', 'auto'."""

    vault_auth_mount: str = Field(default="jwt", alias="VAULT_AUTH_MOUNT")
    """Vault auth method mount path.  Supports custom mounts
    (e.g. 'oidc', 'jwt', 'my-okta-auth')."""

    vault_role: str | None = Field(default=None, alias="VAULT_ROLE")
    """Vault role name for OIDC/JWT or Kubernetes login."""

    vault_path_prefix: str | None = Field(default=None, alias="VAULT_PATH_PREFIX")
    """Path prefix within the KV v2 mount (e.g. 'agents/mcp/')."""

    allowed_origins: str | None = Field(default=None, alias="ALLOWED_ORIGINS")
    """Comma-separated list of allowed CORS origins. Unset disables CORS."""

    cors_allow_credentials: bool = Field(default=False, alias="CORS_ALLOW_CREDENTIALS")
    """Allow browser credentials for explicitly enumerated CORS origins.

    Wildcard origins are rejected when this option is enabled.
    """

    allowed_hosts: str | None = Field(default=None, alias="ALLOWED_HOSTS")
    """Comma-separated Host-header allowlist for ``TrustedHostMiddleware``.

    Non-loopback listeners must configure this explicitly.
    """

    server_tls_certfile: str | None = Field(default=None, alias="SERVER_TLS_CERTFILE")
    server_tls_keyfile: str | None = Field(default=None, alias="SERVER_TLS_KEYFILE")
    server_tls_terminated: bool = Field(default=False, alias="SERVER_TLS_TERMINATED")
    server_trusted_proxy_cidrs: list[str] = Field(
        default_factory=list, alias="SERVER_TRUSTED_PROXY_CIDRS"
    )
    server_max_connections: int = Field(
        default=256, ge=1, le=10_000, alias="SERVER_MAX_CONNECTIONS"
    )

    runtime_workspace_images: list[str] = Field(
        default_factory=list,
        alias="RUNTIME_WORKSPACE_IMAGES",
    )
    runtime_workspace_network: Literal["none", "bridge"] = Field(
        default="none", alias="RUNTIME_WORKSPACE_NETWORK"
    )
    runtime_max_sessions: int = Field(
        default=16, ge=1, le=256, alias="RUNTIME_MAX_SESSIONS"
    )
    runtime_session_ttl_seconds: int = Field(
        default=3_600, ge=60, le=86_400, alias="RUNTIME_SESSION_TTL_SECONDS"
    )
    runtime_max_events: int = Field(
        default=1_000, ge=16, le=10_000, alias="RUNTIME_MAX_EVENTS"
    )

    routing_strategy: str = Field(default="hybrid", alias="ROUTING_STRATEGY")
    graph_persistence_type: str = Field(default="file", alias="GRAPH_PERSISTENCE_TYPE")
    graph_db_connection_profile_ref: str | None = Field(
        default=None,
        alias="GRAPH_DB_CONNECTION_PROFILE_REF",
    )
    """Runtime secret reference resolving to the default graph connection JSON.

    Endpoint, database, identity, credential, TLS, and local-path material never
    enter AgentConfig directly.
    """
    # Optional projections (CONCEPT:AU-KG.backend.mirror-health-repair). The
    # epistemic-graph engine is always the operational authority. Naming one or
    # more mirror connections automatically enables lossless fan-out; external
    # stores remain write projections and never enter the read/ack authority path.
    # Targets resolve against ``kg_connections``
    # (CONCEPT:AU-KG.backend.multi-connection-registry), so endpoint and credential
    # material stays in referenced connection profiles.
    graph_mirror_targets: list[str] | None = Field(
        default=None, alias="GRAPH_MIRROR_TARGETS"
    )
    # Continuous Stardog mirroring (CONCEPT:AU-KG.backend.continuous-stardog-mirror). OFF by default:
    # Stardog is used for EXPLICIT, on-demand per-source push/pull (``stardog_sync``), NOT a
    # live write mirror. Set this to opt IN to continuous mirroring — the engine authority
    # then fans every write out to Stardog (as a first-class fanout mirror, via the same
    # durable outbox + replay machinery), partitioned into ``urn:source:<system>`` named
    # graphs. A Stardog connection must be configured
    # (``kg_connections`` ``stardog`` entry / ``STARDOG_*`` env). This is the ONE switch —
    # no need to also list ``stardog`` in ``GRAPH_MIRROR_TARGETS``.
    continuous_stardog_mirror: bool = Field(
        default=False, alias="CONTINUOUS_STARDOG_MIRROR"
    )
    # Multi-SoR asset mirror (CONCEPT:AU-KG.ingest.enterprise-source-extractor). The
    # canonical :Asset/CI model lives in the graph; each named CMDB sink is a
    # PROJECTION of it. This list (like ``CONTINUOUS_STARDOG_MIRROR`` for Stardog)
    # selects which sinks the ``asset-mirror`` pass fans out to — a subset of
    # ``servicenow``/``erpnext``/``egeria``/``twenty``. Empty by default, so every
    # sink (ServiceNow included) stays available-but-inert until opted in. A listed
    # sink still enforces its own ``<SINK>_ENABLE_WRITE`` for live writes and is
    # dry-run-first, so listing it only enables report-only previews by default.
    asset_mirror_targets: list[str] | None = Field(
        default=None, alias="ASSET_MIRROR_TARGETS"
    )
    # Ingest task-queue selection (CONCEPT:AU-KG.backend.selectable-queue-backend): which durable queue carries
    # KG ingest tasks. Unset (default) = auto: ``postgres`` when ``state_db_uri``
    # is set, else the zero-infra per-host ``sqlite`` file — mirroring the
    # state-externalization convention. An EXPLICIT value is a hard contract:
    # ``kafka``/``postgres`` raise at startup when unreachable (never a silent
    # SQLite degrade). Values: sqlite | postgres | kafka.
    task_queue_backend: str | None = Field(default=None, alias="TASK_QUEUE_BACKEND")
    # Partition count ensured on the ``kg_tasks`` topic at startup when the
    # kafka task queue is selected (CONCEPT:AU-KG.backend.keyed-ingest-partitions). Grow-only: raising it adds
    # partitions idempotently; an existing topic is NEVER shrunk. Bounds the
    # max parallelism of the ``kg-ingest`` consumer group.
    kg_tasks_partitions: int = Field(default=6, alias="KG_TASKS_PARTITIONS")
    # Partitions ensured on the ``agent_turns`` topic when the kafka transport
    # carries dispatched agent turns (CONCEPT:AU-ORCH.dispatch.queue-agent-dispatch). Grow-only, like
    # KG_TASKS_PARTITIONS. Bounds agent-dispatch consumer-group parallelism —
    # i.e. how many sessions can execute concurrently across the worker fleet.
    agent_turns_partitions: int = Field(default=6, alias="AGENT_TURNS_PARTITIONS")
    # Total queued agent-turn bound. Admission is atomic on SQLite/Postgres and
    # fail-closed against authoritative consumer lag on Kafka. This prevents a
    # stalled worker fleet from accepting unbounded durable work.
    agent_dispatch_max_depth: int = Field(
        default=100_000, ge=1, alias="AGENT_DISPATCH_MAX_DEPTH"
    )
    agent_dispatch_claim_ttl_s: float = Field(
        default=120.0,
        ge=60.0,
        le=300.0,
        alias="AGENT_DISPATCH_CLAIM_TTL_S",
    )
    """Renewable dispatch-claim lifetime in seconds.

    The upper bound is the workload contract's recovery-time objective: after a
    worker dies, another worker can reclaim its turn within five minutes without
    an operator-specific override."""
    agent_dispatch_renew_interval_s: float = Field(
        default=30.0,
        ge=1.0,
        le=30.0,
        alias="AGENT_DISPATCH_RENEW_INTERVAL_S",
    )
    """Maximum interval between dispatch lease renewals.

    AgentConfig's bounds make this strictly shorter than every valid dispatch
    claim TTL. Tests that inject a smaller lease use one third of that injected
    TTL so the same invariant still holds."""
    # AgentBus delivery/wakeup plane (CONCEPT:AU-ECO.bus.partitioned-log-delivery, AU-P1-2): which durable
    # partitioned log carries high-volume bus message BODIES (the semantic
    # roster/subscription registry always stays in the KG). The selected
    # backend is required and fails closed when unavailable. Values: engine | kafka.
    agent_bus_log_backend: str = Field(default="engine", alias="AGENT_BUS_LOG_BACKEND")
    # Partitions ensured on the Kafka bus topics (``agent_bus_direct`` /
    # ``agent_bus_topic``) when the Kafka bus-log backend is selected. Grow-only.
    agent_bus_partitions: int = Field(
        default=6, ge=1, le=1024, alias="AGENT_BUS_PARTITIONS"
    )
    agent_bus_max_consumers: int = Field(
        default=32, ge=2, le=4096, alias="AGENT_BUS_MAX_CONSUMERS"
    )
    agent_bus_max_depth: int = Field(default=100_000, ge=1, alias="AGENT_BUS_MAX_DEPTH")
    agent_bus_max_topic_subscribers: int = Field(
        default=1024, ge=1, alias="AGENT_BUS_MAX_TOPIC_SUBSCRIBERS"
    )
    agent_bus_delivery_lease_seconds: int = Field(
        default=300, ge=30, le=3600, alias="AGENT_BUS_DELIVERY_LEASE_SECONDS"
    )
    # Durable-state externalization (CONCEPT:AU-OS.state.unified-durable-state-externalization): ONE flag selects where
    # session/turn/fleet metadata and queue delivery state live. Unset keeps the
    # zero-infra per-host SQLite support stores; a postgresql:// URI moves those
    # planes onto one shared psycopg pool. Native WorkItem checkpoints remain in
    # epistemic-graph and are not selected by this setting.
    state_db_uri: str | None = Field(default=None, alias="STATE_DB_URI")
    # Max connections in the shared state-store pool (min is always 1).
    state_db_pool_size: int = Field(default=8, alias="STATE_DB_POOL_SIZE")
    # Golden-loop breadth ingest roots (CONCEPT:AU-KG.query.vendor-agnostic-traversal): comma-separated paths the
    # one-shot ``loop`` cycle (and the 60-min daemon) auto-ingests — OSS
    # libraries and code repos — so evolution runs end-to-end with no manual
    # ingest. Deployment-specific (set in ``.env``); empty ⇒ breadth is a no-op.
    kg_breadth_library_roots: str = Field(default="", alias="KG_BREADTH_LIBRARY_ROOTS")
    kg_breadth_repo_roots: str = Field(default="", alias="KG_BREADTH_REPO_ROOTS")
    # Loop-engine (autonomous research) parameters. Typed config replaces the
    # scattered bare env reads (CONCEPT:AU-KG.query.vendor-agnostic-traversal). The loop-enable + stage flags are
    # KG_LOOP*; the separate governed auto-merge gate keeps KG_GOLDEN_AUTO_MERGE.
    kg_loop: bool = Field(default=False, alias="KG_LOOP")
    kg_loop_distill: bool = Field(default=False, alias="KG_LOOP_DISTILL")
    # Opt-in (external scholarx calls cost): the intake stage discovers + ingests
    # research papers (LLM concept/fact extraction) at the front of the unified
    # research-intelligence cycle, so the matcher then compares the fresh papers
    # against the ecosystem. Caller-supplied ``papers`` always run regardless.
    # (CONCEPT:AU-KG.research.research-intelligence-loop)
    kg_loop_discover: bool = Field(default=False, alias="KG_LOOP_DISCOVER")
    # On by default: the breadth stage auto-ingests the ecosystem so ``assimilate``
    # has the codebase capability map to compare research against. With no
    # KG_BREADTH_* roots set it self-configures from the XDG workspace.yml, so the
    # default is zero-config; content-addressed ingest makes re-runs cheap. Set
    # KG_LOOP_BREADTH=0 to opt out. (CONCEPT:AU-KG.query.vendor-agnostic-traversal)
    kg_loop_breadth: bool = Field(default=True, alias="KG_LOOP_BREADTH")
    kg_loop_standardize: bool = Field(default=False, alias="KG_LOOP_STANDARDIZE")
    # Discovery-flywheel mining pass (CONCEPT:AU-KG.evolution.mining-flywheel) — runs the
    # engine's graph_mine (associate/anomaly) + graph_learn (fit/predict) surfaces over
    # the KG's concept/capability/article nodes each cycle, writing back typed
    # :AssociationRule/:Anomaly/:PredictedEdge nodes for the evolution flywheel to
    # consume (propose-only — never auto-merges). Default ON: each sub-step is
    # independently best-effort and degrades to an empty/no-op result on a
    # no-mining engine build, so it's safe to leave on everywhere.
    kg_loop_mine_discovery: bool = Field(default=True, alias="KG_LOOP_MINE_DISCOVERY")
    # Confidence propagation + light TMS over Belief nodes, workstream C2
    # (CONCEPT:AU-KG.maintenance.confidence-propagation-belief-revision) —
    # recomputes every ``Belief`` node's confidence from fresh
    # ``ContradictionDetector`` friction plus its already-recorded
    # support/contradiction edges, persisting each outcome as a
    # ``:BeliefRevisionProposal`` (propose-only; never mutates the live belief).
    # Default ON: degrades to a no-op ``skipped`` result with fewer than 2 Belief
    # nodes, so it's safe to leave on everywhere.
    kg_loop_belief_revision: bool = Field(default=True, alias="KG_LOOP_BELIEF_REVISION")
    # Insight Engine closed loop, workstream C4 (CONCEPT:AU-KG.evolution.insight-engine-closed-loop):
    # mined findings (:AssociationRule/:Anomaly/:PredictedEdge, from mine_discovery)
    # above CandidateInsight's confidence floor become reviewable ClaimNodes,
    # EvidenceBundle-packaged, run through the EXISTING promotion-governance +
    # capability-ratchet stack, then gated by action_policy.decide(kind=
    # "promote_mined_claim") — default ON: the stage is itself propose-only
    # (persisting a "proposal" Claim is safe) and the shipped ActionPolicy default
    # for promote_mined_claim is approval_required, so leaving this on everywhere
    # never auto-promotes anything by itself.
    kg_loop_insight_validation: bool = Field(
        default=True, alias="KG_LOOP_INSIGHT_VALIDATION"
    )
    # X3 — opt-in autonomy tier for the Insight Engine (CONCEPT:AU-KG.evolution.insight-engine-closed-loop).
    # OFF by default: even when this is True, a mined claim only auto-promotes
    # if BOTH action_policy.decide(kind="promote_mined_claim") allows (shipped
    # default: approval_required — never allows out of the box) AND the
    # promotion-governance validator (SHACL + capability ratchet + regression
    # gate + constitution rules) is valid. Turning this on wires the EXISTING
    # GovernedAutoMerger + capability_ratchet monotonic-improvement guarantee
    # onto claim promotion for an operator who has ALSO relaxed the
    # promote_mined_claim policy tier — a deliberate two-key turn, not a
    # single flag that silently starts auto-promoting mined claims.
    kg_insight_autonomy: bool = Field(default=False, alias="KG_INSIGHT_AUTONOMY")
    # CONCEPT:AU-OS.config.autonomous-spec-develop-off — autonomous spec→develop. OFF by default = review-first: a
    # distilled spec is persisted as a :SpecProposal in ``pending_review`` and HOLDS
    # for Claude/human approval (graph_loops action=review) before any develop Loop
    # is created. Turning this ON lets the 24/7 loop auto-advance specs through the
    # ``spec_promotion`` ActionPolicy gate (still approval_required by default, so it
    # only auto-develops where an operator has explicitly relaxed that tier).
    kg_loop_auto_develop: bool = Field(default=False, alias="KG_LOOP_AUTO_DEVELOP")
    # DANGEROUS development-only host execution for local validation.
    # OFF by default: graph-carried validation commands otherwise remain data and
    # cannot create host subprocesses. Prefer injecting a governed sandbox runner.
    kg_loop_allow_host_validation: bool = Field(
        default=False, alias="KG_LOOP_ALLOW_HOST_VALIDATION"
    )
    kg_loop_host_validation_executables: str = Field(
        default="pytest,ruff,mypy,pyright,nox,tox,cargo,go",
        alias="KG_LOOP_HOST_VALIDATION_EXECUTABLES",
    )
    # RLM model-code execution boundaries. Disabled means disabled: threshold
    # auto-routing is a separate explicit opt-in, and the in-process executor is
    # never a secure fallback.
    enable_rlm: bool = Field(default=False, alias="ENABLE_RLM")
    rlm_auto_trigger: bool = Field(default=False, alias="RLM_AUTO_TRIGGER")
    rlm_sandbox: str = Field(default="auto", alias="RLM_SANDBOX")
    rlm_container_image_ref: str | None = Field(
        default=None, alias="RLM_CONTAINER_IMAGE_REF"
    )
    rlm_container_memory: str = Field(default="512m", alias="RLM_CONTAINER_MEMORY")
    rlm_container_cpus: float = Field(
        default=1.0, ge=0.1, le=16.0, alias="RLM_CONTAINER_CPUS"
    )
    rlm_container_pids_limit: int = Field(
        default=256, ge=16, le=1024, alias="RLM_CONTAINER_PIDS_LIMIT"
    )
    rlm_container_timeout_seconds: float = Field(
        default=120.0,
        ge=1.0,
        le=600.0,
        alias="RLM_CONTAINER_TIMEOUT_SECONDS",
    )
    # Closed-loop agent mining, workstream C6 (CONCEPT:AU-KG.evolution.insight-engine-closed-loop):
    # mines Episode/OutcomeEvaluation/ToolCall provenance for repeated FAILURE
    # tool-call sequences (``trace_pattern_miner``) and feeds each pattern through
    # the SAME CandidateInsight→Claim→Validation→Action-gate pipeline C4 uses —
    # default ON: the stage is itself propose-only (persisting a "proposal" Claim
    # is safe) and the shipped ActionPolicy default for route_policy_update is
    # approval_required, so leaving this on everywhere never auto-applies a
    # routing/prompt/tool change or records an OutcomeRouter reward by itself.
    kg_loop_trace_mining: bool = Field(default=True, alias="KG_LOOP_TRACE_MINING")
    kg_golden_auto_merge: bool = Field(default=False, alias="KG_GOLDEN_AUTO_MERGE")
    kg_golden_merge_threshold: float | None = Field(
        default=None, alias="KG_GOLDEN_MERGE_THRESHOLD"
    )
    # Evolution→branch bridge (CONCEPT:AU-AHE.harness.evolution-branch-bridge): root directory the
    # LocalBranchPublisher creates fresh git worktrees under when publishing a
    # promoted proposal as a reviewable local branch. Empty (default) resolves
    # to ``data_dir()/evolution_worktrees`` — NEVER the canonical checkout's
    # working tree.
    evolution_worktree_root: str = Field(default="", alias="EVOLUTION_WORKTREE_ROOT")
    kg_loop_interval: float = Field(default=3600.0, alias="KG_LOOP_INTERVAL")
    kg_loop_topics: int = Field(default=5, alias="KG_LOOP_TOPICS")
    # --- Harness-enforced loop-exit conditions (CONCEPT:AU-AHE.harness.
    # loop-exit-conditions). Each is a real, enforced ``run_loop`` exit to a
    # distinct terminal status; safe defaults keep them on without over-firing. ---
    # Exit 5 NO PROGRESS: terminate ``stalled`` when the last N iterations produce
    # an identical (status, output, checkpoint) signature. 0/1 disables.
    kg_loop_no_progress_window: int = Field(
        default=3, alias="KG_LOOP_NO_PROGRESS_WINDOW"
    )
    # Exit 7 ERROR THRESHOLD: terminate ``error_threshold_exceeded`` after N
    # consecutive non-terminal step failures (reset to 0 on any progress). 0
    # disables tripping. Mirrors the engine breaker's 3-5 threshold band.
    kg_loop_max_consecutive_failures: int = Field(
        default=3, alias="KG_LOOP_MAX_CONSECUTIVE_FAILURES"
    )
    # Exit 4 WALL CLOCK: overall wall-clock budget (seconds) for a whole
    # ``run_loop``, checked every iteration in the while-condition independent of
    # the per-substep timeouts. 0 = no overall deadline (only the turn cap /
    # per-substep timeouts apply).
    kg_loop_max_duration_s: float = Field(
        default=0.0, alias="KG_LOOP_MAX_DURATION_S"
    )
    # Exit 1 GOAL MET: rubric-score floor (0..1) at/above which a measured goal
    # evaluation counts as a real pass for research/skill loops.
    kg_loop_goal_eval_threshold: float = Field(
        default=0.7, alias="KG_LOOP_GOAL_EVAL_THRESHOLD"
    )
    # Exit 1 GOAL MET: build the default rubric/LLM-judge goal evaluator for
    # research/skill loops (it degrades to trusting the callee when no model
    # endpoint is reachable, so this is safe to leave on).
    kg_loop_goal_eval_enabled: bool = Field(
        default=True, alias="KG_LOOP_GOAL_EVAL_ENABLED"
    )
    # CONCEPT:AU-KG.research.scholarx-rss-research-feed — ScholarX RSS research-feed loop that grades and fetches new papers.
    # A recurring schedule
    # that grades incoming RSS items (keyword taxonomy + ConceptMatcher novelty),
    # skips already-seen items, and enqueues a prioritized full-paper fetch+ingest
    # only for the high-graded ones. Default-ON (it no-ops safely without ScholarX
    # / network); set KG_RESEARCH_FEED=0 to disable the autonomous fetching.
    kg_research_feed: bool = Field(default=True, alias="KG_RESEARCH_FEED")
    kg_research_feed_interval: float = Field(
        default=1800.0, alias="KG_RESEARCH_FEED_INTERVAL"
    )
    # CONCEPT:AU-KG.ingest.rss-feed-connector — native RSS/Atom feed URLs (comma-separated) the zero-infra
    # `rss` connector ingests through the unified world-model gate. This is the SEED;
    # feeds added at runtime via graph_feeds live as :FeedSource nodes in the KG and
    # are swept too. Empty by default (a deployment opts in its feeds).
    kg_rss_feeds: str = Field(default="", alias="KG_RSS_FEEDS")
    # SAI factory self-specialization (CONCEPT:AU-AHE.harness.sai-controller). LLM-free, bounded, and
    # propose-only (it only persists a SaiFactoryCycle metrics node — nothing is
    # merged or deployed), and a *no-op when there is too little transition history*,
    # so it costs nothing on an idle system. Like the anomaly consumer, that makes it
    # safe to run natively ⇒ ON by default (set KG_SAI_FACTORY=0 to disable). The tick
    # grounds a learned world model in persisted WorldModelTransition history and
    # specializes its config; the same loop is reachable on demand via
    # graph_analyze(action='specialize') through the gateway.
    kg_sai_factory: bool = Field(default=True, alias="KG_SAI_FACTORY")
    kg_sai_factory_interval: float = Field(
        default=3600.0, alias="KG_SAI_FACTORY_INTERVAL"
    )
    # Failure-driven evolution — dormant with no Langfuse credentials and
    # configure-by-default once both credentials are runtime-injected. An
    # explicit false remains an opt-out. Pulls failures into propose-only
    # failure-gap topics the golden loop remediates
    # (CONCEPT:AU-AHE.harness.failure-evolution).
    kg_failure_evolution: bool = Field(default=False, alias="KG_FAILURE_EVOLUTION")
    kg_failure_evolution_interval: float = Field(
        default=3600.0, alias="KG_FAILURE_EVOLUTION_INTERVAL"
    )
    kg_failure_evolution_window: float = Field(
        default=86400.0, alias="KG_FAILURE_EVOLUTION_WINDOW"
    )
    kg_failure_regression_dataset: bool = Field(
        default=False, alias="KG_FAILURE_REGRESSION_DATASET"
    )
    # Optimization sweep (CONCEPT:AU-AHE.optimization.candidate-replaces-incumbent-only).
    # The provider-free native eg-program job is the one optimization path. The
    # scheduled twin of the `graph_evolution action=optimize_component` MCP action:
    # a daemon tick periodically optimizes the
    # self-supervised targets (extraction / concept_match / routing) and records
    # propose-only optimization trajectories (auto-apply stays gated, like
    # KG_GOLDEN_AUTO_MERGE).
    kg_optimization_enabled: bool = Field(default=True, alias="KG_OPTIMIZATION_ENABLED")
    kg_optimization_interval: float = Field(
        default=10800.0, alias="KG_OPTIMIZATION_INTERVAL"
    )
    """Seconds between native optimization sweeps (default 3h)."""
    # Agent-facing auto-apply gate (CONCEPT:AU-AHE.harness.hardening-transparency-surface) — the high-impact half of the
    # hardening loop. A native-optimized *system prompt* that beats baseline on its
    # agent's eval-corpus slice is only written to source (StructuredPrompt.save) when
    # this is True; otherwise the cycle is **propose-only / shadow** — it records a
    # queryable ``ProposedPromptChange`` for human/Claude review and leaves the live
    # prompt untouched. Default OFF (mirrors KG_GOLDEN_AUTO_MERGE): a prompt rewrite is
    # never silent. ``should_promote`` still gates even when this is enabled.
    kg_agent_auto_apply: bool = Field(default=False, alias="KG_AGENT_AUTO_APPLY")
    # PerformanceAnomaly consumer (CONCEPT:AU-AHE.optimization.performance-anomaly-consumer) — drains unconsumed
    # PerformanceAnomaly nodes into failure_gap topics for the golden loop.
    # Default ON: it is LLM-free, bounded, and propose-only (it writes topic
    # nodes; nothing merges without the AHE-3.20 governed auto-merge chain).
    kg_anomaly_consumer: bool = Field(default=True, alias="KG_ANOMALY_CONSUMER")
    # Interval (s) for the leaked-community-tenant GC tick (Phase A2).
    kg_tenant_gc_interval: float = Field(default=300.0, alias="KG_TENANT_GC_INTERVAL")

    kg_engine_pool_size: int = Field(default=8, alias="KG_ENGINE_POOL_SIZE")
    """Max warm per-tenant engine clients kept resident in one process
    (CONCEPT:AU-KG.sharding.elastic-over-kg-shard). The elastic layer over KG-2.58 shard routing: only the N
    most-recently-used tenant graphs stay warm; cold ones are evicted (the
    durable engine store keeps them) and re-hydrated on the next access.

    Default 8 (was 0): per-use construction built a fresh background thread +
    event loop + socket + ``tenants.create`` round-trip on EVERY engine access — a
    connection-setup storm under load. A small warm set amortizes that; a
    single-tenant deployment simply keeps its one graph warm. Eviction is LRU and
    bounded; set ``KG_ENGINE_POOL_DROP_ON_EVICT=1`` to also unload the evicted
    tenant's graph from the engine to reclaim resident memory. ``0`` disables
    per-use behavior."""

    kg_engine_pool_drop_on_evict: bool = Field(
        default=False, alias="KG_ENGINE_POOL_DROP_ON_EVICT"
    )
    """When a tenant is evicted from the engine pool (CONCEPT:AU-KG.sharding.elastic-over-kg-shard), also
    unload its named graph from the engine process to reclaim resident memory
    (``GraphComputeEngine.drop_graph``). **Only safe when engine persistence is
    durable**, so the graph can be re-hydrated on next access;
    otherwise the in-memory graph is lost. Default off (eviction only closes the
    client)."""

    kg_engine_tool_pool_size: int = Field(default=16, alias="KG_ENGINE_TOOL_POOL_SIZE")
    """Max warm raw ``SyncEpistemicGraphClient`` wire connections kept resident
    by the low-level ``engine_<domain>`` MCP tools (AU-P0-6,
    ``mcp/tools/engine_tools.py``). Distinct from :attr:`kg_engine_pool_size`
    (which bounds the resident *compute-engine* pool, a different resource): this one
    bounds the raw per-graph wire client the ``engine_*``/``graph_broker`` etc.
    low-level MCP surface connects with. Previously an unbounded ``dict``
    cached one client PER GRAPH forever — connection/thread/socket count grew
    without limit as graph cardinality grew. LRU-bounded to this size, evicting
    (and closing) the least-recently-used connection past capacity; ``<= 0``
    means passthrough (build + discard a fresh client per call, no caching)."""
    # Fuseki ontology distribution (CONCEPT:AU-KG.ontology.authoritative-tbox) — opt-in daemon tick that
    # pushes the bundled ontology modules to an Apache Jena Fuseki triplestore
    # (KG-2.6 distribution, operationalized). The *publish tick* stays off by
    # default (KG_FUSEKI_PUBLISH, below) because writing to Fuseki is an
    # opt-in action even when an endpoint is reachable; the *endpoint itself*
    # has no environment-specific default. Deployments inject the endpoint through
    # ``KG_FUSEKI_ENDPOINT`` or config.json. A deployment with no Fuseki never flips
    # ``kg_fuseki_publish`` on (see ``_auto_enable_from_dependencies`` below), and
    # the ``jena_fuseki`` backend is never selected unless requested.
    kg_fuseki_publish: bool = Field(default=False, alias="KG_FUSEKI_PUBLISH")
    kg_fuseki_endpoint: str = Field(
        default="",
        alias="KG_FUSEKI_ENDPOINT",
    )
    """THE canonical Fuseki endpoint (CONCEPT:AU-KG.ontology.authoritative-tbox) — the single field every
    Fuseki reader resolves through: the ontology-publish tick
    (``engine_tasks._tick_fuseki_publish``), ``publish_ontology_to_fuseki``'s
    endpoint fallback, the ``fuseki``-kind SPARQL smoke query
    (``database_environment.py``), and the ``jena_fuseki`` query backend
    (``backends/sparql/jena_fuseki_backend.py`` via ``create_backend``).
    Explicit callers may pass an ``endpoint=``/``jena_fuseki_url=`` argument
    to override this per call."""
    graph_fuseki_dataset: str = Field(default="agent_kg", alias="GRAPH_FUSEKI_DATASET")
    graph_fuseki_user: str | None = Field(default=None, alias="GRAPH_FUSEKI_USER")
    graph_fuseki_password_ref: str | None = Field(
        default=None, alias="GRAPH_FUSEKI_PASSWORD_REF"
    )
    """Runtime secret reference for the Fuseki basic-auth password."""
    kg_fuseki_publish_interval: float = Field(
        default=3600.0, alias="KG_FUSEKI_PUBLISH_INTERVAL"
    )
    # Execution-time workflow ontology gate (CONCEPT:AU-ORCH.execution.ontology-validation-execution-path) — SHACL-validate
    # a stored WorkflowDefinition before dispatch. Default ON: it is cheap,
    # LLM-free, and refuses malformed definitions before they burn agent runs.
    kg_workflow_shape_gate: bool = Field(default=True, alias="KG_WORKFLOW_SHAPE_GATE")

    # --- Autonomy control plane (CONCEPT:AU-OS.deployment.fleet-lifecycle-control — OS-5.27) ---

    fleet_mcp_url_template: str | None = Field(
        default=None, alias="FLEET_MCP_URL_TEMPLATE"
    )
    """Optional runtime-only MCP URL template containing ``{server}``.
    It supports both per-host and path-routed fleets without assuming a domain;
    endpoint identities are never included in durable reports or traces."""

    @field_validator("fleet_mcp_url_template", mode="before")
    @classmethod
    def _validate_fleet_mcp_url_template(cls, value: Any) -> str | None:
        return _validated_runtime_http_url(value, require_server_placeholder=True)

    action_policy_path: str = Field(default="", alias="ACTION_POLICY_PATH")
    """Path to the operational ActionPolicy YAML (CONCEPT:AU-OS.deployment.fleet-lifecycle-control). Empty
    (default) resolves to the shipped conservative policy
    (``deploy/action-policy.default.yml`` in a repo checkout, else the
    identical embedded default): every mutating action is approval_required,
    only no-op/diagnostic kinds run automatically. KG ``governance_rule``
    overrides (scope ``action_policy``) win over file rules either way."""

    fleet_reconciler: bool = Field(default=False, alias="FLEET_RECONCILER")
    """Opt-in desired-state fleet reconciler tick (CONCEPT:AU-OS.config.desired-state-fleet-reconciler). Diffs the
    fleet registry (+ optional desired-state override file) against the
    observed fleet and proposes convergence actions through the ActionPolicy
    decision point. Default False until a deployment wires real actuators —
    with the default dry-run actuator it only records intended actions."""

    fleet_reconciler_interval: float = Field(
        default=120.0, alias="FLEET_RECONCILER_INTERVAL"
    )
    """Seconds between fleet-reconciler ticks (leader-only)."""

    fleet_reconciler_max_actions: int = Field(
        default=5, alias="FLEET_RECONCILER_MAX_ACTIONS"
    )
    """Storm guard: max convergence actions processed per reconciler tick;
    further divergences are deferred to the next tick (CONCEPT:AU-OS.config.desired-state-fleet-reconciler)."""

    fleet_registry_path: str = Field(default="", alias="FLEET_REGISTRY_PATH")
    """Path to the fleet service registry YAML. Empty (default) resolves to
    ``deploy/mcp-fleet.registry.yml`` in a repo checkout."""

    fleet_desired_state_path: str = Field(default="", alias="FLEET_DESIRED_STATE_PATH")
    """Optional desired-state override YAML layered on the registry
    (per-service ``replicas`` / ``desired: running|stopped`` / ``version``)."""

    fleet_actuator: str = Field(default="dryrun", alias="FLEET_ACTUATOR")
    """Fleet actuator selection: ``dryrun`` (default — records intended
    actions as KG nodes + notifications, mutates nothing) or ``docker``
    (reference actuator via the docker CLI when available). Real
    Portainer/Swarm actuation is wired at deployment by registering a
    ``FleetActuator`` via ``orchestration.fleet_actuation.set_fleet_actuator``."""

    deploy_watch_window: float = Field(default=300.0, alias="DEPLOY_WATCH_WINDOW")
    """Default health-watch window (seconds) after a deploy/restart action
    (CONCEPT:AU-OS.config.health-gated-deploy-rollback): sustained green inside the window records success, an
    unhealthy observation triggers the policy-gated rollback/escalation."""

    deploy_watch_poll: float = Field(default=15.0, alias="DEPLOY_WATCH_POLL")
    """Seconds between health probes inside a deploy watch window."""

    fleet_autoscaler: bool = Field(default=False, alias="FLEET_AUTOSCALER")
    """Opt-in reactive replica autoscaler tick (CONCEPT:AU-OS.scaling.reactive-replica-autoscaling). For each
    service with a registry/override ``scaling:`` block: read its load signal,
    target-track a desired replica count inside the declared min/max bounds,
    and propose ``scale_service`` through the ActionPolicy gate + actuator
    seam (deploy-watched on scale-up). Default False; with the default
    dry-run actuator it records intent without mutating."""

    fleet_autoscaler_interval: float = Field(
        default=60.0, alias="FLEET_AUTOSCALER_INTERVAL"
    )
    """Seconds between autoscaler ticks (leader-only)."""

    scaling_prometheus_url: str | None = Field(
        default=None, alias="SCALING_PROMETHEUS_URL"
    )
    """Optional Prometheus base URL for autoscaling signals (CONCEPT:AU-OS.scaling.reactive-replica-autoscaling).
    Set → the autoscaler reads signals via instant HTTP queries
    (``PrometheusHttpProvider``); unset (default) → the zero-infra
    ``LocalMetricsProvider`` reads this process's own OS-5.23/KG-2.55 gauges.
    A custom provider injected via
    ``orchestration.scaling_signals.set_scaling_signal_provider`` wins over
    both."""

    @field_validator(
        "kg_failure_evolution",
        "kg_failure_regression_dataset",
        "kg_optimization_enabled",
        "kg_anomaly_consumer",
        "kg_fuseki_publish",
        "kg_workflow_shape_gate",
        "fleet_reconciler",
        "fleet_autoscaler",
        mode="before",
    )
    @classmethod
    def _coerce_failure_flags(cls, v: Any) -> bool:
        """Parse daemon/gate toggles via the canonical ``to_boolean``
        ({t,true,y,yes,1}) so ``"True"``/``"False"`` mcp_config strings behave
        consistently with the rest of the fleet's boolean flags."""
        return to_boolean(v)

    nats_url: str | None = Field(default=None, alias="NATS_URL")
    kafka_bootstrap_servers: str = Field(default="", alias="KAFKA_BOOTSTRAP_SERVERS")
    """Runtime-injected Kafka broker list (service DNS + raw TCP port). Every
    Kafka-consuming code path resolves through this field (``kafka_queue_backend.py``,
    ``bus_log.KafkaBusLog``, ``agent_dispatch.py``, ``ingest_worker.py``) rather than
    a hardcoded deployment hostname. HTTP ingress cannot proxy Kafka's raw
    TCP protocol, so it must never be used here. Selecting Kafka as an ACTIVE
    transport still requires an explicit selection elsewhere
    (``TASK_QUEUE_BACKEND=kafka`` or ``AGENT_BUS_LOG_BACKEND=kafka``); selected
    transports fail loudly when unreachable. No deployment hostname is embedded."""
    graph_compute_backend: str = Field(default="rust", alias="GRAPH_COMPUTE_BACKEND")
    graph_service_endpoints: list[str] | None = Field(
        default=None, alias="GRAPH_SERVICE_ENDPOINTS"
    )
    """Explicit engine coordinator contacts (comma-separated or JSON list).

    Any configured value is a connect-only topology: clients never start a
    local stand-in, even when an endpoint uses loopback or a Unix socket.
    Unset selects the packaged local engine lifecycle. Multiple entries are
    placement-authority contacts; callers never infer graph placement."""

    graph_raft_group_endpoints: dict[str, str] | None = Field(
        default=None, alias="GRAPH_RAFT_GROUP_ENDPOINTS"
    )
    """Optional JSON map from authoritative Raft group id to client endpoint.

    It is required only when a deployment exposes different groups through
    different endpoints. The recommended production topology has one stable
    coordinator in ``GRAPH_SERVICE_ENDPOINTS`` and needs no map. Values use the
    same ``unix://`` or ``tcp://`` endpoint syntax as the contact list.
    """

    @field_validator("graph_service_endpoints", mode="before")
    @classmethod
    def _coerce_endpoint_list(cls, v: Any) -> Any:
        """Accept comma-separated or JSON-encoded GRAPH_SERVICE_ENDPOINTS
        (CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw) via the canonical ``to_list`` so env wiring matches
        the rest of the fleet's list flags."""
        if v is None or isinstance(v, list):
            return v
        items = [str(e).strip() for e in to_list(v) if str(e).strip()]
        return items or None

    @field_validator("graph_raft_group_endpoints", mode="before")
    @classmethod
    def _coerce_group_endpoint_map(cls, v: Any) -> Any:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        if isinstance(v, dict):
            parsed = v
        elif isinstance(v, str):
            import json

            try:
                parsed = json.loads(v)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "GRAPH_RAFT_GROUP_ENDPOINTS must be a JSON object"
                ) from exc
        else:
            raise ValueError("GRAPH_RAFT_GROUP_ENDPOINTS must be a mapping")
        if not isinstance(parsed, dict):
            raise ValueError("GRAPH_RAFT_GROUP_ENDPOINTS must be a JSON object")
        result: dict[str, str] = {}
        for group, endpoint in parsed.items():
            group_text = str(group).strip()
            if not group_text.isdigit():
                raise ValueError("Raft group identifiers must be non-negative integers")
            endpoint_text = str(endpoint).strip()
            if not endpoint_text.startswith(("unix://", "tcp://", "tls://")):
                raise ValueError(
                    "Raft group endpoints require unix://, tcp://, or tls:// schemes"
                )
            if endpoint_text in {"unix://", "tcp://", "tls://"}:
                raise ValueError("Raft group endpoints must include an address")
            result[str(int(group_text))] = endpoint_text
        return result or None

    kg_connections: list[dict[str, Any]] | None = Field(
        default=None, alias="KG_CONNECTIONS"
    )
    """Declarative named graph connections (CONCEPT:AU-KG.backend.multi-connection-registry). A JSON list of
    backend specs, each ``{"name": <str>, "backend": <type>, ...}``. Any
    endpoint, database, identity, credential, TLS, or local-path field must be a
    runtime secret reference. These are registered into
    the multi-connection registry at first use so the SAME graph tools can target
    any one (``target=<name>``) or fan out to all (``target="all"``). The
    zero-infra default is fully preserved: unset → only the ambient ``default``
    connection exists. For Postgres, use ``"backend": "age"`` for native
    openCypher portability."""

    gitlab_instances: list[dict[str, Any]] | None = Field(
        default=None, alias="GITLAB_INSTANCES"
    )
    """GitLab instances to index into the KG (CONCEPT:AU-KG.backend.declared-columns-so-schema). A JSON list of
    ``{"name": <str>, "url": <str>, "token_ref": <runtime-secret-reference>}`` — the
    single source of truth shared by the agent-utilities GitLab indexer
    (``knowledge_graph/core/gitlab_indexer``) and the ``gitlab-api`` connector's
    instance registry, so one config drives multi-tenant indexing and API access.
    Raw ``token`` is not part of the current schema. Unset → falls back to the
    single-host process values ``GITLAB_URL``/``GITLAB_TOKEN``."""

    jira_instances: list[dict[str, Any]] | None = Field(
        default=None, alias="JIRA_INSTANCES"
    )
    """Jira instances to ingest into the KG (CONCEPT:AU-KG.compute.jira-first-class-delta). A JSON list of
    ``{"name": <str>, "server": <atlassian-mcp server name>, "project_keys": [<str>],
    "jql": <optional extra JQL>}`` — each is drained via the ``jira`` mcp_tool preset
    through its named ``atlassian-mcp`` server (which holds the credentials), so two
    Atlassian sites are two server entries + two instances. Unset → one synthetic
    instance over ``atlassian-mcp`` filtered by ``JIRA_PROJECT_KEYS``."""

    confluence_instances: list[dict[str, Any]] | None = Field(
        default=None, alias="CONFLUENCE_INSTANCES"
    )
    """Confluence instances to mirror into the KG (CONCEPT:AU-KG.compute.confluence-first-class-delta). A JSON list of
    ``{"name": <str>, "server": <atlassian-mcp server name>, "spaces": [<space-id>]}``
    — each space is drained via the ``confluence`` mcp_tool preset and ingested as
    full-text ``:ConfluencePage`` Documents. Unset → one synthetic instance over
    ``atlassian-mcp`` across ``CONFLUENCE_SPACE_IDS``."""

    plane_instances: list[dict[str, Any]] | None = Field(
        default=None, alias="PLANE_INSTANCES"
    )
    """Plane instances to ingest into the KG (CONCEPT:AU-KG.compute.plane-first-class-delta). A JSON list of
    ``{"name": <str>, "server": <plane-mcp server name>, "projects": [<project-id>]}``
    — each is drained via the ``plane`` mcp_tool preset through its named ``plane-mcp``
    server, so a SECOND Plane workspace is just a second server entry + instance. Unset
    → one synthetic instance over ``plane-mcp`` across ``PLANE_PROJECT_IDS``."""

    @field_validator(
        "gitlab_instances",
        "jira_instances",
        "confluence_instances",
        "plane_instances",
        mode="before",
    )
    @classmethod
    def _coerce_instance_list(cls, v: Any) -> Any:
        """Accept a JSON-encoded string or an already-parsed list for the
        ``*_INSTANCES`` multi-instance connector configs (CONCEPT:AU-KG.backend.declared-columns-so-schema/2.123-2.125)."""
        if v is None or isinstance(v, list):
            return v
        if isinstance(v, str):
            import json

            s = v.strip()
            if not s:
                return None
            try:
                parsed = json.loads(s)
            except Exception:
                return None
            return parsed if isinstance(parsed, list) else None
        return None

    @field_validator("graph_mirror_targets", mode="before")
    @classmethod
    def _coerce_graph_mirror_targets(cls, v: Any) -> Any:
        """Accept a JSON list, a comma-separated string, or a parsed list for
        GRAPH_MIRROR_TARGETS (CONCEPT:AU-KG.backend.mirror-health-repair)."""
        if v is None or isinstance(v, list):
            return v
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            if s.startswith("["):
                import json

                try:
                    parsed = json.loads(s)
                except Exception:
                    return None
                return parsed if isinstance(parsed, list) else None
            return [x.strip() for x in s.split(",") if x.strip()]
        return v

    @field_validator("kg_connections", mode="before")
    @classmethod
    def _coerce_kg_connections(cls, v: Any) -> Any:
        """Accept a JSON-encoded string or an already-parsed list for
        KG_CONNECTIONS (CONCEPT:AU-KG.backend.multi-connection-registry)."""
        if v is None or isinstance(v, list):
            return v
        if isinstance(v, str):
            import json

            s = v.strip()
            if not s:
                return None
            try:
                parsed = json.loads(s)
            except Exception:
                return None
            return parsed if isinstance(parsed, list) else None
        return None

    @field_validator("kg_connections")
    @classmethod
    def _validate_durable_kg_connections(cls, value: Any) -> Any:
        if value is None:
            return None
        from agent_utilities.knowledge_graph.core.connection_registry import (
            validate_persistable_connection_spec,
        )

        for entry in value:
            if not isinstance(entry, dict):
                raise ValueError("KG_CONNECTIONS entries must be objects")
            validate_persistable_connection_spec(entry)
        return value

    @field_validator("external_graph_connectors", mode="before")
    @classmethod
    def _coerce_external_graph_connectors(cls, value: Any) -> Any:
        if value in (None, ""):
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            import json

            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                raise ValueError(
                    "EXTERNAL_GRAPH_CONNECTORS must be a JSON list"
                ) from None
            if isinstance(parsed, list):
                return parsed
        raise ValueError("EXTERNAL_GRAPH_CONNECTORS must be a list")

    @field_validator("external_graph_connectors")
    @classmethod
    def _validate_external_graph_declaration_identities(
        cls, value: list[ExternalGraphConnectorConfig]
    ) -> list[ExternalGraphConnectorConfig]:
        aliases = [connector.source_alias for connector in value]
        names = [connector.name for connector in value]
        if (
            not all(aliases)
            or len(set(aliases)) != len(aliases)
            or not all(names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "EXTERNAL_GRAPH_CONNECTORS names and source_alias values must be unique"
            )
        return value

    @field_validator("provider_configs", mode="before")
    @classmethod
    def _coerce_provider_configs(cls, value: Any) -> Any:
        if value in (None, ""):
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, str):
            import json

            def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
                parsed: dict[str, Any] = {}
                for key, item in pairs:
                    if key in parsed:
                        raise ValueError("PROVIDER_CONFIGS contains duplicate keys")
                    parsed[key] = item
                return parsed

            try:
                parsed = json.loads(value, object_pairs_hook=reject_duplicates)
            except (TypeError, ValueError):
                raise ValueError(
                    "PROVIDER_CONFIGS must be a strict JSON object"
                ) from None
            if isinstance(parsed, dict):
                return parsed
        raise ValueError("PROVIDER_CONFIGS must be a bounded mapping")

    @field_validator("provider_configs")
    @classmethod
    def _validate_provider_config_identities(
        cls, value: dict[str, ProviderRuntimeProfile]
    ) -> dict[str, ProviderRuntimeProfile]:
        if len(value) > 256:
            raise ValueError("PROVIDER_CONFIGS must contain at most 256 profiles")
        for profile_name in value:
            if _NEUTRAL_ALIAS_RE.fullmatch(profile_name) is None:
                raise ValueError("PROVIDER_CONFIGS contains an invalid profile name")
        return value

    @field_validator("kg_identity_oauth2", mode="before")
    @classmethod
    def _validate_kg_identity_oauth2(cls, value: Any) -> dict[str, Any] | None:
        if value in (None, ""):
            return None
        if isinstance(value, str):
            import json

            try:
                value = json.loads(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("KG_IDENTITY_OAUTH2 must be a JSON object") from exc
        if not isinstance(value, dict):
            raise ValueError("KG_IDENTITY_OAUTH2 must be an object")
        return _validate_oauth2_block(value, "KG_IDENTITY_OAUTH2")

    @field_validator(
        "source_http_allowed_private_hosts",
        "source_http_allowed_redirect_hosts",
        "eunomia_allowed_private_hosts",
        "model_http_allowed_private_hosts",
        "oidc_http_allowed_private_hosts",
        "mcp_http_allowed_private_hosts",
        "qdrant_http_allowed_private_hosts",
    )
    @classmethod
    def _validate_http_host_allowlists(cls, value: list[str]) -> list[str]:
        """Accept only a small, exact, environment-owned hostname set."""
        if len(value) > 256:
            raise ValueError("HTTP host allow-lists may contain at most 256 entries")
        normalized: set[str] = set()
        for raw in value:
            host = str(raw).strip().lower().rstrip(".")
            try:
                host.encode("ascii")
            except UnicodeEncodeError as exc:
                raise ValueError(
                    "HTTP host allow-lists require ASCII hostnames"
                ) from exc
            if (
                not host
                or len(host) > 253
                or any(ord(character) < 33 for character in host)
                or any(character in host for character in "/@*?#[]")
            ):
                raise ValueError("HTTP host allow-lists require exact hostnames")
            try:
                ipaddress.ip_address(host)
            except ValueError:
                labels = host.split(".")
                if any(
                    not label
                    or len(label) > 63
                    or label.startswith("-")
                    or label.endswith("-")
                    or not all(
                        character.isalnum() or character == "-" for character in label
                    )
                    for label in labels
                ):
                    raise ValueError(
                        "HTTP host allow-lists require exact hostnames"
                    ) from None
            normalized.add(host)
        return sorted(normalized)

    @field_validator(
        "tls_profile_ref",
        "tls_profiles_ref",
        "tls_ca_bundle_ref",
        "tls_client_cert_ref",
        "tls_client_key_ref",
        "tls_client_key_password_ref",
        "tls_proxy_url_ref",
        "engine_tls_profile_ref",
        "oidc_client_secret_ref",
        "oidc_tls_profile_ref",
        "model_tls_profile_ref",
        "embedding_tls_profile_ref",
        "oauth2_token_tls_profile_ref",
        "eunomia_api_key_ref",
        "eunomia_tls_profile_ref",
        "vector_db_username_ref",
        "vector_db_password_ref",
        "postgres_tls_profile_ref",
        "qdrant_api_key_ref",
        "qdrant_tls_profile_ref",
        "mongodb_uri_ref",
        "mongodb_tls_profile_ref",
        "redis_connection_profile_ref",
        "redis_tls_profile_ref",
        "mcp_jwt_secret_ref",
        "kg_auth_token_ref",
        "mcp_basic_auth_password_ref",
        "mcp_metrics_token_ref",
        "messaging_alert_intake_token_ref",
        "mcp_static_tokens_ref",
        "fleet_events_token_ref",
        "graph_db_connection_profile_ref",
        "graph_fuseki_password_ref",
        "epistemic_graph_encryption_key_ref",
        "epistemic_graph_sqlite_transfer_root_ref",
        "epistemic_graph_backup_root_ref",
        "persistence_privacy_deny_terms_ref",
        "persistence_identity_hmac_key_ref",
        "memento_raw_encryption_key_ref",
        "rlm_container_image_ref",
        "langfuse_public_key_ref",
        "langfuse_secret_key_ref",
        "langfuse_persistence_hmac_key_ref",
        "otel_exporter_otlp_headers_ref",
        "otel_exporter_otlp_public_key_ref",
        "otel_exporter_otlp_secret_key_ref",
        "otel_tls_profile_ref",
        "langfuse_tls_profile_ref",
        "langfuse_ca_bundle_ref",
        "langfuse_client_cert_ref",
        "langfuse_client_key_ref",
        "langfuse_client_key_password_ref",
        "langfuse_proxy_url_ref",
        "permissions_signing_key_ref",
        "ontology_release_signing_private_key_ref",
    )
    @classmethod
    def _validate_runtime_secret_ref(cls, value: str | None) -> str | None:
        if value is None:
            return None
        rendered = str(value).strip()
        if not _RUNTIME_SECRET_REF_RE.fullmatch(rendered):
            raise ValueError("runtime-only material must use runtime secret refs")
        return rendered

    @field_validator("epistemic_graph_encryption_key_ref")
    @classmethod
    def _validate_engine_encryption_bootstrap_ref(cls, value: str | None) -> str | None:
        """Require an external bootstrap source for the engine data key.

        An engine-backed ``secret://`` value cannot unlock the same engine
        without a circular dependency. The current bootstrap protocol therefore
        accepts only process/runtime-file injection or an external vault.
        """

        if value is not None and not value.startswith(("env://", "vault://")):
            raise ValueError(
                "engine encryption requires an external runtime secret ref"
            )
        return value

    @field_validator("oidc_issuer", "oidc_token_url", mode="before")
    @classmethod
    def _validate_outbound_oidc_urls(cls, value: Any) -> str | None:
        return _validated_runtime_http_url(value)

    @field_validator(
        "oidc_client_id",
        "oidc_audience",
        "oidc_scope",
        "mcp_basic_auth_username",
    )
    @classmethod
    def _validate_outbound_identity_metadata(cls, value: str | None) -> str | None:
        if value is None:
            return None
        rendered = str(value).strip()
        if (
            not rendered
            or len(rendered) > 4_096
            or any(character in rendered for character in "\r\n\x00")
        ):
            raise ValueError("outbound identity metadata is invalid")
        return rendered

    @model_validator(mode="after")
    def _validate_kg_process_identity_source(self) -> "AgentConfig":
        if self.kg_auth_token_ref and self.kg_identity_oauth2:
            raise ValueError(
                "Configure only one graph process identity source: "
                "KG_AUTH_TOKEN_REF or KG_IDENTITY_OAUTH2"
            )
        return self

    @model_validator(mode="after")
    def _validate_otel_auth_source(self) -> "AgentConfig":
        pair = bool(self.otel_exporter_otlp_public_key_ref) == bool(
            self.otel_exporter_otlp_secret_key_ref
        )
        if not pair:
            raise ValueError(
                "OTLP public/secret key references must be configured together"
            )
        if self.otel_exporter_otlp_headers_ref and (
            self.otel_exporter_otlp_public_key_ref
            or self.otel_exporter_otlp_secret_key_ref
        ):
            raise ValueError(
                "Configure one OTLP authentication source: headers ref or key refs"
            )
        return self

    @field_validator("eunomia_remote_url", mode="before")
    @classmethod
    def _validate_eunomia_remote_url(cls, value: Any) -> str | None:
        return _validated_runtime_http_url(value)

    @field_validator("eunomia_policy_file")
    @classmethod
    def _validate_eunomia_policy_file(cls, value: str | None) -> str | None:
        if value is None:
            return None
        rendered = str(value).strip()
        if (
            not rendered
            or len(rendered) > 2_048
            or any(character in rendered for character in "\r\n\x00")
        ):
            raise ValueError("EUNOMIA_POLICY_FILE is invalid")
        return rendered

    @field_validator("runtime_workspace_images")
    @classmethod
    def _validate_runtime_workspace_images(cls, value: list[str]) -> list[str]:
        """Require a small exact allowlist; no implicit executable image exists."""
        if len(value) > 64:
            raise ValueError("RUNTIME_WORKSPACE_IMAGES may contain at most 64 entries")
        normalized: list[str] = []
        for raw in value:
            image = str(raw or "").strip()
            if (
                not image
                or len(image) > 512
                or any(character in image for character in "\r\n\x00 \t")
            ):
                raise ValueError("RUNTIME_WORKSPACE_IMAGES contains an invalid image")
            normalized.append(image)
        return list(dict.fromkeys(normalized))

    @field_validator("auth_jwt_algorithms")
    @classmethod
    def _validate_auth_jwt_algorithms(cls, value: list[str]) -> list[str]:
        allowed = {
            "RS256",
            "RS384",
            "RS512",
            "PS256",
            "PS384",
            "PS512",
            "ES256",
            "ES384",
            "ES512",
            "EdDSA",
        }
        normalized = list(dict.fromkeys(str(item).strip() for item in value))
        if (
            not normalized
            or len(normalized) > len(allowed)
            or not set(normalized) <= allowed
        ):
            raise ValueError(
                "AUTH_JWT_ALGORITHMS must use supported asymmetric algorithms"
            )
        return normalized

    @field_validator("engine_tls_server_name")
    @classmethod
    def _validate_engine_tls_server_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        rendered = str(value).strip()
        if (
            not rendered
            or len(rendered) > 253
            or any(character in rendered for character in "/\\@?#\r\n\t ")
        ):
            raise ValueError("ENGINE_TLS_SERVER_NAME is invalid")
        return rendered

    @field_validator("rlm_sandbox")
    @classmethod
    def _validate_rlm_sandbox(cls, value: str) -> str:
        rendered = str(value or "auto").strip().lower()
        allowed = {
            "auto",
            "docker",
            "firecracker",
            "monty",
            "wasm",
        }
        if rendered not in allowed:
            raise ValueError("RLM_SANDBOX must select an approved confined backend")
        return rendered

    @field_validator("rlm_container_memory")
    @classmethod
    def _validate_rlm_container_memory(cls, value: str) -> str:
        rendered = str(value or "").strip().lower()
        match = re.fullmatch(r"(\d+)([kmg])", rendered)
        if match is None:
            raise ValueError("RLM_CONTAINER_MEMORY is invalid")
        factor = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30}[match.group(2)]
        size = int(match.group(1)) * factor
        if not 64 * (1 << 20) <= size <= 64 * (1 << 30):
            raise ValueError("RLM_CONTAINER_MEMORY is out of range")
        return rendered

    @model_validator(mode="after")
    def _validate_engine_resource_budgets(self) -> "AgentConfig":
        if (
            self.epistemic_graph_ast_max_total_bytes
            < self.epistemic_graph_ast_max_source_bytes
        ):
            raise ValueError(
                "EPISTEMIC_GRAPH_AST_MAX_TOTAL_BYTES must be at least the per-file limit"
            )
        modality_payload = (
            self.epistemic_graph_modality_max_bundle_bytes
            + self.epistemic_graph_modality_max_source_bytes
        )
        if self.epistemic_graph_max_request_bytes < modality_payload:
            raise ValueError(
                "EPISTEMIC_GRAPH_MAX_REQUEST_BYTES must cover the configured modality payload"
            )
        return self

    @model_validator(mode="after")
    def _auto_enable_from_dependencies(self) -> "AgentConfig":
        """Configure-by-default, opt-out: a capability auto-engages once its
        deployment *dependency* is configured, rather than requiring a second
        explicit flag. This keeps the zero-infra default fully intact — with no
        JWT / Fuseki configured, nothing turns on — while a real deployment that
        wires the dependency gets the capability without remembering to also flip
        the flag (the AGENTS.md "detect and self-configure over a knob" rule).

        - ``KG_FUSEKI_PUBLISH`` engages once a Fuseki endpoint is *explicitly*
          configured (``KG_FUSEKI_ENDPOINT`` set via env/config.json) — not
          merely because ``kg_fuseki_endpoint`` carries a real in-cluster
          default value. This keeps a non-cluster/zero-infra deployment from
          auto-flipping the publish tick on just because the field's default
          happens to resolve to a real host.
        - ``LANGFUSE_MCP_ENABLED`` and ``KG_FAILURE_EVOLUTION`` engage once both
          Langfuse credential references are configured. Content
          capture and every auto-apply/merge gate remain off; this only makes
          metadata-only traces and propose-only failure evolution available
          without redundant flags.

        An explicit value for either flag (env/config) always wins — it lands in
        ``model_fields_set`` and is left untouched.
        """
        explicit = self.model_fields_set
        if "kg_fuseki_publish" not in explicit and "kg_fuseki_endpoint" in explicit:
            self.kg_fuseki_publish = True
        langfuse_ready = bool(
            self.langfuse_public_key_ref and self.langfuse_secret_key_ref
        )
        if langfuse_ready and "langfuse_mcp_enabled" not in explicit:
            self.langfuse_mcp_enabled = True
        if langfuse_ready and "kg_failure_evolution" not in explicit:
            self.kg_failure_evolution = True
        return self

    kg_default_graph: str = Field(default="__commons__", alias="KG_DEFAULT_GRAPH")
    """Default named graph for engine clients that don't target an explicit
    graph. In sharded mode (2+ GRAPH_SERVICE_ENDPOINTS) an ambient
    ActorContext tenant maps this default onto its per-tenant graph
    (``tenant__<tenant>__<base>``) before engine-authoritative placement; single-endpoint
    deployments are unaffected (CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw)."""
    graph_schema_pack: str = Field(default="core", alias="GRAPH_SCHEMA_PACK")
    """Registered schema pack selected for the graph ontology runtime."""
    kg_ingest_shard_fanout: bool = Field(default=False, alias="KG_INGEST_SHARD_FANOUT")
    """Within a single routed content source, spread writes across per-shard
    content-keyed sub-graphs (``src:freshrss#0`` … ``#K-1``) instead of one graph
    per source (CONCEPT:AU-KG.ingest.batched-cross-graph-writer). A high-volume source
    (e.g. a large FreshRSS backlog) otherwise pins its whole drain to ONE graph =
    ONE of the engine's K redb shard writers, so K-1 sit idle. Bucketing by a
    content key across ``#0..#K-1`` puts K distinct graph names in flight so the
    memory-gen write stage fans across all K shard writers. OFF (default) keeps
    one graph per source.
    The ``#n`` sub-graphs keep their source prefix so unified read still unions
    them. Pairs with the engine's ``MultiGraphBatchUpdate`` op
    (CONCEPT:EG-KG.storage.multi-graph-batch-write) which commits the K sub-batches in
    one round-trip across the K writers."""
    kg_rerank_model: str | None = Field(default=None, alias="KG_RERANK_MODEL")
    """Remote reranker model served on vLLM (e.g. ``BAAI/bge-reranker-v2-m3``). When set,
    reranking scores (query, passage) on the remote ``/v1/rerank`` endpoint — no local model,
    consistent with embeddings/LLM on vLLM (CONCEPT:AU-KG.retrieval.unset-dependency-free). Unset → the dependency-free
    lexical scorer (or opt-in local neural via ``KG_RERANK_LOCAL_NEURAL``)."""
    kg_rerank_base_url: str | None = Field(default=None, alias="KG_RERANK_BASE_URL")
    """Base URL for the remote reranker endpoint; defaults to ``OPENAI_BASE_URL`` (the vLLM
    endpoint already serving the embedder/LLM)."""
    graph_service_auth_secret: str | None = Field(
        default=None, alias="GRAPH_SERVICE_AUTH_SECRET"
    )
    """HMAC-SHA256 shared secret for service authentication. When unset, a
    per-install secret is generated once and persisted under the XDG data dir
    (``data_dir()/engine_secret``, mode 0600) so every local process — and any
    engine this launcher spawns — agrees (CONCEPT:AU-OS.identity.authenticated-identity-enforcement)."""

    engine_tls_profile: str | None = Field(default=None, alias="ENGINE_TLS_PROFILE")
    engine_tls_profile_ref: str | None = Field(
        default=None, alias="ENGINE_TLS_PROFILE_REF"
    )
    engine_tls_server_name: str | None = Field(
        default=None, alias="ENGINE_TLS_SERVER_NAME"
    )
    engine_lifecycle: str = Field(default="refcounted", alias="ENGINE_LIFECYCLE")
    """Lifecycle of an AUTOSTARTED local engine (CONCEPT:AU-OS.deployment.engine-resolver-auto-provision):

    * ``refcounted`` (default) — reference-counted idle shutdown: the engine
      self-terminates ``engine_idle_shutdown_secs`` seconds after its LAST client
      disconnects (robust to client crashes). The shared-tiny default — auto-stops
      when idle.
    * ``persistent`` — LONG-LIVING: the engine NEVER self-stops, even when idle
      (it runs like a local service). Forces idle-shutdown off regardless of
      ``engine_idle_shutdown_secs``.

    A remote/cluster engine is inherently persistent — the resolver never passes
    an idle-shutdown flag in remote mode."""

    engine_idle_shutdown_secs: int = Field(
        default=60, alias="ENGINE_IDLE_SHUTDOWN_SECS"
    )
    """Idle-shutdown grace (seconds) for a ``refcounted`` autostarted engine
    (CONCEPT:AU-OS.deployment.engine-resolver-auto-provision). ``> 0`` → the autostart leg passes
    ``--idle-shutdown-secs <secs>`` so the engine stops that many seconds after
    its last client disconnects. ``<= 0`` (or ``engine_lifecycle=persistent``) →
    NO flag is passed and the engine is long-living (never auto-stops). Default
    60s. The current packaged engine provides this launch contract."""
    placement_catalog_enabled: bool = Field(
        default=True, alias="PLACEMENT_CATALOG_ENABLED"
    )
    """Consult the engine's authoritative PlacementCatalog when resolving a
    sharded graph's owning endpoint (CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw, DIST-P2-2b —
    ``knowledge_graph.core.placement_catalog.resolve_placement``), instead of
    deciding placement purely from the static client-side HRW ring. Default
    True: the catalog is authoritative WHEN a reachable engine advertises one;
    an engine that doesn't (every endpoint unreachable, or an older engine
    with no placement-route RPC) transparently falls back to the existing HRW
    ring, so today's deployments are unaffected either way. Set False to force
    pure HRW routing and skip the catalog round-trip entirely."""

    epistemic_graph_max_resident_graphs: int = Field(
        default=256, ge=1, le=100_000, alias="EPISTEMIC_GRAPH_MAX_RESIDENT_GRAPHS"
    )
    """Maximum number of materialized graphs retained by an engine process.
    A positive bound is mandatory so cold graphs can be evicted and reopened
    safely; the default is deliberately suitable for a shared workstation."""

    epistemic_graph_lazy_open_page_size: int = Field(
        default=4096, ge=1, le=1_000_000, alias="EPISTEMIC_GRAPH_LAZY_OPEN_PAGE_SIZE"
    )
    """Maximum records loaded per lazy-open page. AgentConfig always supplies a
    bounded value; graph responses remain explicitly partial until recovery and
    maintained index manifests cover the durable source snapshot."""

    epistemic_graph_max_nodes_per_graph: int = Field(
        default=250_000,
        ge=1,
        le=100_000_000,
        alias="EPISTEMIC_GRAPH_MAX_NODES_PER_GRAPH",
    )
    """Maximum resident nodes in one graph before durability-gated bulk eviction.
    This bounds a single hostile or unexpectedly large tenant independently of the
    process-wide memory budget; durable rows remain readable through the engine."""

    epistemic_graph_max_request_bytes: int = Field(
        default=64 * 1024 * 1024,
        ge=1024,
        le=384 * 1024 * 1024,
        alias="EPISTEMIC_GRAPH_MAX_REQUEST_BYTES",
    )
    epistemic_graph_max_response_bytes: int = Field(
        default=64 * 1024 * 1024,
        ge=1024,
        le=384 * 1024 * 1024,
        alias="EPISTEMIC_GRAPH_MAX_RESPONSE_BYTES",
    )
    epistemic_graph_max_msgpack_items: int = Field(
        default=1_000_000,
        ge=1,
        le=4_000_000,
        alias="EPISTEMIC_GRAPH_MAX_MSGPACK_ITEMS",
    )
    """Native frame and MessagePack allocation budgets. The Rust engine also
    enforces immutable hard ceilings; AgentConfig projects deployment policy
    only into a locally autostarted engine."""

    epistemic_graph_connection_io_timeout_secs: int = Field(
        default=120,
        ge=1,
        le=3600,
        alias="EPISTEMIC_GRAPH_CONNECTION_IO_TIMEOUT_SECS",
    )
    epistemic_graph_tls_handshake_timeout_secs: int = Field(
        default=10,
        ge=1,
        le=120,
        alias="EPISTEMIC_GRAPH_TLS_HANDSHAKE_TIMEOUT_SECS",
    )
    """Bound slow-client native I/O and TLS handshakes."""

    epistemic_graph_ast_max_files: int = Field(
        default=4096,
        ge=1,
        le=100_000,
        alias="EPISTEMIC_GRAPH_AST_MAX_FILES",
    )
    epistemic_graph_ast_max_source_bytes: int = Field(
        default=4 * 1024 * 1024,
        ge=1,
        le=64 * 1024 * 1024,
        alias="EPISTEMIC_GRAPH_AST_MAX_SOURCE_BYTES",
    )
    epistemic_graph_ast_max_total_bytes: int = Field(
        default=32 * 1024 * 1024,
        ge=1,
        le=256 * 1024 * 1024,
        alias="EPISTEMIC_GRAPH_AST_MAX_TOTAL_BYTES",
    )
    """Repository-relative AST indexing budgets; callers send bounded logical
    paths and source bytes, never engine-host paths."""

    epistemic_graph_modality_max_bundle_bytes: int = Field(
        default=4 * 1024 * 1024,
        ge=1,
        le=32 * 1024 * 1024,
        alias="EPISTEMIC_GRAPH_MODALITY_MAX_BUNDLE_BYTES",
    )
    epistemic_graph_modality_max_source_bytes: int = Field(
        default=16 * 1024 * 1024,
        ge=1,
        le=256 * 1024 * 1024,
        alias="EPISTEMIC_GRAPH_MODALITY_MAX_SOURCE_BYTES",
    )
    """Governed multimodal bundle and source byte limits."""

    epistemic_graph_encryption_key_ref: str | None = Field(
        default=None, alias="EPISTEMIC_GRAPH_ENCRYPTION_KEY_REF"
    )
    """External runtime secret reference for the durable engine data key.

    The resolved value is never persisted in AgentConfig or projected into the
    parent process environment by the engine launcher. A packaged local ``tiny``
    deployment may generate one stable private XDG key outside production;
    production and non-tiny local engine deployments require this explicit
    bootstrap reference. Remote engines own their encryption configuration.
    """

    epistemic_graph_sqlite_transfer_root_ref: str | None = Field(
        default=None, alias="EPISTEMIC_GRAPH_SQLITE_TRANSFER_ROOT_REF"
    )
    epistemic_graph_sqlite_max_bytes: int = Field(
        default=256 * 1024 * 1024,
        ge=1,
        le=16 * 1024 * 1024 * 1024,
        alias="EPISTEMIC_GRAPH_SQLITE_MAX_BYTES",
    )
    epistemic_graph_sqlite_max_rows: int = Field(
        default=1_000_000,
        ge=1,
        le=100_000_000,
        alias="EPISTEMIC_GRAPH_SQLITE_MAX_ROWS",
    )
    """SQLite file transfer is disabled until a runtime secret reference resolves
    to a dedicated private directory. Configuration never stores its path."""

    epistemic_graph_backup_root_ref: str | None = Field(
        default=None, alias="EPISTEMIC_GRAPH_BACKUP_ROOT_REF"
    )
    """Runtime secret reference for the private backup/restore root. RPC callers
    supply bounded logical bundle names, never host filesystem paths."""

    graph_os_backup_principal: str | None = Field(
        default=None, alias="GRAPH_OS_BACKUP_PRINCIPAL"
    )
    graph_os_backup_tenant: str | None = Field(
        default=None, alias="GRAPH_OS_BACKUP_TENANT"
    )
    graphos_backup_retention_count: int = Field(
        default=2, ge=2, le=1440, alias="GRAPHOS_BACKUP_RETENTION_COUNT"
    )
    epistemic_graph_restore_bin: str = Field(
        default="restore", alias="EPISTEMIC_GRAPH_RESTORE_BIN"
    )
    epistemic_graph_server_bin: str = Field(
        default="epistemic-graph-server", alias="EPISTEMIC_GRAPH_SERVER_BIN"
    )
    restore_validation_port: int = Field(
        default=19_100, ge=1024, le=65_535, alias="RESTORE_VALIDATION_PORT"
    )
    """Production backup/restore settings. Runtime principals and executable
    locations remain environment-owned AgentConfig inputs and are never emitted
    in operational evidence."""

    computer_use_display: str = Field(default=":1", alias="COMPUTER_USE_DISPLAY")
    computer_use_user: str = Field(default="sandbox", alias="COMPUTER_USE_USER")
    computer_use_home: str = Field(default="", alias="COMPUTER_USE_HOME")
    """Runtime identity/environment inside the configured GUI sandbox. Home is
    unset by default so the target runtime supplies it; no host path is embedded."""

    graph_os_analytics_principal: str | None = Field(
        default=None, alias="GRAPH_OS_ANALYTICS_PRINCIPAL"
    )
    graph_os_analytics_tenant: str | None = Field(
        default=None, alias="GRAPH_OS_ANALYTICS_TENANT"
    )
    eg_analytics_worker_capabilities: str = Field(
        default="mining.association,pool:default",
        alias="EG_ANALYTICS_WORKER_CAPABILITIES",
    )
    eg_analytics_worker_slots: int = Field(
        default=1, ge=1, le=64, alias="EG_ANALYTICS_WORKER_SLOTS"
    )
    eg_analytics_worker_lease_ms: int = Field(
        default=60_000,
        ge=5_000,
        le=300_000,
        alias="EG_ANALYTICS_WORKER_LEASE_MS",
    )
    eg_analytics_worker_poll_seconds: float = Field(
        default=0.25,
        ge=0.05,
        le=30.0,
        alias="EG_ANALYTICS_WORKER_POLL_SECONDS",
    )
    """Governed standalone analytics-worker identity, coordinator, capability,
    concurrency, lease, and poll settings."""

    placement_catalog_ttl_s: float = Field(default=5.0, alias="PLACEMENT_CATALOG_TTL_S")
    """How long (seconds) a resolved ``(endpoint, epoch)`` from
    :func:`~agent_utilities.knowledge_graph.core.placement_catalog.resolve_placement`
    is cached before the next call re-queries the engine catalog
    (CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw). Short by design — a partition mid fenced-cutover
    should be rediscovered quickly. A stale-epoch redirect bypasses this TTL
    outright via ``force_refresh=True``, so this only bounds the *unprompted*
    re-check cadence."""

    placement_control_loop_enabled: bool = Field(
        default=False, alias="PLACEMENT_CONTROL_LOOP_ENABLED"
    )
    """Permit automatic or periodic placement-control callers to run.

    The loop remains off by default because a successful governed proposal can
    initiate a real online placement move. An explicit ``graph_loops``
    ``placement_control`` request is itself the operator trigger and enables only
    that invocation; this setting controls callers that have no such request
    boundary. Keeping the flag in ``AgentConfig`` makes it visible to config
    generation, the doctor, and the generated runtime reference."""

    graph_service_persist_on_shutdown: bool = Field(
        default=True, alias="GRAPH_SERVICE_PERSIST_ON_SHUTDOWN"
    )
    """Serialize all graphs to disk on service shutdown."""
    graph_persistence_path: str = Field(
        default_factory=lambda: DEFAULT_DB_PATH, alias="GRAPH_PERSISTENCE_PATH"
    )
    enable_llm_validation: bool = Field(default=False, alias="ENABLE_LLM_VALIDATION")
    graph_router_timeout: float = Field(default=300.0, alias="GRAPH_ROUTER_TIMEOUT")
    graph_verifier_timeout: float = Field(default=300.0, alias="GRAPH_VERIFIER_TIMEOUT")
    enable_kg_embeddings: bool = Field(default=True, alias="ENABLE_KG_EMBEDDINGS")
    kg_backups: int = Field(default=3, alias="KG_BACKUPS")
    kg_ingestion_workers: int | None = Field(default=None, alias="KG_INGESTION_WORKERS")
    kg_llm_concurrency: int = Field(default=4, alias="KG_LLM_CONCURRENCY")
    """Total parallel capacity of the local inference endpoint (e.g. vLLM/LM Studio slots).

    This is the ONE knob for local-model parallelism. CONCEPT:AU-ORCH.execution.reserved-inference-slots — the system always
    reserves ``RESERVED_INTERACTIVE_INSTANCES`` (1) of these slots for the **interactive**
    path (the Telegram/messaging responder and graph-os-spawned pydantic-ai agents, which
    share the default model); all background KG work (fact enrichment, Layer 2/3 analysis,
    embeddings) is bounded to ``background_llm_concurrency()`` = capacity − reserved. So a
    background sweep can never consume the slot you need to get an answer. Set this to your
    endpoint's real parallel capacity and the reservation scales automatically."""

    def background_llm_concurrency(self) -> int:
        """Concurrency ceiling for background KG LLM work — capacity minus the reserved
        interactive slot(s) (CONCEPT:AU-ORCH.execution.reserved-inference-slots). Floors at 1 so background never starves."""
        return max(1, self.kg_llm_concurrency - RESERVED_INTERACTIVE_INSTANCES)

    kg_analysis_max_depth: int = Field(default=2, alias="KG_ANALYSIS_MAX_DEPTH")
    """Maximum recursive depth for background knowledge graph research daemons."""
    knowledge_graph_sync_background: bool = Field(
        default=True, alias="KNOWLEDGE_GRAPH_SYNC_BACKGROUND"
    )
    """Enable or disable background task workers for the Knowledge Graph pipeline."""
    enable_sdd_watcher: bool = Field(default=True, alias="ENABLE_SDD_WATCHER")
    """Enable or disable the background plan/task watcher thread in the KG MCP server."""
    model_registry_path: str | None = Field(default=None, alias="MODEL_REGISTRY_PATH")
    """Path to a YAML or JSON file defining the model registry."""
    role_routing: dict[str, dict] = Field(
        default_factory=dict, alias="MODEL_ROLE_ROUTING"
    )
    """CONCEPT:AU-ORCH.routing.optional-role-override — optional role→{tier,tags} overrides for planner/generator/
    learner/judge model selection. Empty roles fall back to the built-in default map in
    `models/model_registry.py`. Merged into the active registry when no registry-level
    override is present."""
    epistemic_light_default: bool = Field(
        default=True, alias="KG_EPISTEMIC_LIGHT_DEFAULT"
    )
    """Attach the LIGHT epistemic envelope (confidence/source_refs/evidence_refs/
    policy_labels/provenance — CONCEPT:AU-KB-CURRENCY) onto every plain read-path
    row by default (Native by default). This is the cheap, ADDITIVE column-merge
    (`epistemic_row.attach_epistemic_columns`) that keeps a caller's existing
    `list[dict]` shape — never the heavy, type-changing `include_epistemic=True`
    round trip (`EpistemicRow`), which stays opt-in. Set False only for a
    deployment that must skip the extra batched `explain_provenance_by_ids`
    round trip on every read; a row already showing a contested/low-confidence
    signal in its own properties is still resolved regardless (safety auto-on,
    see `epistemic_row.should_attach_epistemic_columns`)."""

    epistemic_light_default: bool = Field(
        default=True, alias="KG_EPISTEMIC_LIGHT_DEFAULT"
    )
    """Attach the LIGHT epistemic envelope (confidence/source_refs/evidence_refs/
    policy_labels/provenance — CONCEPT:AU-KB-CURRENCY) onto every plain read-path
    row by default (Native by default). This is the cheap, ADDITIVE column-merge
    (`epistemic_row.attach_epistemic_columns`) that keeps a caller's existing
    `list[dict]` shape — never the heavy, type-changing `include_epistemic=True`
    round trip (`EpistemicRow`), which stays opt-in. Set False only for a
    deployment that must skip the extra batched `explain_provenance_by_ids`
    round trip on every read; a row already showing a contested/low-confidence
    signal in its own properties is still resolved regardless (safety auto-on,
    see `epistemic_row.should_attach_epistemic_columns`)."""

    sparql_endpoints: list[str] = Field(
        default=["https://query.wikidata.org/sparql"], alias="SPARQL_ENDPOINTS"
    )
    """List of external SPARQL endpoints to federate (CONCEPT:AU-KG.query.vendor-agnostic-traversal)."""

    vllm_base_url: str | None = Field(default=None, alias="VLLM_BASE_URL")
    """Dedicated, runtime-injected base URL for a vLLM inference server."""

    kafka_topic: str | None = Field(default=None, alias="KAFKA_TOPIC")
    """Default Kafka topic for messaging/event ingestion."""

    secrets_backend: Literal["engine", "vault"] = Field(
        default="engine", alias="SECRETS_BACKEND"
    )
    """Secrets storage backend: encrypted engine storage (default) or Vault."""

    custom_skills_directory: str | None = Field(
        default=None, alias="CUSTOM_SKILLS_DIRECTORY"
    )
    skill_types: list[str] | None = Field(default=None, alias="SKILL_TYPES")

    enable_otel: bool = Field(default=False, alias="ENABLE_OTEL")
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None, alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    otel_exporter_otlp_headers_ref: str | None = Field(
        default=None, alias="OTEL_EXPORTER_OTLP_HEADERS_REF"
    )
    otel_exporter_otlp_public_key_ref: str | None = Field(
        default=None, alias="OTEL_EXPORTER_OTLP_PUBLIC_KEY_REF"
    )
    otel_exporter_otlp_secret_key_ref: str | None = Field(
        default=None, alias="OTEL_EXPORTER_OTLP_SECRET_KEY_REF"
    )
    otel_exporter_otlp_protocol: str = Field(
        default="http/protobuf", alias="OTEL_EXPORTER_OTLP_PROTOCOL"
    )
    otel_tls_profile: str | None = Field(default=None, alias="OTEL_TLS_PROFILE")
    otel_tls_profile_ref: str | None = Field(default=None, alias="OTEL_TLS_PROFILE_REF")

    langfuse_public_key_ref: str | None = Field(
        default=None, alias="LANGFUSE_PUBLIC_KEY_REF"
    )
    langfuse_secret_key_ref: str | None = Field(
        default=None, alias="LANGFUSE_SECRET_KEY_REF"
    )
    langfuse_persistence_hmac_key_ref: str | None = Field(
        default=None, alias="LANGFUSE_PERSISTENCE_HMAC_KEY_REF"
    )
    langfuse_host: str = Field(
        default=_LANGFUSE_DEFAULT_HOST,
        alias="LANGFUSE_HOST",
    )

    @field_validator("langfuse_host", mode="before")
    @classmethod
    def _validate_langfuse_host(cls, value: Any) -> str:
        return _validated_langfuse_host(value)

    langfuse_tls_profile: str | None = Field(default=None, alias="LANGFUSE_TLS_PROFILE")
    langfuse_tls_profile_ref: str | None = Field(
        default=None, alias="LANGFUSE_TLS_PROFILE_REF"
    )
    langfuse_ca_bundle_ref: str | None = Field(
        default=None, alias="LANGFUSE_CA_BUNDLE_REF"
    )
    langfuse_client_cert_ref: str | None = Field(
        default=None, alias="LANGFUSE_CLIENT_CERT_REF"
    )
    langfuse_client_key_ref: str | None = Field(
        default=None, alias="LANGFUSE_CLIENT_KEY_REF"
    )
    langfuse_client_key_password_ref: str | None = Field(
        default=None, alias="LANGFUSE_CLIENT_KEY_PASSWORD_REF"
    )
    langfuse_proxy_url_ref: str | None = Field(
        default=None, alias="LANGFUSE_PROXY_URL_REF"
    )

    langfuse_dataset_capture_threshold: float = Field(
        default=0.0, alias="LANGFUSE_DATASET_CAPTURE_THRESHOLD"
    )
    langfuse_latency_baseline_seconds: float = Field(
        default=60.0, alias="LANGFUSE_LATENCY_BASELINE_SECONDS"
    )
    langfuse_token_baseline: int = Field(default=20000, alias="LANGFUSE_TOKEN_BASELINE")
    langfuse_verifier_fallback_limit: int = Field(
        default=1, alias="LANGFUSE_VERIFIER_FALLBACK_LIMIT"
    )
    langfuse_capture_content: bool = Field(
        default=False, alias="LANGFUSE_CAPTURE_CONTENT"
    )
    langfuse_kg_auto_ingest: bool = Field(
        default=False, alias="LANGFUSE_KG_AUTO_INGEST"
    )
    """Opt in to privacy-sanitized trace content; all sinks are metadata-only by default."""
    langfuse_mcp_enabled: bool = Field(default=False, alias="LANGFUSE_MCP_ENABLED")
    """Expose the Langfuse MCP integration when its runtime credentials are injected."""

    # Optional Google Workspace OAuth bootstrap. Values are runtime-only and
    # are never included in graph metadata, traces, or doctor output. The
    # broker is operator-supplied: the codebase ships no environment-specific
    # tenant, client, or endpoint profile.
    google_workspace_oauth_client_id: str | None = Field(
        default=None, alias="GOOGLE_WORKSPACE_OAUTH_CLIENT_ID"
    )
    google_workspace_oauth_broker_url: str | None = Field(
        default=None, alias="GOOGLE_WORKSPACE_OAUTH_BROKER_URL"
    )

    @field_validator("google_workspace_oauth_client_id", mode="before")
    @classmethod
    def _validate_google_workspace_oauth_client_id(cls, value: Any) -> str | None:
        if value in (None, ""):
            return None
        rendered = str(value).strip()
        if not re.fullmatch(r"[A-Za-z0-9._:-]{3,512}", rendered):
            raise ValueError("GOOGLE_WORKSPACE_OAUTH_CLIENT_ID is malformed")
        return rendered

    @field_validator("google_workspace_oauth_broker_url", mode="before")
    @classmethod
    def _validate_google_workspace_oauth_broker_url(cls, value: Any) -> str | None:
        rendered = _validated_runtime_http_url(value)
        if rendered is None:
            return None
        if not rendered.lower().startswith("https://"):
            raise ValueError("GOOGLE_WORKSPACE_OAUTH_BROKER_URL must use HTTPS")
        return rendered

    trace_export_enabled: bool = Field(default=False, alias="TRACE_EXPORT_ENABLED")
    """Enable trace export; the exporter endpoint/credentials remain runtime config."""
    persistence_privacy_deny_terms_ref: str | None = Field(
        default=None, alias="PERSISTENCE_PRIVACY_DENY_TERMS_REF"
    )
    """Secret reference to identity terms removed before KG/log/trace persistence."""

    persistence_identity_hmac_key_ref: str | None = Field(
        default=None, alias="PERSISTENCE_IDENTITY_HMAC_KEY_REF"
    )
    """Secret reference used to HMAC opaque durable identity references."""

    memento_raw_retention_enabled: bool = Field(
        default=False, alias="MEMENTO_RAW_RETENTION_ENABLED"
    )
    """Opt in to encrypted raw Memento retention; disabled by default."""

    memento_raw_retention_policy: str = Field(
        default="", alias="MEMENTO_RAW_RETENTION_POLICY"
    )
    """Must equal the supported versioned approval token before raw retention is allowed."""

    memento_raw_encryption_key_ref: str | None = Field(
        default=None, alias="MEMENTO_RAW_ENCRYPTION_KEY_REF"
    )
    """Secrets-backend reference for the Memento raw-retention encryption key."""

    a2a_broker: Literal["epistemic_graph"] = Field(
        default="epistemic_graph", alias="A2A_BROKER"
    )
    """FastA2A operation delivery uses the native durable engine broker."""

    a2a_storage: Literal["epistemic_graph"] = Field(
        default="epistemic_graph", alias="A2A_STORAGE"
    )
    """FastA2A task/context state uses native durable engine records."""

    a2a_broker_poll_interval_ms: int = Field(
        default=100, ge=10, le=5_000, alias="A2A_BROKER_POLL_INTERVAL_MS"
    )
    a2a_broker_lease_ms: int = Field(
        default=300_000, ge=10_000, le=3_600_000, alias="A2A_BROKER_LEASE_MS"
    )
    a2a_broker_prefetch: int = Field(
        default=1, ge=1, le=128, alias="A2A_BROKER_PREFETCH"
    )
    a2a_broker_message_ttl_ms: int = Field(
        default=86_400_000,
        ge=60_000,
        le=604_800_000,
        alias="A2A_BROKER_MESSAGE_TTL_MS",
    )
    a2a_broker_max_delivery_count: int = Field(
        default=5, ge=1, le=100, alias="A2A_BROKER_MAX_DELIVERY_COUNT"
    )
    a2a_max_payload_bytes: int = Field(
        default=262_144, ge=4_096, le=4_194_304, alias="A2A_MAX_PAYLOAD_BYTES"
    )
    a2a_max_history: int = Field(default=100, ge=1, le=1_000, alias="A2A_MAX_HISTORY")
    a2a_max_artifacts: int = Field(default=50, ge=1, le=500, alias="A2A_MAX_ARTIFACTS")
    a2a_max_context_messages: int = Field(
        default=100, ge=1, le=1_000, alias="A2A_MAX_CONTEXT_MESSAGES"
    )
    a2a_storage_update_retries: int = Field(
        default=4, ge=1, le=16, alias="A2A_STORAGE_UPDATE_RETRIES"
    )
    a2a_dispatch_reconcile_interval_ms: int = Field(
        default=1_000,
        ge=10,
        le=60_000,
        alias="A2A_DISPATCH_RECONCILE_INTERVAL_MS",
    )
    """Interval for recovering persisted A2A dispatches after a process failure."""

    a2a_dispatch_reconcile_limit: int = Field(
        default=64, ge=1, le=1_024, alias="A2A_DISPATCH_RECONCILE_LIMIT"
    )
    """Maximum persisted A2A tasks inspected by one reconciliation tick."""

    a2a_cancellation_poll_interval_ms: int = Field(
        default=1_000,
        ge=10,
        le=60_000,
        alias="A2A_CANCELLATION_POLL_INTERVAL_MS",
    )
    """Maximum interval between durable cross-process cancellation checks."""

    a2a_config: str | None = Field(default=None, alias="A2A_CONFIG")
    """Path to a2a_config.json for external A2A agent discovery (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""
    a2a_refresh_interval: int = Field(default=300, alias="A2A_REFRESH_INTERVAL")
    """Interval in seconds for periodic A2A agent card re-fetch (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    max_tokens: int = Field(default=16384, alias="MAX_TOKENS")
    temperature: float = Field(default=0.7, alias="TEMPERATURE")
    top_p: float = Field(default=1.0, alias="TOP_P")
    timeout: float = Field(default=3600.0, gt=0, le=3_600, alias="TIMEOUT")
    tool_timeout: float = Field(default=3600.0, gt=0, le=3_600, alias="TOOL_TIMEOUT")
    parallel_tool_calls: bool = Field(default=True, alias="PARALLEL_TOOL_CALLS")
    seed: int | None = Field(default=None, alias="SEED")
    presence_penalty: float = Field(default=0.0, alias="PRESENCE_PENALTY")
    frequency_penalty: float = Field(default=0.0, alias="FREQUENCY_PENALTY")

    logit_bias: dict[str, float] | None = Field(default=None, alias="LOGIT_BIAS")
    stop_sequences: list[str] | None = Field(default=None, alias="STOP_SEQUENCES")
    extra_headers: dict[str, str] | None = Field(default=None, alias="EXTRA_HEADERS")
    extra_body: dict[str, Any] | None = Field(default=None, alias="EXTRA_BODY")

    min_confidence: float = Field(default=0.4, alias="MIN_CONFIDENCE")
    validation_mode: bool = Field(default=False, alias="VALIDATION_MODE")
    approval_timeout: float = Field(default=0.0, alias="APPROVAL_TIMEOUT")

    # --- Agent OS Architecture (CONCEPT:AU-OS.state.cognitive-scheduler-preemption) ---

    cognitive_scheduler_enabled: bool = Field(
        default=True, alias="COGNITIVE_SCHEDULER_ENABLED"
    )
    """Enable the Cognitive Scheduler for priority-aware agent management (CONCEPT:AU-OS.state.cognitive-scheduler-preemption)."""

    max_concurrent_agents: int = Field(default=5, alias="MAX_CONCURRENT_AGENTS")
    """Maximum number of concurrently running specialist agents (CONCEPT:AU-OS.state.cognitive-scheduler-preemption)."""

    agent_token_quota: int = Field(default=100_000, alias="AGENT_TOKEN_QUOTA")
    """Default per-agent token budget before preemption (CONCEPT:AU-OS.state.cognitive-scheduler-preemption)."""

    preemption_threshold_pct: float = Field(
        default=0.85, alias="PREEMPTION_THRESHOLD_PCT"
    )
    """Quota usage percentage that triggers preemption warning (CONCEPT:AU-OS.state.cognitive-scheduler-preemption)."""

    agent_policies_path: str | None = Field(default=None, alias="AGENT_POLICIES_PATH")
    """Path to agent_policies.json for identity-based governance (CONCEPT:AU-OS.state.cognitive-scheduler-preemption)."""

    permissions_signing_key_ref: str | None = Field(
        default=None, alias="PERMISSIONS_SIGNING_KEY_REF"
    )
    """Runtime secret reference for stable agent-identity HMAC authority (CONCEPT:AU-OS.state.cognitive-scheduler-preemption)."""

    ontology_release_signing_private_key_ref: str | None = Field(
        default=None, alias="ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF"
    )
    """Runtime secret reference for the stable Ed25519 connector and ontology release signer."""

    ontology_release_trusted_public_keys: str = Field(
        default="", alias="ONTOLOGY_RELEASE_TRUSTED_PUBLIC_KEYS"
    )
    """Comma-separated Ed25519 public keys trusted for connector and ontology release verification."""

    specialist_registry_path: str | None = Field(
        default=None, alias="SPECIALIST_REGISTRY_PATH"
    )
    """Path to local specialist registry directory (CONCEPT:AU-OS.state.cognitive-scheduler-preemption)."""

    # --- Native Messaging Backend (CONCEPT:AU-ECO.messaging.native-backend-abstraction) ---

    messaging_enabled_backends: list[str] = Field(
        default_factory=list, alias="MESSAGING_ENABLED_BACKENDS"
    )
    """List of messaging backend IDs to auto-connect on startup (CONCEPT:AU-ECO.messaging.native-backend-abstraction).
    Example: ["discord", "slack", "telegram"]."""

    messaging_kg_ingest: bool = Field(default=True, alias="MESSAGING_KG_INGEST")
    """Enable automatic Knowledge Graph ingestion for all inbound/outbound messages (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_kg_memory_type: str = Field(
        default="episodic", alias="MESSAGING_KG_MEMORY_TYPE"
    )
    """Default KG memory tier for inbound messages: 'episodic', 'semantic', or 'procedural' (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_route_to_planner: bool = Field(
        default=True, alias="MESSAGING_ROUTE_TO_PLANNER"
    )
    """Route inbound messaging events to the Planner Graph Agent for orchestration (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    # Per-platform tokens (read from config.json or env vars)
    messaging_discord_token: str | None = Field(
        default=None, alias="MESSAGING_DISCORD_TOKEN"
    )
    """Discord bot token. Also reads from DISCORD_BOT_TOKEN (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_slack_token: str | None = Field(
        default=None, alias="MESSAGING_SLACK_TOKEN"
    )
    """Slack bot token (xoxb-...). Also reads from SLACK_BOT_TOKEN (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_slack_app_token: str | None = Field(
        default=None, alias="MESSAGING_SLACK_APP_TOKEN"
    )
    """Slack app-level token (xapp-...) for Socket Mode (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_telegram_token: str | None = Field(
        default=None, alias="MESSAGING_TELEGRAM_TOKEN"
    )
    """Telegram bot token. Also reads from TELEGRAM_BOT_TOKEN (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_whatsapp_token: str | None = Field(
        default=None, alias="MESSAGING_WHATSAPP_TOKEN"
    )
    """WhatsApp API token. Also reads from WHATSAPP_TOKEN (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_whatsapp_phone_number_id: str | None = Field(
        default=None, alias="MESSAGING_WHATSAPP_PHONE_NUMBER_ID"
    )
    """WhatsApp Business API phone number ID (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_whatsapp_use_business_api: bool = Field(
        default=False, alias="MESSAGING_WHATSAPP_USE_BUSINESS_API"
    )
    """Use official WhatsApp Business API (True) or neonize bridge (False) (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_teams_app_id: str | None = Field(
        default=None, alias="MESSAGING_TEAMS_APP_ID"
    )
    """Microsoft Teams Bot Framework app ID (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_teams_app_secret: str | None = Field(
        default=None, alias="MESSAGING_TEAMS_APP_SECRET"
    )
    """Microsoft Teams Bot Framework app password (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_googlechat_service_account: str | None = Field(
        default=None, alias="MESSAGING_GOOGLECHAT_TOKEN"
    )
    """Path to Google Chat service account JSON file (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_googlemeet_service_account: str | None = Field(
        default=None, alias="MESSAGING_GOOGLEMEET_TOKEN"
    )
    """Path to Google Meet service account JSON file (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_mattermost_token: str | None = Field(
        default=None, alias="MESSAGING_MATTERMOST_TOKEN"
    )
    """Mattermost personal access token (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_mattermost_url: str | None = Field(
        default=None, alias="MESSAGING_MATTERMOST_URL"
    )
    """Mattermost server URL (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_matrix_token: str | None = Field(
        default=None, alias="MESSAGING_MATRIX_TOKEN"
    )
    """Matrix access token (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_matrix_homeserver: str | None = Field(
        default=None, alias="MESSAGING_MATRIX_HOMESERVER"
    )
    """Matrix homeserver URL (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_matrix_user_id: str | None = Field(
        default=None, alias="MESSAGING_MATRIX_USER_ID"
    )
    """Matrix user ID (e.g. @bot:matrix.org) (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_irc_server: str | None = Field(default=None, alias="MESSAGING_IRC_SERVER")
    """IRC server hostname (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_irc_port: int = Field(default=6667, alias="MESSAGING_IRC_PORT")
    """IRC server port (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_irc_nickname: str = Field(
        default="agent_bot", alias="MESSAGING_IRC_NICKNAME"
    )
    """IRC nickname (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_irc_channels: list[str] = Field(
        default_factory=list, alias="MESSAGING_IRC_CHANNELS"
    )
    """IRC channels to auto-join (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_signal_phone: str | None = Field(
        default=None, alias="MESSAGING_SIGNAL_TOKEN"
    )
    """Signal phone number for semaphore-bot (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_line_token: str | None = Field(default=None, alias="MESSAGING_LINE_TOKEN")
    """LINE channel access token (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_twitch_token: str | None = Field(
        default=None, alias="MESSAGING_TWITCH_TOKEN"
    )
    """Twitch OAuth token (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_twitch_channels: list[str] = Field(
        default_factory=list, alias="MESSAGING_TWITCH_CHANNELS"
    )
    """Twitch channels to join (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_synology_webhook_url: str | None = Field(
        default=None, alias="MESSAGING_SYNOLOGY_WEBHOOK_URL"
    )
    """Synology Chat incoming webhook URL (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_twilio_account_sid: str | None = Field(
        default=None, alias="MESSAGING_VOICECALL_APP_ID"
    )
    """Twilio account SID for voice/SMS (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_twilio_auth_token: str | None = Field(
        default=None, alias="MESSAGING_VOICECALL_TOKEN"
    )
    """Twilio auth token for voice/SMS (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_twilio_from_number: str | None = Field(
        default=None, alias="MESSAGING_VOICECALL_FROM_NUMBER"
    )
    """Twilio 'from' phone number (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_nextcloud_url: str | None = Field(
        default=None, alias="MESSAGING_NEXTCLOUD_URL"
    )
    """Nextcloud server URL (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_nextcloud_token: str | None = Field(
        default=None, alias="MESSAGING_NEXTCLOUD_TOKEN"
    )
    """Nextcloud app token (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_nextcloud_user: str | None = Field(
        default=None, alias="MESSAGING_NEXTCLOUD_APP_ID"
    )
    """Nextcloud username (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    # --- Parallel Engine (CONCEPT:AU-ORCH.execution.parallel-engine-visualizer) ---

    max_parallel_agents: int = Field(default=60, alias="MAX_PARALLEL_AGENTS")
    """Maximum concurrent agent executions across the engine (CONCEPT:AU-ORCH.execution.parallel-engine-visualizer).
    Acts as a global semaphore. Set higher for cloud deployments with high API limits."""

    worker_pool_size: int = Field(default=8, alias="WORKER_POOL_SIZE")
    """Number of worker processes/threads provisioned per node for executing agent
    turns and graph mutations (CONCEPT:AU-ORCH.execution.parallel-engine-visualizer).

    Scale knob. Together with ``graph_service_endpoints`` (Postgres/L0 shard fan-out
    for the epistemic graph) and ``kafka_bootstrap_servers`` (event-throughput axis),
    this is one of the three horizontal-scale knobs modeled in
    ``docs/scaling/capacity_model.md``:

    * ``worker_pool_size`` x node count -> active-concurrency capacity.
    * ``graph_service_endpoints`` -> resident-population (shard) capacity.
    * ``kafka_bootstrap_servers`` partitions -> event-throughput capacity.
    """

    parallel_batch_size: int = Field(default=25, alias="PARALLEL_BATCH_SIZE")
    """Number of agents per execution wave when batching is needed (CONCEPT:AU-ORCH.execution.parallel-engine-visualizer)."""

    synthesis_strategy: str = Field(default="auto", alias="SYNTHESIS_STRATEGY")
    """Default output synthesis strategy: 'auto', 'flat', 'hierarchical', 'progressive', 'rlm'.
    'auto' selects based on agent count and output size (CONCEPT:AU-ORCH.execution.rlm-synthesis-failed-falling)."""

    synthesis_ratio: int = Field(default=10, alias="SYNTHESIS_RATIO")
    """In hierarchical synthesis, how many outputs per synthesis sub-node (CONCEPT:AU-ORCH.execution.rlm-synthesis-failed-falling)."""

    agent_execution_timeout: float = Field(
        default=120.0, alias="AGENT_EXECUTION_TIMEOUT"
    )
    """Per-agent execution timeout in seconds (CONCEPT:AU-ORCH.execution.parallel-engine-visualizer)."""

    circuit_breaker_threshold: int = Field(default=3, alias="CIRCUIT_BREAKER_THRESHOLD")
    """Number of consecutive failures before disabling an agent type (CONCEPT:AU-ORCH.execution.parallel-engine-visualizer)."""

    enable_progressive_synthesis: bool = Field(
        default=True, alias="ENABLE_PROGRESSIVE_SYNTHESIS"
    )
    """Enable streaming synthesis as agents complete (CONCEPT:AU-ORCH.execution.rlm-synthesis-failed-falling)."""

    # --- Innovation Framework (CONCEPT:AU-OS.state.cognitive-scheduler-preemption through CONCEPT:AU-OS.state.cognitive-scheduler-preemption) ---

    homeostatic_downgrade_enabled: bool = Field(
        default=True, alias="HOMEOSTATIC_DOWNGRADE_ENABLED"
    )
    """Enable automatic model tier downgrade when budget is under pressure (CONCEPT:AU-OS.state.cognitive-scheduler-preemption)."""

    adversarial_verification: bool = Field(
        default=False, alias="ADVERSARIAL_VERIFICATION"
    )
    """Enable adversarial verification pass (opt-in only, doubles verification cost) (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort)."""

    maintenance_token_budget: int = Field(default=0, alias="MAINTENANCE_TOKEN_BUDGET")
    """Token budget for autonomous maintenance cron (0 = unlimited) (CONCEPT:AU-OS.state.cognitive-scheduler-preemption)."""

    maintenance_priority: str = Field(default="LOW", alias="MAINTENANCE_PRIORITY")
    """Priority level for autonomous maintenance tasks (LOW/MEDIUM/HIGH) (CONCEPT:AU-OS.state.cognitive-scheduler-preemption)."""

    watchdog_patterns: list[str] = Field(
        default=[
            "pyproject.toml",
            "mcp_config.json",
            "requirements*.txt",
        ],
        alias="WATCHDOG_PATTERNS",
    )
    """File patterns to monitor for the file watcher trigger (CONCEPT:AU-OS.safety.doom-loop-detection)."""

    tool_guard_mode: Literal["on", "strict"] = Field(
        default="strict", alias="TOOL_GUARD_MODE"
    )
    developer_tool_max_output_bytes: int = Field(
        default=65_536,
        ge=1_024,
        le=4 * 1024 * 1024,
        alias="DEVELOPER_TOOL_MAX_OUTPUT_BYTES",
    )
    developer_tool_max_timeout_seconds: int = Field(
        default=600,
        ge=1,
        le=3_600,
        alias="DEVELOPER_TOOL_MAX_TIMEOUT_SECONDS",
    )
    sensitive_tool_patterns: list[str] = Field(
        default=[
            r".*delete.*",
            r".*remove.*",
            r".*rm_.*",
            r".*rmdir.*",
            r".*drop.*",
            r".*truncate.*",
            r".*prune.*",
            r".*kill.*",
            r".*terminate.*",
            r".*reboot.*",
            r".*shutdown.*",
            r".*install.*",
            r".*uninstall.*",
            r".*redeploy.*",
            r".*bump.*",
            r".*create.*",
            r".*add.*",
            r".*post.*",
            r".*put.*",
            r".*insert.*",
            r".*upload.*",
            r".*ingest.*",
            r".*write.*",
            r".*update.*",
            r".*patch.*",
            r".*set.*",
            r".*reset.*",
            r".*clear.*",
            r".*revert.*",
            r".*replace.*",
            r".*rename.*",
            r".*move.*",
            r".*rotate.*",
            r".*start.*",
            r".*stop.*",
            r".*restart.*",
            r".*pause.*",
            r".*unpause.*",
            r".*execute.*",
            r".*shell.*",
            r".*run_shell.*",
            r".*run_command.*",
            r".*run_script.*",
            r".*run_code.*",
            r".*git_.*",
            r".*clone.*",
            r".*pull.*",
            r".*maintain.*",
            r".*setup.*",
            r".*build.*",
            r".*validate.*",
            r".*sync.*",
            r".*enable.*",
            r".*disable.*",
            r".*activate.*",
            r".*approve.*",
            r".*graphql.*",
            r".*mutation.*",
            r".*http.*",
            r".*eval.*",
            r".*exec.*",
            r".*compile.*",
            r".*socket.*",
            r".*connect.*",
            r".*os\..*",
            r".*subprocess\..*",
            r".*shutil\..*",
        ],
        alias="SENSITIVE_TOOL_PATTERNS",
    )

    def assert_production_safe(self, *, profile: str | None = None) -> None:
        """Raise if this config uses toy defaults under a production profile.

        Delegates to :func:`agent_utilities.core.profile_guard.assert_production_safe`.
        No-op outside a production profile (see ``APP_PROFILE``). See
        ``docs/scaling/capacity_model.md`` for the scale knobs.
        """
        from agent_utilities.core.profile_guard import assert_production_safe

        assert_production_safe(self, profile=profile)


# --- Lazy Configuration Management ---


class AgentConfigProxy:
    """Stable public handle whose validated snapshot swaps under the XDG lock."""

    def __init__(self) -> None:
        object.__setattr__(self, "_target", None)

    def _swap(self, target: AgentConfig) -> None:
        object.__setattr__(self, "_target", target)

    def _current(self) -> AgentConfig:
        target = object.__getattribute__(self, "_target")
        if target is None:
            _init_lazy_config()
            target = object.__getattribute__(self, "_target")
        if target is None:  # pragma: no cover - defensive invariant
            raise RuntimeError("configuration snapshot is unavailable")
        return target

    def reload(self) -> "AgentConfigProxy":
        """Validate and publish a new snapshot while preserving this handle."""
        load_config(reload=True)
        return self

    def __getattr__(self, name: str) -> Any:
        with _xdg_projection_lock:
            return getattr(self._current(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_target":
            object.__setattr__(self, name, value)
            return
        with _xdg_projection_lock:
            setattr(self._current(), name, value)

    def __repr__(self) -> str:
        return "AgentConfigProxy(<redacted-snapshot>)"


class BoundedLRUCache:
    """A bounded, dict-like LRU cache.

    Behaves like a ``dict`` for the subset of operations used by the lazy
    configuration machinery (``__getitem__``, ``__setitem__``, ``__contains__``,
    ``get``), but never grows beyond ``max_size`` entries. When the cap is
    exceeded the least-recently-used entry is evicted.

    Recency is updated on both read (``__getitem__`` / ``get``) and write
    (``__setitem__``). This bounds memory for the process-wide configuration
    cache so it cannot grow without limit (e.g. under repeated reconfiguration
    or many derived keys).
    """

    def __init__(self, max_size: int = 4096) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.max_size = max_size
        self._data: OrderedDict[str, Any] = OrderedDict()

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        # Evict least-recently-used entries until within the cap.
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        self._data.move_to_end(key)
        return value

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._data:
            return self[key]
        return default

    def clear(self) -> None:
        self._data.clear()

    def keys(self):
        return self._data.keys()


# Maximum number of entries retained in the process-wide lazy config cache.
LAZY_CACHE_MAX_SIZE = 4096

_LAZY_CACHE: BoundedLRUCache = BoundedLRUCache(max_size=LAZY_CACHE_MAX_SIZE)
_CONFIG_PROXY = AgentConfigProxy()


def _populate_lazy_config(
    *, existing: AgentConfig | None = None, force: bool = False
) -> None:
    """Populate the currently selected staged lazy-cache generation.

    The public wrapper swaps this staged cache only while holding the projection
    lock. Existing typed objects remain immutable snapshots across hot reload.
    """
    if not force and "_config" in _LAZY_CACHE:
        return

    if force:
        _LAZY_CACHE.clear()

    if existing is None:
        _ensure_env_loaded()
        cfg = AgentConfig()
        # Wire the production guard into the real process configuration path.
        # Direct AgentConfig construction remains available to doctor/generator
        # tooling so it can diagnose an incomplete candidate instead of failing
        # before it can produce a structured report.
        cfg.assert_production_safe(profile=cfg.app_profile)
    else:
        cfg = existing
    _LAZY_CACHE["_config"] = cfg
    _LAZY_CACHE["config"] = _CONFIG_PROXY

    _LAZY_CACHE["DEFAULT_AGENT_NAME"] = cfg.default_agent_name
    _LAZY_CACHE["DEFAULT_AGENT_DESCRIPTION"] = cfg.agent_description
    _LAZY_CACHE["DEFAULT_AGENT_SYSTEM_PROMPT"] = cfg.agent_system_prompt
    _LAZY_CACHE["DEFAULT_DEBUG"] = cfg.debug

    # --- Derive DEFAULT_LLM_* from chat_models / embedding_models registry ---
    _default_chat = cfg.default_chat_model
    _lite_chat = cfg.lite_chat_model
    _super_chat = cfg.super_chat_model
    _default_embed = cfg.default_embedding_model

    _LAZY_CACHE["DEFAULT_LLM_PROVIDER"] = (
        (_default_chat.provider if _default_chat else None)
        or os.getenv("PROVIDER")
        or "openai"
    )
    _LAZY_CACHE["DEFAULT_LLM_MODEL_ID"] = (
        (_default_chat.id if _default_chat else None)
        or os.getenv("MODEL_ID")
        or "qwen/qwen3.6-27b"
    )
    _LAZY_CACHE["DEFAULT_LLM_BASE_URL"] = (
        _default_chat.base_url if _default_chat else None
    )
    _LAZY_CACHE["DEFAULT_LLM_API_KEY"] = (
        _default_chat.api_key_ref if _default_chat else None
    )

    _LAZY_CACHE["DEFAULT_LITE_LLM_PROVIDER"] = (
        _lite_chat.provider if _lite_chat else None
    ) or _LAZY_CACHE["DEFAULT_LLM_PROVIDER"]
    _LAZY_CACHE["DEFAULT_LITE_LLM_MODEL_ID"] = (
        _lite_chat.id if _lite_chat else None
    ) or _LAZY_CACHE["DEFAULT_LLM_MODEL_ID"]
    _LAZY_CACHE["DEFAULT_LITE_LLM_BASE_URL"] = (
        _lite_chat.base_url if _lite_chat else None
    ) or _LAZY_CACHE["DEFAULT_LLM_BASE_URL"]
    _LAZY_CACHE["DEFAULT_LITE_LLM_API_KEY"] = (
        _lite_chat.api_key_ref if _lite_chat else None
    ) or _LAZY_CACHE["DEFAULT_LLM_API_KEY"]

    _LAZY_CACHE["DEFAULT_SUPER_LLM_PROVIDER"] = (
        _super_chat.provider if _super_chat else None
    ) or _LAZY_CACHE["DEFAULT_LLM_PROVIDER"]
    _LAZY_CACHE["DEFAULT_SUPER_LLM_MODEL_ID"] = (
        _super_chat.id if _super_chat else None
    ) or _LAZY_CACHE["DEFAULT_LLM_MODEL_ID"]
    _LAZY_CACHE["DEFAULT_SUPER_LLM_BASE_URL"] = (
        _super_chat.base_url if _super_chat else None
    ) or _LAZY_CACHE["DEFAULT_LLM_BASE_URL"]
    _LAZY_CACHE["DEFAULT_SUPER_LLM_API_KEY"] = (
        _super_chat.api_key_ref if _super_chat else None
    ) or _LAZY_CACHE["DEFAULT_LLM_API_KEY"]

    _LAZY_CACHE["DEFAULT_EMBEDDING_PROVIDER"] = (
        _default_embed.provider if _default_embed else None
    ) or _LAZY_CACHE["DEFAULT_LLM_PROVIDER"]
    _LAZY_CACHE["DEFAULT_EMBEDDING_MODEL_ID"] = (
        _default_embed.id if _default_embed else None
    ) or "text-embedding-nomic-embed-text-v2-moe"
    _LAZY_CACHE["DEFAULT_EMBEDDING_BASE_URL"] = (
        _default_embed.base_url if _default_embed else None
    ) or _LAZY_CACHE["DEFAULT_LLM_BASE_URL"]
    _LAZY_CACHE["DEFAULT_EMBEDDING_API_KEY"] = (
        _default_embed.api_key_ref if _default_embed else None
    ) or _LAZY_CACHE["DEFAULT_LLM_API_KEY"]
    _LAZY_CACHE["DEFAULT_MCP_URL"] = cfg.mcp_url

    _LAZY_CACHE["DEFAULT_MCP_CONFIG"] = cfg.mcp_config
    _LAZY_CACHE["DEFAULT_CUSTOM_SKILLS_DIRECTORY"] = cfg.custom_skills_directory
    _LAZY_CACHE["DEFAULT_SKILL_TYPES"] = cfg.skill_types
    _LAZY_CACHE["DEFAULT_ENABLE_WEB_UI"] = cfg.enable_web_ui
    _LAZY_CACHE["DEFAULT_ENABLE_TERMINAL_UI"] = cfg.enable_terminal_ui
    _LAZY_CACHE["DEFAULT_ENABLE_WEB_LOGS"] = cfg.enable_web_logs
    _LAZY_CACHE["DEFAULT_ENABLE_OTEL"] = cfg.enable_otel
    _LAZY_CACHE["DEFAULT_ENABLE_ACP"] = cfg.enable_acp
    _LAZY_CACHE["DEFAULT_ACP_PORT"] = cfg.acp_port
    _LAZY_CACHE["DEFAULT_ACP_SESSION_ROOT"] = cfg.acp_session_root

    _apply_otel_sdk_policy(cfg.enable_otel)

    _LAZY_CACHE["DEFAULT_OTEL_EXPORTER_OTLP_ENDPOINT"] = cfg.otel_exporter_otlp_endpoint
    _LAZY_CACHE["DEFAULT_OTEL_EXPORTER_OTLP_PROTOCOL"] = cfg.otel_exporter_otlp_protocol

    _LAZY_CACHE["DEFAULT_LANGFUSE_HOST"] = cfg.langfuse_host
    _LAZY_CACHE[
        "DEFAULT_LANGFUSE_DATASET_CAPTURE_THRESHOLD"
    ] = cfg.langfuse_dataset_capture_threshold

    _LAZY_CACHE["DEFAULT_A2A_BROKER"] = cfg.a2a_broker
    _LAZY_CACHE["DEFAULT_A2A_STORAGE"] = cfg.a2a_storage
    _LAZY_CACHE["DEFAULT_A2A_CONFIG"] = cfg.a2a_config
    _LAZY_CACHE["DEFAULT_A2A_REFRESH_INTERVAL"] = cfg.a2a_refresh_interval

    _LAZY_CACHE["DEFAULT_MAX_TOKENS"] = cfg.max_tokens
    _LAZY_CACHE["DEFAULT_TEMPERATURE"] = cfg.temperature
    _LAZY_CACHE["DEFAULT_TOP_P"] = cfg.top_p
    _LAZY_CACHE["DEFAULT_TIMEOUT"] = cfg.timeout
    _LAZY_CACHE["DEFAULT_TOOL_TIMEOUT"] = cfg.tool_timeout
    _LAZY_CACHE["DEFAULT_PARALLEL_TOOL_CALLS"] = cfg.parallel_tool_calls
    _LAZY_CACHE["DEFAULT_SEED"] = cfg.seed
    _LAZY_CACHE["DEFAULT_PRESENCE_PENALTY"] = cfg.presence_penalty
    _LAZY_CACHE["DEFAULT_FREQUENCY_PENALTY"] = cfg.frequency_penalty

    _LAZY_CACHE["DEFAULT_LOGIT_BIAS"] = (
        cfg.logit_bias
        if cfg.logit_bias is not None
        else to_dict(os.getenv("LOGIT_BIAS"))
    )
    _LAZY_CACHE["DEFAULT_STOP_SEQUENCES"] = (
        cfg.stop_sequences
        if cfg.stop_sequences is not None
        else to_list(os.getenv("STOP_SEQUENCES"))
    )
    _LAZY_CACHE["DEFAULT_EXTRA_HEADERS"] = (
        cfg.extra_headers
        if cfg.extra_headers is not None
        else to_dict(os.getenv("EXTRA_HEADERS"))
    )
    _LAZY_CACHE["DEFAULT_EXTRA_BODY"] = (
        cfg.extra_body
        if cfg.extra_body is not None
        else to_dict(os.getenv("EXTRA_BODY"))
    )

    _LAZY_CACHE["DEFAULT_MIN_CONFIDENCE"] = cfg.min_confidence
    _LAZY_CACHE["DEFAULT_VALIDATION_MODE"] = (
        cfg.validation_mode
        or to_boolean(os.getenv("VALIDATION_MODE", "False"))
        or to_boolean(os.getenv("AGENT_UTILITIES_TESTING", "False"))
    )
    _LAZY_CACHE["DEFAULT_APPROVAL_TIMEOUT"] = cfg.approval_timeout
    _LAZY_CACHE["DEFAULT_MAX_CRON_LOG_ENTRIES"] = 50

    _LAZY_CACHE["TOOL_GUARD_MODE"] = cfg.tool_guard_mode
    _LAZY_CACHE["SENSITIVE_TOOL_PATTERNS"] = cfg.sensitive_tool_patterns

    # Router/KG models: find models with can_route/can_kg flags, else fallback to lite
    _router_model = next((m for m in cfg.chat_models if m.can_route), _lite_chat)
    _kg_model = next((m for m in cfg.chat_models if m.can_kg), _lite_chat)
    _LAZY_CACHE["DEFAULT_ROUTER_MODEL"] = (
        _router_model.id if _router_model else None
    ) or _LAZY_CACHE["DEFAULT_LITE_LLM_MODEL_ID"]

    _LAZY_CACHE["DEFAULT_GRAPH_PERSISTENCE_TYPE"] = cfg.graph_persistence_type
    _LAZY_CACHE["DEFAULT_GRAPH_PERSISTENCE_PATH"] = cfg.graph_persistence_path
    _LAZY_CACHE["DEFAULT_ENABLE_LLM_VALIDATION"] = cfg.enable_llm_validation
    _LAZY_CACHE["DEFAULT_ROUTING_STRATEGY"] = cfg.routing_strategy
    _LAZY_CACHE["DEFAULT_GRAPH_ROUTER_TIMEOUT"] = cfg.graph_router_timeout
    _LAZY_CACHE["DEFAULT_GRAPH_VERIFIER_TIMEOUT"] = cfg.graph_verifier_timeout
    _LAZY_CACHE["DEFAULT_ENABLE_KG_EMBEDDINGS"] = cfg.enable_kg_embeddings
    _LAZY_CACHE["DEFAULT_KG_BACKUPS"] = cfg.kg_backups
    _LAZY_CACHE["DEFAULT_KG_INGESTION_WORKERS"] = cfg.kg_ingestion_workers
    _LAZY_CACHE["DEFAULT_KG_LLM_CONCURRENCY"] = cfg.kg_llm_concurrency
    _LAZY_CACHE["DEFAULT_KG_MODEL_ID"] = (
        _kg_model.id if _kg_model else None
    ) or _LAZY_CACHE["DEFAULT_LITE_LLM_MODEL_ID"]
    _LAZY_CACHE["DEFAULT_KG_ANALYSIS_MAX_DEPTH"] = cfg.kg_analysis_max_depth
    _LAZY_CACHE["DEFAULT_KNOWLEDGE_GRAPH_SYNC_BACKGROUND"] = (
        cfg.knowledge_graph_sync_background
    )
    # --- Parallel Engine Defaults ---
    _LAZY_CACHE["DEFAULT_MAX_PARALLEL_AGENTS"] = cfg.max_parallel_agents
    _LAZY_CACHE["DEFAULT_PARALLEL_BATCH_SIZE"] = cfg.parallel_batch_size
    _LAZY_CACHE["DEFAULT_SYNTHESIS_STRATEGY"] = cfg.synthesis_strategy
    _LAZY_CACHE["DEFAULT_SYNTHESIS_RATIO"] = cfg.synthesis_ratio
    _LAZY_CACHE["DEFAULT_AGENT_EXECUTION_TIMEOUT"] = cfg.agent_execution_timeout
    _LAZY_CACHE["DEFAULT_CIRCUIT_BREAKER_THRESHOLD"] = cfg.circuit_breaker_threshold
    _LAZY_CACHE[
        "DEFAULT_ENABLE_PROGRESSIVE_SYNTHESIS"
    ] = cfg.enable_progressive_synthesis

    _LAZY_CACHE["MAX_UPLOAD_SIZE"] = cfg.max_upload_size

    _LAZY_CACHE["SECRETS_BACKEND"] = cfg.secrets_backend
    _LAZY_CACHE["SECRETS_VAULT_URL"] = cfg.vault_url
    _LAZY_CACHE["SECRETS_VAULT_MOUNT"] = cfg.vault_mount

    _LAZY_CACHE["AUTH_JWT_JWKS_URI"] = cfg.auth_jwt_jwks_uri
    _LAZY_CACHE["AUTH_JWT_ISSUER"] = cfg.auth_jwt_issuer
    _LAZY_CACHE["AUTH_JWT_AUDIENCE"] = cfg.auth_jwt_audience
    _LAZY_CACHE["KG_POLICY_VERSION"] = cfg.kg_policy_version
    _LAZY_CACHE["ALLOWED_ORIGINS"] = cfg.allowed_origins
    _LAZY_CACHE["ALLOWED_HOSTS"] = cfg.allowed_hosts

    # Agent OS Architecture defaults
    _LAZY_CACHE["DEFAULT_COGNITIVE_SCHEDULER_ENABLED"] = cfg.cognitive_scheduler_enabled
    _LAZY_CACHE["DEFAULT_MAX_CONCURRENT_AGENTS"] = cfg.max_concurrent_agents
    _LAZY_CACHE["DEFAULT_AGENT_TOKEN_QUOTA"] = cfg.agent_token_quota
    _LAZY_CACHE["DEFAULT_PREEMPTION_THRESHOLD_PCT"] = cfg.preemption_threshold_pct
    _LAZY_CACHE["DEFAULT_AGENT_POLICIES_PATH"] = cfg.agent_policies_path
    _LAZY_CACHE["DEFAULT_PERMISSIONS_SIGNING_KEY_REF"] = cfg.permissions_signing_key_ref
    _LAZY_CACHE["DEFAULT_SPECIALIST_REGISTRY_PATH"] = cfg.specialist_registry_path

    # Innovation Framework defaults
    _LAZY_CACHE["DEFAULT_HOMEOSTATIC_DOWNGRADE"] = cfg.homeostatic_downgrade_enabled
    _LAZY_CACHE["DEFAULT_ADVERSARIAL_VERIFICATION"] = cfg.adversarial_verification
    _LAZY_CACHE["DEFAULT_MAINTENANCE_TOKEN_BUDGET"] = cfg.maintenance_token_budget
    _LAZY_CACHE["DEFAULT_MAINTENANCE_PRIORITY"] = cfg.maintenance_priority
    _LAZY_CACHE["DEFAULT_WATCHDOG_PATTERNS"] = cfg.watchdog_patterns


def _init_lazy_config(
    *, existing: AgentConfig | None = None, force: bool = False
) -> None:
    """Build and publish one complete lazy configuration generation."""
    global _LAZY_CACHE
    with _xdg_projection_lock:
        if not force and "_config" in _LAZY_CACHE:
            return
        previous = _LAZY_CACHE
        staged = BoundedLRUCache(max_size=LAZY_CACHE_MAX_SIZE)
        _LAZY_CACHE = staged
        try:
            _populate_lazy_config(existing=existing, force=False)
            _CONFIG_PROXY._swap(staged["_config"])
        except Exception:
            _LAZY_CACHE = previous
            raise


if TYPE_CHECKING:
    # These names are materialized at runtime via module ``__getattr__`` (PEP 562)
    # from the lazy cache. Declare their concrete types here so importers get
    # real typing instead of ``Any``.
    config: AgentConfig
    SENSITIVE_TOOL_PATTERNS: list[str]
    TOOL_GUARD_MODE: Literal["on", "strict"]
    DEFAULT_EMBEDDING_BASE_URL: str
    DEFAULT_EMBEDDING_MODEL_ID: str
    DEFAULT_KG_ANALYSIS_MAX_DEPTH: int


def __getattr__(name: str) -> Any:
    # Handle the decoupled HOST/PORT directly for instant resolution
    if name == "DEFAULT_HOST":
        return os.environ.get("HOST", "127.0.0.1")
    if name == "DEFAULT_PORT":
        try:
            return int(os.environ.get("PORT", "9000"))
        except ValueError:
            return 9000

    if name.startswith("__"):
        raise AttributeError(name)

    with _xdg_projection_lock:
        _init_lazy_config()

        if name in _LAZY_CACHE:
            return _LAZY_CACHE[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        list(globals().keys())
        + [
            "config",
            "DEFAULT_AGENT_NAME",
            "DEFAULT_AGENT_DESCRIPTION",
            "DEFAULT_AGENT_SYSTEM_PROMPT",
            "DEFAULT_HOST",
            "DEFAULT_PORT",
            "DEFAULT_DEBUG",
            "DEFAULT_LLM_PROVIDER",
            "DEFAULT_LLM_MODEL_ID",
            "DEFAULT_LLM_BASE_URL",
            "DEFAULT_LLM_API_KEY",
            "DEFAULT_LITE_LLM_PROVIDER",
            "DEFAULT_LITE_LLM_MODEL_ID",
            "DEFAULT_LITE_LLM_BASE_URL",
            "DEFAULT_LITE_LLM_API_KEY",
            "DEFAULT_SUPER_LLM_PROVIDER",
            "DEFAULT_SUPER_LLM_MODEL_ID",
            "DEFAULT_SUPER_LLM_BASE_URL",
            "DEFAULT_SUPER_LLM_API_KEY",
            "DEFAULT_EMBEDDING_PROVIDER",
            "DEFAULT_EMBEDDING_MODEL_ID",
            "DEFAULT_EMBEDDING_BASE_URL",
            "DEFAULT_EMBEDDING_API_KEY",
            "DEFAULT_MCP_URL",
            "DEFAULT_MCP_CONFIG",
            "DEFAULT_CUSTOM_SKILLS_DIRECTORY",
            "DEFAULT_SKILL_TYPES",
            "DEFAULT_ENABLE_WEB_UI",
            "DEFAULT_ENABLE_TERMINAL_UI",
            "DEFAULT_ENABLE_WEB_LOGS",
            "DEFAULT_ENABLE_OTEL",
            "DEFAULT_ENABLE_ACP",
            "DEFAULT_ACP_PORT",
            "DEFAULT_ACP_SESSION_ROOT",
            "DEFAULT_OTEL_EXPORTER_OTLP_ENDPOINT",
            "DEFAULT_OTEL_EXPORTER_OTLP_PROTOCOL",
            "DEFAULT_LANGFUSE_HOST",
            "DEFAULT_LANGFUSE_DATASET_CAPTURE_THRESHOLD",
            "DEFAULT_A2A_BROKER",
            "DEFAULT_A2A_STORAGE",
            "DEFAULT_A2A_CONFIG",
            "DEFAULT_A2A_REFRESH_INTERVAL",
            "DEFAULT_MAX_TOKENS",
            "DEFAULT_TEMPERATURE",
            "DEFAULT_TOP_P",
            "DEFAULT_TIMEOUT",
            "DEFAULT_TOOL_TIMEOUT",
            "DEFAULT_PARALLEL_TOOL_CALLS",
            "DEFAULT_SEED",
            "DEFAULT_PRESENCE_PENALTY",
            "DEFAULT_FREQUENCY_PENALTY",
            "DEFAULT_LOGIT_BIAS",
            "DEFAULT_STOP_SEQUENCES",
            "DEFAULT_EXTRA_HEADERS",
            "DEFAULT_EXTRA_BODY",
            "DEFAULT_MIN_CONFIDENCE",
            "DEFAULT_VALIDATION_MODE",
            "DEFAULT_APPROVAL_TIMEOUT",
            "DEFAULT_MAX_CRON_LOG_ENTRIES",
            "TOOL_GUARD_MODE",
            "SENSITIVE_TOOL_PATTERNS",
            "DEFAULT_ROUTER_MODEL",
            "DEFAULT_GRAPH_PERSISTENCE_TYPE",
            "DEFAULT_GRAPH_PERSISTENCE_PATH",
            "DEFAULT_ENABLE_LLM_VALIDATION",
            "DEFAULT_ROUTING_STRATEGY",
            "DEFAULT_GRAPH_ROUTER_TIMEOUT",
            "DEFAULT_GRAPH_VERIFIER_TIMEOUT",
            "DEFAULT_ENABLE_KG_EMBEDDINGS",
            "DEFAULT_KG_BACKUPS",
            "DEFAULT_KG_INGESTION_WORKERS",
            "DEFAULT_KG_LLM_CONCURRENCY",
            "DEFAULT_KG_MODEL_ID",
            "DEFAULT_KG_ANALYSIS_MAX_DEPTH",
            "DEFAULT_KNOWLEDGE_GRAPH_SYNC_BACKGROUND",
            "DEFAULT_MAX_PARALLEL_AGENTS",
            "DEFAULT_PARALLEL_BATCH_SIZE",
            "DEFAULT_SYNTHESIS_STRATEGY",
            "DEFAULT_SYNTHESIS_RATIO",
            "DEFAULT_AGENT_EXECUTION_TIMEOUT",
            "DEFAULT_CIRCUIT_BREAKER_THRESHOLD",
            "DEFAULT_ENABLE_PROGRESSIVE_SYNTHESIS",
            "MAX_UPLOAD_SIZE",
            "SECRETS_BACKEND",
            "SECRETS_VAULT_URL",
            "SECRETS_VAULT_MOUNT",
            "AUTH_JWT_JWKS_URI",
            "AUTH_JWT_ISSUER",
            "AUTH_JWT_AUDIENCE",
            "KG_POLICY_VERSION",
            "ALLOWED_ORIGINS",
            "ALLOWED_HOSTS",
            "DEFAULT_COGNITIVE_SCHEDULER_ENABLED",
            "DEFAULT_MAX_CONCURRENT_AGENTS",
            "DEFAULT_AGENT_TOKEN_QUOTA",
            "DEFAULT_PREEMPTION_THRESHOLD_PCT",
            "DEFAULT_AGENT_POLICIES_PATH",
            "DEFAULT_PERMISSIONS_SIGNING_KEY_REF",
            "DEFAULT_SPECIALIST_REGISTRY_PATH",
            "DEFAULT_HOMEOSTATIC_DOWNGRADE",
            "DEFAULT_ADVERSARIAL_VERIFICATION",
            "DEFAULT_MAINTENANCE_TOKEN_BUDGET",
            "DEFAULT_MAINTENANCE_PRIORITY",
            "DEFAULT_WATCHDOG_PATTERNS",
        ]
    )


# --- Migrated from graph/config_helpers.py ---
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from agent_utilities.base_utilities import to_integer
from agent_utilities.core.workspace import CORE_FILES, get_workspace_path
from agent_utilities.models import (
    MCPAgent,
    MCPAgentRegistryModel,
    MCPConfigModel,
    MCPToolInfo,
)

logger = logging.getLogger(__name__)

import os

# Whole-workflow orchestration execution budget (ms). 10min default (was 20min):
# the client's per-RPC timeout now catches engine hangs in seconds, so this only
# bounds a wedged multi-agent run. Override via GRAPH_TIMEOUT for unusually long
# workflows.
DEFAULT_GRAPH_TIMEOUT = to_integer(os.environ.get("GRAPH_TIMEOUT", "600000"))


# ---------------------------------------------------------------------------
# CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — Session-Scoped Registry Cache
# ---------------------------------------------------------------------------


class _RegistryCache:
    """Session-scoped cache for KG registry data.

    CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — Hot Cache Layer

    Populated on first access, invalidated by explicit event signals.
    No TTL — pure event-driven invalidation from four callsites:

    1. ``agent_manager.sync_mcp_agents()`` (MCP reload)
    2. Pipeline completion (``PipelineRunner.run()``)
    3. ``promote_coalition_to_template()`` (TeamConfig creation)
    4. ``MemoryRetriever.update_after_session()`` (proficiency update)
    """

    _registry: MCPAgentRegistryModel | None = None
    _prompts: dict[str, str] = {}
    _tool_agent_map: dict[str, list[str]] = {}

    @classmethod
    def invalidate(cls) -> None:
        """Clear all cached data.  Called by event-driven signals."""
        cls._registry = None
        cls._prompts.clear()
        cls._tool_agent_map.clear()
        logger.info(
            "[CACHE] Registry cache invalidated (CONCEPT:AU-ORCH.adapter.hot-cache-invalidation)."
        )

    @classmethod
    def get_registry(cls) -> MCPAgentRegistryModel:
        """Return the cached registry, populating on first access."""
        if cls._registry is None:
            cls._registry = _fetch_registry_from_kg()
            logger.info(
                "[CACHE] Registry cache populated: %d agents, %d tools.",
                len(cls._registry.agents),
                len(cls._registry.tools),
            )
        return cls._registry


def invalidate_registry_cache() -> None:
    """Public API to invalidate the hot cache.

    CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — Hot Cache Layer

    Call this after any operation that changes the registry state:
    MCP reload, pipeline ingestion, TeamConfig promotion, or
    Self-Model update.
    """
    _RegistryCache.invalidate()


def _fetch_registry_from_kg() -> MCPAgentRegistryModel:
    """Fetch the full registry from the Knowledge Graph (uncached).

    This is the expensive operation that ``_RegistryCache`` wraps.
    Delegates to focused sub-functions for each data source.
    """
    if __import__("os").getenv("ENABLE_KG_REGISTRY_FETCH", "true").lower() in (
        "false",
        "0",
        "no",
    ):
        logger.info("Registry fetch bypassed via environment variable.")
        return MCPAgentRegistryModel()

    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if not engine:
        from agent_utilities.core.paths import kg_db_path

        db_path = str(kg_db_path())
        engine = IntelligenceGraphEngine.get_or_create(db_path=db_path)

    if not engine or not engine.backend:
        return MCPAgentRegistryModel()

    agents: list[MCPAgent] = []
    agents.extend(_fetch_prompt_agents(engine))
    agents.extend(_fetch_specialist_agents(engine))

    tools = _fetch_tools(engine)
    agents.extend(_synthesize_partition_agents(tools, {a.name for a in agents}))

    return MCPAgentRegistryModel(agents=agents, tools=tools)


def _fetch_prompt_agents(engine: Any) -> list[MCPAgent]:
    """Fetch Prompt-based agents from the KG."""
    agents: list[MCPAgent] = []
    try:
        prompt_rows = engine.backend.execute(
            "MATCH (p:Prompt) RETURN p.name AS name, p.description AS descriptionription, p.capabilities AS capabilities, p.system_prompt AS system_prompt, p.json_blueprint AS json_blueprint"
        )
        for row in prompt_rows:
            blueprint = row.get("json_blueprint")
            if isinstance(blueprint, str):
                try:
                    blueprint = json.loads(blueprint)
                except (TypeError, json.JSONDecodeError):
                    logger.debug("Rejected non-JSON prompt blueprint")
                    continue

            if blueprint and not isinstance(blueprint, dict):
                logger.debug("Rejected non-object prompt blueprint")
                continue

            parsed_blueprint: dict[str, Any] | None = (
                blueprint if isinstance(blueprint, dict) else None
            )
            if parsed_blueprint is not None:
                from agent_utilities.prompting.structured import validate_canonical

                if validate_canonical(parsed_blueprint):
                    logger.debug("Rejected non-canonical prompt blueprint")
                    continue
            agents.append(
                MCPAgent(
                    name=row.get("name", ""),
                    description=row.get("description", ""),
                    agent_type="specialist",
                    capabilities=row.get("capabilities", []),
                    system_prompt=row.get("system_prompt", ""),
                    json_blueprint=parsed_blueprint,
                )
            )
    except Exception as e:
        logger.debug(f"Failed to fetch Prompt nodes: {e}")
    return agents


def _fetch_specialist_agents(engine: Any) -> list[MCPAgent]:
    """Fetch Agent-type specialist nodes from the KG."""
    agents: list[MCPAgent] = []
    try:
        agent_rows = engine.backend.execute(
            "MATCH (a:Agent) RETURN a.name AS name, a.description AS descriptionription, a.agent_type AS agent_type, a.system_prompt AS system_prompt, a.tool_count AS tool_count, a.mcp_server AS mcp_server"
        )
        for row in agent_rows:
            agent_type = str(row.get("agent_type") or "")
            if agent_type not in {"specialist", "a2a"}:
                raise RuntimeError("stored Agent uses a non-current agent_type")
            agents.append(
                MCPAgent(
                    name=row.get("name", "unknown"),
                    description=row.get("description", ""),
                    agent_type=agent_type,
                    system_prompt=row.get("system_prompt", ""),
                    tool_count=row.get("tool_count", 0),
                    mcp_server=row.get("mcp_server"),
                )
            )
    except Exception as e:
        logger.debug(f"Failed to fetch specialist agents from KG: {e}")
    return agents


def _fetch_tools(engine: Any) -> list[MCPToolInfo]:
    """Fetch Tool nodes from the KG."""
    tools: list[MCPToolInfo] = []
    try:
        tool_rows = engine.backend.execute(
            "MATCH (t:Tool) RETURN t.name, t.description, t.mcp_server, t.relevance_score, t.tags, t.requires_approval"
        )
        for row in tool_rows:
            tools.append(
                MCPToolInfo(
                    name=row.get("t.name", ""),
                    description=row.get("t.description", ""),
                    mcp_server=row.get("t.mcp_server", "unknown"),
                    relevance_score=row.get("t.relevance_score", 0),
                    all_tags=row.get("t.tags", []),
                    requires_approval=row.get("t.requires_approval", False),
                )
            )
    except Exception as e:
        logger.debug(f"Failed to fetch Tool nodes: {e}")
    return tools


def _synthesize_partition_agents(
    tools: list[MCPToolInfo],
    existing_agent_names: set[str],
) -> list[MCPAgent]:
    """Synthesize partition-based agents from tool tags.

    CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — Re-derive Server Agents from Tools (Dynamic Partitioning at read-time)
    """
    partitions: dict[str, list[MCPToolInfo]] = {}
    for t in tools:
        tags = t.all_tags if t.all_tags else ([t.tag] if t.tag else [])
        server_tag = (
            t.mcp_server.lower()
            .replace("-mcp", "")
            .replace("_mcp", "")
            .replace("-manager", "")
            .replace("-agent", "")
            .replace("-server", "")
        )
        if not tags or tags == ["general"]:
            all_partition_tags = {f"{t.mcp_server}_general"}
        else:
            all_partition_tags = set(tags)
            all_partition_tags.add(server_tag)

        for tag in all_partition_tags:
            if tag not in partitions:
                partitions[tag] = []
            partitions[tag].append(t)

    agents: list[MCPAgent] = []
    for tag, partition_tools in partitions.items():
        if tag in existing_agent_names:
            continue

        mcp_servers = list(set(t.mcp_server for t in partition_tools))
        primary_server = mcp_servers[0] if mcp_servers else "unknown"

        agents.append(
            MCPAgent(
                name=tag,
                description=f"Dynamically synthesized agent for {tag} capabilities.",
                agent_type="specialist",
                system_prompt=f"You are the {tag} specialist.",
                tool_count=len(partition_tools),
                mcp_server=primary_server,
                tools=[t.name for t in partition_tools],
                capabilities=list(
                    set(
                        c_tag
                        for t in partition_tools
                        for c_tag in (
                            t.all_tags if t.all_tags else ([t.tag] if t.tag else [])
                        )
                    )
                ),
            )
        )

    return agents


def get_discovery_registry() -> MCPAgentRegistryModel:
    """Load the unified agent discovery registry (cached).

    CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — Hot Cache Layer

    Returns the registry from the in-memory cache.  On first call,
    populates the cache from the Knowledge Graph.  Subsequent calls
    are O(1) until ``invalidate_registry_cache()`` is called.

    Returns:
        The populated MCPAgentRegistryModel.
    """
    return _RegistryCache.get_registry()


def get_relevant_specialists(
    query: str,
    engine: Any | None = None,
    top_n: int = 7,
) -> list[MCPAgent]:
    """Return the top-N adaptive_agent_router most relevant to a query.

    CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — Hot Cache Layer

    Uses KG discovery results (hybrid search + tool matching) to filter
    the full specialist list down to the most relevant agents for a
    given query.  Falls back to the full list if KG discovery returns
    nothing or the engine is unavailable.

    Args:
        query: The user query to match against.
        engine: Optional ``IntelligenceGraphEngine`` for hybrid search.
        top_n: Maximum number of adaptive_agent_router to return.

    Returns:
        A list of the most relevant ``MCPAgent`` objects.
    """
    registry = get_discovery_registry()
    all_agents = registry.agents

    if not all_agents:
        return []

    if not engine or not query:
        return all_agents[:top_n]

    # Use hybrid search to find relevant nodes
    try:
        results = engine.search_hybrid(query, top_k=top_n * 3)
        matched_names: set[str] = set()
        for r in results:
            name = r.get("name", "")
            if name:
                matched_names.add(name.lower())
            # Also check the node type for agent/prompt matches
            node_type = str(r.get("type", "")).lower()
            if node_type in ("agent", "prompt"):
                matched_names.add(name.lower())

        # Score agents by whether they appear in search results
        relevant = [a for a in all_agents if a.name.lower() in matched_names]

        if relevant:
            return relevant[:top_n]
    except Exception as e:
        logger.debug(f"Hybrid search for adaptive_agent_router failed: {e}")

    # Fallback: return all agents (capped)
    return all_agents[:top_n]


def load_mcp_config() -> MCPConfigModel:
    """Retrieve the global MCP server configuration from the workspace.

    Loads the mcp_config.json file which contains the definitions of
    external MCP servers (e.g., Docker, GitHub) and their connection
    parameters.

    Returns:
        An MCPConfigModel object containing server definitions and settings.

    """
    path = get_workspace_path(CORE_FILES["MCP_CONFIG"])
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return MCPConfigModel.model_validate(data)
        except Exception:
            return MCPConfigModel()
    return MCPConfigModel()


def save_mcp_config(config: MCPConfigModel):
    """Persist the MCP configuration model back to the workspace file.

    Args:
        config: The MCPConfigModel to be saved.

    """
    path = get_workspace_path(CORE_FILES["MCP_CONFIG"])
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")


def emit_graph_event(eq: asyncio.Queue[Any] | None, event_type: str, **kwargs):
    """Emit a standardized graph event for real-time UI visualization.

    Formats the event data as a sideband part compatible with the
    Agentic UI streaming protocol, allowing the frontend to visualize
    graph progression and tool activity.  Also emits a structured log
    line so the full execution trace is visible in server-side logs
    without requiring the UI.

    Args:
        eq: The asynchronous event queue to publish to.
        event_type: A string identifier for the event category.
        **kwargs: Additional metadata to include in the event payload.

    """
    ts = time.time()
    trace_kwargs = {k: v for k, v in kwargs.items() if k != "timestamp"}
    _log_graph_trace(event_type, ts, **trace_kwargs)

    if not eq:
        return

    try:
        eq.put_nowait(
            {
                "type": "data-graph-event",
                "data": {
                    "event": event_type,
                    "timestamp": ts,
                    **kwargs,
                },
            }
        )
    except Exception as e:
        logger.warning(f"Failed to emit graph event '{event_type}': {e}")


# ---------------------------------------------------------------------------
# Structured graph trace logging
# ---------------------------------------------------------------------------

_graph_trace_logger = logging.getLogger("agent_utilities.graph.trace")

_PHASE_MAP: dict[str, str] = {
    # ── Lifecycle ──────────────────────────────────────────────────────
    "graph_start": "LIFECYCLE",
    "graph_complete": "LIFECYCLE",
    "node_start": "LIFECYCLE",
    "node_complete": "LIFECYCLE",
    # ── Safety & Policy ───────────────────────────────────────────────
    "safety_warning": "SAFETY",
    # ── Routing & Planning ────────────────────────────────────────────
    "routing_started": "ROUTING",
    "routing_completed": "ROUTING",
    "plan_created": "PLANNING",
    "replanning_started": "REPLANNING",
    "replanning_completed": "REPLANNING",
    # ── Dispatch ──────────────────────────────────────────────────────
    "step_dispatched": "DISPATCH",
    "batch_dispatched": "DISPATCH",
    # ── Context Enrichment ────────────────────────────────────────────
    "context_gap_detected": "ENRICHMENT",
    # ── Specialist Execution ──────────────────────────────────────────
    "specialist_enter": "EXECUTION",
    "specialist_exit": "EXECUTION",
    "specialist_fallback": "FALLBACK",
    "expert_metadata": "EXECUTION",
    "expert_thinking": "EXECUTION",
    "expert_warning": "EXECUTION",
    "expert_text": "EXECUTION",
    "expert_complete": "EXECUTION",
    "tools_bound": "EXECUTION",
    "subagent_started": "EXECUTION",
    "subagent_completed": "EXECUTION",
    "subagent_thought": "EXECUTION",
    # ── Tool Calls ────────────────────────────────────────────────────
    "expert_tool_call": "TOOL_CALL",
    "subagent_tool_call": "TOOL_CALL",
    "tool_result": "TOOL_RESULT",
    # ── Parallel / Orthogonal Regions ─────────────────────────────────
    "orthogonal_regions_start": "PARALLEL",
    "orthogonal_regions_complete": "PARALLEL",
    "region_start": "PARALLEL",
    "region_complete": "PARALLEL",
    # ── Verification & Synthesis ──────────────────────────────────────
    "verification_result": "VERIFICATION",
    "agent_node_delta": "SYNTHESIS",
    "synthesis_fallback": "SYNTHESIS",
    # ── Human-in-the-Loop ─────────────────────────────────────────────
    "approval_required": "APPROVAL",
    "approval_resolved": "APPROVAL",
    "elicitation": "APPROVAL",
    # ── Recovery & Termination ────────────────────────────────────────
    "error_recovery_replan": "RECOVERY",
    "error_recovery_terminal": "RECOVERY",
    "graph_force_terminated": "TERMINATION",
    # ── Council Deliberation ──────────────────────────────────────────
    "council_started": "COUNCIL",
    "council_stage": "COUNCIL",
    "council_advisor_complete": "COUNCIL",
    "council_reviewer_complete": "COUNCIL",
    "council_completed": "COUNCIL",
    # ── KG-Driven Graph Materialization (CONCEPT:AU-ORCH.adapter.kg-graph-materialization) ─────────────
    "kg_query_start": "KG_BRIDGE",
    "kg_query_complete": "KG_BRIDGE",
    "kg_template_resolved": "KG_BRIDGE",
    "kg_prompt_injected": "KG_BRIDGE",
    "kg_topology_materialized": "KG_BRIDGE",
}


def _log_graph_trace(event_type: str, timestamp: float, **kwargs):
    """Emit a structured log line for a graph event."""
    phase = _PHASE_MAP.get(event_type, "GRAPH")
    detail_parts: list[str] = []

    for key in ("agent", "expert", "node_id", "id", "domain", "server"):
        if key in kwargs:
            detail_parts.append(f"{key}={kwargs[key]}")
    for key in ("count", "score", "batch_size", "attempt", "duration_ms"):
        if key in kwargs:
            detail_parts.append(f"{key}={kwargs[key]}")
    if "tool_name" in kwargs:
        detail_parts.append(f"tool={kwargs['tool_name']}")
    if "success" in kwargs:
        detail_parts.append(f"ok={kwargs['success']}")
    if "message" in kwargs and event_type in ("expert_warning", "safety_warning"):
        detail_parts.append(f"msg={kwargs['message'][:120]}")

    detail = " ".join(detail_parts) if detail_parts else ""
    _graph_trace_logger.info(f"[{phase}] {event_type} {detail}".rstrip())


def _render_prompt_payload(data: dict[str, Any]) -> str:
    """Render a prompt blueprint dict to the string the LLM should see.

    Every blueprint is validated against the one current
    :class:`StructuredPrompt` contract before it reaches ``system_prompt=``.
    """
    from agent_utilities.prompting.structured import StructuredPrompt

    return StructuredPrompt.model_validate(data).render()


def load_specialized_prompts(prompt_name: str) -> str:
    """Load a specialized agent persona prompt from the registry defined path.

    The loader checks, in order:

    1. A matching agent in the Knowledge Graph registry with a
       ``json_blueprint`` payload.
    2. An agent whose ``prompt_file`` points at a local ``*.json`` file.
    3. A fallback ``agent_utilities/prompts/<prompt_name>.json`` file.

    Args:
        prompt_name: The slugified name/tag of the expert (e.g. ``router``).

    Returns:
        The specialized system prompt serialized as a JSON string.

    """
    registry = get_discovery_registry()
    agent = next((a for a in registry.agents if a.name == prompt_name), None)

    if agent:
        if agent.json_blueprint:
            return _render_prompt_payload(dict(agent.json_blueprint))

        if agent.prompt_file:
            # Check if it's a JSON file
            prompt_path = (Path(__file__).parent.parent / agent.prompt_file).resolve()
            if prompt_path.suffix == ".json" and prompt_path.exists():
                data = json.loads(prompt_path.read_text(encoding="utf-8"))
                return _render_prompt_payload(data)

    # Unified JSON loading from prompts/
    json_path = (
        Path(__file__).parent.parent / "prompts" / f"{prompt_name}.json"
    ).resolve()
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            return _render_prompt_payload(data)
        except Exception as e:
            logger.warning(
                f"Failed to load structured prompt JSON for '{prompt_name}': {e}"
            )

    logger.warning(
        f"Specialized prompt for '{prompt_name}' not found in registry "
        "or prompts/*.json."
    )
    return f"You are a helpful assistant specialized in {prompt_name}."


# --- Migrated from mcp/config_loader.py ---
import os
import shutil
import tempfile


def load_mcp_servers_from_config(config_path: str | Path) -> list[Any]:
    """Load and expand environment variables in an MCP config file.

    Reads the specified mcp_config.json, expands any environment variable
    placeholders (e.g., ${API_KEY}), performs robust pre-validation of
    executable commands in the PATH, and initializes the server objects.

    Args:
        config_path: Path to the mcp_config.json file.

    Returns:
        A list of initialized pydantic_ai.mcp.MCPServer objects (technically
        MCPToolSet in newer versions, but returned as list of servers here).

    """
    from pydantic_ai.mcp import load_mcp_toolsets

    from agent_utilities.base_utilities import expand_env_vars

    try:
        path = Path(config_path)
        if not path.exists():
            return []

        content = path.read_text()
        expanded_content = expand_env_vars(content)

        # Robust Validation: Check if commands exist before pydantic-ai tries to start them
        try:
            config_data = json.loads(expanded_content)
            mcp_servers = config_data.get("mcpServers", {})
            modified = False

            for name, cfg in mcp_servers.items():
                command = cfg.get("command")
                if command:
                    # Resolve command path with explicit ~/.local/bin support
                    search_path = os.environ.get("PATH", "")
                    local_bin = str(Path.home() / ".local" / "bin")
                    if local_bin not in search_path:
                        search_path = f"{local_bin}:{search_path}"

                    resolved = shutil.which(command, path=search_path)
                    if not resolved:
                        logger.warning(
                            f"MCP Config: Command '{command}' for server '{name}' NOT FOUND in PATH ({search_path}). Startup will likely fail."
                        )
                    else:
                        logger.debug(
                            f"MCP Config: Resolved command '{command}' to '{resolved}'"
                        )

                    # Ensure PATH and PYTHONPATH are preserved if not explicitly set
                    if "env" not in cfg:
                        cfg["env"] = {}

                    if "PATH" not in cfg["env"]:
                        cfg["env"]["PATH"] = search_path
                    if "PYTHONPATH" not in cfg["env"] and "PYTHONPATH" in os.environ:
                        cfg["env"]["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")

                    # Suppress RequestsDependencyWarning in subprocesses
                    if "PYTHONWARNINGS" not in cfg["env"]:
                        cfg["env"][
                            "PYTHONWARNINGS"
                        ] = "ignore:urllib3 (2.3.0) or chardet"
                    else:
                        if "ignore:urllib3" not in cfg["env"]["PYTHONWARNINGS"]:
                            cfg["env"][
                                "PYTHONWARNINGS"
                            ] += ",ignore:urllib3 (2.3.0) or chardet"

                    # Token forwarding: propagate user session token to
                    # MCP subprocesses for delegated authentication.
                    # CONCEPT:AU-OS.config.secrets-authentication — Secrets & Authentication
                    if "AGENT_USER_TOKEN" not in cfg["env"]:
                        _user_token = os.environ.get("AGENT_USER_TOKEN")
                        if not _user_token:
                            try:
                                from agent_utilities.security.secrets_client import (
                                    create_secrets_client,
                                )

                                _sc = create_secrets_client()
                                _user_token = _sc.get("session_token")
                            except Exception:  # nosec B110
                                pass
                        if _user_token:
                            cfg["env"]["AGENT_USER_TOKEN"] = _user_token

                    modified = True

            if modified:
                expanded_content = json.dumps(config_data)
        except Exception as e:
            logger.warning(f"MCP Config: Pre-validation failed: {e}")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp.write(expanded_content)
            tmp_path = tmp.name

        try:
            servers = load_mcp_toolsets(tmp_path)
            # Re-attach IDs from config
            config_data = json.loads(expanded_content)
            mcp_servers_cfg = config_data.get("mcpServers", {})

            # Match by command and args as a heuristic if pydantic-ai doesn't preserve order or names
            for ts in servers:
                # pydantic-ai objects might not have a clean way to match back,
                # but they usually follow the order in the JSON.
                pass

            # Better: If we have a list, and the config had a dict, they MIGHT match by order
            # However, pydantic-ai load_mcp_servers is internal.
            # I'll just set the .id if they are list components.
            # `AbstractToolset.id` is a read-only abstract property on most concrete
            # pydantic-ai toolsets (no setter) — best-effort only; a toolset that
            # rejects the assignment keeps its own id rather than failing the whole
            # load (this used to raise AttributeError here and silently return []
            # for every real toolset).
            for i, (name, cfg) in enumerate(mcp_servers_cfg.items()):
                if i < len(servers):
                    try:
                        servers[i].id = name  # type: ignore[misc]
                    except AttributeError:
                        logger.debug(
                            f"MCP Config: toolset for '{name}' has a read-only id; "
                            "keeping its own"
                        )
                        continue
                    logger.debug(f"MCP Config: Loaded server '{name}'")

            return servers
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as e:
        logger.error("Failed to load MCP configuration (%s)", type(e).__name__)
        return []
