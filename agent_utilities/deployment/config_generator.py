#!/usr/bin/python
"""Complete config generation + validation for full agent-utilities deployments.

The framework has ~261 :class:`AgentConfig` fields. Operators (and Claude setting
itself up) shouldn't hand-copy a template or reason about each flag. This module:

- :func:`generate_config` — emits a COMPLETE ``config.json`` covering every field at
  its default, then layers a per-profile preset (the handful of deployment-varying
  keys that actually differ between ``tiny`` / ``single-node-prod`` / ``enterprise``)
  and blanks secret-like values so a template never leaks a credential.
- :func:`config_reference` — every option grouped by the subsystem section it lives
  under in ``core/config.py`` (env name, type, default) for a one-page reference.
- :func:`config_doctor` — validates a config (a file, or the live process) for
  completeness/health against the chosen profile, reusing the existing
  :func:`collect_production_violations` durability rules.

Generation/validation operate only over the typed schema; they never invent an
ad-hoc deployment setting. Keys are the canonical environment names from the
typed schema, which the config loader uppercases into ``os.environ`` at startup.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Recognized deployment profiles (rungs of docs/guides/deployment-configurations.md).
PROFILES = ("tiny", "single-node-prod", "enterprise")

# Suffixes that mark a key as a credential VALUE holder — blanked in generated
# templates so a committed/shared config.json never carries a secret. Suffix-precise
# so config keys like SECRETS_BACKEND / SECRETS_VAULT_URL (not credentials) are NOT
# blanked, while raw credential-value fields remain excluded from current config.
_SECRET_SUFFIXES = (
    "API_KEY",
    "PASSWORD",
    "PRIVATE_KEY",
    "SECRET_KEY",
    "ACCESS_KEY",
    "_SECRET",
    "_TOKEN",
    "TOKEN",
)

# Host-local defaults are resolved by AgentConfig at runtime and omitted from a
# generated template. Serializing an evaluated XDG/home default would bind a
# portable configuration to the machine that generated it.
_PATH_KEY_PARTS = frozenset({"PATH", "ROOT", "DIR", "DIRECTORY", "FILE", "SOCKET"})


def _is_host_local_default(env_key: str, value: Any) -> bool:
    parts = frozenset(str(env_key).upper().split("_"))
    if not parts.intersection(_PATH_KEY_PARTS):
        return False
    if isinstance(value, str):
        return bool(re.match(r"^(?:/|[A-Za-z]:[\\/])", value.strip()))
    if isinstance(value, list | tuple):
        return any(_is_host_local_default(env_key, item) for item in value)
    if isinstance(value, dict):
        return any(_is_host_local_default(env_key, item) for item in value.values())
    return False


# ── Per-profile presets — ONLY the deployment-varying keys that differ from the
# zero-infra default. Everything else stays at the schema default. Placeholder
# DSNs/URIs are obvious and meant to be edited (or replaced with vault:// refs).
_PLACEHOLDER_OTEL = "https://telemetry.example.test:4317"

# Production profiles run the bounded, propose-only learning stages explicitly.
# Any setting that can apply/merge/develop a change remains fail-closed, and the
# content-bearing regression-dataset path remains disabled.  Keeping this map in
# one place prevents the single-host and enterprise postures from drifting.
_SAFE_PROPOSE_ONLY_EVOLUTION: dict[str, Any] = {
    "KG_LOOP": True,
    "KG_LOOP_BREADTH": True,
    "KG_LOOP_MINE_DISCOVERY": True,
    "KG_LOOP_BELIEF_REVISION": True,
    "KG_LOOP_INSIGHT_VALIDATION": True,
    "KG_LOOP_TRACE_MINING": True,
    "KG_OPTIMIZATION_ENABLED": True,
    "KG_FAILURE_EVOLUTION": True,
    "KG_FAILURE_REGRESSION_DATASET": False,
    # External/cost-bearing or filesystem-writing stages stay explicit opt-ins.
    "KG_LOOP_DISCOVER": False,
    "KG_LOOP_DISTILL": False,
    "KG_LOOP_STANDARDIZE": False,
    # Mutating autonomy remains review-first even when learning is enabled.
    "KG_GOLDEN_AUTO_MERGE": False,
    "KG_AGENT_AUTO_APPLY": False,
    "KG_LOOP_AUTO_DEVELOP": False,
    "KG_LOOP_ALLOW_HOST_VALIDATION": False,
    "KG_INSIGHT_AUTONOMY": False,
}

_PRODUCTION_READINESS_EXPECTED: dict[str, bool] = {
    "ENABLE_OTEL": True,
    "LANGFUSE_MCP_ENABLED": True,
    "TRACE_EXPORT_ENABLED": True,
    "LANGFUSE_CAPTURE_CONTENT": False,
    **_SAFE_PROPOSE_ONLY_EVOLUTION,
}

_PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "tiny": {
        # Zero-infra: with no GRAPH_SERVICE_ENDPOINTS, the authoritative packaged
        # epistemic-graph engine is provisioned locally on first use.
        "APP_PROFILE": "dev",
        "DEPLOYMENT_PROFILE": "tiny",
        "SECRETS_BACKEND": "engine",
    },
    "single-node-prod": {
        # One host: the engine is the authority; pg-age is an async mirror (interop/
        # BI/DR). Gateway hardened, secrets in encrypted engine storage.
        "APP_PROFILE": "production",
        "DEPLOYMENT_PROFILE": "single-node-prod",
        "GRAPH_DB_CONNECTION_PROFILE_REF": None,
        "GRAPH_MIRROR_TARGETS": ["age"],
        "GRAPH_SERVICE_ENDPOINTS": [],
        # Identity authority is deployment-owned and required by doctor; no
        # audience, issuer, or endpoint is inferred by a portable template.
        "AUTH_JWT_JWKS_URI": None,
        "AUTH_JWT_ISSUER": None,
        "AUTH_JWT_AUDIENCE": None,
        "KG_POLICY_VERSION": "baseline-v1",
        "EPISTEMIC_GRAPH_MAX_RESIDENT_GRAPHS": 1024,
        "EPISTEMIC_GRAPH_LAZY_OPEN_PAGE_SIZE": 4096,
        "EPISTEMIC_GRAPH_MAX_NODES_PER_GRAPH": 250000,
        "EPISTEMIC_GRAPH_ENCRYPTION_KEY_REF": "env://ENGINE_DATA_KEY",
        "GATEWAY_METRICS": "1",
        "USAGE_CONTENT_RETENTION": "metadata",
        "LANGFUSE_CAPTURE_CONTENT": False,
        "PERSISTENCE_IDENTITY_HMAC_KEY_REF": "env://PERSISTENCE_IDENTITY_HMAC_KEY",
        "SECRETS_BACKEND": "engine",
        "KAFKA_BOOTSTRAP_SERVERS": "",  # optional on one host; doctor will note it
        "ENABLE_OTEL": True,
        "OTEL_EXPORTER_OTLP_ENDPOINT": _PLACEHOLDER_OTEL,
        "LANGFUSE_PUBLIC_KEY_REF": "env://LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY_REF": "env://LANGFUSE_SECRET_KEY",
        "LANGFUSE_MCP_ENABLED": True,
        "TRACE_EXPORT_ENABLED": True,
        **_SAFE_PROPOSE_ONLY_EVOLUTION,
    },
    "enterprise": {
        # Multi-node: shared/remote engine authority + pg-age mirror, durable state,
        # queue dispatch, auth fail-closed, vault, event backbone, observability.
        # Hand off multi-node wiring to the deployment workflow skill.
        "APP_PROFILE": "production",
        "DEPLOYMENT_PROFILE": "enterprise",
        "GRAPH_DB_CONNECTION_PROFILE_REF": None,
        "GRAPH_MIRROR_TARGETS": ["age"],
        "GRAPH_SERVICE_ENDPOINTS": [],
        "EPISTEMIC_GRAPH_MAX_RESIDENT_GRAPHS": 1024,
        "EPISTEMIC_GRAPH_LAZY_OPEN_PAGE_SIZE": 4096,
        "EPISTEMIC_GRAPH_MAX_NODES_PER_GRAPH": 250000,
        "STATE_DB_URI": "",
        "TASK_QUEUE_BACKEND": "kafka",
        "KAFKA_BOOTSTRAP_SERVERS": "redpanda:9092",
        # Identity authority is deployment-owned and required by doctor; no
        # audience, issuer, or endpoint is inferred by a portable template.
        "AUTH_JWT_JWKS_URI": None,
        "AUTH_JWT_ISSUER": None,
        "AUTH_JWT_AUDIENCE": None,
        "KG_POLICY_VERSION": "baseline-v1",
        "SECRETS_BACKEND": "vault",
        "SECRETS_VAULT_URL": "https://vault.example.test:8200",
        "VAULT_AUTH_METHOD": "approle",
        "GATEWAY_METRICS": "1",
        "USAGE_CONTENT_RETENTION": "metadata",
        "LANGFUSE_CAPTURE_CONTENT": False,
        "PERSISTENCE_IDENTITY_HMAC_KEY_REF": "env://PERSISTENCE_IDENTITY_HMAC_KEY",
        "OTEL_EXPORTER_OTLP_ENDPOINT": _PLACEHOLDER_OTEL,
        "ENABLE_OTEL": True,
        "LANGFUSE_PUBLIC_KEY_REF": "env://LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY_REF": "env://LANGFUSE_SECRET_KEY",
        "LANGFUSE_MCP_ENABLED": True,
        "TRACE_EXPORT_ENABLED": True,
        **_SAFE_PROPOSE_ONLY_EVOLUTION,
    },
}

# Keys each profile genuinely *requires* an operator to set (doctor checks these).
_PROFILE_REQUIRED: dict[str, tuple[str, ...]] = {
    "tiny": (),
    "single-node-prod": (
        "GRAPH_DB_CONNECTION_PROFILE_REF",
        "EPISTEMIC_GRAPH_ENCRYPTION_KEY_REF",
        "AUTH_JWT_JWKS_URI",
        "AUTH_JWT_ISSUER",
        "AUTH_JWT_AUDIENCE",
        "KG_POLICY_VERSION",
    ),
    "enterprise": (
        "GRAPH_DB_CONNECTION_PROFILE_REF",
        "GRAPH_SERVICE_ENDPOINTS",
        "STATE_DB_URI",
        "AUTH_JWT_JWKS_URI",
        "AUTH_JWT_ISSUER",
        "AUTH_JWT_AUDIENCE",
        "KG_POLICY_VERSION",
        "KAFKA_BOOTSTRAP_SERVERS",
    ),
}


def _is_secret(env_key: str) -> bool:
    up = env_key.upper()
    return any(up.endswith(suffix) for suffix in _SECRET_SUFFIXES)


#: Settings that only take effect on an engine/daemon rebuild — a live `set_config`
#: persists + updates the value but cannot apply it to the running process; callers
#: should restart the daemon (CONCEPT:AU-KG.backend.connection-registry).
_RESTART_REQUIRED: frozenset[str] = frozenset(
    {
        "GRAPH_MIRROR_TARGETS",
        "GRAPH_DB_CONNECTION_PROFILE_REF",
        "STATE_DB_URI",
        "GRAPH_SERVICE_ENDPOINTS",
        "GRAPH_SERVICE_AUTH_SECRET",
        "EPISTEMIC_GRAPH_ENCRYPTION_KEY_REF",
        "KG_DAEMON_ROLE",
        "TASK_QUEUE_BACKEND",
        "AGENT_UTILITIES_CONFIG_DIR",
    }
)


def is_restart_required(env_key: str) -> bool:
    """True if changing ``env_key`` needs a daemon restart to take effect.

    Engine/daemon-rebuild settings (backend, durable DSN, auth secret, sharding,
    queue backend) are wired at startup; everything else is read live via
    ``config.setting`` / re-parsed fields (CONCEPT:AU-KG.backend.connection-registry)."""
    up = (env_key or "").upper()
    return up in _RESTART_REQUIRED or up.startswith(("AUTH_", "GRAPH_SERVICE_"))


def _base_dump() -> dict[str, Any]:
    """Portable AgentConfig defaults keyed by env-var alias.

    Host-local XDG/home paths are absent and resolve on the target machine when
    AgentConfig loads.
    """
    from agent_utilities.core.config import AgentConfig

    # by_alias=True → canonical env names; round-trips through json with default=str.
    data = AgentConfig.model_construct().model_dump(by_alias=True)
    return {
        key: value
        for key, value in data.items()
        if not _is_host_local_default(key, value)
    }


_SECRET_REFERENCE_RE = re.compile(
    r"^(?:"
    r"env://[A-Za-z_][A-Za-z0-9_]{0,127}"
    r"|(?:vault|secret)://[A-Za-z0-9][A-Za-z0-9_./#-]{0,511}"
    r")$"
)
_SENSITIVE_MAPPING_KEYS = frozenset(
    {"authorization", "cookie", "set-cookie", "x-api-key"}
)
_HEADER_CONTAINER_KEYS = frozenset({"headers", "extra_headers", "custom_headers"})


def _sanitize_generated_value(value: Any, *, key: str = "", parent: str = "") -> Any:
    """Remove raw credential material and preserve reference-only fields."""
    normalized = key.strip().lower().replace("-", "_")
    if normalized in _HEADER_CONTAINER_KEYS:
        return {}
    sensitive = _is_secret(normalized) or normalized in {
        item.replace("-", "_") for item in _SENSITIVE_MAPPING_KEYS
    }
    if sensitive:
        reference_field = normalized.endswith("_ref") or (
            normalized == "client_secret" and parent == "oauth2"
        )
        if (
            reference_field
            and isinstance(value, str)
            and _SECRET_REFERENCE_RE.fullmatch(value.strip())
        ):
            return value.strip()
        if value is None:
            return None
        if isinstance(value, list):
            return []
        if isinstance(value, dict):
            return {}
        return ""
    if isinstance(value, dict):
        return {
            str(child_key): _sanitize_generated_value(
                child_value,
                key=str(child_key),
                parent=normalized,
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_generated_value(item) for item in value]
    return value


def generate_config(profile: str = "tiny") -> dict[str, Any]:
    """Return a portable config dict for ``profile`` (schema defaults + overlay).

    Args:
        profile: one of :data:`PROFILES`.
        Generated output always removes credential values recursively. References
        remain intact so the operator can resolve them at runtime.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose one of {PROFILES}.")
    data = _base_dump()
    # JSON-normalize (drops non-serializable defaults to their str form).
    data = json.loads(json.dumps(data, default=str))
    data.update(_PROFILE_PRESETS[profile])
    return _sanitize_generated_value(data)


def write_config(
    profile: str = "tiny",
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate and write a complete ``config.json`` for ``profile`` to ``path``.

    ``path`` defaults to the XDG config location agent-utilities loads at startup.
    """
    cfg = generate_config(profile)
    out = Path(path) if path else _default_config_path()
    from agent_utilities.core.config import _write_private_configuration_mapping

    _write_private_configuration_mapping(out, dict(sorted(cfg.items())))
    return {
        "status": "success",
        "profile": profile,
        "destination": "explicit" if path else "xdg",
        "keys": len(cfg),
        "presets_applied": sorted(_PROFILE_PRESETS[profile]),
    }


def _default_config_path() -> Path:
    from agent_utilities.core.paths import config_dir

    return config_dir() / "config.json"


# ──────────────────────────────────────────────────────────────────────────
# Grouped reference — every option under its config.py subsystem section.
# ──────────────────────────────────────────────────────────────────────────
_SECTION_RE = re.compile(r"^\s*#\s*[-─=]{2,}\s*(.+?)\s*[-─=]{2,}\s*$")
_FIELD_RE = re.compile(r"^    ([a-z_][a-z0-9_]*)\s*:")


def _field_sections() -> dict[str, str]:
    """Map each AgentConfig field NAME to the subsystem section it's defined under.

    Parsed from ``core/config.py`` (the section ``# --- Title ---`` comments). Only
    the ``class AgentConfig`` body is scanned; fields before/after fall back to a
    catch-all so generation never depends on perfect parsing.
    """
    import agent_utilities.core.config as cfgmod

    src = Path(cfgmod.__file__).read_text().splitlines()
    sections: dict[str, str] = {}
    in_class = False
    current = "General"
    for line in src:
        if line.startswith("class AgentConfig"):
            in_class = True
            current = "General"
            continue
        if in_class and line.startswith("class ") and not line.startswith("    "):
            break  # left the AgentConfig body
        if not in_class:
            continue
        m = _SECTION_RE.match(line)
        if m:
            current = m.group(1).strip()
            continue
        fm = _FIELD_RE.match(line)
        if fm:
            sections.setdefault(fm.group(1), current)
    return sections


def config_reference() -> list[dict[str, Any]]:
    """Every config option grouped by subsystem: ``[{section, fields:[...]}, ...]``.

    Each field carries its env name, python type, and default — the full inventory
    in one structure for a reference table or an LLM to scan.
    """
    from agent_utilities.core.config import AgentConfig

    field_section = _field_sections()
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for name, info in AgentConfig.model_fields.items():
        section = field_section.get(name, "General")
        if section not in grouped:
            grouped[section] = []
            order.append(section)
        type_name = getattr(info.annotation, "__name__", str(info.annotation))
        default = info.default
        try:
            json.dumps(default, default=str)
            default_repr = default
        except Exception:  # noqa: BLE001
            default_repr = str(default)
        grouped[section].append(
            {
                "name": name,
                "env": info.alias or name.upper(),
                "type": type_name,
                "default": default_repr
                if not _is_secret(info.alias or name)
                else "***",
                "secret": _is_secret(info.alias or name),
            }
        )
    return [{"section": s, "fields": grouped[s]} for s in order]


# ──────────────────────────────────────────────────────────────────────────
# Doctor — validate a deployment's config completeness/health.
# ──────────────────────────────────────────────────────────────────────────
def unknown_configuration_keys(mapping: Mapping[str, Any]) -> list[str]:
    """Keys in ``mapping`` the strict production loader would reject as unknown.

    These are keys that are neither an ``AgentConfig`` field alias nor a retired
    key — e.g. connector/service settings the central production ``config.json``
    schema does not model. Reported (never auto-stripped) because they usually
    carry real configuration a caller must relocate (to env/secret store),
    not lose.
    """
    from agent_utilities.core.config import (
        AgentConfig,
        retired_configuration_keys,
    )

    aliases = {
        str(field.alias or name.upper()).upper()
        for name, field in AgentConfig.model_fields.items()
    }
    retired = retired_configuration_keys()
    return sorted(
        str(key)
        for key in mapping
        if str(key).strip().upper() not in aliases
        and str(key).strip().upper() not in retired
    )


def migrate_config_file(
    config_path: str | Path, *, backup: bool = True, strip_unknown: bool = False
) -> dict[str, Any]:
    """Reconcile a stale ``config.json`` so it loads under the current schema.

    A stale ``config.json`` fails the load two ways: **retired** keys (rejected
    with ``retired durable configuration key(s) are not accepted``) and, under
    the strict production path, **unknown** keys not modelled by ``AgentConfig``.
    This operates on the raw JSON — before any ``AgentConfig`` validation:

    * Always strips retired keys (safe: they are removed capabilities).
    * With ``strip_unknown=True`` also strips unknown/unmodelled keys (e.g. stray
      connector settings) — these are reported either way so a caller can relocate
      them to env/the secret store instead of losing them.

    Writes a one-time backup and returns a value-free report (key *names* only).
    """
    from agent_utilities.core.config import strip_retired_configuration_keys

    path = Path(config_path)
    if not path.exists():
        return {"status": "skip", "reason": "no_config_file", "path": str(path)}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"status": "error", "error": "config_unreadable", "path": str(path)}
    if not isinstance(raw, dict):
        return {"status": "error", "error": "config_not_object", "path": str(path)}

    cleaned, removed = strip_retired_configuration_keys(raw)
    unknown = unknown_configuration_keys(cleaned)
    unknown_removed: list[str] = []
    if strip_unknown and unknown:
        cleaned = {
            k: v
            for k, v in cleaned.items()
            if str(k) not in set(unknown)
        }
        unknown_removed = unknown
        unknown = []

    if not removed and not unknown_removed:
        return {
            "status": "ok",
            "removed": [],
            "unknown_present": unknown,
            "path": str(path),
        }
    backup_path: Path | None = None
    if backup:
        backup_path = path.with_name(path.name + ".pre-migrate.bak")
        backup_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    path.write_text(json.dumps(cleaned, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "migrated",
        "removed": removed,
        "unknown_removed": unknown_removed,
        "unknown_present": unknown,
        "backup": str(backup_path) if backup_path else None,
        "path": str(path),
    }


def config_doctor(
    profile: str | None = None,
    config_path: str | Path | None = None,
    *,
    migrate: bool = False,
) -> dict[str, Any]:
    """Validate config completeness/health for ``profile``.

    Loads config from ``config_path`` (a generated ``config.json``) if given, else
    evaluates the **live** process config. Checks: required-for-profile keys are set,
    secret refs are resolvable, and durability rules hold (reusing
    :func:`collect_production_violations`). Returns a structured report; never raises.
    """
    from agent_utilities.core.config import AgentConfig
    from agent_utilities.core.profile_guard import collect_production_violations

    # Pre-validation: a stale config.json carrying retired keys fails the load
    # outright, so detect them on the raw JSON first and (when migrate=True) strip
    # them before the AgentConfig validation below ever sees them.
    if config_path is not None and Path(config_path).exists():
        from agent_utilities.core.config import strip_retired_configuration_keys

        try:
            _raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
            _, retired_present = (
                strip_retired_configuration_keys(_raw)
                if isinstance(_raw, dict)
                else ({}, [])
            )
        except Exception:  # noqa: BLE001
            retired_present = []
        if retired_present and migrate:
            migrate_config_file(config_path)
            retired_present = []
        if retired_present:
            return {
                "status": "needs_migration",
                "profile": profile,
                "healthy": False,
                "checks": [
                    {
                        "check": "retired_configuration_keys",
                        "ok": False,
                        "keys": retired_present,
                        "remediation": (
                            "call config_doctor(config_path=..., migrate=True) or "
                            "`agent-utilities-doctor --migrate-config` to remove them"
                        ),
                    }
                ],
                "summary": (
                    f"{len(retired_present)} retired configuration key(s) present — "
                    "migrate to load cleanly"
                ),
            }

    # Build the AgentConfig under evaluation.
    if config_path:
        try:
            from agent_utilities.core.config import (
                _canonicalize_xdg_configuration,
                _mapping_selects_production,
                _read_configuration_mapping,
                _validate_agent_config_without_settings,
            )

            raw = _read_configuration_mapping(
                config_path,
                source_type="xdg",
                strict=False,
            )
            raw = _canonicalize_xdg_configuration(raw)
            if _mapping_selects_production(raw):
                raw = _read_configuration_mapping(
                    config_path,
                    source_type="xdg",
                    strict=True,
                )
                raw = _canonicalize_xdg_configuration(raw)
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "healthy": False,
                "error": "configuration_source_unreadable",
                "error_class": type(exc).__name__,
            }
        # config.json keys are env aliases; AgentConfig accepts them via populate.
        try:
            cfg = _validate_agent_config_without_settings(
                {k: v for k, v in raw.items() if v not in (None, "")}
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "healthy": False,
                "error": "configuration_schema_invalid",
                "error_class": type(exc).__name__,
                "checks": [{"check": "schema", "ok": False}],
            }
        prof = profile or raw.get("DEPLOYMENT_PROFILE")
        app_profile = str(raw.get("APP_PROFILE") or cfg.app_profile).strip().casefold()
        profile_source = (
            "argument"
            if profile
            else ("configuration" if raw.get("DEPLOYMENT_PROFILE") else "default")
        )
    else:
        try:
            cfg = AgentConfig()
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "healthy": False,
                "error": "configuration_schema_invalid",
                "error_class": type(exc).__name__,
                "checks": [{"check": "schema", "ok": False}],
            }
        from agent_utilities.core.config import setting

        configured_profile = setting("DEPLOYMENT_PROFILE", "")
        prof = profile or configured_profile
        app_profile = (
            str(setting("APP_PROFILE", cfg.app_profile) or "").strip().casefold()
        )
        profile_source = (
            "argument"
            if profile
            else ("configuration" if configured_profile else "default")
        )

    # APP_PROFILE is a runtime posture, not a deployment-topology identity. A
    # production posture is deliberately ambiguous between single-node and
    # enterprise and therefore requires DEPLOYMENT_PROFILE (or an explicit
    # function argument). The zero-configuration development posture remains tiny.
    if not prof and app_profile in {"prod", "production"}:
        return {
            "status": "error",
            "healthy": False,
            "error": "deployment_profile_required",
            "checks": [
                {
                    "check": "deployment_profile",
                    "ok": False,
                    "reason": "production_posture_is_ambiguous",
                }
            ],
        }
    norm = str(prof or "tiny").strip()
    if norm not in PROFILES:
        return {
            "status": "error",
            "healthy": False,
            "error": "deployment_profile_invalid",
            "checks": [{"check": "deployment_profile", "ok": False}],
        }

    checks: list[dict[str, Any]] = [
        {
            "check": "deployment_profile",
            "profile": norm,
            "source": profile_source,
            "ok": True,
        }
    ]

    def _set(env: str) -> bool:
        val = getattr(cfg, _alias_to_field(env), None)
        return bool(str(val).strip()) if val not in (None, False) else False

    # 1. Required-for-profile keys.
    missing = [k for k in _PROFILE_REQUIRED.get(norm, ()) if not _set(k)]
    checks.append(
        {
            "check": "required_keys",
            "profile": norm,
            "ok": not missing,
            "missing": missing,
        }
    )

    # 2. Durability / production-safety rules (always evaluated, advisory for tiny).
    violations = collect_production_violations(cfg)
    checks.append(
        {
            "check": "durability",
            "ok": not violations or norm == "tiny",
            "violations": violations,
            "advisory": norm == "tiny",
        }
    )

    # 3. Secret references resolvable (vault://, secret://, env://).
    try:
        unresolved = _unresolved_secret_refs(cfg, resolve=config_path is None)
    except Exception as exc:  # noqa: BLE001 - aggregate, privacy-safe boundary
        checks.append(
            {
                "check": "secret_refs",
                "ok": False,
                "evaluation_error": type(exc).__name__,
                "unresolved_count": 0,
                "redacted": True,
            }
        )
    else:
        checks.append(
            {
                "check": "secret_refs",
                "ok": not unresolved,
                "unresolved_count": len(unresolved),
                "redacted": True,
            }
        )

    # 4. Outbound fleet identity is declaration-only here. Doctor validates
    # metadata and a secret reference, never resolves or reports the secret.
    outbound_mode = cfg.mcp_client_auth
    outbound_missing: list[str] = []
    if outbound_mode == "oidc-client-credentials":
        for env, value in (
            ("OIDC_CLIENT_ID", cfg.oidc_client_id),
            ("OIDC_CLIENT_SECRET_REF", cfg.oidc_client_secret_ref),
            ("OIDC_AUDIENCE", cfg.oidc_audience),
        ):
            if not value:
                outbound_missing.append(env)
        if not (cfg.oidc_token_url or cfg.oidc_issuer):
            outbound_missing.append("OIDC_TOKEN_URL_OR_OIDC_ISSUER")
    elif outbound_mode == "basic":
        for env, value in (
            ("MCP_BASIC_AUTH_USERNAME", cfg.mcp_basic_auth_username),
            ("MCP_BASIC_AUTH_PASSWORD_REF", cfg.mcp_basic_auth_password_ref),
        ):
            if not value:
                outbound_missing.append(env)
    checks.append(
        {
            "check": "outbound_mcp_auth",
            "ok": not outbound_missing,
            "mode": outbound_mode,
            "missing": sorted(outbound_missing),
            "redacted": True,
        }
    )

    # 5. Raw Memento retention has no permissive/partial configuration. The runtime also enforces
    # this gate, while doctor makes the reason visible before deployment without exposing a key or
    # secret reference in its report.
    retention_enabled = bool(getattr(cfg, "memento_raw_retention_enabled", False))
    retention_issues: list[str] = []
    if retention_enabled:
        if (
            str(getattr(cfg, "memento_raw_retention_policy", "") or "").strip()
            != "approved-encrypted-v1"
        ):
            retention_issues.append("approved_policy_required")
        if not str(getattr(cfg, "memento_raw_encryption_key_ref", "") or "").strip():
            retention_issues.append("encryption_key_reference_required")
    checks.append(
        {
            "check": "memento_raw_retention",
            "ok": not retention_issues,
            "enabled": retention_enabled,
            "issues": retention_issues,
        }
    )

    # 6. Dispatch crash recovery must remain inside the published five-minute
    # workload RTO, and renewal must happen before the lease expires.
    dispatch_claim_ttl_s = float(cfg.agent_dispatch_claim_ttl_s)
    dispatch_renew_interval_s = float(cfg.agent_dispatch_renew_interval_s)
    checks.append(
        {
            "check": "dispatch_lease_recovery",
            "ok": (
                dispatch_claim_ttl_s <= 300.0
                and dispatch_renew_interval_s < dispatch_claim_ttl_s
            ),
            "claim_ttl_seconds": dispatch_claim_ttl_s,
            "renew_interval_seconds": dispatch_renew_interval_s,
            "rto_target_seconds": 300.0,
        }
    )

    # 7. Production feature posture. Report only setting names, never configured
    # values, endpoints, identities, secret references, or trace content.
    readiness_mismatches: list[str] = []
    if norm != "tiny":
        for env, expected in _PRODUCTION_READINESS_EXPECTED.items():
            actual = getattr(cfg, _alias_to_field(env), None)
            if actual is not expected:
                readiness_mismatches.append(env)
    checks.append(
        {
            "check": "propose_only_observability",
            "ok": not readiness_mismatches,
            "applicable": norm != "tiny",
            "mismatched": sorted(readiness_mismatches),
            "redacted": True,
        }
    )

    ok = all(c["ok"] for c in checks)
    return {
        "status": "success",
        "profile": norm,
        "healthy": ok,
        "checks": checks,
        "summary": "ready" if ok else "needs attention — see checks",
    }


def _alias_to_field(env: str) -> str:
    """Map an env alias back to the AgentConfig field name (best-effort)."""
    from agent_utilities.core.config import AgentConfig

    for name, info in AgentConfig.model_fields.items():
        if (info.alias or name.upper()) == env:
            return name
    return env.lower()


def _unresolved_secret_refs(cfg: Any, *, resolve: bool = True) -> list[str]:
    """Return opaque markers for unresolved refs anywhere in the typed config.

    Reference names and values deliberately never leave this function. Backend
    construction failure raises so doctor cannot report a false healthy state.
    """
    from collections.abc import Mapping

    if hasattr(cfg, "model_dump"):
        root = cfg.model_dump(mode="python", by_alias=True)
    elif isinstance(cfg, Mapping):
        root = dict(cfg)
    else:
        root = vars(cfg)

    reference_pattern = re.compile(
        r"^(?:env://[A-Za-z_][A-Za-z0-9_]{0,127}|"
        r"(?:vault|secret)://[A-Za-z0-9][A-Za-z0-9_./#-]{0,511})$"
    )
    references: set[str] = set()
    pending: list[Any] = [root]
    visited = 0
    while pending:
        value = pending.pop()
        visited += 1
        if visited > 100_000:
            raise ValueError("secret reference inventory exceeds its bound")
        if isinstance(value, Mapping):
            pending.extend(value.values())
        elif isinstance(value, list | tuple):
            pending.extend(value)
        elif isinstance(value, str) and reference_pattern.fullmatch(value.strip()):
            references.add(value.strip())

    if not references:
        return []
    if not resolve:
        return ["unresolved"] * len(references)
    client: Any | None = None
    unresolved: list[str] = []
    for reference in sorted(references):
        try:
            if reference.startswith("env://"):
                from agent_utilities.core._env import setting

                resolved = setting(reference.removeprefix("env://"))
            else:
                if client is None:
                    from agent_utilities.security.secrets_client import (
                        create_secrets_client,
                    )

                    client = create_secrets_client()
                resolved = client.resolve_ref(reference)
        except Exception:  # noqa: BLE001 - only an opaque marker escapes
            resolved = None
        if not resolved:
            unresolved.append("unresolved")
    return unresolved
