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

No new env flags are introduced — generation/validation operate over the *existing*
schema. Keys are the canonical env var names (``GRAPH_BACKEND``), which the config
loader uppercases into ``os.environ`` at startup.
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
# blanked, while OIDC_CLIENT_SECRET / *_PASSWORD / *_API_KEY / *_TOKEN are.
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

# ── Per-profile presets — ONLY the deployment-varying keys that differ from the
# zero-infra default. Everything else stays at the schema default. Placeholder
# DSNs/URIs are obvious and meant to be edited (or replaced with vault:// refs).
_PLACEHOLDER_PG = "postgresql://agent:CHANGE_ME@localhost:5432/agent_kg"

_PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "tiny": {
        # Zero-infra: the epistemic-graph engine IS the whole database (compute +
        # cache + semantic + durable), auto-provisioned as a lifecycle-coupled child
        # the first time the KG is touched. Nothing external.
        "GRAPH_BACKEND": "epistemic_graph",
        "EPISTEMIC_GRAPH_AUTOSTART": "1",
        "APP_PROFILE": "dev",
        "SECRETS_BACKEND": "inmemory",
    },
    "single-node-prod": {
        # One host: the engine is the authority; pg-age is an async mirror (interop/
        # BI/DR). Gateway hardened, secrets in sqlite/vault.
        "APP_PROFILE": "production",
        "GRAPH_BACKEND": "fanout",
        "GRAPH_DB_URI": _PLACEHOLDER_PG,
        "GRAPH_PG_AGE": "1",
        "GRAPH_MIRROR_TARGETS": "age",
        "GATEWAY_METRICS": "1",
        "SECRETS_BACKEND": "sqlite",
        "KAFKA_BOOTSTRAP_SERVERS": "",  # optional on one host; doctor will note it
    },
    "enterprise": {
        # Multi-node: shared/remote engine authority + pg-age mirror, durable state,
        # queue dispatch, auth fail-closed, vault, event backbone, observability.
        # Hand off swarm to the agent-os-genesis (day0) skill.
        "APP_PROFILE": "production",
        "GRAPH_BACKEND": "fanout",
        "GRAPH_DB_URI": _PLACEHOLDER_PG,
        "GRAPH_PG_AGE": "1",
        "GRAPH_MIRROR_TARGETS": "age",
        "STATE_DB_URI": _PLACEHOLDER_PG,
        "TASK_QUEUE_BACKEND": "kafka",
        "AGENT_DISPATCH_BACKEND": "queue",
        "KAFKA_BOOTSTRAP_SERVERS": "redpanda:9092",
        "KG_AUTH_REQUIRED": "1",
        "AUTH_JWT_JWKS_URI": "https://keycloak.example/realms/agent-os/protocol/openid-connect/certs",
        "SECRETS_BACKEND": "vault",
        "SECRETS_VAULT_URL": "https://vault.example:8200",
        "VAULT_AUTH_METHOD": "approle",
        "GATEWAY_METRICS": "1",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel-collector:4317",
    },
}

# Keys each profile genuinely *requires* an operator to set (doctor checks these).
_PROFILE_REQUIRED: dict[str, tuple[str, ...]] = {
    "tiny": (),
    "single-node-prod": ("GRAPH_DB_URI",),
    "enterprise": (
        "GRAPH_DB_URI",
        "STATE_DB_URI",
        "KG_AUTH_REQUIRED",
        "AUTH_JWT_JWKS_URI",
        "KAFKA_BOOTSTRAP_SERVERS",
    ),
}


def _is_secret(env_key: str) -> bool:
    up = env_key.upper()
    return any(up.endswith(suffix) for suffix in _SECRET_SUFFIXES)


#: Settings that only take effect on an engine/daemon rebuild — a live `set_config`
#: persists + updates the value but cannot apply it to the running process; callers
#: should restart the daemon (CONCEPT:KG-2.89).
_RESTART_REQUIRED: frozenset[str] = frozenset(
    {
        "GRAPH_BACKEND",
        "GRAPH_MIRROR_TARGETS",
        "GRAPH_DB_URI",
        "STATE_DB_URI",
        "GRAPH_AUTHORITY",
        "GRAPH_SERVICE_ENDPOINTS",
        "GRAPH_SERVICE_AUTH_SECRET",
        "KG_DAEMON_ROLE",
        "TASK_QUEUE_BACKEND",
        "AGENT_UTILITIES_CONFIG_DIR",
    }
)


def is_restart_required(env_key: str) -> bool:
    """True if changing ``env_key`` needs a daemon restart to take effect.

    Engine/daemon-rebuild settings (backend, durable DSN, auth secret, sharding,
    queue backend) are wired at startup; everything else is read live via
    ``config.setting`` / re-parsed fields (CONCEPT:KG-2.89)."""
    up = (env_key or "").upper()
    return up in _RESTART_REQUIRED or up.startswith(("AUTH_", "GRAPH_SERVICE_"))


def _base_dump() -> dict[str, Any]:
    """All AgentConfig fields at their defaults, keyed by env-var alias."""
    from agent_utilities.core.config import AgentConfig

    # by_alias=True → canonical env names; round-trips through json with default=str.
    return AgentConfig().model_dump(by_alias=True)


def generate_config(
    profile: str = "tiny", *, redact_secrets: bool = True
) -> dict[str, Any]:
    """Return a COMPLETE config dict for ``profile`` (every field + preset overlay).

    Args:
        profile: one of :data:`PROFILES`.
        redact_secrets: blank secret-like keys (API keys, passwords, tokens) so the
            template is safe to share/commit; the operator fills them via env/vault.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose one of {PROFILES}.")
    data = _base_dump()
    # JSON-normalize (drops non-serializable defaults to their str form).
    data = json.loads(json.dumps(data, default=str))
    if redact_secrets:
        for key in list(data):
            if _is_secret(key) and data[key] not in (None, "", [], {}):
                data[key] = ""
    data.update(_PROFILE_PRESETS[profile])
    return data


def write_config(
    profile: str = "tiny",
    path: str | Path | None = None,
    *,
    redact_secrets: bool = True,
) -> dict[str, Any]:
    """Generate and write a complete ``config.json`` for ``profile`` to ``path``.

    ``path`` defaults to the XDG config location agent-utilities loads at startup.
    """
    cfg = generate_config(profile, redact_secrets=redact_secrets)
    out = Path(path) if path else _default_config_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=4, sort_keys=True))
    return {
        "status": "success",
        "profile": profile,
        "path": str(out),
        "keys": len(cfg),
        "presets_applied": sorted(_PROFILE_PRESETS[profile]),
    }


def _default_config_path() -> Path:
    import platformdirs

    return (
        Path(platformdirs.user_config_path("agent-utilities", "knuckles-team"))
        / "config.json"
    )


# ──────────────────────────────────────────────────────────────────────────
# Minimal mcp_config.json (mcpServers) — doctor-driven, NOT hand-written.
# ──────────────────────────────────────────────────────────────────────────
def generate_mcp_config(profile: str = "tiny", *, fleet: bool = True) -> dict[str, Any]:
    """Return the minimal ``{"mcpServers": {...}}`` an IDE registers.

    CONCEPT:OS-5.65 doctor-driven minimal mcp_config — graph-os plus mcp-multiplexer

    Two console scripts front the platform, and which one you register decides what
    the agent sees:

    - **graph-os** = *just the Knowledge Graph* — the ``go__*`` tools of one KG
      backend (``uv run graph-os``). Register this for a single, self-contained KG.
    - **mcp-multiplexer** = *the whole fleet* — runs in dynamic mode and fronts
      graph-os **plus every ``*-mcp`` server on demand** (``uv run mcp-multiplexer``;
      only meta-tools + always-on servers are visible, the rest load via
      ``find_tools`` / ``load_tools``). Register this for the fleet.

    ``graph-os`` is ALWAYS included; ``mcp-multiplexer`` is included when ``fleet``
    (the default) so the emitted config offers both — pick by deleting the one you
    don't want. Envs stay MINIMAL: only workspace path / agent id (and the
    multiplexer's mode + its child-fleet ``MCP_CONFIG`` pointer) live here — model
    selection, routing, and secrets live in the XDG ``config.json``, NOT the MCP env.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose one of {PROFILES}.")
    servers: dict[str, Any] = {
        "graph-os": {
            "command": "uv",
            "args": ["run", "graph-os"],
            "env": {
                "AGENT_ID": "local-developer",
                "WORKSPACE_PATH": "${workspaceFolder}",
            },
        },
    }
    if fleet:
        servers["mcp-multiplexer"] = {
            "command": "uv",
            "args": ["run", "mcp-multiplexer"],
            "env": {
                "AGENT_ID": "local-developer",
                "WORKSPACE_PATH": "${workspaceFolder}",
                "MCP_MULTIPLEXER_MODE": "dynamic",
                "MCP_CONFIG": "${workspaceFolder}/mcp_config.json",
            },
        }
    return {"mcpServers": servers}


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
def config_doctor(
    profile: str | None = None, config_path: str | Path | None = None
) -> dict[str, Any]:
    """Validate config completeness/health for ``profile``.

    Loads config from ``config_path`` (a generated ``config.json``) if given, else
    evaluates the **live** process config. Checks: required-for-profile keys are set,
    secret refs are resolvable, and durability rules hold (reusing
    :func:`collect_production_violations`). Returns a structured report; never raises.
    """
    from agent_utilities.core.config import AgentConfig
    from agent_utilities.core.profile_guard import collect_production_violations

    # Build the AgentConfig under evaluation.
    if config_path:
        try:
            raw = json.loads(Path(config_path).read_text())
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": f"unreadable config: {exc}"}
        # config.json keys are env aliases; AgentConfig accepts them via populate.
        cfg = AgentConfig(**{k: v for k, v in raw.items() if v not in (None, "")})
        prof = profile or raw.get("APP_PROFILE")
    else:
        cfg = AgentConfig()
        from agent_utilities.core.config import setting

        prof = profile or setting("APP_PROFILE", "")

    prof = (prof or "tiny").strip()
    # Normalize APP_PROFILE values (prod/production) to a known profile name.
    norm = (
        "enterprise"
        if prof in ("prod", "production", "enterprise")
        else ("single-node-prod" if prof in ("single-node-prod", "single") else "tiny")
    )

    checks: list[dict[str, Any]] = []

    def _set(env: str) -> bool:
        val = getattr(cfg, _alias_to_field(env), None)
        return (
            bool(str(val).strip())
            if val not in (None, False)
            else (env == "KG_AUTH_REQUIRED" and bool(val))
        )

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

    # 3. Secret references resolvable (vault://, env://, sqlite://).
    unresolved = _unresolved_secret_refs(cfg)
    checks.append(
        {"check": "secret_refs", "ok": not unresolved, "unresolved": unresolved}
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


def _unresolved_secret_refs(cfg: Any) -> list[str]:
    """Return env names whose value is a vault://-style ref that won't resolve."""
    refs: list[str] = []
    try:
        from agent_utilities.security.secrets_client import create_secrets_client

        client = create_secrets_client()
    except Exception:  # noqa: BLE001 — no secrets backend → can't resolve, report none
        return refs
    from agent_utilities.core.config import AgentConfig

    for name, info in AgentConfig.model_fields.items():
        val = getattr(cfg, name, None)
        if (
            isinstance(val, str)
            and "://" in val
            and val.split("://", 1)[0]
            in (
                "vault",
                "env",
                "sqlite",
            )
        ):
            try:
                resolved = client.resolve_ref(val)
                if not resolved:
                    refs.append(info.alias or name.upper())
            except Exception:  # noqa: BLE001
                refs.append(info.alias or name.upper())
    return refs
