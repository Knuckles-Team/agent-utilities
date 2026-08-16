"""Governed read/describe/diff/reload/set core for ``AgentConfig``.

CONCEPT:AU-OS.config.two-surfaces-by-default

The engine behind the ``graph_config`` MCP tool. It is a separate, importable
core (never logic living inside a tool body) so the same behaviour is testable
without an MCP server and reusable by the REST/CLI surfaces.

Three properties are load-bearing, in priority order:

1. **Everything is derived from the pydantic model.** Field names, env aliases,
   types, defaults and docstrings all come from ``AgentConfig`` itself. There is
   no hand-maintained list of settable keys — a hand-maintained list goes stale
   silently and then lies to the agent reading it.
2. **Secrets are redacted by reference, never by value.** A value that is a
   runtime *reference* (``vault://…``, ``env://…``, ``${VAR}``) is exactly what
   an operator needs to see and is returned verbatim; a sensitive key holding a
   literal is replaced with ``"***"``. Nothing in this module ever resolves a
   reference to its secret.
3. **Refusal is the default for ``set``.** An unknown key, a value the model
   rejects, or an inline secret is refused before anything is written. A tool
   that can silently change production configuration is a liability; the write
   path is validated, policy-gated, routed through the ONE existing write-back
   (``save_config_item``), and recorded as provenance.
"""

from __future__ import annotations

import ast
import json
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent_utilities.security.config_sensitivity import (
    configuration_key_is_sensitive,
    runtime_reference,
)

logger = logging.getLogger(__name__)

#: Substituted for any sensitive value that is a literal rather than a
#: reference. Deliberately valueless — it carries no length, prefix or shape
#: information about the secret it replaces.
REDACTED = "***"

ACTIONS = ("get", "set", "describe", "reload", "diff")


class ConfigAdminError(Exception):
    """A refusal. Carries a machine-readable ``code`` for the tool surface."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# ──────────────────────────────────────────────────────────────────────────
# Model introspection — the single source of what exists and what it means.
# ──────────────────────────────────────────────────────────────────────────
def field_index() -> dict[str, str]:
    """``{ENV_ALIAS: field_name}`` for every ``AgentConfig`` field."""
    from agent_utilities.core.config import AgentConfig

    return {
        str(info.alias or name).upper(): name
        for name, info in AgentConfig.model_fields.items()
    }


@lru_cache(maxsize=1)
def _attribute_docstrings() -> dict[str, str]:
    """``{field_name: docstring}`` parsed from ``AgentConfig``'s own source.

    ``AgentConfig`` documents its fields with the PEP 258 attribute-docstring
    convention (a bare string literal following the assignment) rather than
    ``Field(description=...)``, so the text is not on ``FieldInfo``. This reads
    it back off the class's own AST — still derived from the model, never a
    second hand-maintained table that could disagree with it.

    Fails soft to ``{}``: a missing docstring degrades ``describe`` to
    type/default/alias, it never breaks the surface.
    """
    try:
        import agent_utilities.core.config as cfgmod

        source = Path(cfgmod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception as exc:  # noqa: BLE001 - documentation is never critical
        logger.debug("AgentConfig attribute docstrings unavailable: %s", exc)
        return {}

    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "AgentConfig":
            continue
        pending: str | None = None
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                pending = stmt.target.id
                continue
            if (
                pending
                and isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            ):
                out[pending] = " ".join(stmt.value.value.split())
            pending = None
        break
    return out


def _type_name(annotation: Any) -> str:
    return getattr(annotation, "__name__", None) or str(annotation)


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


# ──────────────────────────────────────────────────────────────────────────
# Redaction — applied identically by get, describe and diff.
# ──────────────────────────────────────────────────────────────────────────
def redact(env_key: str, value: Any, *, secret: bool | None = None) -> tuple[Any, bool]:
    """``(safe_value, was_redacted)`` for one configuration value.

    A sensitive key whose value is a runtime REFERENCE is returned verbatim —
    that reference is the answer an operator wants ("which vault path is this
    pointing at?") and it discloses nothing. A sensitive key holding a literal
    is replaced wholesale. Non-string sensitive values (a list of endpoints, a
    dict of credentials) are always replaced: there is no safe partial view of
    a structure whose key says it holds secrets.
    """
    sensitive = (
        configuration_key_is_sensitive(env_key) if secret is None else bool(secret)
    )
    if not sensitive or value is None:
        return _jsonable(value), False
    if runtime_reference(value):
        return value, False
    return REDACTED, True


# ──────────────────────────────────────────────────────────────────────────
# Actions.
# ──────────────────────────────────────────────────────────────────────────
def _resolve_key(key: str) -> tuple[str, str]:
    """``(ENV_ALIAS, field_name)``, accepting either form. Refuses unknowns."""
    text = str(key or "").strip()
    if not text:
        raise ConfigAdminError("key_required", "an AgentConfig key is required")
    index = field_index()
    env_key = text.upper()
    if env_key in index:
        return env_key, index[env_key]
    from agent_utilities.core.config import AgentConfig

    lowered = text.lower()
    if lowered in AgentConfig.model_fields:
        info = AgentConfig.model_fields[lowered]
        return str(info.alias or lowered).upper(), lowered
    raise ConfigAdminError(
        "unknown_key",
        f"{text!r} is not an AgentConfig field or alias; "
        "call graph_config(action='describe') to list what is settable",
    )


def _effective(field_name: str) -> Any:
    """The live effective value of one field, from the process's config."""
    from agent_utilities.core.config import config as agent_config

    return getattr(agent_config, field_name, None)


def _default(field_name: str) -> Any:
    from agent_utilities.core.config import AgentConfig

    info = AgentConfig.model_fields[field_name]
    try:
        return info.get_default(call_default_factory=True)
    except TypeError:  # pydantic < 2.10 factories taking validated data
        return info.get_default()


def describe(key: str = "", *, contains: str = "") -> dict[str, Any]:
    """Field docstring, type, default, current value and env alias.

    With ``key`` — one field, fully described. Without — the whole inventory
    (optionally narrowed by ``contains``, matched against alias and field
    name), so an agent can discover what is settable without reading source.
    Every value passes through :func:`redact`.
    """
    from agent_utilities.core.config import AgentConfig
    from agent_utilities.deployment import is_restart_required

    docs = _attribute_docstrings()

    def _one(env_key: str, field_name: str) -> dict[str, Any]:
        info = AgentConfig.model_fields[field_name]
        current, current_redacted = redact(env_key, _effective(field_name))
        default, default_redacted = redact(env_key, _default(field_name))
        return {
            "key": env_key,
            "field": field_name,
            "type": _type_name(info.annotation),
            "description": docs.get(field_name),
            "default": _jsonable(default),
            "current": _jsonable(current),
            "secret": configuration_key_is_sensitive(env_key),
            "redacted": current_redacted or default_redacted,
            "restart_required": is_restart_required(env_key),
        }

    if key:
        env_key, field_name = _resolve_key(key)
        return {"action": "describe", "field": _one(env_key, field_name)}

    needle = str(contains or "").strip().lower()
    fields = [
        _one(env_key, field_name)
        for env_key, field_name in sorted(field_index().items())
        if not needle or needle in env_key.lower() or needle in field_name
    ]
    return {"action": "describe", "count": len(fields), "fields": fields}


def get(key: str) -> dict[str, Any]:
    """The effective value of one field, redacted."""
    from agent_utilities.deployment import is_restart_required

    env_key, field_name = _resolve_key(key)
    value, was_redacted = redact(env_key, _effective(field_name))
    return {
        "action": "get",
        "key": env_key,
        "field": field_name,
        "value": _jsonable(value),
        "redacted": was_redacted,
        "secret": configuration_key_is_sensitive(env_key),
        "restart_required": is_restart_required(env_key),
    }


def diff() -> dict[str, Any]:
    """Every field whose effective value differs from its shipped default.

    The fastest answer to "why is this deployment behaving differently from
    that one". Both sides are redacted, so a deployment whose only difference
    is a credential reports the KEY as changed without disclosing either value.
    """
    from agent_utilities.deployment import is_restart_required

    changed: list[dict[str, Any]] = []
    for env_key, field_name in sorted(field_index().items()):
        effective = _effective(field_name)
        default = _default(field_name)
        if effective == default:
            continue
        safe_effective, eff_redacted = redact(env_key, effective)
        safe_default, def_redacted = redact(env_key, default)
        changed.append(
            {
                "key": env_key,
                "field": field_name,
                "default": _jsonable(safe_default),
                "effective": _jsonable(safe_effective),
                "redacted": eff_redacted or def_redacted,
                "restart_required": is_restart_required(env_key),
            }
        )
    return {
        "action": "diff",
        "count": len(changed),
        "total_fields": len(field_index()),
        "changed": changed,
    }


def reload() -> dict[str, Any]:
    """Re-read configuration from disk/env without restarting the process.

    Explicit about the limit: fields the restart classifier marks
    ``restart_required`` are wired into the engine/daemon at startup, so their
    NEW value is loaded here but is NOT in effect until a restart. They are
    returned by name rather than being silently reported as applied.
    """
    from agent_utilities.core.config import load_config
    from agent_utilities.deployment import is_restart_required

    load_config()
    pending = sorted(k for k in field_index() if is_restart_required(k))
    return {
        "action": "reload",
        "reloaded": True,
        "live_fields": len(field_index()) - len(pending),
        "restart_required_fields": pending,
        "note": (
            "restart_required fields were re-read but are wired at startup; "
            "their new values take effect only after a daemon restart"
        ),
    }


def _validate_against_model(field_name: str, value: Any) -> Any:
    """Validate one value against the field's own pydantic type. Refuses on
    failure — an unvalidated write is exactly what this surface must not do."""
    from pydantic import TypeAdapter, ValidationError

    from agent_utilities.core.config import AgentConfig

    info = AgentConfig.model_fields[field_name]
    try:
        return TypeAdapter(info.annotation).validate_python(value)
    except ValidationError as exc:
        raise ConfigAdminError(
            "validation_failed",
            f"value rejected by the AgentConfig schema for {field_name}: "
            f"{exc.error_count()} error(s); expected {_type_name(info.annotation)}",
        ) from exc
    except Exception as exc:  # noqa: BLE001 - unvalidatable is still a refusal
        raise ConfigAdminError(
            "validation_failed",
            f"value could not be validated against {field_name}: {type(exc).__name__}",
        ) from exc


def _coerce_input(value: Any) -> Any:
    """Parse a JSON-looking string argument; leave anything else alone."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text[:1] in "[{" or text in ("true", "false", "null"):
        try:
            return json.loads(text)
        except ValueError:
            return value
    return value


def _record_provenance(
    env_key: str, *, redacted: bool, reason: str, decision: Any
) -> str | None:
    """Best-effort ``:ConfigChange`` provenance node for one applied write.

    Best-effort by design: a provenance backend that is down must not roll back
    a configuration change the operator was authorised to make and that has
    already been persisted. The failure is logged at ERROR so it is findable.
    The VALUE is never recorded for a sensitive key — provenance answers who
    changed what and when, not what the secret was.
    """
    node_id = f"configchange:{env_key}:{int(time.time() * 1000)}"
    try:
        from agent_utilities.mcp import kg_server

        kg_server._get_engine().add_node(
            node_id,
            {
                "node_type": "ConfigChange",
                "config_key": env_key,
                "value_redacted": bool(redacted),
                "reason": reason or "",
                "decision": str(decision or ""),
                "changed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": "graph_config",
            },
        )
        return node_id
    except Exception as exc:  # noqa: BLE001 - provenance must not undo a write
        logger.error(
            "graph_config applied %s but its provenance event could NOT be "
            "recorded (error=%s); the change is live and unlogged in the KG",
            env_key,
            type(exc).__name__,
        )
        return None


def _gate(env_key: str, reason: str) -> tuple[bool, dict[str, Any]]:
    """ActionPolicy decision for a ``config.set``.

    Uses the same gate every other governed mutation goes through rather than
    a second authorization notion. A gate that cannot be consulted DENIES —
    an unavailable policy engine must never read as permission.
    """
    try:
        from agent_utilities.mcp import kg_server
        from agent_utilities.orchestration.action_policy import (
            ActionRequest,
            get_action_policy,
        )

        decision = get_action_policy(kg_server._get_engine()).decide(
            ActionRequest(
                kind="config.set",
                target=env_key,
                source="mcp",
                reason=reason or f"set {env_key}",
            )
        )
    except Exception as exc:  # noqa: BLE001 - fail closed
        logger.error(
            "config.set policy gate unavailable for %s; refusing (error=%s)",
            env_key,
            type(exc).__name__,
        )
        return False, {
            "decision": "denied",
            "reason": f"policy gate unavailable ({type(exc).__name__})",
        }
    return decision.allowed, {
        "decision": decision.decision,
        "tier": decision.tier,
        "reason": decision.reason,
        "approval_id": decision.approval_id,
    }


def set_value(key: str, value: Any, *, reason: str = "") -> dict[str, Any]:
    """Governed write of one ``AgentConfig`` field.

    Order matters and is not negotiable: resolve the key against the model →
    refuse inline secrets → validate against the field's pydantic type →
    ActionPolicy gate → write through ``save_config_item`` (the one existing
    config/secret precedence path) → record provenance. Every refusal happens
    BEFORE anything is persisted.
    """
    from agent_utilities.core.config import save_config_item
    from agent_utilities.deployment import is_restart_required

    env_key, field_name = _resolve_key(key)
    parsed = _coerce_input(value)

    # A sensitive setting may only ever be given a REFERENCE, and only on a
    # field whose contract is to hold one. This mirrors the pre-existing
    # graph_configure(set_config) rule; graph_config must not be the easier
    # door into the same disclosure.
    if configuration_key_is_sensitive(env_key):
        if not env_key.endswith("_REF") or not runtime_reference(parsed):
            raise ConfigAdminError(
                "inline_secret_refused",
                "sensitive settings cannot be persisted inline; store the value "
                "in the secret store and set a reference-capable *_REF key to a "
                "vault:// / env:// / secret:// reference",
            )

    validated = _validate_against_model(field_name, parsed)

    allowed, decision = _gate(env_key, reason)
    if not allowed:
        return {
            "action": "set",
            "key": env_key,
            "applied": False,
            "error": "policy_denied",
            **decision,
        }

    save_config_item(env_key, validated)
    restart = is_restart_required(env_key)
    safe_value, was_redacted = redact(env_key, validated)
    provenance_id = _record_provenance(
        env_key,
        redacted=was_redacted,
        reason=reason,
        decision=decision.get("decision"),
    )
    return {
        "action": "set",
        "key": env_key,
        "field": field_name,
        "applied": True,
        "value": _jsonable(safe_value),
        "redacted": was_redacted,
        # BUG-065: this used to be named ``applied_live`` and asserted a
        # fleet-wide fact ("this setting is now in effect") from evidence
        # ``is_restart_required`` can only ever have about ONE process — this
        # process's own startup-cached fields. In this fleet's multi-pod
        # topology, a caller reading ``applied_live: True`` had no way to know
        # whether any OTHER replica had even seen the write, let alone
        # reloaded it. Same overclaim shape BUG-050 named for
        # ``load_tools``'s old ``notified`` field ("a field asserting a
        # fleet-wide fact from process-local evidence"): rename to what this
        # process can actually observe rather than fabricate a poll this
        # module never performs. A real fleet-wide "did every replica pick
        # this up" answer needs an actual cross-replica poll/broadcast+ack —
        # a distinct, not-yet-built capability — not a renamed field.
        "applied_in_this_process": not restart,
        "restart_required": restart,
        "provenance_id": provenance_id,
        **decision,
    }


def dispatch(
    action: str, *, key: str = "", value: Any = "", reason: str = "", contains: str = ""
) -> dict[str, Any]:
    """Route one ``graph_config`` action. Unknown actions are refused."""
    if action == "describe":
        return describe(key, contains=contains)
    if action == "get":
        return get(key)
    if action == "diff":
        return diff()
    if action == "reload":
        return reload()
    if action == "set":
        return set_value(key, value, reason=reason)
    raise ConfigAdminError(
        "unknown_action",
        f"unknown action {action!r}; expected one of {', '.join(ACTIONS)}",
    )
