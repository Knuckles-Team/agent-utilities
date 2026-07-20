"""Privacy and identity normalization for the durable usage store.

The usage database is an analytics fact store, not a transcript archive.  This
module is the single boundary every SQL backend uses before persistence:

* tenant, run, correlation, parent, tool-use, and dedup identities become
  stable non-reversible references;
* run identities are tenant-qualified, preventing equal source IDs in two
  tenants from colliding;
* local filesystem locations and host names are never retained;
* free-text prompts, thinking text, and tool inputs are metadata-only by
  default.  A deployment may explicitly select ``sanitized`` retention for a
  separately governed store, but the persistence privacy guard still runs.

Only privacy category/count evidence may be logged by callers.  Raw values are
never included in errors or diagnostics from this module.
"""

from __future__ import annotations

import json
from typing import Any

from agent_utilities.core.config import setting
from agent_utilities.security.persistence_privacy import (
    PersistencePrivacyGuard,
    persistence_reference,
)

from .models import (
    ParsedSessionBundle,
    UsageEvent,
    UsageMessage,
    UsageSession,
    UsageToolCall,
)

__all__ = [
    "normalize_bundle",
    "normalize_run_id",
    "normalize_tenant_id",
    "normalize_tool_call",
    "normalize_usage_event",
    "sanitize_query_text",
]

_METADATA_ONLY = frozenset({"", "metadata", "none", "off", "false", "0"})
_SANITIZED = frozenset({"sanitized", "redacted"})


def _retains_sanitized_content() -> bool:
    mode = str(setting("USAGE_CONTENT_RETENTION", "metadata") or "metadata")
    normalized = mode.strip().lower()
    if normalized in _METADATA_ONLY:
        return False
    if normalized in _SANITIZED:
        return True
    raise ValueError(
        "USAGE_CONTENT_RETENTION must be 'metadata' or 'sanitized'; raw "
        "transcript retention is not supported by the usage analytics store"
    )


def normalize_tenant_id(value: str | None) -> str:
    """Return the stable opaque tenant reference persisted by usage storage."""

    return persistence_reference("tenant", value or "")


def normalize_run_id(value: str | None, *, tenant_id: str | None = "") -> str:
    """Return a stable tenant-qualified opaque run/trace identity."""

    tenant_ref = normalize_tenant_id(tenant_id)
    return persistence_reference("run", value or "", namespace=tenant_ref)


def _reference(kind: str, value: str | None, tenant_ref: str) -> str:
    return persistence_reference(kind, value or "", namespace=tenant_ref)


def _safe_text(value: Any, guard: PersistencePrivacyGuard) -> str:
    clean, _report = guard.sanitize_text(str(value or ""))
    return clean


def sanitize_query_text(value: str) -> str:
    """Sanitize a transient search term without retaining the source value."""

    return _safe_text(value, PersistencePrivacyGuard())


def _safe_json(value: str | None, guard: PersistencePrivacyGuard) -> str | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return _safe_text(value, guard)
    clean, _report = guard.sanitize(parsed)
    return json.dumps(clean, sort_keys=True, separators=(",", ":"))


def _normalize_message(
    message: UsageMessage,
    *,
    run_ref: str,
    retain_content: bool,
    guard: PersistencePrivacyGuard,
) -> UsageMessage:
    return message.model_copy(
        update={
            "session_id": run_ref,
            "content": _safe_text(message.content, guard) if retain_content else "",
            "thinking_text": (
                _safe_text(message.thinking_text, guard) if retain_content else ""
            ),
            "model": _safe_text(message.model, guard),
        }
    )


def normalize_tool_call(
    call: UsageToolCall,
    *,
    authoritative_tenant: str | None = None,
    authoritative_run_id: str | None = None,
) -> UsageToolCall:
    """Normalize a tool-call row before direct or bundled persistence."""

    guard = PersistencePrivacyGuard()
    raw_tenant = (
        authoritative_tenant if authoritative_tenant is not None else call.tenant_id
    )
    tenant_ref = normalize_tenant_id(raw_tenant)
    run_ref = normalize_run_id(
        authoritative_run_id if authoritative_run_id is not None else call.session_id,
        tenant_id=raw_tenant,
    )
    retain_content = _retains_sanitized_content()
    return call.model_copy(
        update={
            "session_id": run_ref,
            "tenant_id": tenant_ref,
            "correlation_id": _reference(
                "correlation", call.correlation_id, tenant_ref
            ),
            "tool_use_id": _reference("tool_use", call.tool_use_id, tenant_ref),
            "subagent_session_id": normalize_run_id(
                call.subagent_session_id, tenant_id=raw_tenant
            ),
            "tool_name": _safe_text(call.tool_name, guard),
            "skill_name": (
                _safe_text(call.skill_name, guard) if call.skill_name else None
            ),
            "input_json": (
                _safe_json(call.input_json, guard) if retain_content else None
            ),
            "status": _safe_text(call.status, guard),
        }
    )


def normalize_usage_event(
    event: UsageEvent,
    *,
    authoritative_tenant: str | None = None,
    authoritative_run_id: str | None = None,
) -> UsageEvent:
    """Normalize a token/cost fact before direct or bundled persistence."""

    guard = PersistencePrivacyGuard()
    raw_tenant = (
        authoritative_tenant if authoritative_tenant is not None else event.tenant_id
    )
    tenant_ref = normalize_tenant_id(raw_tenant)
    run_ref = normalize_run_id(
        authoritative_run_id if authoritative_run_id is not None else event.session_id,
        tenant_id=raw_tenant,
    )
    return event.model_copy(
        update={
            "session_id": run_ref,
            "tenant_id": tenant_ref,
            "correlation_id": _reference(
                "correlation", event.correlation_id, tenant_ref
            ),
            "dedup_key": _reference("usage_dedup", event.dedup_key, tenant_ref),
            "source": _safe_text(event.source, guard),
            "model": _safe_text(event.model, guard),
            "cost_status": _safe_text(event.cost_status, guard),
            "cost_source": _safe_text(event.cost_source, guard),
        }
    )


def normalize_bundle(bundle: ParsedSessionBundle) -> ParsedSessionBundle:
    """Return a privacy-safe, tenant-qualified copy of a session bundle."""

    guard = PersistencePrivacyGuard()
    source = bundle.session
    raw_tenant = source.tenant_id
    tenant_ref = normalize_tenant_id(raw_tenant)
    run_ref = normalize_run_id(source.id, tenant_id=raw_tenant)
    retain_content = _retains_sanitized_content()

    session: UsageSession = source.model_copy(
        update={
            "id": run_ref,
            "tenant_id": tenant_ref,
            "correlation_id": _reference(
                "correlation", source.correlation_id, tenant_ref
            ),
            "parent_session_id": normalize_run_id(
                source.parent_session_id, tenant_id=raw_tenant
            ),
            "project": _safe_text(source.project, guard),
            "agent": _safe_text(source.agent, guard),
            "machine": "unattributed",
            "first_message": (
                _safe_text(source.first_message, guard) if retain_content else ""
            ),
            "file_path": None,
        }
    )
    messages = [
        _normalize_message(
            message,
            run_ref=run_ref,
            retain_content=retain_content,
            guard=guard,
        )
        for message in bundle.messages
    ]
    tool_calls = [
        normalize_tool_call(
            call,
            authoritative_tenant=raw_tenant,
            authoritative_run_id=source.id,
        )
        for call in bundle.tool_calls
    ]
    usage_events = [
        normalize_usage_event(
            event,
            authoritative_tenant=raw_tenant,
            authoritative_run_id=source.id,
        )
        for event in bundle.usage_events
    ]
    return ParsedSessionBundle(
        session=session,
        messages=messages,
        tool_calls=tool_calls,
        usage_events=usage_events,
    )
