"""Privacy-safe identifiers and payloads for durable AgentBus boundaries.

Raw participant, tenant, topic, host, session, and actor values are runtime
inputs.  They must not become graph identifiers, broker routing keys, logs, or
trace attributes.  This module converts them to stable, non-reversible
references and applies the shared persistence sanitizer to message content.

Production requires an operator-managed HMAC key referenced through the
secrets layer.  Development remains zero-config by using an unkeyed digest;
that fallback is pseudonymous, but deliberately rejected by production.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from functools import lru_cache
from typing import Any

from agent_utilities.core.config import setting
from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

__all__ = ["bus_reference", "sanitize_bus_content"]


_REFERENCE = re.compile(r"^busref_[a-z0-9_]+_[0-9a-f]{64}$")


@lru_cache(maxsize=1)
def _identity_key() -> bytes | None:
    reference = str(setting("BUS_IDENTITY_HMAC_KEY_REF", "") or "").strip()
    if reference:
        from agent_utilities.security.secrets_client import create_secrets_client

        value = create_secrets_client().resolve_ref(reference)
        if value:
            return str(value).encode("utf-8")

    # Reuse an already provisioned service-auth secret when present.  This is
    # configuration, never a source-code trust bypass or hard-coded verifier.
    configured = str(setting("GRAPH_SERVICE_AUTH_SECRET", "") or "").strip()
    if configured:
        return configured.encode("utf-8")

    from agent_utilities.core.profile_guard import is_production_profile

    if is_production_profile():
        raise RuntimeError(
            "production AgentBus requires BUS_IDENTITY_HMAC_KEY_REF "
            "or GRAPH_SERVICE_AUTH_SECRET"
        )
    return None


def bus_reference(kind: str, value: str, *, tenant: str = "") -> str:
    """Return an idempotent stable reference without retaining ``value``."""

    text = str(value or "")
    if not text:
        return ""
    if _REFERENCE.fullmatch(text):
        return text
    namespace = re.sub(r"[^a-z0-9_]+", "_", str(kind).lower()).strip("_") or "value"
    framed = b"\x00".join(
        (
            b"agent-utilities:bus-reference:v1",
            namespace.encode("utf-8"),
            str(tenant or "").encode("utf-8"),
            text.encode("utf-8"),
        )
    )
    key = _identity_key()
    digest = (
        hmac.new(key, framed, hashlib.sha256).hexdigest()
        if key is not None
        else hashlib.sha256(framed).hexdigest()
    )
    return f"busref_{namespace}_{digest}"


def sanitize_bus_content(
    payload: Any, metadata: dict[str, Any] | None
) -> tuple[str, str, dict[str, Any]]:
    """Sanitize content before any graph, broker, or telemetry persistence."""

    guard = PersistencePrivacyGuard()
    clean_payload, payload_report = guard.sanitize_text(str(payload or ""))
    clean_metadata, metadata_report = guard.sanitize(metadata or {})
    if not isinstance(clean_metadata, dict):
        clean_metadata = {}
    report = {
        "redactions": payload_report.redactions + metadata_report.redactions,
        "detected_types": sorted(
            set(payload_report.detected_types) | set(metadata_report.detected_types)
        ),
    }
    return (
        clean_payload,
        json.dumps(clean_metadata, sort_keys=True, separators=(",", ":")),
        report,
    )
