"""Pseudonymized security-audit records (EH-410, operator ruling 2026-09-24).

Security audit feeds (Keycloak user/admin events, OpenBao audit, CrowdSec
decisions) enter the knowledge graph ONLY in this form:

* identities (user ids, usernames, entity ids, display names) and secret paths
  become keyed HMAC references -- stable, so one principal's events still join,
  but not reversible without the key, which lives in OpenBao
  (``AUDIT_PSEUDONYM_HMAC_KEY_REF``);
* IP addresses are truncated to their /24 (IPv4) or /48 (IPv6) network;
* CrowdSec ban targets are kept whole (the decision IS about that address);
* everything else is dropped unless the feed's :class:`AuditFieldPolicy`
  allowlists it -- secret values and request/response bodies never pass.

There is no unkeyed fallback: without the key the feed refuses to ingest.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

#: The setting naming the OpenBao reference of the HMAC key.
KEY_SETTING = "AUDIT_PSEUDONYM_HMAC_KEY_REF"
_DOMAIN = b"agent-utilities:audit-pseudonym:v1"
_PREFIX_BITS = {4: 24, 6: 48}


class AuditPseudonymUnavailable(RuntimeError):
    """No pseudonymization key is configured or resolvable: refuse the feed."""


def _dotted(record: Mapping[str, Any], path: str) -> Any:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _scalar(value: Any) -> Any:
    """Allowlisted passthrough keeps scalars only -- never a nested body."""
    return value if isinstance(value, str | int | float | bool) else None


def ip_network(value: Any) -> str:
    """The /24 (IPv4) or /48 (IPv6) network of an address; ``""`` if invalid."""
    text = str(value or "").strip()
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return ""
    prefix = _PREFIX_BITS[address.version]
    return str(ipaddress.ip_network(f"{address}/{prefix}", strict=False))


@dataclass(frozen=True, slots=True)
class AuditFieldPolicy:
    """Which source fields a feed keeps, and how (dotted paths)."""

    keep: tuple[str, ...] = ()
    identities: tuple[str, ...] = ()
    secret_paths: tuple[str, ...] = ()
    ips: tuple[str, ...] = ()
    whole_ips: tuple[str, ...] = field(default=())


@dataclass(frozen=True, slots=True)
class AuditPseudonymizer:
    """Keyed, deterministic pseudonyms for one deployment."""

    key: bytes

    @classmethod
    def from_settings(cls) -> AuditPseudonymizer:
        """Resolve the key from OpenBao; raise when it is not available."""
        from agent_utilities.core.config import setting

        reference = str(setting("AUDIT_PSEUDONYM_HMAC_KEY_REF", "") or "").strip()
        if not reference:
            raise AuditPseudonymUnavailable(f"{KEY_SETTING} is not configured")
        from agent_utilities.security.secrets_client import create_secrets_client

        value = create_secrets_client().resolve_ref(reference)
        if not value:
            raise AuditPseudonymUnavailable(f"{KEY_SETTING} did not resolve")
        return cls(str(value).encode("utf-8"))

    def reference(self, kind: str, value: Any) -> str:
        """``apseud_<kind>_<hmac>``; ``""`` for an empty value."""
        text = str(value or "").strip()
        if not text:
            return ""
        framed = b"\x00".join((_DOMAIN, kind.encode(), text.encode("utf-8")))
        digest = hmac.new(self.key, framed, hashlib.sha256).hexdigest()[:32]
        return f"apseud_{kind}_{digest}"

    def identity(self, value: Any) -> str:
        return self.reference("identity", value)

    def secret_path(self, value: Any) -> str:
        return self.reference("secret_path", str(value or "").strip("/"))

    def record(self, record: Mapping[str, Any], policy: AuditFieldPolicy) -> dict:
        """The allowlisted, pseudonymized projection of one audit record."""
        out: dict[str, Any] = {}
        transforms = (
            (policy.keep, _scalar),
            (policy.identities, self.identity),
            (policy.secret_paths, self.secret_path),
            (policy.ips, ip_network),
            (policy.whole_ips, lambda v: str(v or "").strip()),
        )
        for paths, transform in transforms:
            for path in paths:
                value = _dotted(record, path)
                result = None if value in (None, "") else transform(value)
                if result not in (None, ""):
                    out[path.replace(".", "_")] = result
        return out


__all__ = [
    "KEY_SETTING",
    "AuditFieldPolicy",
    "AuditPseudonymUnavailable",
    "AuditPseudonymizer",
    "ip_network",
]
