"""Privacy guard for payloads that cross a persistence or telemetry boundary.

The guard is deliberately dependency-free and deterministic. It removes known
identifier formats, secret-bearing fields, personal-name fields, and machine-
specific filesystem locations before a value can enter the Knowledge Graph,
logs, or an external trace sink. Runtime deny terms are resolved from a secret
reference; the terms themselves are never committed or included in reports.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import socket
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent_utilities.core.config import setting

__all__ = [
    "PersistencePrivacyGuard",
    "PrivacyReport",
    "persistence_reference",
    "sanitize_for_persistence",
]


_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("tax_id", re.compile(r"\b\d{2}-\d{7}\b")),
    (
        "credit_card",
        re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    ),
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        "api_token",
        re.compile(
            r"(?i)\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{16,}|"
            r"gh[pousr]_[A-Za-z0-9]{20,})\b"
        ),
    ),
    (
        "bearer_token",
        re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    ),
    (
        "phone",
        re.compile(
            r"(?<!\w)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\w)"
        ),
    ),
    (
        "windows_user_path",
        re.compile(
            r"(?i)(?:[A-Z]:[\\/]|/mnt/[a-z]/)(?:Users|Documents and Settings)[\\/][^\\/\s]+(?:[\\/][^\s]*)?"
        ),
    ),
    (
        "windows_absolute_path",
        re.compile(r"(?i)\b[A-Z]:[\\/][^\r\n\t]+"),
    ),
    (
        "posix_user_path",
        re.compile(r"(?<![\w.])/(?:home|Users)/[^/\s]+(?:/[^\s]*)?"),
    ),
    (
        "posix_local_path",
        re.compile(
            r"(?<![\w.])/(?:mnt|opt|private/tmp|root|srv|tmp|var/tmp|workspace|workspaces)(?:/[^\s]*)?"
        ),
    ),
    ("file_uri", re.compile(r"(?i)\bfile://[^\s]+")),
    (
        "ipv4",
        re.compile(
            r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])"
        ),
    ),
    (
        "mac_address",
        re.compile(r"(?i)(?<![0-9a-f])(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}(?![0-9a-f])"),
    ),
    (
        "iban",
        re.compile(r"(?i)\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}\b"),
    ),
)

_SECRET_FIELDS = frozenset(
    {
        "authorization",
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "client_secret",
        "password",
        "secret",
        "token",
    }
)
_LOCATION_FIELDS = frozenset(
    {
        "base_url",
        "canonical_url",
        "dsn",
        "endpoint",
        "endpoint_url",
        "file_path",
        "home",
        "host",
        "hostname",
        "href",
        "html_url",
        "local_path",
        "path",
        "pdf_url",
        "root_path",
        "source_uri",
        "uri",
        "url",
        "web_url",
        "weburl",
        "workspace_path",
    }
)
_PERSON_FIELDS = frozenset(
    {
        "assignee",
        "author",
        "author_name",
        "birth_date",
        "created_by",
        "date_of_birth",
        "display_name",
        "employee_id",
        "first_name",
        "full_name",
        "ip_address",
        "last_name",
        "owner",
        "owner_name",
        "person_name",
        "requester",
        "updated_by",
        "user_email",
        "user_name",
        "username",
    }
)
_PERSON_CONTEXT = frozenset({"author", "employee", "owner", "person", "user"})
_REFERENCE_RE = re.compile(r"^pref_[a-z0-9_]+_[0-9a-f]{64}$")


@lru_cache(maxsize=1)
def _persistence_identity_key() -> bytes | None:
    reference = str(setting("PERSISTENCE_IDENTITY_HMAC_KEY_REF", "") or "").strip()
    if reference:
        from agent_utilities.security.secrets_client import create_secrets_client

        value = create_secrets_client().resolve_ref(reference)
        if value:
            return str(value).encode("utf-8")

    configured = str(setting("GRAPH_SERVICE_AUTH_SECRET", "") or "").strip()
    if configured:
        return configured.encode("utf-8")

    from agent_utilities.core.profile_guard import is_production_profile

    if is_production_profile():
        raise RuntimeError(
            "production persistence references require "
            "PERSISTENCE_IDENTITY_HMAC_KEY_REF or GRAPH_SERVICE_AUTH_SECRET"
        )
    return None


def persistence_reference(kind: str, value: Any, *, namespace: str = "") -> str:
    """Return a stable non-reversible reference for a durable identity field."""

    text = str(value or "")
    if not text:
        return ""
    if _REFERENCE_RE.fullmatch(text):
        return text
    label = re.sub(r"[^a-z0-9_]+", "_", str(kind).lower()).strip("_") or "value"
    framed = b"\x00".join(
        (
            b"agent-utilities:persistence-reference:v1",
            label.encode("utf-8"),
            str(namespace or "").encode("utf-8"),
            text.encode("utf-8"),
        )
    )
    key = _persistence_identity_key()
    digest = (
        hmac.new(key, framed, hashlib.sha256).hexdigest()
        if key is not None
        else hashlib.sha256(framed).hexdigest()
    )
    return f"pref_{label}_{digest}"


def _field_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _runtime_deny_terms() -> tuple[str, ...]:
    """Derive local identity terms in memory, then resolve optional policy terms.

    Runtime account/home/host identifiers are never written to configuration or
    reports. They exist only long enough to compile the in-process redaction
    expressions.
    """
    candidates = {
        str(os.environ.get(name) or "").strip()
        for name in ("LOGNAME", "USER", "USERNAME")
    }
    try:
        candidates.add(Path.home().name.strip())
    except Exception:
        pass
    try:
        candidates.add(socket.gethostname().strip())
    except Exception:
        pass
    ignored = {
        "",
        "admin",
        "administrator",
        "cache",
        "config",
        "data",
        "home",
        "root",
        "runner",
        "system",
        "tmp",
        "user",
        "workspace",
    }
    terms = {
        value
        for value in candidates
        if len(value) >= 3 and value.casefold() not in ignored
    }

    ref = str(setting("PERSISTENCE_PRIVACY_DENY_TERMS_REF", "") or "").strip()
    if not ref:
        return tuple(sorted(terms))
    try:
        from agent_utilities.security.secrets_client import create_secrets_client

        raw = create_secrets_client().resolve_ref(ref)
        if not raw:
            return tuple(sorted(terms))
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            value = [part.strip() for part in str(raw).split(",")]
        if not isinstance(value, list):
            return tuple(sorted(terms))
        terms.update(
            term.strip()
            for term in value
            if isinstance(term, str) and len(term.strip()) >= 3
        )
        return tuple(sorted(terms))
    except Exception:
        # A missing optional deny-list never exposes the secret reference or its
        # resolution error. Deterministic pattern/field redaction still applies.
        return tuple(sorted(terms))


@dataclass(frozen=True)
class PrivacyReport:
    """Non-sensitive summary of a sanitization pass."""

    redactions: int
    detected_types: tuple[str, ...]

    @property
    def changed(self) -> bool:
        return self.redactions > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "redactions": self.redactions,
            "detected_types": list(self.detected_types),
        }


class PersistencePrivacyGuard:
    """Deep sanitizer for data entering durable or external observability stores."""

    def __init__(
        self, *, deny_terms: tuple[str, ...] | list[str] | None = None
    ) -> None:
        terms = tuple(deny_terms) if deny_terms is not None else _runtime_deny_terms()
        self._deny_terms = tuple(
            ("identity_term", re.compile(re.escape(term), re.IGNORECASE))
            for term in terms
            if isinstance(term, str) and len(term.strip()) >= 3
        )

    def sanitize(self, value: Any) -> tuple[Any, PrivacyReport]:
        counts: dict[str, int] = {}
        clean = self._sanitize(value, counts, ())
        return clean, PrivacyReport(
            redactions=sum(counts.values()),
            detected_types=tuple(sorted(counts)),
        )

    def sanitize_text(self, value: str) -> tuple[str, PrivacyReport]:
        clean, report = self.sanitize(value)
        return str(clean), report

    def _sanitize(
        self,
        value: Any,
        counts: dict[str, int],
        context: tuple[str, ...],
    ) -> Any:
        if isinstance(value, str):
            return self._sanitize_string(value, counts)
        if isinstance(value, dict):
            clean: dict[str, Any] = {}
            for raw_key, item in value.items():
                key = str(raw_key)
                field = _field_name(key)
                populated = not (
                    item is None
                    or (isinstance(item, str) and not item)
                    or (
                        isinstance(item, (dict, list, tuple, set, frozenset))
                        and not item
                    )
                )
                if field in _SECRET_FIELDS and populated:
                    counts["secret_field"] = counts.get("secret_field", 0) + 1
                    clean[key] = "[REDACTED_SECRET]"
                    continue
                if field in _LOCATION_FIELDS and populated:
                    counts["location_field"] = counts.get("location_field", 0) + 1
                    clean[key] = "[REDACTED_LOCATION]"
                    continue
                personal = field in _PERSON_FIELDS or (
                    field == "name" and any(part in _PERSON_CONTEXT for part in context)
                )
                if personal and populated:
                    counts["personal_field"] = counts.get("personal_field", 0) + 1
                    clean[key] = "[REDACTED_PERSON]"
                    continue
                clean[key] = self._sanitize(item, counts, (*context, field))
            return clean
        if isinstance(value, (list, tuple, set, frozenset)):
            return [self._sanitize(item, counts, context) for item in value]
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        # Never serialize an arbitrary object's repr: it commonly embeds memory
        # addresses, credentials, or machine-specific paths.
        counts["opaque_object"] = counts.get("opaque_object", 0) + 1
        return f"[REDACTED_OBJECT:{type(value).__name__}]"

    def _sanitize_string(self, value: str, counts: dict[str, int]) -> str:
        clean = value
        for label, pattern in (*_PATTERNS, *self._deny_terms):
            clean, count = pattern.subn(f"[REDACTED_{label.upper()}]", clean)
            if count:
                counts[label] = counts.get(label, 0) + count
        return clean


def sanitize_for_persistence(
    value: Any, *, deny_terms: tuple[str, ...] | list[str] | None = None
) -> tuple[Any, PrivacyReport]:
    """Sanitize ``value`` and return only a count/type report alongside it."""
    return PersistencePrivacyGuard(deny_terms=deny_terms).sanitize(value)
