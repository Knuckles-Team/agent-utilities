"""Fail-closed TLS and native MCP configuration for Langfuse.

The Langfuse Python SDK and ``requests``-based ``langfuse-agent`` client use
different trust environment variables.  This module resolves one
operator-supplied trust profile, validates its CA material, and projects it into
those runtime-only variables. A bundle may contain one private root, a chain,
or independent roots in a trust store. Paths, certificate subjects,
credentials, and hosts are never returned in status objects or logged.
"""

from __future__ import annotations

import os
import re
import ssl
import sys
import warnings
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import (
    dsa,
    ec,
    ed448,
    ed25519,
    padding,
    rsa,
)
from cryptography.utils import CryptographyDeprecationWarning
from cryptography.x509.oid import ExtensionOID
from packaging.version import InvalidVersion, Version

from agent_utilities.core.config import resolve_langfuse_host, setting

if TYPE_CHECKING:
    from agent_utilities.core.config import AgentConfig
    from agent_utilities.core.transport_security import ResolvedTLSProfile

__all__ = [
    "LangfuseTrustError",
    "LangfuseTrustStatus",
    "configure_langfuse_trust",
    "langfuse_credentials_configured",
    "langfuse_parent_kg_ingestion_enabled",
    "langfuse_provider_contract_ready",
    "native_langfuse_mcp_config",
    "prepare_langfuse_mcp_config",
    "resolve_langfuse_credentials",
    "resolve_langfuse_host",
    "resolve_langfuse_persistence_hmac_key",
    "resolve_langfuse_requests_transport",
    "validate_ca_bundle",
]

_PEM_CERT_RE = re.compile(
    rb"-----BEGIN CERTIFICATE-----\s+.*?\s+-----END CERTIFICATE-----",
    re.DOTALL,
)
_RUNTIME_PLACEHOLDER_RE = re.compile(
    r"^(?:\$\{(?:env:)?[A-Za-z_][A-Za-z0-9_]*\}|%[A-Za-z_][A-Za-z0-9_]*%|<[^>]+>)$"
)
_RUNTIME_PLACEHOLDER_FRAGMENT_RE = re.compile(
    r"(?:\$\{(?:env:)?[A-Za-z_][A-Za-z0-9_]*\}|"
    r"%[A-Za-z_][A-Za-z0-9_]*%|\{\{[^{}]+\}\}|<[^>\r\n]+>)"
)
_CREDENTIAL_SENTINELS = frozenset(
    {
        "changeme",
        "change_me",
        "example",
        "example_value",
        "masked",
        "none",
        "null",
        "placeholder",
        "public_key",
        "redacted",
        "replace_me",
        "secret_key",
        "unset",
        "your_api_key",
        "your_key",
        "your_langfuse_public_key",
        "your_langfuse_secret_key",
        "your_secret",
    }
)
_SECRET_REF_RE = re.compile(r"^(?:vault|secret|env)://[A-Za-z0-9_./#-]+$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_LANGFUSE_SERVER_NAMES = frozenset({"langfuse-agent", "langfuse-mcp"})
_MATERIALIZED_TRUST_FLAG = "LANGFUSE_TRUST_MATERIALIZED"
_PARENT_KG_INGESTION_FLAG = "_graphos_parent_kg_ingestion"
_PARENT_TRUST_SELECTORS = frozenset(
    {
        "LANGFUSE_TLS_PROFILE",
        "LANGFUSE_TLS_PROFILE_REF",
        "TLS_PROFILE",
        "TLS_PROFILE_REF",
        "TLS_PROFILES_REF",
        "TLS_PROFILES",
        "LANGFUSE_CA_BUNDLE_REF",
        "LANGFUSE_CA_BUNDLE",
        "LANGFUSE_CLIENT_CERT_REF",
        "LANGFUSE_CLIENT_KEY_REF",
        "LANGFUSE_CLIENT_KEY_PASSWORD_REF",
        "LANGFUSE_PROXY_URL_REF",
        "LANGFUSE_PROXY_URL",
        "LANGFUSE_NO_PROXY",
    }
)
_MAX_BUNDLE_BYTES = 2_000_000
_TRUST_STORE_CERTIFICATE_THRESHOLD = 10
_LANGFUSE_PROVIDER_MIN_VERSION = Version("1.0.3")
_LANGFUSE_PROVIDER_MAX_VERSION = Version("2")
_AGENT_CONFIG_FIELDS = {
    "LANGFUSE_HOST": "langfuse_host",
    "LANGFUSE_PUBLIC_KEY_REF": "langfuse_public_key_ref",
    "LANGFUSE_SECRET_KEY_REF": "langfuse_secret_key_ref",
    "LANGFUSE_PERSISTENCE_HMAC_KEY_REF": "langfuse_persistence_hmac_key_ref",
    "LANGFUSE_TLS_PROFILE": "langfuse_tls_profile",
    "LANGFUSE_TLS_PROFILE_REF": "langfuse_tls_profile_ref",
    "LANGFUSE_CA_BUNDLE_REF": "langfuse_ca_bundle_ref",
    "LANGFUSE_CLIENT_CERT_REF": "langfuse_client_cert_ref",
    "LANGFUSE_CLIENT_KEY_REF": "langfuse_client_key_ref",
    "LANGFUSE_CLIENT_KEY_PASSWORD_REF": "langfuse_client_key_password_ref",
    "LANGFUSE_PROXY_URL_REF": "langfuse_proxy_url_ref",
    "LANGFUSE_CAPTURE_CONTENT": "langfuse_capture_content",
    "LANGFUSE_KG_AUTO_INGEST": "langfuse_kg_auto_ingest",
    "LANGFUSE_MCP_ENABLED": "langfuse_mcp_enabled",
}


class LangfuseTrustError(RuntimeError):
    """A stable, non-sensitive trust failure."""

    _CATEGORIES = {
        "langfuse_host_invalid": "host",
        "langfuse_credentials_missing": "credentials",
        "langfuse_credentials_invalid": "credentials",
        "langfuse_persistence_hmac_key_invalid": "persistence",
        "langfuse_ca_bundle_invalid": "transport_security",
        "langfuse_requests_transport_invalid": "transport_security",
    }

    def __init__(self, reason: str) -> None:
        safe_reason = (
            reason if reason in self._CATEGORIES else "langfuse_configuration_invalid"
        )
        self.reason = safe_reason
        self.category = self._CATEGORIES.get(safe_reason, "configuration")
        super().__init__(safe_reason)


@dataclass(frozen=True)
class LangfuseTrustStatus:
    """Path- and identity-free trust readiness summary."""

    configured: bool
    valid: bool
    certificate_count: int = 0
    source: str = "system"
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "valid": self.valid,
            "certificate_count": self.certificate_count,
            "source": self.source,
            "reason": self.reason,
        }


def _value(
    environ: MutableMapping[str, str],
    name: str,
    agent_config: AgentConfig | None = None,
) -> str:
    value = environ.get(name)
    if value not in (None, ""):
        return str(value).strip()
    field_name = _AGENT_CONFIG_FIELDS.get(name)
    if agent_config is not None and field_name:
        value = getattr(agent_config, field_name, None)
        if value not in (None, ""):
            return str(value).strip()
    if environ is not os.environ:
        return ""
    return str(setting(name, "") or "").strip()


def langfuse_credentials_configured(
    *,
    environ: MutableMapping[str, str] | None = None,
    agent_config: AgentConfig | None = None,
) -> bool:
    """Return credential-pair readiness without resolving or exposing secrets."""
    target_env = environ if environ is not None else os.environ
    public_ready = bool(_value(target_env, "LANGFUSE_PUBLIC_KEY_REF", agent_config))
    secret_ready = bool(_value(target_env, "LANGFUSE_SECRET_KEY_REF", agent_config))
    return public_ready and secret_ready


def langfuse_provider_contract_ready() -> bool:
    """Return whether the installed MCP child implements the current contract.

    Merely finding ``langfuse_agent.mcp_server`` is insufficient. The current
    certified provider floor exposes the metadata-only ``runtime_posture`` proof
    and current trust contract. GraphOS launches this exact interpreter, so reject
    a stale or incomplete provider before reporting the child as available.
    """

    try:
        installed = Version(package_version("langfuse-agent"))
        surfaces_ready = all(
            find_spec(module) is not None
            for module in (
                "langfuse_agent.mcp_server",
                "langfuse_agent.runtime_posture",
            )
        )
    except (
        AttributeError,
        ImportError,
        InvalidVersion,
        ModuleNotFoundError,
        PackageNotFoundError,
        TypeError,
        ValueError,
    ):
        return False
    return bool(
        surfaces_ready
        and _LANGFUSE_PROVIDER_MIN_VERSION <= installed < _LANGFUSE_PROVIDER_MAX_VERSION
    )


def _resolve_secret_reference(
    reference: str,
    *,
    environ: MutableMapping[str, str],
    resolver: Callable[[str], str | None] | None,
) -> str | bytes | None:
    """Resolve one reference within the caller's explicit runtime boundary."""

    if reference.startswith("env://"):
        target = reference.removeprefix("env://")
        if _ENV_NAME_RE.fullmatch(target) is None:
            return None
        return environ.get(target)
    if resolver is not None:
        return resolver(reference)
    from agent_utilities.security.secrets_client import create_secrets_client

    return create_secrets_client().resolve_ref(reference)


def _credential_material_is_sentinel(value: str) -> bool:
    """Reject obvious masks/templates without assuming a provider key length."""

    rendered = value.strip()
    if not rendered:
        return True
    if _RUNTIME_PLACEHOLDER_FRAGMENT_RE.search(rendered):
        return True
    if re.fullmatch(r"(?:\*+|#+|x{4,})", rendered, flags=re.IGNORECASE):
        return True
    normalized = re.sub(r"[^a-z0-9]+", "_", rendered.lower()).strip("_")
    if normalized in _CREDENTIAL_SENTINELS:
        return True
    tokens = set(re.findall(r"[a-z0-9]+", rendered.lower()))
    if (
        tokens.intersection(
            {"changeme", "example", "masked", "placeholder", "redacted", "your"}
        )
        or {"replace", "me"} <= tokens
    ):
        return True
    if re.fullmatch(r"[A-Z][A-Z0-9_ -]{3,}", rendered):
        words = set(re.findall(r"[A-Z]+", rendered))
        if words.intersection(
            {
                "CHANGEME",
                "EXAMPLE",
                "MASKED",
                "PLACEHOLDER",
                "REDACTED",
                "REPLACE",
                "YOUR",
            }
        ):
            return True
    return False


def resolve_langfuse_credentials(
    *,
    environ: MutableMapping[str, str] | None = None,
    agent_config: AgentConfig | None = None,
    resolver: Callable[[str], str | None] | None = None,
) -> tuple[str, str]:
    """Materialize the Langfuse key pair in memory from strict secret refs."""
    target_env = environ if environ is not None else os.environ
    resolved: list[str] = []
    for key in ("LANGFUSE_PUBLIC_KEY_REF", "LANGFUSE_SECRET_KEY_REF"):
        reference = _concrete_runtime_value(_value(target_env, key, agent_config))
        if _SECRET_REF_RE.fullmatch(reference) is None or ".." in reference.partition(
            "://"
        )[2].split("/"):
            raise LangfuseTrustError("langfuse_credentials_missing")
        try:
            material = _resolve_secret_reference(
                reference,
                environ=target_env,
                resolver=resolver,
            )
            if isinstance(material, bytes):
                material = material.decode("utf-8")
            value = _concrete_runtime_value(material)
        except Exception:
            value = ""
        if (
            not value
            or len(value.encode("utf-8")) > 16_384
            or any(ord(character) < 32 for character in value)
        ):
            raise LangfuseTrustError("langfuse_credentials_missing")
        if _credential_material_is_sentinel(value):
            raise LangfuseTrustError("langfuse_credentials_invalid")
        resolved.append(value)
    return resolved[0], resolved[1]


def resolve_langfuse_persistence_hmac_key(
    *,
    environ: MutableMapping[str, str] | None = None,
    agent_config: AgentConfig | None = None,
    resolver: Callable[[str], str | None] | None = None,
) -> str | None:
    """Resolve the optional dedicated persistence key from one strict ref."""
    target_env = environ if environ is not None else os.environ
    reference = _concrete_runtime_value(
        _value(target_env, "LANGFUSE_PERSISTENCE_HMAC_KEY_REF", agent_config)
    )
    if not reference:
        return None
    if _SECRET_REF_RE.fullmatch(reference) is None or ".." in reference.partition(
        "://"
    )[2].split("/"):
        raise LangfuseTrustError("langfuse_persistence_hmac_key_invalid")
    try:
        material = _resolve_secret_reference(
            reference,
            environ=target_env,
            resolver=resolver,
        )
        if isinstance(material, bytes):
            material = material.decode("utf-8")
        value = _concrete_runtime_value(material)
    except Exception:
        value = ""
    encoded = value.encode("utf-8")
    if (
        len(encoded) < 32
        or len(encoded) > 16_384
        or any(ord(character) < 32 for character in value)
    ):
        raise LangfuseTrustError("langfuse_persistence_hmac_key_invalid")
    return value


def _concrete_runtime_value(value: Any) -> str:
    """Return a configured value, treating launcher placeholders as unresolved."""

    rendered = str(value or "").strip()
    if not rendered or _RUNTIME_PLACEHOLDER_RE.fullmatch(rendered):
        return ""
    return rendered


def _signature_is_valid(
    certificate: x509.Certificate, issuer: x509.Certificate
) -> bool:
    """Verify one certificate link without exposing certificate identities."""
    key = issuer.public_key()
    try:
        if isinstance(key, rsa.RSAPublicKey):
            key.verify(
                certificate.signature,
                certificate.tbs_certificate_bytes,
                padding.PKCS1v15(),
                certificate.signature_hash_algorithm,
            )
        elif isinstance(key, ec.EllipticCurvePublicKey):
            key.verify(
                certificate.signature,
                certificate.tbs_certificate_bytes,
                ec.ECDSA(certificate.signature_hash_algorithm),
            )
        elif isinstance(key, dsa.DSAPublicKey):
            key.verify(
                certificate.signature,
                certificate.tbs_certificate_bytes,
                certificate.signature_hash_algorithm,
            )
        elif isinstance(key, (ed25519.Ed25519PublicKey, ed448.Ed448PublicKey)):
            key.verify(certificate.signature, certificate.tbs_certificate_bytes)
        else:
            return False
    except (InvalidSignature, TypeError, ValueError):
        return False
    return True


def validate_ca_bundle(value: bytes | str | Path) -> LangfuseTrustStatus:
    """Validate a bounded PEM trust store containing at least one CA.

    Trust stores are sets, not necessarily a single chain. OpenSSL performs
    connection-time path construction; this check proves only bounded,
    parseable, currently valid CA material and valid self-signed roots.
    """
    try:
        if isinstance(value, Path):
            if not value.is_file() or value.stat().st_size > _MAX_BUNDLE_BYTES:
                return LangfuseTrustStatus(
                    configured=True,
                    valid=False,
                    source="file",
                    reason="bundle_unavailable",
                )
            payload = value.read_bytes()
        elif isinstance(value, bytes):
            payload = value
        else:
            payload = value.encode("utf-8")
    except Exception:
        return LangfuseTrustStatus(
            configured=True,
            valid=False,
            source="file",
            reason="bundle_unavailable",
        )

    if len(payload) > _MAX_BUNDLE_BYTES:
        return LangfuseTrustStatus(
            configured=True,
            valid=False,
            source="inline",
            reason="bundle_too_large",
        )

    blocks = _PEM_CERT_RE.findall(payload)
    certificates: list[x509.Certificate] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CryptographyDeprecationWarning)
            certificates = [x509.load_pem_x509_certificate(block) for block in blocks]
    except Exception:
        return LangfuseTrustStatus(
            configured=True,
            valid=False,
            source="inline",
            reason="invalid_pem",
        )

    if not certificates:
        return LangfuseTrustStatus(
            configured=True,
            valid=False,
            source="inline",
            reason="invalid_pem",
        )

    now = datetime.now(UTC)
    if any(certificate.serial_number <= 0 for certificate in certificates):
        return LangfuseTrustStatus(
            configured=True,
            valid=False,
            certificate_count=len(certificates),
            source="inline",
            reason="invalid_certificate_serial",
        )
    if any(
        cert.not_valid_before_utc > now or cert.not_valid_after_utc <= now
        for cert in certificates
    ):
        return LangfuseTrustStatus(
            configured=True,
            valid=False,
            certificate_count=len(certificates),
            source="inline",
            reason="certificate_outside_validity_window",
        )

    ca_count = 0
    for certificate in certificates:
        try:
            basic = certificate.extensions.get_extension_for_oid(
                ExtensionOID.BASIC_CONSTRAINTS
            ).value
            ca_count += int(bool(basic.ca))
        except x509.ExtensionNotFound:
            continue
    if ca_count < 1 or any(
        certificate.subject == certificate.issuer
        and not _signature_is_valid(certificate, certificate)
        for certificate in certificates
    ):
        return LangfuseTrustStatus(
            configured=True,
            valid=False,
            certificate_count=len(certificates),
            source="inline",
            reason="invalid_ca_bundle",
        )
    return LangfuseTrustStatus(
        configured=True,
        valid=True,
        certificate_count=len(certificates),
        source="inline",
    )


def _validate_platform_trust_store(path: Path) -> LangfuseTrustStatus:
    """Validate a multi-root platform store without requiring one linear chain."""
    try:
        payload = path.read_bytes()
        count = len(_PEM_CERT_RE.findall(payload))
        if (
            len(payload) > _MAX_BUNDLE_BYTES
            or count < _TRUST_STORE_CERTIFICATE_THRESHOLD
        ):
            raise ValueError("not_platform_store")
        ssl.create_default_context(cafile=str(path))
    except Exception:
        return LangfuseTrustStatus(
            configured=True,
            valid=False,
            source="environment_trust_store",
            reason="trust_store_invalid",
        )
    return LangfuseTrustStatus(
        configured=True,
        valid=True,
        certificate_count=count,
        source="environment_trust_store",
    )


def _resolve_langfuse_trust(
    *,
    environ: MutableMapping[str, str] | None = None,
    agent_config: AgentConfig | None = None,
    resolver: Callable[[str], str | None] | None = None,
    destination_root: Path | None = None,
) -> tuple[LangfuseTrustStatus, ResolvedTLSProfile | None]:
    """Resolve and validate the generic Langfuse TLS profile once.

    Named ``LANGFUSE_TLS_PROFILE``/``LANGFUSE_TLS_PROFILE_REF`` settings and
    the shared ``TLS_PROFILES_REF`` catalog are preferred. Direct runtime CA
    references and standard trust environment variables are also supported.
    """
    target_env = environ if environ is not None else os.environ
    resolution_env = dict(target_env)
    materialized_trust = _is_true(target_env.get(_MATERIALIZED_TRUST_FLAG, ""))
    if materialized_trust:
        # ``load_config()`` can project the parent's selector/ref fields back
        # into ``os.environ`` after process start. They are intentionally not
        # valid in the isolated child: only the concrete CA/proxy variables
        # materialized by the parent are. Remove selectors before resolving.
        for key in _PARENT_TRUST_SELECTORS:
            resolution_env.pop(key, None)
    relevant_keys = (
        "LANGFUSE_TLS_PROFILE",
        "LANGFUSE_TLS_PROFILE_REF",
        "TLS_PROFILE",
        "TLS_PROFILE_REF",
        "TLS_PROFILES_REF",
        "TLS_PROFILES",
        "LANGFUSE_CA_BUNDLE_REF",
        "LANGFUSE_CA_BUNDLE",
        "LANGFUSE_CLIENT_CERT_REF",
        "LANGFUSE_CLIENT_KEY_REF",
        "LANGFUSE_CLIENT_KEY_PASSWORD_REF",
        "LANGFUSE_PROXY_URL_REF",
        "LANGFUSE_PROXY_URL",
        "LANGFUSE_NO_PROXY",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    )
    for key in relevant_keys:
        # A parent process may already have resolved a named/ref-backed profile
        # into private runtime files for this child. In that case the concrete
        # child environment is authoritative: consulting the child's persisted
        # AgentConfig again can resurrect a profile name without its secret
        # catalog and incorrectly reject valid materialized trust.
        if materialized_trust and key in _PARENT_TRUST_SELECTORS:
            continue
        configured_value = (
            target_env.get(key, "")
            if materialized_trust
            else _value(target_env, key, agent_config)
        )
        value = _concrete_runtime_value(configured_value)
        if value:
            resolution_env[key] = value

    from agent_utilities.core.transport_security import (
        TransportSecurityError,
        resolve_tls_profile,
    )

    try:
        trust = resolve_tls_profile(
            "LANGFUSE",
            environ=resolution_env,
            resolver=resolver,
            destination_root=destination_root,
        )
    except TransportSecurityError:
        return (
            LangfuseTrustStatus(
                configured=True,
                valid=False,
                source="profile",
                reason="trust_profile_invalid",
            ),
            None,
        )
    if not trust.verify_enabled:
        return (
            LangfuseTrustStatus(
                configured=True,
                valid=False,
                source="profile",
                reason="insecure_transport_unsupported",
            ),
            None,
        )

    configured = bool(
        trust.configured
        or trust.ca_bundle_path is not None
        or trust.ca_directory is not None
        or trust.client_bundle_path is not None
        or trust.proxy_url
    )
    count = 0
    source = trust.source
    if _concrete_runtime_value(resolution_env.get("LANGFUSE_CA_BUNDLE_REF")):
        source = "secret_ref"
    reason: str | None = None
    if trust.ca_bundle_path is not None:
        explicit_profile = any(
            _concrete_runtime_value(resolution_env.get(key))
            for key in (
                "LANGFUSE_TLS_PROFILE",
                "LANGFUSE_TLS_PROFILE_REF",
                "TLS_PROFILE",
                "TLS_PROFILE_REF",
                "TLS_PROFILES_REF",
                "TLS_PROFILES",
            )
        )
        explicit_langfuse_bundle = any(
            _concrete_runtime_value(resolution_env.get(key))
            for key in ("LANGFUSE_CA_BUNDLE", "LANGFUSE_CA_BUNDLE_REF")
        )
        environment_store = bool(
            not explicit_profile
            and not explicit_langfuse_bundle
            and any(
                _concrete_runtime_value(resolution_env.get(key))
                for key in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE")
            )
        )
        checked = (
            _validate_platform_trust_store(trust.ca_bundle_path)
            if environment_store
            else validate_ca_bundle(trust.ca_bundle_path)
        )
        if environment_store and not checked.valid:
            checked = validate_ca_bundle(trust.ca_bundle_path)
        if not checked.valid:
            trust.cleanup()
            return (
                LangfuseTrustStatus(
                    configured=True,
                    valid=False,
                    certificate_count=checked.certificate_count,
                    source=source,
                    reason=checked.reason,
                ),
                None,
            )
        count = checked.certificate_count
        if checked.source == "environment_trust_store":
            source = checked.source

    return (
        LangfuseTrustStatus(
            configured=configured,
            valid=True,
            certificate_count=count,
            source=source if configured else "system",
            reason=reason,
        ),
        trust,
    )


def _project_langfuse_trust(
    trust: ResolvedTLSProfile, target_env: MutableMapping[str, str]
) -> None:
    """Project only concrete runtime trust material into one process boundary."""
    for key, value in trust.child_env(service="LANGFUSE").items():
        if key == "UV_NATIVE_TLS":
            continue
        target_env[key] = value


def configure_langfuse_trust(
    *,
    environ: MutableMapping[str, str] | None = None,
    agent_config: AgentConfig | None = None,
    resolver: Callable[[str], str | None] | None = None,
    destination_root: Path | None = None,
) -> LangfuseTrustStatus:
    """Validate and export Langfuse trust without returning sensitive paths."""
    target_env = environ if environ is not None else os.environ
    status, trust = _resolve_langfuse_trust(
        environ=target_env,
        agent_config=agent_config,
        resolver=resolver,
        destination_root=destination_root,
    )
    if status.valid and trust is not None:
        _project_langfuse_trust(trust, target_env)
    return status


def resolve_langfuse_requests_transport(
    *,
    environ: MutableMapping[str, str] | None = None,
    agent_config: AgentConfig | None = None,
    resolver: Callable[[str], str | None] | None = None,
    destination_root: Path | None = None,
) -> dict[str, Any]:
    """Return verified Requests kwargs for CA, proxy, and mTLS policy.

    Secret references are resolved exactly once. Returned values are concrete
    process-lifetime adapters and must never be persisted or logged.
    """
    target_env = environ if environ is not None else os.environ
    status, trust = _resolve_langfuse_trust(
        environ=target_env,
        agent_config=agent_config,
        resolver=resolver,
        destination_root=destination_root,
    )
    if not status.valid or trust is None:
        raise LangfuseTrustError("langfuse_requests_transport_invalid")
    try:
        request_kwargs = trust.requests_kwargs()
    except Exception:
        trust.cleanup()
        raise LangfuseTrustError("langfuse_requests_transport_invalid") from None
    _project_langfuse_trust(trust, target_env)
    return request_kwargs


def _is_true(value: str) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def langfuse_parent_kg_ingestion_enabled(config: dict[str, Any]) -> bool:
    """Return the process-local GraphOS mediation decision for one child.

    Langfuse children are untrusted data-source adapters, not graph principals.
    They therefore never receive a graph token or write directly to the engine.
    ``prepare_langfuse_mcp_config`` records the operator's opt-in on the parent
    configuration and forces the child-side flag off. The GraphOS multiplexer
    consumes this marker while its caller-minted ``GraphSession`` is active.
    """

    return (
        config.get(_PARENT_KG_INGESTION_FLAG) is True
        and str((config.get("env") or {}).get("LANGFUSE_KG_AUTO_INGEST", ""))
        .strip()
        .casefold()
        == "false"
    )


def prepare_langfuse_mcp_config(
    config: dict[str, Any],
    *,
    environ: MutableMapping[str, str] | None = None,
    agent_config: AgentConfig | None = None,
    resolver: Callable[[str], str | None] | None = None,
    destination_root: Path | None = None,
) -> dict[str, Any]:
    """Project canonical Graph-OS settings into a Langfuse MCP child config."""
    target_env = environ if environ is not None else os.environ
    prepared = dict(config)
    if _PARENT_KG_INGESTION_FLAG in prepared:
        raise LangfuseTrustError("langfuse_configuration_invalid")
    child_env = {
        str(key): str(value) for key, value in (prepared.get("env") or {}).items()
    }
    if any(
        _concrete_runtime_value(child_env.get(key))
        for key in ("LANGFUSE_BASE_URL", "LANGFUSE_URL")
    ):
        raise LangfuseTrustError("langfuse_host_invalid")
    if any(
        _concrete_runtime_value(child_env.get(key))
        for key in ("LANGFUSE_CLIENT_CERT", "LANGFUSE_CLIENT_KEY")
    ):
        raise LangfuseTrustError("langfuse_requests_transport_invalid")
    if any(
        _concrete_runtime_value(child_env.get(key))
        for key in (
            "LANGFUSE_PERSISTENCE_HMAC_KEY",
            "LANGFUSE_PERSISTENCE_HMAC_MATERIALIZED",
        )
    ):
        raise LangfuseTrustError("langfuse_persistence_hmac_key_invalid")
    host = next(
        (
            value
            for value in (
                _concrete_runtime_value(child_env.get("LANGFUSE_HOST")),
                _concrete_runtime_value(
                    _value(target_env, "LANGFUSE_HOST", agent_config)
                ),
            )
            if value
        ),
        "",
    )
    try:
        host = resolve_langfuse_host(environ={"LANGFUSE_HOST": host})
    except ValueError:
        raise LangfuseTrustError("langfuse_host_invalid") from None
    if not host:
        raise LangfuseTrustError("langfuse_host_invalid")
    child_env["LANGFUSE_HOST"] = host
    policies: dict[str, bool] = {}
    for policy_name in ("LANGFUSE_CAPTURE_CONTENT", "LANGFUSE_KG_AUTO_INGEST"):
        policy_value = (
            _value(target_env, policy_name, agent_config) or "false"
        ).casefold()
        if policy_value in {"1", "true", "yes", "on"}:
            policies[policy_name] = True
        elif policy_value in {"0", "false", "no", "off"}:
            policies[policy_name] = False
        else:
            raise LangfuseTrustError("langfuse_configuration_invalid")
    child_env["LANGFUSE_CAPTURE_CONTENT"] = (
        "true" if policies["LANGFUSE_CAPTURE_CONTENT"] else "false"
    )
    # The child is an API adapter and has no graph authority. Keep graph writes
    # in the authenticated GraphOS parent, where the request's least-privilege
    # GraphSession is already active. No bearer, claims, or engine credential is
    # copied into the child environment.
    prepared[_PARENT_KG_INGESTION_FLAG] = policies["LANGFUSE_KG_AUTO_INGEST"]
    child_env["LANGFUSE_KG_AUTO_INGEST"] = "false"
    credential_env = dict(target_env)
    for key in (
        "LANGFUSE_PUBLIC_KEY_REF",
        "LANGFUSE_SECRET_KEY_REF",
    ):
        value = _concrete_runtime_value(child_env.get(key))
        if value:
            credential_env[key] = value
    public_key, secret_key = resolve_langfuse_credentials(
        environ=credential_env,
        agent_config=agent_config,
        resolver=resolver,
    )
    child_env["LANGFUSE_PUBLIC_KEY"] = public_key
    child_env["LANGFUSE_SECRET_KEY"] = secret_key
    child_env.pop("LANGFUSE_PUBLIC_KEY_REF", None)
    child_env.pop("LANGFUSE_SECRET_KEY_REF", None)
    persistence_env = dict(target_env)
    persistence_ref = _concrete_runtime_value(
        child_env.get("LANGFUSE_PERSISTENCE_HMAC_KEY_REF")
    )
    if persistence_ref:
        persistence_env["LANGFUSE_PERSISTENCE_HMAC_KEY_REF"] = persistence_ref
    persistence_hmac_key = resolve_langfuse_persistence_hmac_key(
        environ=persistence_env,
        agent_config=agent_config,
        resolver=resolver,
    )
    child_env.pop("LANGFUSE_PERSISTENCE_HMAC_KEY_REF", None)
    if persistence_hmac_key is not None:
        child_env["LANGFUSE_PERSISTENCE_HMAC_KEY"] = persistence_hmac_key
        child_env["LANGFUSE_PERSISTENCE_HMAC_MATERIALIZED"] = "true"

    trust_env = dict(target_env)
    for key in (
        "LANGFUSE_TLS_PROFILE",
        "LANGFUSE_TLS_PROFILE_REF",
        "TLS_PROFILE",
        "TLS_PROFILE_REF",
        "TLS_PROFILES_REF",
        "TLS_PROFILES",
        "LANGFUSE_CA_BUNDLE_REF",
        "LANGFUSE_CA_BUNDLE",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "LANGFUSE_CLIENT_CERT_REF",
        "LANGFUSE_CLIENT_KEY_REF",
        "LANGFUSE_CLIENT_KEY_PASSWORD_REF",
        "LANGFUSE_PROXY_URL_REF",
        "LANGFUSE_PROXY_URL",
        "LANGFUSE_NO_PROXY",
    ):
        value = _concrete_runtime_value(child_env.get(key)) or _concrete_runtime_value(
            _value(target_env, key, agent_config)
        )
        if value:
            trust_env[key] = value
    if (
        child_env.get("SSL_CERT_FILE")
        and not child_env.get("LANGFUSE_CA_BUNDLE_REF")
        and not child_env.get("LANGFUSE_CA_BUNDLE")
        and not child_env.get("REQUESTS_CA_BUNDLE")
    ):
        trust_env["LANGFUSE_CA_BUNDLE"] = child_env["SSL_CERT_FILE"]
    status = configure_langfuse_trust(
        environ=trust_env,
        agent_config=agent_config,
        resolver=resolver,
        destination_root=destination_root,
    )
    if not status.valid:
        raise LangfuseTrustError("langfuse_ca_bundle_invalid")
    for key in (
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "NO_PROXY",
        "LANGFUSE_CLIENT_CERT",
        "LANGFUSE_CLIENT_KEY",
    ):
        value = trust_env.get(key)
        if value:
            child_env[key] = value
    child_env[_MATERIALIZED_TRUST_FLAG] = "true"
    child_env.pop("LANGFUSE_CA_BUNDLE_REF", None)
    child_env.pop("LANGFUSE_CA_BUNDLE", None)
    for key in (
        "LANGFUSE_TLS_PROFILE",
        "LANGFUSE_TLS_PROFILE_REF",
        "TLS_PROFILE",
        "TLS_PROFILE_REF",
        "TLS_PROFILES_REF",
        "TLS_PROFILES",
        "LANGFUSE_CLIENT_CERT_REF",
        "LANGFUSE_CLIENT_KEY_REF",
        "LANGFUSE_CLIENT_KEY_PASSWORD_REF",
        "LANGFUSE_PROXY_URL_REF",
    ):
        child_env.pop(key, None)
    child_env.pop("UV_NATIVE_TLS", None)
    prepared["env"] = child_env
    return prepared


def native_langfuse_mcp_config(
    *,
    environ: MutableMapping[str, str] | None = None,
    agent_config: AgentConfig | None = None,
) -> dict[str, Any] | None:
    """Build a lazy MCP entry when credentials are present (explicitly opt-out)."""
    target_env = environ if environ is not None else os.environ
    credentials_ready = langfuse_credentials_configured(
        environ=target_env,
        agent_config=agent_config,
    )
    if not credentials_ready:
        return None
    enabled = _value(target_env, "LANGFUSE_MCP_ENABLED", agent_config)
    if enabled and not _is_true(enabled):
        return None
    if not langfuse_provider_contract_ready():
        return None
    # The serving extra installs the provider into this same environment.  Launch
    # that exact artifact so a self-contained GraphOS never reaches the network,
    # resolves a different provider release, or depends on shell PATH ordering.
    config: dict[str, Any] = {
        "command": sys.executable,
        "args": ["-m", "langfuse_agent.mcp_server"],
    }
    config.update(
        {
            "prefix": "langfuse",
            # Covers dependency import + MCP initialization on constrained
            # hosts. Operators can still override this in an explicit child
            # catalog entry; the multiplexer enforces the global 3600 s cap.
            "timeout": 120,
            "max_concurrency": 4,
        }
    )
    return prepare_langfuse_mcp_config(
        config,
        environ=target_env,
        agent_config=agent_config,
    )


def is_langfuse_server(name: str, config: dict[str, Any]) -> bool:
    """Recognize explicit Langfuse child entries without inspecting secrets."""
    if name.casefold() in _LANGFUSE_SERVER_NAMES:
        return True
    command = str(config.get("command") or "").casefold()
    args = " ".join(str(item) for item in config.get("args") or []).casefold()
    return (
        "langfuse-mcp" in command
        or "langfuse-mcp" in args
        or "langfuse_agent.mcp_server" in args
    )
