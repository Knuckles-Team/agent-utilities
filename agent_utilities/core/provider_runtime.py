"""Reference-only runtime profiles for external provider integrations.

Provider-specific packages consume this boundary instead of defining durable
endpoint, credential, or certificate fields. The configuration model contains
only runtime references and named trust selectors; resolved values live only
for the lifetime of :class:`ResolvedProviderRuntime`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from agent_utilities.core.config import AgentConfig, ProviderRuntimeProfile
    from agent_utilities.core.transport_security import ResolvedTLSProfile

__all__ = [
    "ProviderRuntimeError",
    "PreparedProviderChildRuntime",
    "ResolvedProviderRuntime",
    "get_provider_runtime_profile",
    "prepare_provider_runtime_child_environment",
    "prepare_resolved_provider_runtime_child_environment",
    "resolve_provider_runtime_profile",
    "resolve_selected_provider_runtime_profile",
]

_PROFILE_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{1,62}$")
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_MAX_ENDPOINT_BYTES = 8_192
_MAX_SELECTOR_BYTES = 262_144
_MAX_CHILD_VALUE_BYTES = 65_536
_MAX_CHILD_ENVIRONMENT_BYTES = 512 * 1_024
_CHILD_PROFILE_KEY = "AGENT_PROVIDER_PROFILE"
_CHILD_CONFIG_KEY = "PROVIDER_CONFIGS"
_CHILD_ENDPOINT_KEY = "AGENT_PROVIDER_RUNTIME_ENDPOINT"
_CHILD_TLS_KEY = "AGENT_PROVIDER_RUNTIME_TLS_PROFILE"


class ProviderRuntimeError(RuntimeError):
    """Stable provider-profile failure without deployment material."""


def _profile_name(value: object) -> str:
    rendered = str(value or "").strip()
    if _PROFILE_NAME_RE.fullmatch(rendered) is None:
        raise ProviderRuntimeError("provider_profile_invalid")
    return rendered


def _secure_endpoint(value: str) -> str:
    """Return a canonical credential-free HTTPS or loopback HTTP endpoint."""

    rendered = str(value or "").strip().rstrip("/")
    if (
        not rendered
        or len(rendered.encode("utf-8")) > _MAX_ENDPOINT_BYTES
        or any(character.isspace() or ord(character) < 32 for character in rendered)
    ):
        raise ProviderRuntimeError("provider_endpoint_invalid")
    try:
        parsed = urlsplit(rendered)
        port = parsed.port
    except ValueError:
        raise ProviderRuntimeError("provider_endpoint_invalid") from None
    host = (parsed.hostname or "").casefold()
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.netloc
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65_535)
        or (parsed.scheme.casefold() == "http" and host not in _LOOPBACK_HOSTS)
    ):
        raise ProviderRuntimeError("provider_endpoint_invalid")
    return rendered


def get_provider_runtime_profile(
    profile_name: str,
    *,
    config: AgentConfig | None = None,
    require_enabled: bool = True,
) -> ProviderRuntimeProfile:
    """Return one validated reference-only profile without resolving material."""

    name = _profile_name(profile_name)
    if config is None:
        from agent_utilities.core.config import AgentConfig

        config = AgentConfig()
    profiles = getattr(config, "provider_configs", None)
    if not isinstance(profiles, dict):
        raise ProviderRuntimeError("provider_profiles_invalid")
    profile = profiles.get(name)
    if profile is None:
        raise ProviderRuntimeError("provider_profile_unavailable")
    if require_enabled and not profile.enabled:
        raise ProviderRuntimeError("provider_profile_disabled")
    return profile


@dataclass(slots=True, repr=False)
class ResolvedProviderRuntime:
    """Ephemeral provider values with deterministic TLS-material cleanup."""

    endpoint: str | None = field(default=None, repr=False)
    credentials: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({}), repr=False
    )
    selectors: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({}), repr=False
    )
    tls: ResolvedTLSProfile | None = field(default=None, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __repr__(self) -> str:
        return "<ResolvedProviderRuntime redacted>"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        trust = self.tls
        self.endpoint = None
        self.credentials = MappingProxyType({})
        self.selectors = MappingProxyType({})
        self.tls = None
        if trust is not None:
            trust.cleanup()

    def __enter__(self) -> ResolvedProviderRuntime:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@dataclass(slots=True, repr=False)
class PreparedProviderChildRuntime:
    """One isolated child projection that owns its ephemeral TLS material."""

    environment: Mapping[str, str] = field(repr=False)
    runtime: ResolvedProviderRuntime = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __repr__(self) -> str:
        return "<PreparedProviderChildRuntime redacted>"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.environment = MappingProxyType({})
        self.runtime.close()


def resolve_provider_runtime_profile(
    profile_name: str,
    *,
    config: AgentConfig | None = None,
) -> ResolvedProviderRuntime:
    """Resolve one enabled provider profile at its exact runtime boundary."""

    if config is None:
        from agent_utilities.core.config import AgentConfig

        config = AgentConfig()
    profile = get_provider_runtime_profile(profile_name, config=config)
    from agent_utilities.security.cli_secrets import (
        resolve_runtime_secret_reference,
    )

    try:
        endpoint = (
            _secure_endpoint(resolve_runtime_secret_reference(profile.endpoint_ref))
            if profile.endpoint_ref
            else None
        )
        credentials = {
            alias: resolve_runtime_secret_reference(reference)
            for alias, reference in profile.credential_refs.items()
        }
        selectors: dict[str, str] = {}
        for alias, reference in profile.selector_refs.items():
            value = resolve_runtime_secret_reference(reference)
            if (
                len(value.encode("utf-8")) > _MAX_SELECTOR_BYTES
                or "\x00" in value
                or "\r" in value
                or "\n" in value
            ):
                raise ProviderRuntimeError("provider_selector_invalid")
            selectors[alias] = value
    except ProviderRuntimeError:
        raise
    except Exception:
        raise ProviderRuntimeError("provider_runtime_reference_unavailable") from None

    tls = None
    if profile.tls_profile or profile.tls_profile_ref:
        try:
            from agent_utilities.core.transport_security import (
                resolve_configured_tls_profile,
            )

            tls = resolve_configured_tls_profile(
                _profile_name(profile_name),
                profile_name=profile.tls_profile,
                profile_ref=profile.tls_profile_ref,
                config=config,
            )
        except Exception:
            raise ProviderRuntimeError("provider_tls_profile_invalid") from None
    return ResolvedProviderRuntime(
        endpoint=endpoint,
        credentials=MappingProxyType(credentials),
        selectors=MappingProxyType(selectors),
        tls=tls,
    )


def prepare_provider_runtime_child_environment(
    profile_name: str,
    *,
    config: AgentConfig | None = None,
) -> PreparedProviderChildRuntime:
    """Preflight and project one selected profile into an isolated child.

    Store references are resolved by the trusted parent and rewritten to
    isolated ephemeral ``env://`` aliases. The child never receives ambient
    Vault/engine authority, original secret references, or the parent's config
    root.
    """

    name = _profile_name(profile_name)
    if config is None:
        from agent_utilities.core.config import AgentConfig

        config = AgentConfig()
    runtime = resolve_provider_runtime_profile(name, config=config)
    return prepare_resolved_provider_runtime_child_environment(name, runtime)


def prepare_resolved_provider_runtime_child_environment(
    profile_name: str,
    runtime: ResolvedProviderRuntime,
) -> PreparedProviderChildRuntime:
    """Project an already-resolved profile into one isolated child.

    Runtime child policies use this boundary when they must validate additional
    provider-specific invariants before launch. Reusing the resolved object
    avoids a second secret-store/TLS resolution and guarantees that validation
    and child materialization consume the same ephemeral values.
    """

    name = _profile_name(profile_name)
    if runtime._closed:
        raise ProviderRuntimeError("provider_runtime_closed")
    projected: dict[str, str] = {}
    projected_names: set[str] = set()
    total_bytes = 0

    def project(key: str, value: str) -> None:
        nonlocal total_bytes
        normalized = key.upper()
        encoded_value = value.encode("utf-8")
        if (
            normalized in projected_names
            or len(encoded_value) > _MAX_CHILD_VALUE_BYTES
            or "\x00" in value
        ):
            raise ProviderRuntimeError("provider_child_environment_invalid")
        total_bytes += len(key.encode("utf-8")) + len(encoded_value)
        if total_bytes > _MAX_CHILD_ENVIRONMENT_BYTES:
            raise ProviderRuntimeError("provider_child_environment_too_large")
        projected_names.add(normalized)
        projected[key] = value

    try:
        project(_CHILD_PROFILE_KEY, name)
        rewritten_credentials: dict[str, str] = {}
        rewritten_selectors: dict[str, str] = {}
        endpoint_ref = None
        if runtime.endpoint is not None:
            project(_CHILD_ENDPOINT_KEY, runtime.endpoint)
            endpoint_ref = f"env://{_CHILD_ENDPOINT_KEY}"
        for index, (alias, value) in enumerate(runtime.credentials.items()):
            child_key = f"AGENT_PROVIDER_RUNTIME_CREDENTIAL_{index:02d}"
            project(child_key, value)
            rewritten_credentials[alias] = f"env://{child_key}"
        for index, (alias, value) in enumerate(runtime.selectors.items()):
            child_key = f"AGENT_PROVIDER_RUNTIME_SELECTOR_{index:02d}"
            project(child_key, value)
            rewritten_selectors[alias] = f"env://{child_key}"

        tls_profile_ref = None
        if runtime.tls is not None:
            trust = runtime.tls
            tls_payload: dict[str, object] = {
                "system_trust": trust.system_trust,
                "trust_env": trust.trust_env,
            }
            for key, material_value in (
                ("ca_bundle_path", trust.ca_bundle_path),
                ("ca_directory", trust.ca_directory),
                ("client_cert_path", trust.client_cert_path),
                ("client_key_path", trust.client_key_path),
                ("client_key_password", trust.client_key_password),
                ("proxy_url", trust.proxy_url),
                ("no_proxy", trust.no_proxy),
            ):
                if material_value is not None:
                    tls_payload[key] = str(material_value)
            project(
                _CHILD_TLS_KEY,
                json.dumps(tls_payload, sort_keys=True, separators=(",", ":")),
            )
            tls_profile_ref = f"env://{_CHILD_TLS_KEY}"

        child_profile = {
            "enabled": True,
            "endpoint_ref": endpoint_ref,
            "credential_refs": rewritten_credentials,
            "selector_refs": rewritten_selectors,
            "tls_profile_ref": tls_profile_ref,
        }
        project(
            _CHILD_CONFIG_KEY,
            json.dumps(
                {name: child_profile},
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    except Exception:
        runtime.close()
        raise
    return PreparedProviderChildRuntime(
        environment=MappingProxyType(projected),
        runtime=runtime,
    )


def resolve_selected_provider_runtime_profile(
    *, config: AgentConfig | None = None
) -> ResolvedProviderRuntime:
    """Resolve the profile explicitly selected by the GraphOS child launcher."""

    from agent_utilities.core.config import setting

    selected = setting("AGENT_PROVIDER_PROFILE")
    if selected in (None, ""):
        raise ProviderRuntimeError("provider_profile_not_selected")
    return resolve_provider_runtime_profile(str(selected), config=config)
