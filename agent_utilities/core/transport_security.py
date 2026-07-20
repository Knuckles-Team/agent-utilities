"""Runtime-only outbound transport security profiles.

The same named profile can be consumed by HTTPX, Requests, GraphQL, Langfuse,
and database drivers without copying machine paths or certificate material into
durable application configuration.  Profiles may use the platform trust store,
a CA file/directory, inline PEM supplied at runtime, or secret references.  mTLS
and proxy settings are optional; certificate/key material resolved from secrets
is written only to a private runtime directory and removed at process exit.

Certificate and hostname verification are mandatory. Trust anchors, mTLS
material, and proxies remain runtime-configurable; boolean verification
downgrades are not part of the current contract.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import ssl
import tempfile
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agent_utilities.core.paths import runtime_dir

__all__ = [
    "ResolvedTLSProfile",
    "TransportSecurityError",
    "resolve_configured_tls_profile",
    "resolve_tls_profile",
    "tls_environment_from_config",
]

_MAX_PROFILE_BYTES = 256_000
_MAX_PEM_BYTES = 4_000_000
_PEM_CERTIFICATE = "-----BEGIN CERTIFICATE-----"
_PEM_PRIVATE_KEY = re.compile(r"-----BEGIN (?:ENCRYPTED |RSA |EC )?PRIVATE KEY-----")
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_MATERIALIZED_PATHS: set[Path] = set()


def tls_environment_from_config(
    config: Any,
    *,
    base_environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Render only runtime TLS selectors from ``AgentConfig`` for resolution.

    This closes the gap between process environment and XDG-backed AgentConfig:
    callers can use one resolver without copying endpoints, certificate bodies,
    secret values, or local paths into durable configuration or diagnostics.
    """

    # Preserve standard/runtime CA variables so uvx, Requests, HTTPX, and the
    # application can share one complete-chain PEM bundle.  Ambient proxy and
    # unrelated environment variables are deliberately not copied into this
    # trust-only view.
    inherited = os.environ if base_environ is None else base_environ
    inherited_keys = {
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "TLS_PROFILES",
    }
    for prefix in (
        "MODEL",
        "EMBEDDING",
        "OAUTH2_TOKEN",
        "OTEL",
        "POSTGRES",
        "QDRANT",
        "MONGODB",
        "REDIS",
    ):
        inherited_keys.update(
            {
                f"{prefix}_CA_BUNDLE",
                f"{prefix}_CA_BUNDLE_REF",
                f"{prefix}_CA_DIRECTORY",
                f"{prefix}_CLIENT_CERT",
                f"{prefix}_CLIENT_CERT_REF",
                f"{prefix}_CLIENT_KEY",
                f"{prefix}_CLIENT_KEY_REF",
                f"{prefix}_CLIENT_KEY_PASSWORD_REF",
            }
        )
    result: dict[str, str] = {
        key: str(inherited[key])
        for key in inherited_keys
        if inherited.get(key) not in (None, "")
    }
    scalar_fields = {
        "TLS_PROFILE": "tls_profile",
        "TLS_PROFILE_REF": "tls_profile_ref",
        "TLS_PROFILES_REF": "tls_profiles_ref",
        "TLS_CA_BUNDLE_REF": "tls_ca_bundle_ref",
        "TLS_CLIENT_CERT_REF": "tls_client_cert_ref",
        "TLS_CLIENT_KEY_REF": "tls_client_key_ref",
        "TLS_CLIENT_KEY_PASSWORD_REF": "tls_client_key_password_ref",
        "TLS_PROXY_URL_REF": "tls_proxy_url_ref",
        "MODEL_TLS_PROFILE": "model_tls_profile",
        "MODEL_TLS_PROFILE_REF": "model_tls_profile_ref",
        "EMBEDDING_TLS_PROFILE": "embedding_tls_profile",
        "EMBEDDING_TLS_PROFILE_REF": "embedding_tls_profile_ref",
        "OAUTH2_TOKEN_TLS_PROFILE": "oauth2_token_tls_profile",
        "OAUTH2_TOKEN_TLS_PROFILE_REF": "oauth2_token_tls_profile_ref",
        "OTEL_TLS_PROFILE": "otel_tls_profile",
        "OTEL_TLS_PROFILE_REF": "otel_tls_profile_ref",
        "POSTGRES_TLS_PROFILE": "postgres_tls_profile",
        "POSTGRES_TLS_PROFILE_REF": "postgres_tls_profile_ref",
        "QDRANT_TLS_PROFILE": "qdrant_tls_profile",
        "QDRANT_TLS_PROFILE_REF": "qdrant_tls_profile_ref",
        "MONGODB_TLS_PROFILE": "mongodb_tls_profile",
        "MONGODB_TLS_PROFILE_REF": "mongodb_tls_profile_ref",
        "REDIS_TLS_PROFILE": "redis_tls_profile",
        "REDIS_TLS_PROFILE_REF": "redis_tls_profile_ref",
        "CERT_PROMETHEUS_TLS_PROFILE": "cert_prometheus_tls_profile",
        "CERT_PROMETHEUS_TLS_PROFILE_REF": "cert_prometheus_tls_profile_ref",
    }
    for environment_key, field_name in scalar_fields.items():
        value = getattr(config, field_name, None)
        if value not in (None, ""):
            result[environment_key] = str(value)

    # A rendered trust view is an isolated runtime boundary. Copy only the
    # environment values explicitly named by configured env:// references so
    # resolution never falls through to unrelated ambient process state.
    for reference in tuple(result.values()):
        rendered = str(reference or "").strip()
        if not rendered.startswith("env://"):
            continue
        target = rendered.removeprefix("env://")
        if _ENV_NAME_RE.fullmatch(target) and inherited.get(target) not in (None, ""):
            result[target] = str(inherited[target])

    boolean_fields = {
        "TLS_SYSTEM_TRUST": ("tls_system_trust", True),
        "TLS_TRUST_ENV": ("tls_trust_env", True),
    }
    for environment_key, (field_name, default) in boolean_fields.items():
        value = bool(getattr(config, field_name, default))
        result[environment_key] = "true" if value else "false"
    return result


def resolve_configured_tls_profile(
    service: str,
    *,
    profile_name: str | None = None,
    profile_ref: str | None = None,
    profile: Mapping[str, Any] | None = None,
    config: Any | None = None,
    resolver: Callable[[str], str | None] | None = None,
    destination_root: Path | None = None,
) -> ResolvedTLSProfile:
    """Resolve one service's TLS policy through the active ``AgentConfig``.

    Outbound clients use this entrypoint instead of reading transport variables
    or reducing TLS policy to a boolean.  The resulting profile retains the
    mandatory peer/hostname verification invariant while applying runtime-only
    trust anchors, mTLS material, and proxy policy.
    """

    if config is None:
        from agent_utilities.core.config import config as active_config

        config = active_config
    runtime_environ = tls_environment_from_config(config)

    # Explicit call-site profiles (for example an OAuth token endpoint's
    # profile_ref) are not fields on AgentConfig. Project only the env://
    # targets they name into the same isolated runtime view.
    def project_references(value: Any) -> None:
        if isinstance(value, Mapping):
            for item in value.values():
                project_references(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                project_references(item)
            return
        rendered = str(value or "").strip()
        if not rendered.startswith("env://"):
            return
        target = rendered.removeprefix("env://")
        if _ENV_NAME_RE.fullmatch(target) and os.environ.get(target) not in (None, ""):
            runtime_environ[target] = str(os.environ[target])

    project_references(profile_ref)
    project_references(profile)
    return resolve_tls_profile(
        service,
        profile_name=profile_name,
        profile_ref=profile_ref,
        profile=profile,
        environ=runtime_environ,
        resolver=resolver,
        destination_root=destination_root,
    )


class TransportSecurityError(RuntimeError):
    """Stable failure that deliberately omits paths, hosts, and secret refs."""


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    rendered = str(value).strip().casefold()
    if rendered in {"1", "true", "yes", "on"}:
        return True
    if rendered in {"0", "false", "no", "off"}:
        return False
    raise TransportSecurityError("tls_profile_boolean_invalid")


def _service_prefix(service: str) -> str:
    normalized = "".join(
        character if character.isalnum() else "_"
        for character in str(service or "").strip().upper()
    ).strip("_")
    if not normalized:
        raise TransportSecurityError("tls_service_invalid")
    return normalized


def _first(environ: Mapping[str, str], *keys: str) -> str:
    for key in keys:
        value = environ.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _resolve_secret(
    ref: str,
    resolver: Callable[[str], str | None] | None,
    environ: Mapping[str, str],
) -> str:
    reference = str(ref or "").strip()
    if not reference:
        raise TransportSecurityError("tls_secret_ref_invalid")
    if reference.startswith("env://"):
        target = reference.removeprefix("env://")
        if _ENV_NAME_RE.fullmatch(target) is None:
            raise TransportSecurityError("tls_secret_ref_invalid")
        value = environ.get(target)
    elif resolver is None:
        # Backend-backed references use the fleet's canonical runtime resolver;
        # env:// was already resolved exclusively from this runtime boundary.
        from agent_utilities.security.cli_secrets import (
            resolve_runtime_secret_reference,
        )

        try:
            value = resolve_runtime_secret_reference(reference)
        except Exception:
            value = None
    else:
        try:
            value = resolver(reference)
        except Exception:
            value = None
    rendered = str(value or "")
    if not rendered:
        raise TransportSecurityError("tls_secret_unavailable")
    return rendered


def _parse_profile_json(raw: str) -> dict[str, Any]:
    if len(raw.encode("utf-8")) > _MAX_PROFILE_BYTES:
        raise TransportSecurityError("tls_profile_too_large")
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        raise TransportSecurityError("tls_profile_invalid") from None
    if not isinstance(value, dict):
        raise TransportSecurityError("tls_profile_invalid")
    return value


def _select_named_profile(catalog: dict[str, Any], name: str) -> dict[str, Any]:
    profiles = catalog.get("profiles", catalog)
    if not isinstance(profiles, Mapping):
        raise TransportSecurityError("tls_profile_catalog_invalid")
    selected = profiles.get(name)
    if not isinstance(selected, Mapping):
        raise TransportSecurityError("tls_profile_not_found")
    return dict(selected)


def _profile_from_environment(
    prefix: str, environ: Mapping[str, str]
) -> dict[str, Any]:
    def value(suffix: str, global_name: str = "") -> str:
        keys = [f"{prefix}_{suffix}"]
        if global_name:
            keys.append(global_name)
        return _first(environ, *keys)

    profile: dict[str, Any] = {
        "system_trust": value("TLS_SYSTEM_TRUST", "TLS_SYSTEM_TRUST") or True,
        "trust_env": value("TLS_TRUST_ENV", "TLS_TRUST_ENV") or True,
    }
    aliases = {
        "ca_bundle_ref": ("CA_BUNDLE_REF", "TLS_CA_BUNDLE_REF"),
        "ca_bundle_path": ("CA_BUNDLE", "TLS_CA_BUNDLE"),
        "ca_directory": ("CA_DIRECTORY", "TLS_CA_DIRECTORY"),
        "client_cert_ref": ("CLIENT_CERT_REF", "TLS_CLIENT_CERT_REF"),
        "client_cert_path": ("CLIENT_CERT", "TLS_CLIENT_CERT"),
        "client_key_ref": ("CLIENT_KEY_REF", "TLS_CLIENT_KEY_REF"),
        "client_key_path": ("CLIENT_KEY", "TLS_CLIENT_KEY"),
        "client_key_password_ref": (
            "CLIENT_KEY_PASSWORD_REF",
            "TLS_CLIENT_KEY_PASSWORD_REF",
        ),
        "proxy_url_ref": ("PROXY_URL_REF", "TLS_PROXY_URL_REF"),
        "proxy_url": ("PROXY_URL", "TLS_PROXY_URL"),
        "no_proxy": ("NO_PROXY", "NO_PROXY"),
    }
    for key, (suffix, global_name) in aliases.items():
        resolved = value(suffix, global_name)
        if resolved:
            profile[key] = resolved

    # Interoperate with the standard variables used by HTTPX and Requests when
    # no service profile was selected.  The values remain runtime-only.
    if not any(
        profile.get(key) for key in ("ca_bundle_ref", "ca_bundle_path", "ca_directory")
    ):
        standard_file = _first(environ, "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")
        standard_dir = _first(environ, "SSL_CERT_DIR")
        if standard_file:
            profile["ca_bundle_path"] = standard_file
        elif standard_dir:
            profile["ca_directory"] = standard_dir
    if not profile.get("proxy_url"):
        proxy = _first(
            environ,
            "HTTPS_PROXY",
            "https_proxy",
            "ALL_PROXY",
            "all_proxy",
        )
        if proxy:
            profile["proxy_url"] = proxy
    return profile


def _pick(profile: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = profile.get(name)
        if value not in (None, ""):
            return value
    return None


def _path(value: Any, *, directory: bool = False) -> Path:
    rendered = os.path.expandvars(str(value or "").strip())
    if not rendered:
        raise TransportSecurityError("tls_material_path_invalid")
    candidate = Path(rendered).expanduser()
    try:
        valid = candidate.is_dir() if directory else candidate.is_file()
        if not valid:
            raise TransportSecurityError("tls_material_unavailable")
        if not directory and candidate.stat().st_size > _MAX_PEM_BYTES:
            raise TransportSecurityError("tls_material_too_large")
    except TransportSecurityError:
        raise
    except Exception:
        raise TransportSecurityError("tls_material_unavailable") from None
    return candidate


def _private_runtime_root(
    environ: Mapping[str, str], destination_root: Path | None
) -> Path:
    if destination_root is not None:
        root = Path(destination_root)
    else:
        xdg_runtime = str(environ.get("XDG_RUNTIME_DIR") or "").strip()
        root = Path(xdg_runtime) / "agent-utilities" if xdg_runtime else runtime_dir()
    target = root / "transport-security"
    target.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        target.chmod(0o700)
    except OSError:
        pass
    return target


def _cleanup_materialized() -> None:
    for path in tuple(_MATERIALIZED_PATHS):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue
        _MATERIALIZED_PATHS.discard(path)


atexit.register(_cleanup_materialized)


def _materialize(
    payload: str,
    *,
    kind: str,
    environ: Mapping[str, str],
    destination_root: Path | None,
) -> Path:
    encoded = payload.encode("utf-8")
    if not encoded or len(encoded) > _MAX_PEM_BYTES:
        raise TransportSecurityError("tls_material_invalid")
    marker_valid = (
        _PEM_CERTIFICATE in payload
        if kind in {"ca", "cert"}
        else bool(_PEM_PRIVATE_KEY.search(payload))
    )
    if not marker_valid:
        raise TransportSecurityError("tls_material_invalid")
    root = _private_runtime_root(environ, destination_root)
    descriptor: int | None = None
    path: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=f"tls-{kind}-", suffix=".pem", dir=root
        )
        path = Path(raw_path)
        handle = os.fdopen(descriptor, "wb")
        descriptor = None
        with handle:
            handle.write(encoded)
        path.chmod(0o600)
    except Exception:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                _MATERIALIZED_PATHS.add(path)
        raise TransportSecurityError("tls_materialization_failed") from None
    assert path is not None
    _MATERIALIZED_PATHS.add(path)
    return path


def _material(
    profile: Mapping[str, Any],
    *,
    kind: str,
    resolver: Callable[[str], str | None] | None,
    environ: Mapping[str, str],
    destination_root: Path | None,
) -> tuple[Path | None, bool]:
    ref = _pick(profile, f"{kind}_ref")
    pem = _pick(profile, f"{kind}_pem")
    path_value = _pick(profile, f"{kind}_path")
    populated = sum(value not in (None, "") for value in (ref, pem, path_value))
    if populated > 1:
        raise TransportSecurityError("tls_material_source_ambiguous")
    material_kind = {
        "ca_bundle": "ca",
        "client_cert": "cert",
        "client_key": "key",
    }[kind]
    if ref:
        return (
            _materialize(
                _resolve_secret(str(ref), resolver, environ),
                kind=material_kind,
                environ=environ,
                destination_root=destination_root,
            ),
            True,
        )
    if pem:
        return (
            _materialize(
                str(pem),
                kind=material_kind,
                environ=environ,
                destination_root=destination_root,
            ),
            True,
        )
    if path_value:
        return _path(path_value), False
    return None, False


def _proxy_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme.casefold() not in {"http", "https", "socks5", "socks5h"}:
        raise TransportSecurityError("tls_proxy_invalid")
    if not parsed.hostname:
        raise TransportSecurityError("tls_proxy_invalid")
    return value


@dataclass(frozen=True)
class ResolvedTLSProfile:
    """A path-safe summary plus client-specific runtime adapters."""

    name: str
    source: str
    configured: bool
    system_trust: bool
    trust_env: bool
    ssl_context: ssl.SSLContext = field(repr=False)
    ca_bundle_path: Path | None = field(default=None, repr=False)
    ca_directory: Path | None = field(default=None, repr=False)
    client_cert_path: Path | None = field(default=None, repr=False)
    client_key_path: Path | None = field(default=None, repr=False)
    client_bundle_path: Path | None = field(default=None, repr=False)
    client_key_password: str | None = field(default=None, repr=False)
    proxy_url: str | None = field(default=None, repr=False)
    no_proxy: str | None = field(default=None, repr=False)
    materialized: tuple[Path, ...] = field(default=(), repr=False)

    @property
    def verify_enabled(self) -> bool:
        """Verification is an invariant of every resolved profile."""
        return True

    def httpx_kwargs(self) -> dict[str, Any]:
        """Arguments accepted by ``httpx.Client``/``AsyncClient``."""
        kwargs: dict[str, Any] = {
            "verify": self.ssl_context,
            "trust_env": self.trust_env,
        }
        if self.proxy_url:
            kwargs["proxy"] = self.proxy_url
        return kwargs

    def requests_kwargs(self) -> dict[str, Any]:
        """Arguments accepted by ``requests.request`` and ``Session.request``."""
        verify: bool | str
        if self.ca_bundle_path is not None:
            verify = str(self.ca_bundle_path)
        elif self.ca_directory is not None:
            verify = str(self.ca_directory)
        else:
            verify = True
        kwargs: dict[str, Any] = {"verify": verify}
        if self.client_bundle_path is not None:
            # Requests cannot accept a private-key password. Resolution creates a
            # private, process-lifetime bundle (decrypting the key when needed),
            # so every Requests consumer receives the same current TLS contract.
            kwargs["cert"] = str(self.client_bundle_path)
        if self.proxy_url:
            kwargs["proxies"] = {
                "http": self.proxy_url,
                "https": self.proxy_url,
            }
        return kwargs

    def configure_requests_session(self, session: Any) -> Any:
        """Apply trust-env and TLS settings to a Requests-compatible session."""
        kwargs = self.requests_kwargs()
        session.trust_env = self.trust_env
        session.verify = kwargs["verify"]
        if "cert" in kwargs:
            session.cert = kwargs["cert"]
        if "proxies" in kwargs:
            session.proxies.update(kwargs["proxies"])
        return session

    def psycopg_kwargs(self) -> dict[str, Any]:
        """Strict libpq/psycopg TLS parameters for a database connection."""
        if self.proxy_url:
            raise TransportSecurityError("postgres_tls_proxy_unsupported")
        if self.ca_directory is not None:
            raise TransportSecurityError("postgres_ca_directory_unsupported")
        kwargs: dict[str, Any] = {"sslmode": "verify-full"}
        if self.ca_bundle_path is not None:
            kwargs["sslrootcert"] = str(self.ca_bundle_path)
        if self.client_cert_path is not None:
            kwargs["sslcert"] = str(self.client_cert_path)
            kwargs["sslkey"] = str(self.client_key_path)
        if self.client_key_password is not None:
            kwargs["sslpassword"] = self.client_key_password
        return kwargs

    def pymongo_kwargs(self) -> dict[str, Any]:
        """Strict PyMongo TLS parameters for a database connection."""
        if self.proxy_url:
            raise TransportSecurityError("mongodb_tls_proxy_unsupported")
        if self.ca_directory is not None:
            raise TransportSecurityError("mongodb_ca_directory_unsupported")
        kwargs: dict[str, Any] = {"tls": True}
        if self.ca_bundle_path is not None:
            kwargs["tlsCAFile"] = str(self.ca_bundle_path)
        if self.client_bundle_path is not None:
            kwargs["tlsCertificateKeyFile"] = str(self.client_bundle_path)
        if self.client_key_password is not None:
            kwargs["tlsCertificateKeyFilePassword"] = self.client_key_password
        return kwargs

    def redis_kwargs(self) -> dict[str, Any]:
        """Strict redis-py TLS parameters for a ``rediss://`` connection."""
        if self.proxy_url:
            raise TransportSecurityError("redis_tls_proxy_unsupported")
        kwargs: dict[str, Any] = {
            "ssl_cert_reqs": "required",
            "ssl_check_hostname": True,
        }
        if self.ca_bundle_path is not None:
            kwargs["ssl_ca_certs"] = str(self.ca_bundle_path)
        if self.ca_directory is not None:
            kwargs["ssl_ca_path"] = str(self.ca_directory)
        if self.client_cert_path is not None:
            kwargs["ssl_certfile"] = str(self.client_cert_path)
            kwargs["ssl_keyfile"] = str(self.client_key_path)
        if self.client_key_password is not None:
            kwargs["ssl_password"] = self.client_key_password
        return kwargs

    def child_env(self, *, service: str | None = None) -> dict[str, str]:
        """Portable environment projection for HTTP and isolated child tools.

        A service name additionally projects an mTLS bundle through that
        service's current runtime variables. The bundle is a private temporary
        file and contains no machine-specific durable configuration.
        """
        values: dict[str, str] = {}
        if self.ca_bundle_path is not None:
            rendered = str(self.ca_bundle_path)
            values["REQUESTS_CA_BUNDLE"] = rendered
            values["SSL_CERT_FILE"] = rendered
        if self.ca_directory is not None:
            values["SSL_CERT_DIR"] = str(self.ca_directory)
        if self.system_trust:
            values["UV_NATIVE_TLS"] = "true"
        if self.proxy_url:
            values["HTTPS_PROXY"] = self.proxy_url
            values["HTTP_PROXY"] = self.proxy_url
        if self.no_proxy:
            values["NO_PROXY"] = self.no_proxy
        if service and self.client_bundle_path is not None:
            prefix = _service_prefix(service)
            bundle = str(self.client_bundle_path)
            # The generic profile accepts separate cert/key paths. Supplying the
            # same combined PEM for each lets the child reconstruct an SSLContext
            # without receiving a password or a secret reference.
            values[f"{prefix}_CLIENT_CERT"] = bundle
            values[f"{prefix}_CLIENT_KEY"] = bundle
        return values

    def cleanup(self) -> None:
        """Remove only secret/inline material created by this resolution."""
        for path in self.materialized:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue
            _MATERIALIZED_PATHS.discard(path)


def resolve_tls_profile(
    service: str,
    *,
    profile_name: str | None = None,
    profile_ref: str | None = None,
    profile: Mapping[str, Any] | None = None,
    environ: MutableMapping[str, str] | Mapping[str, str] | None = None,
    resolver: Callable[[str], str | None] | None = None,
    destination_root: Path | None = None,
) -> ResolvedTLSProfile:
    """Resolve a named or inline runtime TLS profile without persisting it.

    Resolution order is explicit profile, explicit/service profile reference,
    named profile catalog (``TLS_PROFILES_REF`` preferred over
    ``TLS_PROFILES``), then service/global environment variables.
    """

    target_env: Mapping[str, str] = environ if environ is not None else os.environ
    prefix = _service_prefix(service)
    # Once a caller supplies either selector, the pair is an isolated trust
    # decision.  Never fill the missing half from ambient service/global state:
    # doing so can silently combine two unrelated provider trust policies.
    explicit_selector = bool(profile_name or profile_ref)
    selected_name = str(
        profile_name or ""
        if explicit_selector
        else _first(target_env, f"{prefix}_TLS_PROFILE", "TLS_PROFILE")
    ).strip()
    selected_ref = str(
        profile_ref or ""
        if explicit_selector
        else _first(target_env, f"{prefix}_TLS_PROFILE_REF", "TLS_PROFILE_REF")
    ).strip()
    source = "environment"
    explicitly_configured = profile is not None or bool(selected_name or selected_ref)

    if profile is not None:
        resolved_profile = dict(profile)
        source = "inline"
    elif selected_ref:
        resolved = _parse_profile_json(
            _resolve_secret(selected_ref, resolver, target_env)
        )
        resolved_profile = (
            _select_named_profile(resolved, selected_name)
            if selected_name
            else resolved
        )
        source = "secret_ref"
    elif selected_name:
        if not _NAME_RE.fullmatch(selected_name):
            raise TransportSecurityError("tls_profile_name_invalid")
        catalog_ref = _first(target_env, "TLS_PROFILES_REF")
        catalog_raw = (
            _resolve_secret(catalog_ref, resolver, target_env)
            if catalog_ref
            else _first(target_env, "TLS_PROFILES")
        )
        if not catalog_raw:
            raise TransportSecurityError("tls_profile_catalog_unavailable")
        resolved_profile = _select_named_profile(
            _parse_profile_json(catalog_raw), selected_name
        )
        source = "secret_catalog" if catalog_ref else "runtime_catalog"
    else:
        resolved_profile = _profile_from_environment(prefix, target_env)
        direct_keys = (
            f"{prefix}_TLS_SYSTEM_TRUST",
            f"{prefix}_TLS_TRUST_ENV",
            f"{prefix}_CA_BUNDLE_REF",
            f"{prefix}_CA_BUNDLE",
            f"{prefix}_CA_DIRECTORY",
            f"{prefix}_CLIENT_CERT_REF",
            f"{prefix}_CLIENT_CERT",
            f"{prefix}_CLIENT_KEY_REF",
            f"{prefix}_CLIENT_KEY",
            f"{prefix}_PROXY_URL_REF",
            f"{prefix}_PROXY_URL",
            "TLS_CA_BUNDLE_REF",
            "TLS_CA_BUNDLE",
            "TLS_CA_DIRECTORY",
            "TLS_CLIENT_CERT_REF",
            "TLS_CLIENT_CERT",
            "TLS_CLIENT_KEY_REF",
            "TLS_CLIENT_KEY",
            "TLS_PROXY_URL_REF",
            "TLS_PROXY_URL",
        )
        explicitly_configured = any(
            target_env.get(key) not in (None, "") for key in direct_keys
        )

    retired_controls = {"verify", "allow_insecure"}.intersection(resolved_profile)
    if retired_controls:
        raise TransportSecurityError("tls_verification_control_retired")
    system_trust = _as_bool(resolved_profile.get("system_trust"), default=True)
    trust_env = _as_bool(resolved_profile.get("trust_env"), default=True)

    ca_path: Path | None = None
    ca_materialized = False
    ca_directory: Path | None = None
    cert_path: Path | None = None
    cert_materialized = False
    key_path: Path | None = None
    key_materialized = False
    client_key_password: str | None = None
    proxy_url: str | None = None
    no_proxy: str | None = None
    client_bundle_path: Path | None = None
    try:
        ca_profile = dict(resolved_profile)
        for alias, canonical in (
            ("ca_ref", "ca_bundle_ref"),
            ("ca_pem", "ca_bundle_pem"),
            ("ca_path", "ca_bundle_path"),
        ):
            if ca_profile.get(canonical) in (
                None,
                "",
            ) and ca_profile.get(alias) not in (None, ""):
                ca_profile[canonical] = ca_profile[alias]
        ca_path, ca_materialized = _material(
            ca_profile,
            kind="ca_bundle",
            resolver=resolver,
            environ=target_env,
            destination_root=destination_root,
        )
        directory_value = _pick(resolved_profile, "ca_directory", "ca_dir")
        ca_directory = (
            _path(directory_value, directory=True) if directory_value else None
        )
        if ca_path is not None and ca_directory is not None:
            raise TransportSecurityError("tls_ca_source_ambiguous")
        if not system_trust and ca_path is None and ca_directory is None:
            raise TransportSecurityError("tls_trust_anchor_missing")

        cert_path, cert_materialized = _material(
            resolved_profile,
            kind="client_cert",
            resolver=resolver,
            environ=target_env,
            destination_root=destination_root,
        )
        key_path, key_materialized = _material(
            resolved_profile,
            kind="client_key",
            resolver=resolver,
            environ=target_env,
            destination_root=destination_root,
        )
        if (cert_path is None) != (key_path is None):
            raise TransportSecurityError("tls_client_certificate_incomplete")

        password_ref = _pick(resolved_profile, "client_key_password_ref")
        password_value = _pick(resolved_profile, "client_key_password")
        if password_ref and password_value:
            raise TransportSecurityError("tls_client_key_password_ambiguous")
        client_key_password = (
            _resolve_secret(str(password_ref), resolver, target_env)
            if password_ref
            else (str(password_value) if password_value else None)
        )

        proxy_ref = _pick(resolved_profile, "proxy_url_ref", "proxy_ref")
        proxy_value = _pick(resolved_profile, "proxy_url", "proxy")
        if proxy_ref and proxy_value:
            raise TransportSecurityError("tls_proxy_source_ambiguous")
        if proxy_ref:
            proxy_value = _resolve_secret(str(proxy_ref), resolver, target_env)
        proxy_url = _proxy_url(str(proxy_value)) if proxy_value else None
        no_proxy = str(_pick(resolved_profile, "no_proxy") or "").strip() or None

        context = (
            ssl.create_default_context()
            if system_trust
            else ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        )
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        if ca_path is not None:
            context.load_verify_locations(cafile=str(ca_path))
        elif ca_directory is not None:
            context.load_verify_locations(capath=str(ca_directory))
        if cert_path is not None:
            assert key_path is not None
            context.load_cert_chain(
                certfile=str(cert_path),
                keyfile=str(key_path),
                password=client_key_password,
            )
            private_key_pem = key_path.read_bytes()
            if client_key_password:
                from cryptography.hazmat.primitives import serialization

                private_key = serialization.load_pem_private_key(
                    private_key_pem,
                    password=client_key_password.encode("utf-8"),
                )
                private_key_pem = private_key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.PKCS8,
                    serialization.NoEncryption(),
                )
            client_bundle_path = _materialize(
                cert_path.read_text(encoding="utf-8")
                + "\n"
                + private_key_pem.decode("ascii"),
                kind="cert",
                environ=target_env,
                destination_root=destination_root,
            )
    except Exception as exc:
        for candidate, created in (
            (ca_path, ca_materialized),
            (cert_path, cert_materialized),
            (key_path, key_materialized),
            (client_bundle_path, client_bundle_path is not None),
        ):
            if candidate is not None and created:
                try:
                    candidate.unlink(missing_ok=True)
                except OSError:
                    continue
                _MATERIALIZED_PATHS.discard(candidate)
        if isinstance(exc, TransportSecurityError):
            raise
        raise TransportSecurityError("tls_profile_material_invalid") from None

    materialized = tuple(
        path
        for path, created in (
            (ca_path, ca_materialized),
            (cert_path, cert_materialized),
            (key_path, key_materialized),
            (client_bundle_path, client_bundle_path is not None),
        )
        if path is not None and created
    )
    profile_label = selected_name if _NAME_RE.fullmatch(selected_name) else "default"
    return ResolvedTLSProfile(
        name=profile_label,
        source=source,
        configured=explicitly_configured,
        system_trust=system_trust,
        trust_env=trust_env,
        ssl_context=context,
        ca_bundle_path=ca_path,
        ca_directory=ca_directory,
        client_cert_path=cert_path,
        client_key_path=key_path,
        client_bundle_path=client_bundle_path,
        client_key_password=client_key_password,
        proxy_url=proxy_url,
        no_proxy=no_proxy,
        materialized=materialized,
    )
