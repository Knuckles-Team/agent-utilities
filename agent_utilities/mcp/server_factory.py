from __future__ import annotations

"""MCP Server Factory.

Handles CLI argument parsing for MCP servers, automated server initialization
with middleware stacks, and the ``create_mcp_server`` convenience constructor.

CONCEPT:AU-ECO.mcp.standardized-interfaces — MCP Standardized Interfaces
"""


import argparse
import asyncio
import contextlib
import ipaddress
import logging
import os
import re
import sys
from typing import Any, cast
from urllib.parse import urlsplit

from agent_utilities._version import __version__
from agent_utilities.base_utilities import to_boolean
from agent_utilities.core.config import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    setting,
)
from agent_utilities.security.cli_secrets import (
    RuntimeSecretReferenceAction,
    RuntimeSecretReferenceError,
    resolve_runtime_secret_reference,
)

logger = logging.getLogger(__name__)


# MCP-specific auth/delegation config (not in config.py)
mcp_auth_config = {
    "enable_delegation": to_boolean(setting("ENABLE_DELEGATION", "False")),
    "audience": setting("AUDIENCE", None),
    "delegated_scopes": setting("DELEGATED_SCOPES", "api"),
    "token_endpoint": None,
    "oidc_client_id": setting("OIDC_CLIENT_ID", None),
    "oidc_client_secret": None,
    "oidc_config_url": setting("OIDC_CONFIG_URL", None),
    "jwt_jwks_uri": setting("FASTMCP_SERVER_AUTH_JWT_JWKS_URI", None),
    "jwt_issuer": setting("FASTMCP_SERVER_AUTH_JWT_ISSUER", None),
    "jwt_audience": setting("FASTMCP_SERVER_AUTH_JWT_AUDIENCE", None),
    "jwt_algorithm": setting("FASTMCP_SERVER_AUTH_JWT_ALGORITHM", None),
    "jwt_secret": setting("FASTMCP_SERVER_AUTH_JWT_PUBLIC_KEY", None),
    "jwt_required_scopes": setting("FASTMCP_SERVER_AUTH_JWT_REQUIRED_SCOPES", None),
}  # nosec B105

DEFAULT_TRANSPORT = setting("TRANSPORT", "stdio")

_NETWORK_TRANSPORTS = frozenset({"streamable-http", "sse"})
_ALL_TRANSPORTS = ("stdio", "streamable-http", "sse")
_FILTER_VALUE_RE = re.compile(r"^[A-Za-z0-9_.:/-]{1,128}$")
_NO_HTTP_REQUEST = "No active HTTP request found."


class UnsupportedAuthTypeError(ValueError):
    """``--auth-type``/``AUTH_TYPE`` named a mode this factory cannot configure.

    Fail closed (BUG-CX-023). ``_configure_auth`` used to fall off the end of
    its dispatch chain and return ``None`` -- which FastMCP reads as "no auth"
    -- so an unrecognised value silently produced an UNAUTHENTICATED server.
    ``--auth-type`` carries ``choices=``, but argparse does **not** validate a
    *default*, and the default is ``setting("AUTH_TYPE", "none")``: a typo'd
    ``AUTH_TYPE`` environment variable therefore reached the dispatch chain
    completely unchecked. Refusing is the only safe reading of an auth mode we
    do not understand.

    Subclasses :class:`ValueError` to match this module's existing idiom for an
    invalid configuration value (and ``kg_server.UnsupportedToolFieldError``).
    """


def _bounded_filter_values(values: list[str]) -> list[str]:
    """Parse bounded client visibility filters; these may only narrow access."""
    parsed: list[str] = []
    for raw in values:
        if len(str(raw).encode("utf-8")) > 16_384:
            raise ValueError("MCP visibility filter is too large")
        for part in str(raw).split(","):
            value = part.strip()
            if not value:
                continue
            if value.lower() == "all" or not _FILTER_VALUE_RE.fullmatch(value):
                raise ValueError("MCP visibility filter is invalid")
            parsed.append(value)
            if len(parsed) > 256:
                raise ValueError("MCP visibility filter has too many values")
    return list(dict.fromkeys(parsed))


def _narrow_values(current: list[str], requested: list[str]) -> list[str]:
    """Intersect an existing allowlist, or establish one when none exists."""
    if not requested:
        return current
    if not current:
        return requested
    requested_set = set(requested)
    narrowed = [value for value in current if value in requested_set]
    return narrowed or ["__empty_filter_intersection__"]


def _union_values(current: list[str], additions: list[str]) -> list[str]:
    """Union deny lists without allowing a caller to remove server policy."""
    return list(dict.fromkeys([*current, *additions]))


def _split_csv(raw: str) -> list[str]:
    """Split a comma-separated env/CLI/query value into trimmed, non-empty parts."""
    return [x.strip() for x in raw.split(",") if x.strip()]


def _csv_setting(key: str) -> list[str]:
    """Read a comma-separated env-var default via `setting()`, split and trimmed."""
    raw = setting(key)
    return _split_csv(raw) if raw else []


def _env_filter_defaults() -> tuple[list[str], list[str], list[str], list[str]]:
    """Starting (enabled_tools, disabled_tools, enabled_tags, disabled_tags)
    lists from the MCP_*_TOOLS / MCP_*_TAGS environment-variable defaults."""
    return (
        _csv_setting("MCP_ENABLED_TOOLS"),
        _csv_setting("MCP_DISABLED_TOOLS"),
        _csv_setting("MCP_ENABLED_TAGS"),
        _csv_setting("MCP_DISABLED_TAGS"),
    )


def _single_or_empty(value: str | None) -> list[str]:
    return [value] if value else []


def _collect_multi_values(source: Any, keys: list[str]) -> list[str]:
    """Collect comma-separated values across query-param/header ``keys``,
    using whichever of ``getlist``/``get`` the source (Starlette QueryParams
    or Headers) implements."""
    collected: list[str] = []
    for key in keys:
        vals = (
            source.getlist(key)
            if hasattr(source, "getlist")
            else _single_or_empty(source.get(key))
        )
        for val in vals:
            if val:
                collected.extend(_split_csv(val))
    return collected


def _first_present(source: Any, keys: list[str]) -> str | None:
    """First truthy value for any of ``keys`` on a query-param/header source."""
    for key in keys:
        val = source.get(key)
        if val:
            return val
    return None


def _query_param_filters(
    q_params: Any,
) -> tuple[list[str], list[str], list[str], list[str], str | None]:
    """(enabled_tools, disabled_tools, enabled_tags, disabled_tags, query) from request query params."""
    return (
        _collect_multi_values(q_params, ["tools", "toolsets"]),
        _collect_multi_values(q_params, ["disabled_tools", "disabled_toolsets"]),
        _collect_multi_values(q_params, ["tags"]),
        _collect_multi_values(q_params, ["disabled_tags"]),
        _first_present(q_params, ["q", "query", "search"]),
    )


def _header_filters(
    headers: Any,
) -> tuple[list[str], list[str], list[str], list[str], str | None]:
    """(enabled_tools, disabled_tools, enabled_tags, disabled_tags, query) from request headers."""
    return (
        _collect_multi_values(
            headers, ["x-mcp-enabled-tools", "x-mcp-enabled-components"]
        ),
        _collect_multi_values(
            headers, ["x-mcp-disabled-tools", "x-mcp-disabled-components"]
        ),
        _collect_multi_values(headers, ["x-mcp-enabled-tags"]),
        _collect_multi_values(headers, ["x-mcp-disabled-tags"]),
        _first_present(headers, ["x-mcp-query", "x-mcp-search"]),
    )


def _apply_narrow_union_overrides(
    enabled_tools: list[str],
    disabled_tools: list[str],
    enabled_tags: list[str],
    disabled_tags: list[str],
    query_filter: str | None,
    extracted: tuple[list[str], list[str], list[str], list[str], str | None],
) -> tuple[list[str], list[str], list[str], list[str], str | None]:
    """Layer one extracted (query-param or header) filter set on top of the
    running lists: enabled/tags narrow, disabled/tags union, query replaces."""
    tools, disabled, tags, new_disabled_tags, query = extracted
    if tools:
        enabled_tools = _narrow_values(enabled_tools, _bounded_filter_values(tools))
    if disabled:
        disabled_tools = _union_values(disabled_tools, _bounded_filter_values(disabled))
    if tags:
        enabled_tags = _narrow_values(enabled_tags, _bounded_filter_values(tags))
    if new_disabled_tags:
        disabled_tags = _union_values(
            disabled_tags, _bounded_filter_values(new_disabled_tags)
        )
    if query:
        query_filter = query
    return enabled_tools, disabled_tools, enabled_tags, disabled_tags, query_filter


def _apply_request_overrides(
    enabled_tools: list[str],
    disabled_tools: list[str],
    enabled_tags: list[str],
    disabled_tags: list[str],
    query_filter: str | None,
) -> tuple[list[str], list[str], list[str], list[str], str | None, bool]:
    """Read the active HTTP request (query params, then headers) and layer
    its filters on top of the given lists. A malformed client filter must
    never widen the surface or silently fall back to the full component
    set, so any failure here rejects everything instead.

    Returns (enabled_tools, disabled_tools, enabled_tags, disabled_tags,
    query_filter, reject_all).
    """
    try:
        req = _optional_http_request()
        if not req:
            return (
                enabled_tools,
                disabled_tools,
                enabled_tags,
                disabled_tags,
                query_filter,
                False,
            )

        enabled_tools, disabled_tools, enabled_tags, disabled_tags, query_filter = (
            _apply_narrow_union_overrides(
                enabled_tools,
                disabled_tools,
                enabled_tags,
                disabled_tags,
                query_filter,
                _query_param_filters(req.query_params),
            )
        )
        if req.headers:
            enabled_tools, disabled_tools, enabled_tags, disabled_tags, query_filter = (
                _apply_narrow_union_overrides(
                    enabled_tools,
                    disabled_tools,
                    enabled_tags,
                    disabled_tags,
                    query_filter,
                    _header_filters(req.headers),
                )
            )
    except Exception as exc:
        logger.warning(
            "Rejected MCP visibility filter (exception_type=%s)",
            type(exc).__name__,
        )
        return (
            enabled_tools,
            disabled_tools,
            enabled_tags,
            disabled_tags,
            query_filter,
            True,
        )

    return (
        enabled_tools,
        disabled_tools,
        enabled_tags,
        disabled_tags,
        query_filter,
        False,
    )


def _semantic_kg_matches(query_filter: str, name: str) -> list[str] | None:
    """Resolve a `q=` semantic filter through the active KG engine.

    Returns the matched tool names (possibly empty, meaning "no match"), or
    ``None`` if the engine/orchestrator import or resolution itself failed.
    """
    try:
        from agent_utilities.knowledge_graph.core.engine import (
            IntelligenceGraphEngine,
        )
        from agent_utilities.tools.dynamic_tool_orchestrator import (
            DynamicToolOrchestrator,
        )

        engine = IntelligenceGraphEngine.get_active()
        if not engine:
            return []
        orchestrator = DynamicToolOrchestrator(engine)
        return list(orchestrator.resolve_mcp_tools(query_filter, server_name=name))
    except Exception as exc:
        logger.debug(
            "Failed to filter components using Knowledge Graph (exception_type=%s)",
            type(exc).__name__,
        )
        return None


def _apply_semantic_filter(
    query_filter: str | None, enabled_tools_list: list[str], name: str
) -> tuple[list[str], bool]:
    """A requested semantic filter is a least-privilege boundary: no match,
    no active graph, or any resolution failure exposes no components -- it
    must never widen back to the configured surface.

    Returns (enabled_tools_list, reject_all).
    """
    if not query_filter:
        return enabled_tools_list, False
    kg_matched = _semantic_kg_matches(query_filter, name)
    if not kg_matched:
        return enabled_tools_list, True
    if enabled_tools_list and "all" not in enabled_tools_list:
        narrowed = [tool for tool in enabled_tools_list if tool in kg_matched]
        return (narrowed, False) if narrowed else (narrowed, True)
    return kg_matched, False


def _finalize_filter_sets(
    enabled_tags_list: list[str],
    disabled_tags_list: list[str],
    enabled_tools_list: list[str],
    disabled_tools_list: list[str],
) -> tuple[set[str] | None, set[str] | None, set[str] | None, set[str] | None]:
    """Convert the resolved lists to sets, applying the "'all' or empty
    means expose everything" fallback to the enabled-tools set."""
    enabled_tags = set(enabled_tags_list) if enabled_tags_list else None
    disabled_tags = set(disabled_tags_list) if disabled_tags_list else None
    enabled_names = set(enabled_tools_list) if enabled_tools_list else None
    disabled_names = set(disabled_tools_list) if disabled_tools_list else None
    if enabled_names is not None and ("all" in enabled_names or not enabled_names):
        enabled_names = None
    return enabled_tags, disabled_tags, enabled_names, disabled_names


def _component_tag_visible(
    component: Any, enabled_tags: set[str] | None, disabled_tags: set[str] | None
) -> bool:
    """Tag-visibility rule for ONE already name-permitted component."""
    if hasattr(component, "tags") and component.tags:
        if enabled_tags is not None and not (component.tags & enabled_tags):
            return False
        return not (disabled_tags is not None and component.tags & disabled_tags)
    return enabled_tags is None


def _component_passes(
    component: Any,
    enabled_names: set[str] | None,
    disabled_names: set[str] | None,
    enabled_tags: set[str] | None,
    disabled_tags: set[str] | None,
) -> bool:
    """Whether ONE component survives the resolved name/tag filters.

    Fail closed (BUG-CX-022): a component carrying neither ``name`` nor ``uri``
    is EXCLUDED, not exempted. It used to return ``True`` here, skipping *all*
    visibility filtering -- name allow/deny lists and tag restrictions alike --
    so something that should have been hidden was handed to the caller.

    Exclusion (rather than raising) is the right refusal for this shape: the
    function answers one yes/no visibility question per component, and it is
    reached from ``DynamicVisibilityTransform._filter_components`` on the
    ``list_tools``/``list_resources``/``list_resource_templates`` path, where
    raising would turn one malformed entry into a failed listing for every
    caller. Every real FastMCP component (Tool, Resource, ResourceTemplate,
    Prompt) inherits a ``name``, so an object with neither identifier is not a
    component this server can vouch for exposing in the first place.
    """
    c_name = getattr(component, "name", None) or getattr(component, "uri", None)
    if not c_name:
        logger.warning(
            "Excluding an MCP component with neither name nor uri: it cannot be "
            "checked against the configured visibility filters"
        )
        return False
    if enabled_names is not None and c_name not in enabled_names:
        return False
    if disabled_names is not None and c_name in disabled_names:
        return False
    return _component_tag_visible(component, enabled_tags, disabled_tags)


def _optional_http_request() -> Any | None:
    """Return the active FastMCP HTTP request, if this is an HTTP invocation.

    FastMCP deliberately raises when ``list_tools`` is called in-process (for
    example by connector certification) or over stdio because neither path has
    an HTTP request context.  That expected absence is not a rejected client
    filter.  Every other exception still propagates to the fail-closed filter
    guard below.
    """
    from fastmcp.server.dependencies import get_http_request

    try:
        return get_http_request()
    except RuntimeError as exc:
        if exc.args == (_NO_HTTP_REQUEST,):
            return None
        raise


def _is_malformed_numeric_claim(value: Any) -> bool:
    """True if `value` cannot be trusted as a finite numeric timestamp claim
    (bool masquerading as int, wrong type, or non-finite)."""
    import math

    return (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    )


def _exp_claim_invalid(claims: dict, now: float) -> bool:
    # `Any`, not the inferred `Any | None`: `_is_malformed_numeric_claim`
    # short-circuits every non-numeric value (None included) before the
    # `float(...)`, but that narrowing lives in the callee, so the annotation
    # has to say what a JWT claim actually is — untyped.
    exp: Any = claims.get("exp")
    return _is_malformed_numeric_claim(exp) or float(exp) < now


def _nbf_iat_claims_invalid(claims: dict, now: float) -> bool:
    for field in ("nbf", "iat"):
        value = claims.get(field)
        if value is None:
            continue
        if _is_malformed_numeric_claim(value) or float(value) > now + 30.0:
            return True
    return False


def _hardened_jwt_verifier(**kwargs: Any) -> Any:
    """Build FastMCP's verifier with pinned JWKS I/O and privacy-safe logging."""
    from fastmcp.server.auth.providers.jwt import JWTVerifier

    class _Verifier(JWTVerifier):
        async def _fetch_jwks(self) -> dict[str, Any]:
            if not self.jwks_uri:
                raise ValueError("JWKS URI is not configured")
            from agent_utilities.security.auth import _fetch_jwks

            document = await _fetch_jwks(str(self.jwks_uri))
            return dict(document)

        async def verify_token(self, token: str) -> Any:
            if not isinstance(token, str) or not token or len(token.encode()) > 16_384:
                return None
            result = await super().verify_token(token)
            if result is None:
                return None
            claims = getattr(result, "claims", None)
            if not isinstance(claims, dict):
                return None
            import time

            now = time.time()
            if _exp_claim_invalid(claims, now) or _nbf_iat_claims_invalid(claims, now):
                return None
            return result

    class _PrivacyLogger:
        def debug(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            logger.debug("JWT verifier diagnostic")

        def info(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            logger.info("JWT verifier event")

        def warning(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            logger.warning("JWT verifier rejected a credential")

        def error(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            logger.error("JWT verifier error")

        def exception(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            logger.error("JWT verifier error")

        def isEnabledFor(self, level: int) -> bool:  # noqa: N802
            return logger.isEnabledFor(level)

    verifier = _Verifier(**kwargs)
    # _PrivacyLogger is a deliberate duck-typed shim (redacts JWT-adjacent log
    # content) satisfying the same debug/info/warning/error/isEnabledFor surface
    # JWTVerifier calls -- not a real logging.Logger subclass, so it is cast
    # rather than nominally typed.
    verifier.logger = cast("logging.Logger", _PrivacyLogger())
    return verifier


def _secure_auth_url(value: Any, *, field: str) -> str:
    rendered = str(value or "").strip()
    if len(rendered) > 8_192:
        raise ValueError(f"{field} is too large")
    parsed = urlsplit(rendered)
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError(f"{field} is invalid")
    if parsed.scheme.lower() == "http" and parsed.hostname.lower() not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise ValueError(f"{field} requires HTTPS outside loopback")
    return rendered


def _validated_redirect_uris(value: str | None) -> list[str] | None:
    if not value:
        return None
    uris = [part.strip() for part in value.split(",") if part.strip()]
    if not uris or len(uris) > 32:
        raise ValueError("redirect URI allowlist is invalid")
    return [_secure_auth_url(uri, field="redirect URI") for uri in uris]


def _wrap_with_resource_metadata(
    verifier: Any, args: argparse.Namespace, issuer_values: list[str]
) -> Any:
    """Compose *verifier* with RFC 9728 protected-resource metadata when a public
    base URL is configured (CONCEPT:AU-OS.identity.protected-resource-metadata).

    A bare ``TokenVerifier`` (what plain ``jwt`` auth builds) publishes zero OAuth
    routes and never appends ``resource_metadata`` to its 401 challenge, so an
    RFC 9728-aware MCP client (e.g. Claude Code) has no way to discover the
    authorization server and refresh its own token — it is handed a static JWT
    that silently expires. Wrapping the same verifier in ``RemoteAuthProvider``
    adds the ``/.well-known/oauth-protected-resource`` route and the
    ``resource_metadata`` challenge param with no change to token verification
    itself (``RemoteAuthProvider.verify_token`` just delegates to *verifier*).

    With no public base URL configured, *verifier* is returned unwrapped — today's
    exact behavior, so this is additive and fully backward compatible.
    """
    import sys as _sys

    base_url = str(getattr(args, "public_base_url", "") or "").strip()
    if not base_url:
        return verifier
    try:
        base_url = _secure_auth_url(base_url, field="MCP public base URL")
    except ValueError:
        logger.error("Error: MCP public base URL policy is invalid")
        _sys.exit(1)

    from fastmcp.server.auth import RemoteAuthProvider
    from pydantic import AnyHttpUrl

    return RemoteAuthProvider(
        token_verifier=verifier,
        authorization_servers=[AnyHttpUrl(u) for u in issuer_values],
        base_url=base_url,
    )


def _is_loopback_bind(host: Any) -> bool:
    """Return true only for an explicit loopback bind target.

    Hostnames other than ``localhost`` are deliberately treated as remote. We
    do not resolve DNS while validating startup configuration because a name
    can be rebound after validation.
    """
    value = str(host or "").strip().lower()
    if value in {"localhost", "localhost."}:
        return True
    value = value.strip("[]").split("%", 1)[0]
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _requires_network_exposure_validation(args: argparse.Namespace) -> bool:
    transport = str(getattr(args, "transport", "stdio") or "stdio").lower()
    if transport not in _NETWORK_TRANSPORTS:
        return False
    return not _is_loopback_bind(getattr(args, "host", ""))


def _validate_network_auth_present(args: argparse.Namespace) -> None:
    auth_type = str(getattr(args, "auth_type", "none") or "none").lower()
    if auth_type == "none":
        logger.error(
            "Refusing an unauthenticated MCP network listener outside loopback"
        )
        raise SystemExit(1)


def _validate_network_tls_material(args: argparse.Namespace) -> bool:
    """Validate cert/key policy; returns whether direct TLS is configured."""
    certfile = str(getattr(args, "tls_certfile", "") or "").strip()
    keyfile = str(getattr(args, "tls_keyfile", "") or "").strip()
    if bool(certfile) != bool(keyfile):
        logger.error("MCP server TLS requires both certificate and key material")
        raise SystemExit(1)
    direct_tls = bool(certfile and keyfile)
    if direct_tls and not (os.path.isfile(certfile) and os.path.isfile(keyfile)):
        logger.error("MCP server TLS material is unavailable")
        raise SystemExit(1)
    return direct_tls


def _validate_network_tls_boundary(args: argparse.Namespace, direct_tls: bool) -> None:
    terminated = bool(getattr(args, "tls_terminated", False))
    proxy_cidrs = str(getattr(args, "trusted_proxy_cidrs", "") or "").strip()
    if terminated and not proxy_cidrs:
        logger.error(
            "MCP_TLS_TERMINATED requires MCP_TRUSTED_PROXY_CIDRS so plaintext "
            "traffic is accepted only from the TLS ingress"
        )
        raise SystemExit(1)
    if not direct_tls and not terminated:
        logger.error(
            "A non-loopback MCP listener requires direct TLS or an explicitly "
            "trusted TLS-terminating ingress"
        )
        raise SystemExit(1)


def _validate_network_allowed_hosts(args: argparse.Namespace) -> None:
    hosts = _split_csv(str(getattr(args, "allowed_hosts", "") or ""))
    if not hosts or any("*" in value for value in hosts):
        logger.error("A non-loopback MCP listener requires exact MCP_ALLOWED_HOSTS")
        raise SystemExit(1)


def _validate_network_exposure(args: argparse.Namespace) -> None:
    """Fail closed unless a remote MCP listener has auth and a TLS boundary."""
    if not _requires_network_exposure_validation(args):
        return
    _validate_network_auth_present(args)
    direct_tls = _validate_network_tls_material(args)
    _validate_network_tls_boundary(args, direct_tls)
    _validate_network_allowed_hosts(args)


def _resolve_mcp_parser_secret_defaults() -> tuple[
    str | None, str | None, str | None, str | None, str | None, str | None
]:
    """(default_oidc_config, default_oidc_client_id, default_oidc_client_secret,
    default_oauth_upstream_client_secret, default_openapi_password,
    default_openapi_client_secret) -- resolved from Keycloak/env-var defaults
    and runtime-secret-ref lookups, for create_mcp_parser's flag defaults."""
    keycloak_url = setting("KEYCLOAK_URL")
    keycloak_realm = setting("KEYCLOAK_REALM", "master")
    default_oidc_config = setting("OIDC_CONFIG_URL")
    if not default_oidc_config and keycloak_url:
        default_oidc_config = (
            f"{keycloak_url}/realms/{keycloak_realm}/.well-known/openid-configuration"
        )

    default_oidc_client_id = setting("OIDC_CLIENT_ID") or setting("KEYCLOAK_CLIENT_ID")
    default_oidc_client_secret_ref = setting("OIDC_CLIENT_SECRET_REF") or setting(
        "KEYCLOAK_CLIENT_SECRET_REF"
    )
    try:
        default_oidc_client_secret = (
            resolve_runtime_secret_reference(default_oidc_client_secret_ref)
            if default_oidc_client_secret_ref
            else None
        )
        oauth_upstream_client_secret_ref = setting("OAUTH_UPSTREAM_CLIENT_SECRET_REF")
        default_oauth_upstream_client_secret = (
            resolve_runtime_secret_reference(oauth_upstream_client_secret_ref)
            if oauth_upstream_client_secret_ref
            else None
        )
        openapi_password_ref = setting("OPENAPI_PASSWORD_REF")
        default_openapi_password = (
            resolve_runtime_secret_reference(openapi_password_ref)
            if openapi_password_ref
            else None
        )
        openapi_client_secret_ref = setting("OPENAPI_CLIENT_SECRET_REF")
        default_openapi_client_secret = (
            resolve_runtime_secret_reference(openapi_client_secret_ref)
            if openapi_client_secret_ref
            else None
        )
    except RuntimeSecretReferenceError as exc:
        raise RuntimeError("configured authentication secret is unavailable") from exc

    return (
        default_oidc_config,
        default_oidc_client_id,
        default_oidc_client_secret,
        default_oauth_upstream_client_secret,
        default_openapi_password,
        default_openapi_client_secret,
    )


def _validate_transport_choices(transport_choices: tuple[str, ...]) -> None:
    if not transport_choices or any(
        transport not in _ALL_TRANSPORTS for transport in transport_choices
    ):
        raise ValueError(
            "transport_choices must be a non-empty subset of supported transports"
        )


def create_mcp_parser(
    *, transport_choices: tuple[str, ...] = _ALL_TRANSPORTS
) -> argparse.ArgumentParser:
    """Create a standard argument parser for MCP servers.

    Defines a comprehensive set of CLI flags for transport selection,
    host/port configuration, authentication (JWT, OIDC, OAuth), and Eunomia
    policy enforcement. Callers may narrow ``transport_choices`` when a server
    intentionally exposes a smaller current transport surface.

    Returns:
        An argparse.ArgumentParser instance.

    """
    (
        default_oidc_config,
        default_oidc_client_id,
        default_oidc_client_secret,
        default_oauth_upstream_client_secret,
        default_openapi_password,
        default_openapi_client_secret,
    ) = _resolve_mcp_parser_secret_defaults()
    _validate_transport_choices(transport_choices)

    parser = argparse.ArgumentParser(add_help=False, description="MCP Server")
    parser.add_argument(
        "-t",
        "--transport",
        default=DEFAULT_TRANSPORT,
        choices=list(transport_choices),
        help=f"Transport method: {', '.join(transport_choices)} (default: {DEFAULT_TRANSPORT})",
    )
    parser.add_argument(
        "-H",
        "--host",
        default=DEFAULT_HOST,
        help="Host address for HTTP transport (default: 127.0.0.1)",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port number for HTTP transport (default: 8000)",
    )
    parser.add_argument(
        "--tls-certfile",
        default=setting("MCP_TLS_CERTFILE"),
        help="Runtime path to the inbound TLS certificate (prefer secret projection)",
    )
    parser.add_argument(
        "--tls-keyfile",
        default=setting("MCP_TLS_KEYFILE"),
        help="Runtime path to the inbound TLS private key (prefer secret projection)",
    )
    parser.add_argument(
        "--tls-terminated",
        action="store_true",
        default=to_boolean(setting("MCP_TLS_TERMINATED", "False")),
        help="TLS is terminated by a peer restricted with --trusted-proxy-cidrs",
    )
    parser.add_argument(
        "--trusted-proxy-cidrs",
        default=setting("MCP_TRUSTED_PROXY_CIDRS"),
        help="Comma-separated exact ingress peer CIDRs used with --tls-terminated",
    )
    parser.add_argument(
        "--allowed-hosts",
        default=setting("MCP_ALLOWED_HOSTS"),
        help="Comma-separated exact Host-header allowlist for network transports",
    )
    parser.add_argument(
        "--allowed-origins",
        default=setting("MCP_ALLOWED_ORIGINS"),
        help="Comma-separated exact browser/WebSocket origins; unset blocks Origin requests",
    )
    parser.add_argument(
        "--max-request-bytes",
        type=int,
        default=int(setting("MCP_MAX_REQUEST_BYTES", str(4 * 1024 * 1024))),
        help="Maximum buffered MCP HTTP request body",
    )
    parser.add_argument(
        "--auth-type",
        default=setting("AUTH_TYPE", "none"),
        choices=["none", "static", "jwt", "oauth-proxy", "oidc-proxy", "remote-oauth"],
        help="Authentication type for MCP server: 'none' (disabled), 'static' (internal), 'jwt' (external token verification), 'oauth-proxy', 'oidc-proxy', 'remote-oauth' (external) (default: none)",
    )
    parser.add_argument(
        "--static-tokens-ref",
        default=setting("FASTMCP_SERVER_AUTH_STATIC_TOKENS_REF"),
        help="Secret reference containing the JSON map used by static authentication",
    )
    parser.add_argument(
        "--token-jwks-uri",
        default=setting("FASTMCP_SERVER_AUTH_JWT_JWKS_URI"),
        help="JWKS URI for JWT verification",
    )
    parser.add_argument(
        "--token-issuer",
        default=setting("FASTMCP_SERVER_AUTH_JWT_ISSUER"),
        help="Issuer for JWT verification",
    )
    parser.add_argument(
        "--token-audience",
        default=setting("FASTMCP_SERVER_AUTH_JWT_AUDIENCE"),
        help="Audience for JWT verification",
    )
    parser.add_argument(
        "--token-algorithm",
        default=setting("FASTMCP_SERVER_AUTH_JWT_ALGORITHM"),
        choices=[
            "HS256",
            "HS384",
            "HS512",
            "RS256",
            "RS384",
            "RS512",
            "ES256",
            "ES384",
            "ES512",
        ],
        help="JWT signing algorithm (required for HMAC or static key). Auto-detected for JWKS.",
    )
    parser.add_argument(
        "--token-secret-ref",
        default=setting("FASTMCP_SERVER_AUTH_JWT_SECRET_REF"),
        help="Runtime secret reference for an HMAC JWT verification secret",
    )
    parser.add_argument(
        "--token-public-key",
        default=setting("FASTMCP_SERVER_AUTH_JWT_PUBLIC_KEY"),
        help="Path to PEM public key file or inline PEM string (for static asymmetric keys).",
    )
    parser.add_argument(
        "--required-scopes",
        default=setting("FASTMCP_SERVER_AUTH_JWT_REQUIRED_SCOPES"),
        help="Comma-separated list of required scopes (e.g., gitlab.read,gitlab.write).",
    )
    parser.add_argument(
        "--oauth-upstream-auth-endpoint",
        default=None,
        help="Upstream authorization endpoint for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-upstream-token-endpoint",
        default=None,
        help="Upstream token endpoint for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-upstream-client-id",
        default=None,
        help="Upstream client ID for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-upstream-client-secret-ref",
        dest="oauth_upstream_client_secret",
        action=RuntimeSecretReferenceAction,
        default=default_oauth_upstream_client_secret,
        help="Runtime secret reference for the OAuth proxy client secret",
    )
    parser.add_argument(
        "--oauth-base-url", default=None, help="Base URL for OAuth Proxy"
    )
    parser.add_argument(
        "--oidc-config-url",
        default=default_oidc_config,
        help="OIDC configuration URL",
    )
    parser.add_argument(
        "--oidc-client-id",
        default=default_oidc_client_id,
        help="OIDC client ID",
    )
    parser.add_argument(
        "--oidc-client-secret-ref",
        dest="oidc_client_secret",
        action=RuntimeSecretReferenceAction,
        default=default_oidc_client_secret,
        help="Runtime secret reference for the OIDC client secret",
    )
    parser.add_argument(
        "--oidc-base-url",
        default=setting("OIDC_BASE_URL"),
        help="Base URL for OIDC Proxy",
    )
    parser.add_argument(
        "--public-base-url",
        default=setting("MCP_PUBLIC_BASE_URL"),
        help="Public base URL of this MCP server, used to advertise RFC 9728 "
        "protected-resource metadata for JWT auth (e.g. https://graph-os.example) "
        "so an RFC 9728-aware client can discover the authorization server and "
        "refresh its own token instead of a static JWT",
    )
    parser.add_argument(
        "--remote-auth-servers",
        default=None,
        help="Comma-separated list of authorization servers for Remote OAuth",
    )
    parser.add_argument(
        "--remote-base-url", default=None, help="Base URL for Remote OAuth"
    )
    parser.add_argument(
        "--allowed-client-redirect-uris",
        default=None,
        help="Comma-separated list of allowed client redirect URIs",
    )
    parser.add_argument(
        "--eunomia-type",
        default=setting("EUNOMIA_TYPE", "none"),
        choices=["none", "embedded", "remote"],
        help="Eunomia authorization type: 'none' (disabled), 'embedded' (built-in), 'remote' (external) (default: none)",
    )
    parser.add_argument(
        "--eunomia-policy-file",
        default=setting("EUNOMIA_POLICY_FILE", "mcp_policies.json"),
        help="Policy file for embedded Eunomia (default: mcp_policies.json)",
    )
    parser.add_argument(
        "--eunomia-remote-url",
        default=setting("EUNOMIA_REMOTE_URL", None),
        help="URL for remote Eunomia server",
    )
    parser.add_argument(
        "--eunomia-api-key-ref",
        default=setting("EUNOMIA_API_KEY_REF", None),
        help="Runtime secret reference for the remote Eunomia API key",
    )
    parser.add_argument(
        "--enable-delegation",
        action="store_true",
        default=to_boolean(setting("ENABLE_DELEGATION", "False")),
        help="Enable OIDC token delegation",
    )
    parser.add_argument(
        "--audience",
        default=setting("AUDIENCE", None),
        help="Audience for the delegated token",
    )
    parser.add_argument(
        "--delegated-scopes",
        default=setting("DELEGATED_SCOPES", "api"),
        help="Scopes for the delegated token (space-separated)",
    )
    parser.add_argument(
        "--openapi-file",
        default=None,
        help="Path to the OpenAPI JSON file to import additional tools from",
    )
    parser.add_argument(
        "--openapi-base-url",
        default=None,
        help="Base URL for the OpenAPI client (overrides instance URL)",
    )
    parser.add_argument(
        "--openapi-use-token",
        action="store_true",
        help="Use the incoming Bearer token (from MCP request) to authenticate OpenAPI import",
    )

    parser.add_argument(
        "--openapi-username",
        default=setting("OPENAPI_USERNAME"),
        help="Username for basic auth during OpenAPI import",
    )

    parser.add_argument(
        "--openapi-password-ref",
        dest="openapi_password",
        action=RuntimeSecretReferenceAction,
        default=default_openapi_password,
        help="Runtime secret reference for OpenAPI basic authentication",
    )

    parser.add_argument(
        "--openapi-client-id",
        default=setting("OPENAPI_CLIENT_ID"),
        help="OAuth client ID for OpenAPI import",
    )

    parser.add_argument(
        "--openapi-client-secret-ref",
        dest="openapi_client_secret",
        action=RuntimeSecretReferenceAction,
        default=default_openapi_client_secret,
        help="Runtime secret reference for the OpenAPI OAuth client secret",
    )

    parser.add_argument(
        "--tools",
        "--toolsets",
        default=setting("MCP_ENABLED_TOOLS"),
        help="Comma-separated list of enabled tools or toolsets to expose",
    )
    parser.add_argument(
        "--disabled-tools",
        "--disabled-toolsets",
        default=setting("MCP_DISABLED_TOOLS"),
        help="Comma-separated list of disabled tools or toolsets to exclude",
    )

    parser.add_argument("--help", action="store_true", help="Show usage")
    return parser


def _apply_mcp_auth_config_overrides(args: argparse.Namespace) -> None:
    """Layer parsed CLI auth/delegation flags onto the module-level
    ``mcp_auth_config`` dict (unchanged from the original inline block)."""
    mcp_auth_config["enable_delegation"] = args.enable_delegation
    mcp_auth_config["audience"] = args.audience or mcp_auth_config["audience"]
    mcp_auth_config["delegated_scopes"] = (
        args.delegated_scopes or mcp_auth_config["delegated_scopes"]
    )
    if hasattr(args, "oidc_config_url"):
        mcp_auth_config["oidc_config_url"] = (
            args.oidc_config_url or mcp_auth_config["oidc_config_url"]
        )
    if hasattr(args, "oidc_client_id"):
        mcp_auth_config["oidc_client_id"] = (
            args.oidc_client_id or mcp_auth_config["oidc_client_id"]
        )
    if hasattr(args, "oidc_client_secret"):
        mcp_auth_config["oidc_client_secret"] = (
            args.oidc_client_secret or mcp_auth_config["oidc_client_secret"]
        )


def _discover_delegation_token_endpoint(config_url: str) -> str:
    """OIDC-discover the token endpoint for token delegation. Raises
    ValueError on any failure (unavailable issuer, no token_endpoint)."""
    if not isinstance(config_url, str):
        raise ValueError("oidc_config_url must be a string")
    suffix = "/.well-known/openid-configuration"
    issuer = config_url[: -len(suffix)] if config_url.endswith(suffix) else config_url

    from agent_utilities.security.oidc_discovery import discover

    oidc_config = discover(issuer)
    token_endpoint = oidc_config.get("token_endpoint")
    if not token_endpoint:
        raise ValueError("No token_endpoint found in OIDC configuration")
    return token_endpoint


def _enforce_delegation_requirements(args: argparse.Namespace) -> None:
    """When token delegation is enabled, require oidc-proxy + a complete
    OIDC configuration, and resolve the delegation token endpoint."""
    if not mcp_auth_config["enable_delegation"]:
        return
    if args.auth_type != "oidc-proxy":
        logger.error("Error: Token delegation requires auth-type=oidc-proxy")
        sys.exit(1)
    if not mcp_auth_config["audience"]:
        logger.error("Error: audience is required for delegation")
        sys.exit(1)
    if not all(
        [
            mcp_auth_config["oidc_config_url"],
            mcp_auth_config["oidc_client_id"],
            mcp_auth_config["oidc_client_secret"],
        ]
    ):
        logger.error("Error: Delegation requires complete OIDC configuration")
        sys.exit(1)
    try:
        mcp_auth_config["token_endpoint"] = _discover_delegation_token_endpoint(
            mcp_auth_config["oidc_config_url"]
        )
    except Exception as exc:
        logger.error(
            "Failed to fetch OIDC configuration (exception_type=%s)",
            type(exc).__name__,
        )
        sys.exit(1)


def _validate_allowed_redirect_uris(args: argparse.Namespace) -> list[str] | None:
    try:
        return _validated_redirect_uris(args.allowed_client_redirect_uris)
    except ValueError:
        logger.error("Error: allowed client redirect URI policy is invalid")
        sys.exit(1)


def _resolve_static_tokens_ref(reference: str) -> Any:
    if reference.startswith("env://"):
        return setting(reference[len("env://") :])
    from agent_utilities.security.secrets_client import create_secrets_client

    return create_secrets_client().resolve_ref(reference)


def _validate_static_token_string(token: Any) -> None:
    if not isinstance(token, str) or len(token) < 32 or len(token) > 4096:
        raise ValueError("invalid token")


def _validate_static_claims_shape(claims: Any) -> tuple[str, list]:
    if not isinstance(claims, dict):
        raise ValueError("invalid claims")
    client_id = claims.get("client_id")
    scopes = claims.get("scopes", [])
    if not isinstance(client_id, str) or not 1 <= len(client_id) <= 256:
        raise ValueError("invalid client id")
    if not isinstance(scopes, list) or not all(
        isinstance(scope, str) and 1 <= len(scope) <= 256 for scope in scopes
    ):
        raise ValueError("invalid scopes")
    return client_id, scopes


def _validate_static_expiry(claims: dict) -> Any:
    expires_at = claims.get("expires_at")
    if expires_at is None:
        return None
    import math

    if (
        isinstance(expires_at, bool)
        or not isinstance(expires_at, int | float)
        or not math.isfinite(float(expires_at))
    ):
        raise ValueError("invalid expiry")
    return expires_at


def _validate_one_static_token_entry(token: Any, claims: Any) -> tuple[str, dict]:
    _validate_static_token_string(token)
    client_id, scopes = _validate_static_claims_shape(claims)
    expires_at = _validate_static_expiry(claims)
    entry: dict[str, Any] = {"client_id": client_id, "scopes": scopes}
    if expires_at is not None:
        entry["expires_at"] = expires_at
    return token, entry


def _validate_static_token_map(token_map: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(token_map, dict) or not token_map:
        raise ValueError("empty token map")
    validated: dict[str, dict[str, Any]] = {}
    for token, claims in token_map.items():
        key, entry = _validate_one_static_token_entry(token, claims)
        validated[key] = entry
    return validated


def _configure_static_auth(args: argparse.Namespace) -> Any:
    reference = str(getattr(args, "static_tokens_ref", "") or "").strip()
    if not reference:
        logger.error(
            "Error: static auth requires --static-tokens-ref; inline and "
            "built-in tokens are not permitted"
        )
        sys.exit(1)
    try:
        import json as _json

        raw_tokens = _resolve_static_tokens_ref(reference)
        token_map = _json.loads(str(raw_tokens or ""))
        validated = _validate_static_token_map(token_map)
    except Exception as exc:
        logger.error(
            "Error: static authentication token reference is unavailable or invalid (%s)",
            type(exc).__name__,
        )
        sys.exit(1)

    import hashlib as _hashlib
    import hmac as _hmac
    import secrets as _secrets
    import time as _time

    from fastmcp.server.auth import AccessToken, TokenVerifier

    class _ConstantTimeStaticVerifier(TokenVerifier):
        def __init__(self, tokens: dict[str, dict[str, Any]]) -> None:
            super().__init__()
            self._key = _secrets.token_bytes(32)
            self._entries = [
                (
                    _hmac.new(
                        self._key,
                        token.encode("utf-8"),
                        _hashlib.sha256,
                    ).digest(),
                    dict(claims),
                )
                for token, claims in tokens.items()
            ]

        async def verify_token(self, token: str) -> Any:
            if not isinstance(token, str) or len(token) > 4_096:
                return None
            candidate = _hmac.new(
                self._key,
                token.encode("utf-8"),
                _hashlib.sha256,
            ).digest()
            matched: dict[str, Any] | None = None
            for expected, claims in self._entries:
                if _secrets.compare_digest(candidate, expected):
                    matched = claims
            if matched is None:
                return None
            expires_at = matched.get("expires_at")
            if expires_at is not None and float(expires_at) < _time.time():
                return None
            return AccessToken(
                token=token,
                client_id=matched["client_id"],
                scopes=list(matched.get("scopes", [])),
                expires_at=int(expires_at) if expires_at is not None else None,
                claims=matched,
            )

    return _ConstantTimeStaticVerifier(validated)


def _configure_oauth_proxy_auth(
    args: argparse.Namespace, allowed_uris: list[str] | None
) -> Any:
    from fastmcp.server.auth import OAuthProxy

    if not all(
        [
            args.oauth_upstream_auth_endpoint,
            args.oauth_upstream_token_endpoint,
            args.oauth_upstream_client_id,
            args.oauth_upstream_client_secret,
            args.oauth_base_url,
            args.token_jwks_uri,
            args.token_issuer,
            args.token_audience,
        ]
    ):
        logger.error(
            "Error: oauth-proxy requires all upstream endpoints and JWT params"
        )
        sys.exit(1)
    try:
        upstream_auth = _secure_auth_url(
            args.oauth_upstream_auth_endpoint,
            field="OAuth authorization endpoint",
        )
        upstream_token = _secure_auth_url(
            args.oauth_upstream_token_endpoint,
            field="OAuth token endpoint",
        )
        base_url = _secure_auth_url(args.oauth_base_url, field="OAuth base URL")
    except ValueError:
        logger.error("Error: OAuth proxy URL policy is invalid")
        sys.exit(1)
    token_verifier = _hardened_jwt_verifier(
        jwks_uri=args.token_jwks_uri,
        issuer=args.token_issuer,
        audience=args.token_audience,
    )
    return OAuthProxy(
        upstream_authorization_endpoint=upstream_auth,
        upstream_token_endpoint=upstream_token,
        upstream_client_id=args.oauth_upstream_client_id,
        upstream_client_secret=args.oauth_upstream_client_secret,
        token_verifier=token_verifier,
        base_url=base_url,
        allowed_client_redirect_uris=allowed_uris,
    )


def _configure_oidc_proxy_auth(
    args: argparse.Namespace, allowed_uris: list[str] | None
) -> Any:
    from fastmcp.server.auth.oidc_proxy import OIDCProxy

    if not all(
        [
            args.oidc_config_url,
            args.oidc_client_id,
            args.oidc_client_secret,
            args.oidc_base_url,
        ]
    ):
        logger.error(
            "Error: oidc-proxy requires OIDC metadata, client identity, "
            "a resolved client-secret reference, and base URL"
        )
        sys.exit(1)
    inbound_audience = args.token_audience or args.audience
    if not inbound_audience:
        logger.error("Error: oidc-proxy requires an explicit token audience")
        sys.exit(1)
    try:
        config_url = _secure_auth_url(
            args.oidc_config_url, field="OIDC configuration URL"
        )
        base_url = _secure_auth_url(args.oidc_base_url, field="OIDC base URL")
    except ValueError:
        logger.error("Error: OIDC proxy URL policy is invalid")
        sys.exit(1)
    return OIDCProxy(
        config_url=config_url,
        client_id=args.oidc_client_id,
        client_secret=args.oidc_client_secret,
        audience=inbound_audience,
        base_url=base_url,
        allowed_client_redirect_uris=allowed_uris,
    )


def _configure_remote_oauth_auth(args: argparse.Namespace) -> Any:
    from fastmcp.server.auth import RemoteAuthProvider
    from pydantic import AnyHttpUrl

    if not all(
        [
            args.remote_auth_servers,
            args.remote_base_url,
            args.token_jwks_uri,
            args.token_issuer,
            args.token_audience,
        ]
    ):
        logger.error(
            "Error: remote-oauth requires remote-auth-servers, remote-base-url, and JWT params"
        )
        sys.exit(1)
    try:
        auth_servers = [
            _secure_auth_url(url.strip(), field="authorization server")
            for url in args.remote_auth_servers.split(",")
            if url.strip()
        ]
        remote_base_url = _secure_auth_url(
            args.remote_base_url, field="remote OAuth base URL"
        )
    except ValueError:
        logger.error("Error: remote OAuth URL policy is invalid")
        sys.exit(1)
    if not auth_servers or len(auth_servers) > 16:
        logger.error("Error: remote authorization-server policy is invalid")
        sys.exit(1)
    token_verifier = _hardened_jwt_verifier(
        jwks_uri=args.token_jwks_uri,
        issuer=args.token_issuer,
        audience=args.token_audience,
    )
    return RemoteAuthProvider(
        token_verifier=token_verifier,
        authorization_servers=[AnyHttpUrl(u) for u in auth_servers],
        base_url=remote_base_url,
    )


def _configure_auth(args: argparse.Namespace) -> Any:
    """Configure authentication provider based on parsed CLI args.

    Returns the auth provider instance, or ``None`` only for the explicitly
    recognised "no authentication" values (``"none"`` / empty). Any other
    unrecognised ``auth_type`` raises :class:`UnsupportedAuthTypeError` rather
    than degrading to an unauthenticated server (BUG-CX-023).
    """
    if args.auth_type == "none" or not args.auth_type:
        return None

    _apply_mcp_auth_config_overrides(args)
    _enforce_delegation_requirements(args)
    allowed_uris = _validate_allowed_redirect_uris(args)

    if args.auth_type == "none":
        return None
    if args.auth_type == "static":
        return _configure_static_auth(args)
    if args.auth_type == "jwt":
        return _configure_jwt_auth(args)
    if args.auth_type == "oauth-proxy":
        return _configure_oauth_proxy_auth(args, allowed_uris)
    if args.auth_type == "oidc-proxy":
        return _configure_oidc_proxy_auth(args, allowed_uris)
    if args.auth_type == "remote-oauth":
        return _configure_remote_oauth_auth(args)
    logger.error(
        "Refusing to build an MCP server for unsupported auth type %r", args.auth_type
    )
    raise UnsupportedAuthTypeError(
        f"unsupported auth type {args.auth_type!r}; expected one of "
        "none, static, jwt, oauth-proxy, oidc-proxy, remote-oauth"
    )


def _resolve_jwt_basic_params(
    args: argparse.Namespace,
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    """(jwks_uri, issuer, audience, algorithm, secret_or_key) from CLI args
    with their environment-variable fallbacks. OIDC_ISSUER is the
    canonical, provider-agnostic var; FASTMCP_SERVER_AUTH_JWT_ISSUER is the
    per-server alias. Either may be a comma-separated multi-issuer list
    (OS-5.45)."""
    jwks_uri = args.token_jwks_uri or setting("FASTMCP_SERVER_AUTH_JWT_JWKS_URI")
    issuer = (
        setting("OIDC_ISSUER")
        or args.token_issuer
        or setting("FASTMCP_SERVER_AUTH_JWT_ISSUER")
    )
    audience = args.token_audience or setting("FASTMCP_SERVER_AUTH_JWT_AUDIENCE")
    algorithm = args.token_algorithm
    secret_or_key = args.token_public_key
    return jwks_uri, issuer, audience, algorithm, secret_or_key


def _oidc_discover_jwks_uris(issuer: str) -> list[str | None]:
    from agent_utilities.security.oidc_discovery import jwks_uri_for

    return [jwks_uri_for(i.strip()) for i in str(issuer).split(",") if i.strip()]


def _maybe_discover_jwks(
    jwks_uri: str | None,
    secret_or_key: str | None,
    args: argparse.Namespace,
    issuer: str | None,
) -> str | None:
    """CONCEPT:AU-OS.identity.resolve-token-endpoint-from -- IdP-agnostic:
    with no explicit JWKS URI (and not a static key), resolve it from each
    issuer's OIDC discovery document, so config carries only the issuer
    (Keycloak, Okta, Auth0, Entra, ...) -- never a vendor-specific path.

    ``args.token_secret_ref`` is read lazily, INSIDE this same short-
    circuited condition (not pre-extracted at the call site), matching the
    original inline expression's evaluation order exactly: a real
    argparse.Namespace always defines the attribute, but a caller with a
    truthy ``jwks_uri``/``secret_or_key`` (as in this repo's own JWT test
    fixtures) never needed to.
    """
    if jwks_uri or secret_or_key or args.token_secret_ref or not issuer:
        return jwks_uri
    resolved = _oidc_discover_jwks_uris(issuer)
    if resolved and all(resolved):
        return ",".join(u for u in resolved if u)
    if any(resolved):
        logger.warning(
            "OIDC discovery resolved JWKS for only part of the issuer policy; "
            "configure the JWKS policy explicitly"
        )
    return jwks_uri


def _validate_jwt_key_material_present(
    jwks_uri: str | None, secret_or_key: str | None, args: argparse.Namespace
) -> None:
    """``args.token_secret_ref`` is read lazily -- see :func:`_maybe_discover_jwks`."""
    if not (jwks_uri or secret_or_key or args.token_secret_ref):
        logger.error(
            "Error: JWT auth requires --token-jwks-uri, --token-public-key, "
            "or --token-secret-ref"
        )
        sys.exit(1)


def _validate_jwt_issuer_audience_present(
    issuer: str | None, audience: str | None
) -> None:
    if not (issuer and audience):
        logger.error("Error: JWT requires --token-issuer and --token-audience")
        sys.exit(1)


def _validate_jwt_audience_format(audience: Any) -> None:
    if (
        not isinstance(audience, str)
        or not 1 <= len(audience) <= 512
        or any(character in audience for character in "\r\n\x00")
    ):
        logger.error("Error: JWT audience is invalid")
        sys.exit(1)


def _normalize_jwt_issuer_and_jwks(
    issuer: str | None, jwks_uri: str | None
) -> tuple[str, str | None, list[str]]:
    try:
        issuer_values = [
            _secure_auth_url(value.strip(), field="JWT issuer")
            for value in str(issuer).split(",")
            if value.strip()
        ]
        if not issuer_values:
            raise ValueError("missing issuer")
        issuer = ",".join(issuer_values)
        if jwks_uri:
            jwks_values = [
                _secure_auth_url(value.strip(), field="JWKS URI")
                for value in str(jwks_uri).split(",")
                if value.strip()
            ]
            if not jwks_values:
                raise ValueError("missing JWKS URI")
            jwks_uri = ",".join(jwks_values)
    except ValueError:
        logger.error("Error: JWT issuer/JWKS URL policy is invalid")
        sys.exit(1)
    return issuer, jwks_uri, issuer_values


def _resolve_jwt_public_key_pem(args: argparse.Namespace) -> str | None:
    if args.token_public_key and os.path.isfile(args.token_public_key):
        try:
            with open(args.token_public_key) as f:
                return f.read()
        except Exception as exc:
            logger.error(
                "Failed to read public key file (exception_type=%s)",
                type(exc).__name__,
            )
            sys.exit(1)
    elif args.token_public_key:
        return args.token_public_key
    return None


def _resolve_jwt_secret_or_key(
    args: argparse.Namespace, algorithm: str | None, public_key_pem: str | None
) -> Any:
    if not (algorithm and algorithm.startswith("HS")):
        return public_key_pem
    secret_ref = str(getattr(args, "token_secret_ref", "") or "").strip()
    if not secret_ref:
        logger.error("Error: HMAC JWT verification requires --token-secret-ref")
        sys.exit(1)
    try:
        from agent_utilities.security.secrets_client import create_secrets_client

        public_key = create_secrets_client().resolve_ref(secret_ref)
        if not isinstance(public_key, str) or len(public_key.encode()) < 32:
            raise ValueError("weak secret")
    except Exception:
        logger.error("Error: HMAC JWT secret reference is unavailable")
        sys.exit(1)
    return public_key


def _parse_required_scopes(args: argparse.Namespace) -> list[str] | None:
    if not args.required_scopes:
        return None
    return _split_csv(args.required_scopes)


def _build_multi_realm_verifiers(
    jwks_uris: list[str],
    issuers: list[str],
    audience: str | None,
    algorithm: str | None,
    required_scopes: list[str] | None,
) -> list[Any]:
    return [
        _hardened_jwt_verifier(
            jwks_uri=u,
            issuer=i,
            audience=audience,
            algorithm=algorithm or "RS256",
            required_scopes=required_scopes,
        )
        for u, i in zip(jwks_uris, issuers, strict=False)
    ]


def _configure_multi_realm_jwt(
    jwks_uris: list[str],
    issuers: list[str],
    audience: str | None,
    algorithm: str | None,
    required_scopes: list[str] | None,
    args: argparse.Namespace,
) -> Any:
    """CONCEPT:AU-OS.identity.native-multi-realm-jwt -- native multi-realm
    JWT trust. FASTMCP_SERVER_AUTH_JWT_ISSUER and _JWKS_URI may each be a
    comma-separated, aligned-by-index list (one Keycloak realm per entry).
    FastMCP's JWTVerifier accepts an issuer list but only a SINGLE
    jwks_uri (one realm's signing keys), so during a realm migration the
    fleet must trust two realms' signing keys at once. Build one
    JWTVerifier per realm and accept a token if ANY verifies it --
    enabling a zero-downtime issuer cutover (add new realm -> flip the
    minter -> drop the old realm) with no signature gap and no auth
    lock-out window."""
    if len(jwks_uris) != len(issuers):
        logger.error(
            "Multi-realm JWT auth requires FASTMCP_SERVER_AUTH_JWT_ISSUER and "
            "FASTMCP_SERVER_AUTH_JWT_JWKS_URI to be comma-separated lists of EQUAL length "
            "(one entry per realm, aligned by index)."
        )
        sys.exit(1)

    from fastmcp.server.auth import TokenVerifier

    class _MultiIssuerVerifier(TokenVerifier):
        """Accept a JWT validated by ANY wrapped per-realm JWTVerifier (CONCEPT:AU-OS.identity.native-multi-realm-jwt)."""

        def __init__(self, verifiers: list, *, required_scopes: Any = None) -> None:
            super().__init__(required_scopes=required_scopes)
            self._verifiers = list(verifiers)

        async def verify_token(self, token: str) -> Any:
            for _v in self._verifiers:
                try:
                    _result = await _v.verify_token(token)
                except Exception:  # noqa: BLE001 - try the next realm
                    _result = None
                if _result is not None:
                    return _result
            return None

    try:
        verifiers = _build_multi_realm_verifiers(
            jwks_uris, issuers, audience, algorithm, required_scopes
        )
        logger.info("JWT auth: native multi-issuer trust configured")
        return _wrap_with_resource_metadata(
            _MultiIssuerVerifier(verifiers, required_scopes=required_scopes),
            args,
            issuers,
        )
    except Exception as exc:
        logger.error(
            "Failed to initialize multi-realm JWTVerifier (exception_type=%s)",
            type(exc).__name__,
        )
        sys.exit(1)


def _configure_single_realm_jwt(
    jwks_uri: str | None,
    public_key: Any,
    issuer: str,
    audience: str | None,
    algorithm: str | None,
    required_scopes: list[str] | None,
    args: argparse.Namespace,
    issuer_values: list[str],
) -> Any:
    try:
        verifier = _hardened_jwt_verifier(
            jwks_uri=jwks_uri,
            public_key=public_key,
            issuer=issuer,
            audience=audience,
            algorithm=algorithm or "RS256",
            required_scopes=required_scopes,
        )
        return _wrap_with_resource_metadata(verifier, args, issuer_values)
    except Exception as exc:
        logger.error(
            "Failed to initialize JWTVerifier (exception_type=%s)",
            type(exc).__name__,
        )
        sys.exit(1)


def _configure_jwt_auth(args: argparse.Namespace) -> Any:
    """Configure JWT authentication from CLI args."""
    jwks_uri, issuer, audience, algorithm, secret_or_key = _resolve_jwt_basic_params(
        args
    )
    jwks_uri = _maybe_discover_jwks(jwks_uri, secret_or_key, args, issuer)
    _validate_jwt_key_material_present(jwks_uri, secret_or_key, args)
    _validate_jwt_issuer_audience_present(issuer, audience)
    _validate_jwt_audience_format(audience)
    issuer, jwks_uri, issuer_values = _normalize_jwt_issuer_and_jwks(issuer, jwks_uri)

    public_key_pem = _resolve_jwt_public_key_pem(args)
    public_key = _resolve_jwt_secret_or_key(args, algorithm, public_key_pem)
    required_scopes = _parse_required_scopes(args)

    issuers = _split_csv(str(issuer)) if issuer else []
    jwks_uris = _split_csv(str(jwks_uri)) if jwks_uri else []
    if len(jwks_uris) > 1 or len(issuers) > 1:
        return _configure_multi_realm_jwt(
            jwks_uris, issuers, audience, algorithm, required_scopes, args
        )
    return _configure_single_realm_jwt(
        jwks_uri,
        public_key,
        issuer,
        audience,
        algorithm,
        required_scopes,
        args,
        issuer_values,
    )


def _rate_limit_client_id(context: Any) -> str:
    """Per-caller bucket key for :class:`RateLimitingMiddleware`.

    Without this, ``RateLimitingMiddleware(get_client_id=None)`` buckets
    EVERY request from EVERY caller under the single literal key
    ``"global"`` (``fastmcp.server.middleware.rate_limiting
    .RateLimitingMiddleware._get_client_identifier``) — one shared
    20-token / 10-req/s budget for the WHOLE server, across every session
    and every caller, covering every request type (``initialize``,
    ``notifications/initialized``, ``tools/list``, ``tools/call`` — not
    just tool calls). This fleet routinely runs several concurrent MCP
    clients against graph-os at once (multiple agent lanes, the harness,
    service-account bridges); a handful of them handshaking within the
    same second exhausts the shared bucket, and each rejected request
    surfaces as a JSON-RPC ``-32000 "Rate limit exceeded for client:
    global"`` error. A caller that does not check the ``error`` field
    before reading ``result.tools`` mistakes that for a genuinely empty
    ``tools/list`` — this module's own multiplexer already documents a
    fleet child tripping the identical "Rate limit exceeded for client:
    global" message under a bulk probe (see the skill-harvest backoff
    comment in ``multiplexer.py``), so this is a known, live failure
    shape for this exact middleware, not a hypothetical one.

    Keying per authenticated caller preserves the same PER-CALLER budget
    (a single runaway/abusive client is still throttled) while stopping
    unrelated legitimate callers from starving each other on a shared
    bucket none of them knows exists. Falls back to one anonymous bucket
    only for requests with no resolvable identity (stdio, or an
    unauthenticated HTTP caller) — those already share fate under the
    same trust boundary.
    """
    import hashlib

    from fastmcp.server.dependencies import get_access_token

    try:
        token = get_access_token()
    except Exception:
        token = None
    if token is None:
        return "anonymous"
    claims = getattr(token, "claims", None) or {}
    raw = "\x00".join(
        str(value or "")
        for value in (
            getattr(token, "client_id", None),
            claims.get("sub") if isinstance(claims, dict) else None,
            claims.get("tenant_id") if isinstance(claims, dict) else None,
        )
    )
    if not raw.strip("\x00"):
        return "anonymous"
    return "caller_" + hashlib.blake2s(raw.encode("utf-8"), digest_size=16).hexdigest()


def _import_optional_middlewares() -> tuple[Any, Any, Any, Any, Any]:
    """(UserTokenMiddleware, JWTClaimsLoggingMiddleware, EntityLinkingMiddleware,
    ToolMetricsMiddleware, ActorContextMiddleware), each None if the optional
    middlewares package is unavailable."""
    try:
        from agent_utilities.mcp.middlewares import (
            ActorContextMiddleware,
            EntityLinkingMiddleware,
            JWTClaimsLoggingMiddleware,
            ToolMetricsMiddleware,
            UserTokenMiddleware,
        )
    except ImportError:
        return None, None, None, None, None
    return (
        UserTokenMiddleware,
        JWTClaimsLoggingMiddleware,
        EntityLinkingMiddleware,
        ToolMetricsMiddleware,
        ActorContextMiddleware,
    )


def _append_optional_middlewares(
    middlewares: list[Any],
    server_name: str,
    optional_classes: tuple[Any, Any, Any, Any, Any],
) -> None:
    (
        UserTokenMiddleware,
        JWTClaimsLoggingMiddleware,
        EntityLinkingMiddleware,
        ToolMetricsMiddleware,
        ActorContextMiddleware,
    ) = optional_classes

    # Scope every tool call to the caller's validated OIDC (Okta/Keycloak)
    # identity so servers can auto-load resources and inherit authz per-caller
    # (CONCEPT:AU-OS.identity.idp-agnostic-role-inheritance). No-op when the
    # request carries no validated token (loopback/stdio trust only) --
    # EXCEPT for graph-os itself, which fails closed instead (BUG-036/GOC-15,
    # see ActorContextMiddleware/`_configure_middleware` docstrings).
    if ActorContextMiddleware is not None:
        middlewares.append(
            ActorContextMiddleware(require_verified_session=(server_name == "graph-os"))
        )

    # Per-tool Prometheus metrics (count/latency/error) for this server, scraped
    # from its own /metrics route (CONCEPT:AU-OS.observability.no-op-without-metrics). No-op without the metrics extra.
    if ToolMetricsMiddleware is not None:
        middlewares.append(ToolMetricsMiddleware())

    if JWTClaimsLoggingMiddleware is not None:
        pass  # Also do not add this as it logs to stdout

    if EntityLinkingMiddleware is not None:
        middlewares.append(EntityLinkingMiddleware())

    if mcp_auth_config["enable_delegation"] and UserTokenMiddleware is not None:
        # Keep the privacy-safe error middleware outermost so a delegation
        # rejection cannot bypass the standardized exception surface.
        middlewares.insert(1, UserTokenMiddleware(config=mcp_auth_config))


def _configure_eunomia_middleware(args: argparse.Namespace) -> Any | None:
    """None when Eunomia is not configured for this server; the built
    middleware when it is; exits the process on a configuration/build
    failure (never returns a sentinel for that case)."""
    if args.eunomia_type not in ["embedded", "remote"]:
        return None
    try:
        from agent_utilities.mcp.eunomia_principal import create_eunomia_middleware

        require_verified = str(getattr(args, "auth_type", "none")) != "none"
        if args.eunomia_type == "remote":
            return create_eunomia_middleware(
                policy_file=None,
                use_remote_eunomia=True,
                eunomia_endpoint=args.eunomia_remote_url,
                api_key_ref=getattr(args, "eunomia_api_key_ref", None),
                require_verified_principal=require_verified,
            )
        policy_file = args.eunomia_policy_file or "mcp_policies.json"
        return create_eunomia_middleware(
            policy_file=policy_file,
            use_remote_eunomia=False,
            require_verified_principal=require_verified,
        )
    except Exception as exc:
        logger.error(
            "Failed to load Eunomia middleware (exception_type=%s)",
            type(exc).__name__,
        )
        sys.exit(1)


def _configure_middleware(
    args: argparse.Namespace, *, server_name: str = ""
) -> list[Any]:
    """Build the standard middleware stack for an MCP server.

    ``server_name`` (the same ``name`` passed to :func:`create_mcp_server`)
    selects :class:`~agent_utilities.mcp.middlewares.ActorContextMiddleware`'s
    fail-closed mode (BUG-036/GOC-15): only the ``"graph-os"`` server —
    the one MCP server in the fleet whose tools reach privileged Knowledge-
    Graph reads/writes — gets ``require_verified_session=True``. Every other
    fleet server (~60 independently-owned ``agents/*-mcp`` packages using this
    same factory) keeps the prior no-op-without-a-token behavior unchanged;
    making this fail closed fleet-wide is a separable, larger change that
    needs its own per-package audit, not a side effect of closing BUG-036.
    """
    from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
    from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware

    middlewares: list[Any] = [
        ErrorHandlingMiddleware(include_traceback=False, transform_errors=True),
        RateLimitingMiddleware(
            max_requests_per_second=10.0,
            burst_capacity=20,
            get_client_id=_rate_limit_client_id,
        ),
    ]
    _append_optional_middlewares(
        middlewares, server_name, _import_optional_middlewares()
    )

    eunomia_mw = _configure_eunomia_middleware(args)
    if eunomia_mw is not None:
        middlewares.append(eunomia_mw)

    return middlewares


# ── Stdio JSON-RPC purity ────────────────────────────────────────────────────
#
# There used to be a ``protect_stdio_jsonrpc()`` here: a permanent, process-wide
# monkeypatch of ``builtins.print``/``warnings.showwarning`` applied once and
# never undone (B-19). It intercepted every caller in the process to protect one
# file descriptor — global where the concern was local, unscoped, un-nestable,
# no teardown — and it is the confirmed root cause of a cross-file test-pollution
# incident: a test that drove the real stdio entrypoint left ``print`` silently
# redirected for the rest of the session, breaking ~23 unrelated tests in 11
# files with no plausible causal link.
#
# It also turned out to be solving an already-solved problem. On the stdio
# transport, ``mcp.run(transport="stdio")`` (FastMCP's ``run_stdio_async``, see
# ``fastmcp/server/mixins/transport.py``) enters ``mcp.server.stdio.stdio_server()``
# (vendored MCP SDK, ``mcp/server/stdio.py``), which ALREADY gives the protocol
# writer exclusive, fd-level ownership of real stdout for the scope of serving:
# it ``os.dup()``s fd 1 to a private descriptor for the JSON-RPC writer, then
# ``os.dup2()``s a duplicate of stderr onto fd 1 so every OTHER writer in the
# process — ``print()``, a bare ``sys.stdout.write()``, a C extension writing
# directly to the fd, a subprocess that inherits it — lands on stderr instead,
# and restores both descriptors in a ``finally`` on exit. Because the diversion
# is at the OS file-descriptor level, not a Python-object patch, it needs no
# teardown bookkeeping here: nothing in this codebase ever touches
# ``builtins.print`` or ``sys.stdout``, so there is nothing to save or restore,
# and a second concurrent ``stdio_server()`` in one process raises loudly
# instead of silently double-diverting (the SDK's own ``_claims`` registry).
# ``warnings.showwarning`` needed no interception either — its stdlib default
# already targets ``sys.stderr`` when no explicit ``file=`` is given.
#
# What this repo still owns: never introduce a ``print()`` in the served MCP
# surface in the first place (there is no runtime net for code that runs BEFORE
# ``mcp.run()`` claims the descriptor, e.g. engine bootstrap or a co-service
# thread started moments earlier). That is enforced statically, in the fast
# pre-commit tier, by ``scripts/check_no_stdout_writes.py`` (CONCEPT:AU-ECO.mcp.stdio-static-purity-gate) —
# catching the offender at authoring time instead of corrupting a frame at 3am.
# Diagnostics in served code route through the standard ``logging`` module
# (stderr by default), never ``print``.


def _resolve_network_allowed_hosts(
    args: argparse.Namespace, loopback: bool, normalize_host_authorities: Any
) -> list[str]:
    hosts = _split_csv(str(getattr(args, "allowed_hosts", "") or ""))
    if not hosts:
        if not loopback:
            raise RuntimeError("MCP_ALLOWED_HOSTS is required")
        port = int(getattr(args, "port", 8000))
        hosts = [
            f"localhost:{port}",
            f"127.0.0.1:{port}",
            f"[::1]:{port}",
            "testserver",
        ]
    try:
        return sorted(normalize_host_authorities(hosts))
    except ValueError:
        raise RuntimeError("MCP_ALLOWED_HOSTS must contain exact authorities") from None


def _build_network_base_middleware(
    hosts: list[str],
    origin_values: list[str],
    max_bytes: int,
    Middleware: Any,
    ExactHostAuthorityMiddleware: Any,
    OriginPolicyMiddleware: Any,
    BoundedRequestBodyMiddleware: Any,
) -> list[Any]:
    return [
        Middleware(ExactHostAuthorityMiddleware, allowed_hosts=hosts),
        Middleware(OriginPolicyMiddleware, allowed_origins=origin_values),
        Middleware(BoundedRequestBodyMiddleware, max_bytes=max_bytes),
    ]


def _maybe_prepend_trusted_proxy_middleware(
    args: argparse.Namespace,
    middleware: list[Any],
    parse_cidrs: Any,
    Middleware: Any,
    TrustedProxyPeerMiddleware: Any,
) -> None:
    if not bool(getattr(args, "tls_terminated", False)):
        return
    cidrs = _split_csv(str(getattr(args, "trusted_proxy_cidrs", "") or ""))
    parse_cidrs(cidrs)
    middleware.insert(0, Middleware(TrustedProxyPeerMiddleware, trusted_cidrs=cidrs))


def _resolve_network_connection_bounds() -> tuple[int, int]:
    max_connections = int(setting("MCP_MAX_CONNECTIONS", "128"))
    backlog = int(setting("MCP_LISTEN_BACKLOG", "256"))
    if not 1 <= max_connections <= 10_000 or not 1 <= backlog <= 65_535:
        raise RuntimeError("MCP listener resource bounds are invalid")
    return max_connections, backlog


def _build_uvicorn_config(
    args: argparse.Namespace, max_connections: int, backlog: int
) -> dict[str, Any]:
    uvicorn_config: dict[str, Any] = {
        # Never trust Forwarded/X-Forwarded-* from the network. The direct peer
        # address remains authoritative for the trusted-ingress CIDR gate.
        "proxy_headers": False,
        "timeout_keep_alive": 5,
        "timeout_graceful_shutdown": 15,
        "limit_concurrency": max_connections,
        "backlog": backlog,
        "h11_max_incomplete_event_size": 65_536,
    }
    certfile = str(getattr(args, "tls_certfile", "") or "").strip()
    keyfile = str(getattr(args, "tls_keyfile", "") or "").strip()
    if certfile and keyfile:
        uvicorn_config.update(ssl_certfile=certfile, ssl_keyfile=keyfile)
    return uvicorn_config


def mcp_network_run_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Return the one hardened FastMCP/Uvicorn network-serving configuration."""
    from starlette.middleware import Middleware

    from agent_utilities.security.http_boundary import (
        BoundedRequestBodyMiddleware,
        ExactHostAuthorityMiddleware,
        OriginPolicyMiddleware,
        TrustedProxyPeerMiddleware,
        normalize_host_authorities,
        normalize_origins,
        parse_cidrs,
    )

    transport = str(getattr(args, "transport", "stdio") or "stdio").lower()
    if transport not in _NETWORK_TRANSPORTS:
        return {}
    host = str(getattr(args, "host", "") or "")
    loopback = _is_loopback_bind(host)
    hosts = _resolve_network_allowed_hosts(args, loopback, normalize_host_authorities)

    origin_values = _split_csv(str(getattr(args, "allowed_origins", "") or ""))
    normalize_origins(origin_values)  # validate before the listener starts

    max_bytes = int(getattr(args, "max_request_bytes", 4 * 1024 * 1024))
    middleware = _build_network_base_middleware(
        hosts,
        origin_values,
        max_bytes,
        Middleware,
        ExactHostAuthorityMiddleware,
        OriginPolicyMiddleware,
        BoundedRequestBodyMiddleware,
    )
    _maybe_prepend_trusted_proxy_middleware(
        args, middleware, parse_cidrs, Middleware, TrustedProxyPeerMiddleware
    )

    max_connections, backlog = _resolve_network_connection_bounds()
    uvicorn_config = _build_uvicorn_config(args, max_connections, backlog)
    return {"middleware": middleware, "uvicorn_config": uvicorn_config}


# ── Fleet server registry self-registration (CONCEPT:EG-KG.sharding.server-registry, W2.5) ──
#
# Every server built via ``create_mcp_server`` self-registers a REAL, queryable
# ``:Server`` graph node with the engine at startup and renews its lease on a
# heartbeat cadence for the server's whole lifetime — wired here (the ONE
# factory ~62 fleet MCP servers already call), not per-server, so it is native
# by default with zero per-server opt-in. Killing the process simply stops the
# heartbeats; the engine's own stale-lease reaper expires the row once the
# lease lapses (`epistemic-graph` `src/server/registry_reaper.rs`).
#
# The au config-file sync (`knowledge_graph.core.engine_ingestion.
# ingest_mcp_server`) remains a RECONCILER over the SAME `:Server` shape — it
# repairs a hand-broken or never-self-registered row — but this is now the
# PRIMARY, live-identity writer.
_FLEET_REGISTRATION_DEFAULT_TTL_SECS = 300
_FLEET_REGISTRATION_MIN_TTL_SECS = 60


def _fleet_registration_ttl_secs() -> int:
    """Positive lease TTL, seconds (``MCP_FLEET_REGISTRATION_TTL_SECS`` overrides).

    Heartbeats renew at roughly a third of the TTL (see
    :func:`_register_and_heartbeat_forever`), so a transient miss or two never
    expires a healthy server.
    """
    try:
        value = int(
            setting(
                "MCP_FLEET_REGISTRATION_TTL_SECS",
                _FLEET_REGISTRATION_DEFAULT_TTL_SECS,
            )
        )
    except (TypeError, ValueError):
        return _FLEET_REGISTRATION_DEFAULT_TTL_SECS
    return (
        value
        if value >= _FLEET_REGISTRATION_MIN_TTL_SECS
        else (_FLEET_REGISTRATION_DEFAULT_TTL_SECS)
    )


def _fleet_registration_endpoint_reference(args: argparse.Namespace, name: str) -> str:
    """A bounded, privacy-safe reference to how this server is reached.

    Never a raw credentialed URL — an opaque marker the engine and the au
    reconciler both treat as a reference, not a dereferenceable address (the
    same convention au's ``persistence_reference``/``mcp-ref://`` values use).
    """
    transport = str(getattr(args, "transport", "stdio") or "stdio")
    if transport in _NETWORK_TRANSPORTS:
        host = str(getattr(args, "host", "") or DEFAULT_HOST)
        port = getattr(args, "port", 0) or 0
        return f"{transport}://{host}:{port}"
    return f"stdio://{name}"


async def _register_and_heartbeat_forever(name: str, url: str, ttl_secs: int) -> None:
    """Self-register, then renew ``name``'s lease forever on a cadence well
    inside ``ttl_secs`` (CONCEPT:EG-KG.sharding.server-registry, W2.5).

    Best-effort: an unreachable engine (a ``tiny`` profile with no graph
    access yet, a dev/test process, a cold start racing the engine's own
    boot) never crashes or blocks the server — a failed attempt is logged at
    debug level and retried on the SAME cadence, self-healing the moment the
    engine becomes reachable.
    """
    interval = max(1, ttl_secs // 3)
    resources = {
        "transport": str(setting("TRANSPORT", "stdio") or "stdio"),
        "pid": os.getpid(),
    }
    while True:
        try:
            from agent_utilities.knowledge_graph.core.graph_compute import (
                GraphComputeEngine,
            )

            engine = GraphComputeEngine.get_or_create()
            await engine.async_client.server_registry.register(
                name, url, resources=resources, ttl_secs=ttl_secs
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — self-registration retry loop, will retry next iteration
            logger.debug(
                "Fleet self-registration attempt failed, will retry: %s",
                type(exc).__name__,
            )
        await asyncio.sleep(interval)


def _fleet_registration_lifespan_factory(args: argparse.Namespace, name: str):
    """Build the ``FastMCP(..., lifespan=...)`` ASGI lifespan that starts/stops
    the self-registration heartbeat task (CONCEPT:EG-KG.sharding.server-registry, W2.5).

    A closure (not a bare module-level lifespan) so it captures THIS server's
    own ``args``/``name`` without stashing state on the app object. Opt out
    with ``MCP_FLEET_REGISTRATION=false`` for a run that genuinely has no
    engine access (e.g. an isolated unit-test harness).
    """

    @contextlib.asynccontextmanager
    async def _fleet_registration_lifespan(_app: Any):
        task: asyncio.Task[None] | None = None
        if to_boolean(setting("MCP_FLEET_REGISTRATION", "True")):
            url = _fleet_registration_endpoint_reference(args, name)
            ttl_secs = _fleet_registration_ttl_secs()
            task = asyncio.create_task(
                _register_and_heartbeat_forever(name, url, ttl_secs)
            )
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            # GraphOS owns one process transport shared by all graph-scoped
            # views.  A pod/process drain must stop new admissions, wait only
            # the configured bounded interval for in-flight calls, and then
            # close the transport.  A timeout is visible and is never
            # presented as session continuity; a restarted pod must mint a
            # fresh GraphSession and re-read ClusterMembers.
            try:
                from agent_utilities.knowledge_graph.core.graph_compute import (
                    GraphComputeEngine,
                )

                engine = GraphComputeEngine.get_active()
                if engine is not None:
                    status = await asyncio.to_thread(engine.drain)
                    if status is not None and getattr(status, "timed_out", False):
                        logger.error(
                            "GraphOS transport drain timed out with %s active request(s); "
                            "continuity is not claimed",
                            getattr(status, "active_requests", "unknown"),
                        )
                    engine.close()
            except Exception as exc:  # noqa: BLE001 - lifecycle teardown must continue
                logger.error(
                    "GraphOS transport drain failed; continuity is not claimed (%s)",
                    type(exc).__name__,
                )

    return _fleet_registration_lifespan


def _handle_help_and_port_validation(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if hasattr(args, "help") and args.help:
        parser.print_help()
        sys.exit(0)
    if args.port < 0 or args.port > 65535:
        logger.error(f"Error: Port {args.port} is out of valid range (0-65535).")
        sys.exit(1)


def _is_remote_network(args: argparse.Namespace) -> bool:
    return str(
        getattr(args, "transport", "stdio") or "stdio"
    ).lower() in _NETWORK_TRANSPORTS and not _is_loopback_bind(
        getattr(args, "host", "")
    )


def _resolve_metrics_token() -> str | None:
    metrics_ref = str(setting("MCP_METRICS_TOKEN_REF", "") or "").strip()
    if not metrics_ref:
        return None
    try:
        if metrics_ref.startswith("env://"):
            metrics_token = str(setting(metrics_ref[len("env://") :], "") or "")
        else:
            from agent_utilities.security.secrets_client import create_secrets_client

            metrics_token = str(create_secrets_client().resolve_ref(metrics_ref) or "")
        if not 32 <= len(metrics_token) <= 4_096 or any(
            character in metrics_token for character in "\r\n\x00"
        ):
            return None
        return metrics_token
    except Exception:
        return None


def _mount_tasks_extension_if_available(mcp: Any, name: str) -> None:
    # CONCEPT:AU-ECO.mcp.tasks-workitem-bridge -- mount the native WorkItem-backed
    # Tasks extension (agent_utilities/mcp/tasks_extension.py), NOT
    # fastmcp_tasks.extension.TasksExtension: that package's engine is
    # hard-wired to Docket/Redis, a second job system this codebase's one
    # WorkItem state machine (AU-P1-1) forbids duplicating. This makes the
    # same WorkItem-backed contract available to any MCP protocol version
    # this server negotiates over its one `/mcp` endpoint, including
    # 2026-07-28 (BUG-069: the isolated mcp_v2_gateway sidecar that used to
    # carry an equivalent projection over its own hop was retired once the
    # `mcp<2` isolation premise it depended on expired -- see
    # docs/architecture/mcp-2026-protocol-surface.md).
    #
    # `fastmcp.server.extensions` (what the Tasks extension mounts through)
    # is fastmcp-4-only; the fleet is EXPLICITLY mixed-version (D-SH-3: child
    # images still ship fastmcp 3.4.4, hostPath-mounted over this same
    # working tree -- D-W2C2-2). `tasks_extension` itself already guards the
    # import, degrading `TASKS_EXTENSION_AVAILABLE` to False and logging the
    # cause; this only needs to skip mounting so a fastmcp-3 image still
    # builds a working server minus tasks/get, tasks/update, tasks/cancel.
    from agent_utilities.mcp.tasks_extension import (
        TASKS_EXTENSION_AVAILABLE,
        WorkItemTasksExtension,
    )

    if TASKS_EXTENSION_AVAILABLE:
        # The owning server identity is part of the native Tasks response
        # metadata.  GraphOS's multiplexer uses it to route follow-up
        # tasks/get/update/cancel requests without introducing another task
        # registry or scheduler.
        mcp.add_extension(WorkItemTasksExtension(server_id=name))


def _register_operational_routes(
    mcp: Any, remote_network: bool, metrics_token: str | None
) -> None:
    """Health is a generic readiness result. Metrics are local-only unless a
    remote listener has a runtime-resolved bearer token; otherwise that
    route is absent. Wrapped defensively so older FastMCP builds without
    custom_route still produce a working server."""
    try:
        from starlette.requests import Request as _Request
        from starlette.responses import JSONResponse as _JSONResponse
        from starlette.responses import Response as _Response

        # render_metrics_with_engine() merges the Python gateway's own
        # Prometheus exposition with the embedded epistemic-graph engine's
        # (fetched from its loopback-only listener — the engine refuses any
        # non-loopback auxiliary listener outright, so this process is the
        # only thing that CAN re-serve it). This is the live /metrics path
        # for the collapsed single-container graph-os pod: this FastMCP
        # process never mounts the REST gateway's own /metrics
        # (agent_utilities.gateway.graph_api.register_graph_routes), so this
        # custom_route is the only exposition actually served.
        from agent_utilities.observability.gateway_metrics import (
            render_metrics_with_engine as _render_metrics_with_engine,
        )

        if not remote_network or metrics_token is not None:

            @mcp.custom_route("/metrics", methods=["GET"])
            async def _metrics_route(request: _Request) -> _Response:
                if metrics_token is not None:
                    import secrets as _secrets

                    values = [
                        value.decode("latin-1")
                        for key, value in request.scope.get("headers", ())
                        if key.lower() == b"authorization"
                    ]
                    expected = f"Bearer {metrics_token}"
                    if len(values) != 1 or not _secrets.compare_digest(
                        values[0], expected
                    ):
                        return _Response(
                            status_code=401,
                            headers={
                                "WWW-Authenticate": "Bearer",
                                "Cache-Control": "no-store",
                            },
                        )
                body, content_type = await _render_metrics_with_engine()
                return _Response(
                    content=body,
                    media_type=content_type,
                    headers={"Cache-Control": "no-store"},
                )

        @mcp.custom_route("/health", methods=["GET"])
        async def _health_route(request: _Request) -> _JSONResponse:  # noqa: ARG001
            return _JSONResponse(
                {"status": "ok"}, headers={"Cache-Control": "no-store"}
            )

    except Exception as _route_exc:  # pragma: no cover - defensive
        logger.warning(
            "Could not register metrics and health routes (exception_type=%s)",
            type(_route_exc).__name__,
        )


def create_mcp_server(
    name: str = "MCP Server",
    version: str = __version__,
    instructions: str = "",
    command_args: list[str] | None = None,
    transport_choices: tuple[str, ...] = _ALL_TRANSPORTS,
):
    """Initialize a FastMCP server with a standard middleware and auth stack.

    This helper consolidates the steps of creating a parser, configuring
    authentication providers (JWT, OIDC, etc.), and assembling standard
    middleware (Logging, Timing, Rate Limiting). It handles CLI flag
    parsing and will exit the process if help is requested or configuration
    is invalid.

    Args:
        name: The human-readable name of the MCP server.
        version: Semantic version string for the server.
        instructions: System instructions specific to this MCP server's
            tools, providing context for the LLM.
        command_args: Optional list of CLI arguments (default: sys.argv).
        transport_choices: Current transports exposed by this server.

    Returns:
        A tuple containing:
            - args: The parsed argparse.Namespace object.
            - mcp: The initialized FastMCP server instance.
            - middlewares: A list of configured middleware instances.

    """
    import logging

    from fastmcp import FastMCP

    # Force all logging to stderr to prevent JSON-RPC corruption over stdio
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, force=True)

    parser = create_mcp_parser(transport_choices=transport_choices)
    args, _ = parser.parse_known_args(command_args)
    if args.transport not in transport_choices:
        parser.error(
            f"transport {args.transport!r} is not supported by this server; "
            f"choose from {', '.join(transport_choices)}"
        )

    # NOTE: stdout purity on the stdio transport needs no code here (or anywhere in
    # this module) — see the module docstring above ``mcp_network_run_kwargs`` for
    # why. Building a server does not dedicate the process to stdio in the first
    # place; the transport that does is claimed, fd-level, by the MCP SDK itself.

    _handle_help_and_port_validation(args, parser)

    _validate_network_exposure(args)

    auth = _configure_auth(args)
    middlewares = _configure_middleware(args, server_name=name)

    remote_network = _is_remote_network(args)
    metrics_token = _resolve_metrics_token()

    import os

    os.environ["FASTMCP_LOG_LEVEL"] = "CRITICAL"
    mcp = FastMCP(
        name,
        version=version,
        auth=auth,
        instructions=instructions,
        lifespan=_fleet_registration_lifespan_factory(args, name),
        # `tasks=` only sets the DEFAULT task-mode for individual `@mcp.tool()`
        # registrations (fastmcp.utilities.tasks.TaskConfig) -- it does not by
        # itself mount the `io.modelcontextprotocol/tasks` extension's
        # `tasks/get`/`tasks/update`/`tasks/cancel` methods (that requires a
        # `ServerExtension`, added below). Kept False: no tool here is
        # registered with `task=True`, so there is nothing for a per-tool
        # default to apply to yet.
        tasks=False,
    )
    _mount_tasks_extension_if_available(mcp, name)
    _register_operational_routes(mcp, remote_network, metrics_token)

    # Inject dynamic visibility transform for dynamic tag/tool filtering
    try:
        from fastmcp.server.transforms import Transform

        class DynamicVisibilityTransform(Transform):
            """Enforces environment-variable and header-based tag and tool filters dynamically across all components."""

            def _filter_components(self, components):
                (
                    enabled_tools_list,
                    disabled_tools_list,
                    enabled_tags_list,
                    disabled_tags_list,
                ) = _env_filter_defaults()

                # 1.5. Override/Append with parsed CLI args if specified
                cli_tools = getattr(args, "tools", None)
                if cli_tools:
                    enabled_tools_list = _split_csv(cli_tools)
                cli_disabled_tools = getattr(args, "disabled_tools", None)
                if cli_disabled_tools:
                    disabled_tools_list = _split_csv(cli_disabled_tools)

                # 2. Extract request query parameters and headers if HTTP/SSE transport is active
                query_filter = None
                (
                    enabled_tools_list,
                    disabled_tools_list,
                    enabled_tags_list,
                    disabled_tags_list,
                    query_filter,
                    reject_all,
                ) = _apply_request_overrides(
                    enabled_tools_list,
                    disabled_tools_list,
                    enabled_tags_list,
                    disabled_tags_list,
                    query_filter,
                )

                if query_filter and not reject_all:
                    enabled_tools_list, reject_all = _apply_semantic_filter(
                        query_filter, enabled_tools_list, name
                    )

                if reject_all:
                    return []

                # 3. Convert lists to sets, then filter each component.
                enabled_tags, disabled_tags, enabled_names, disabled_names = (
                    _finalize_filter_sets(
                        enabled_tags_list,
                        disabled_tags_list,
                        enabled_tools_list,
                        disabled_tools_list,
                    )
                )
                return [
                    c
                    for c in components
                    if _component_passes(
                        c, enabled_names, disabled_names, enabled_tags, disabled_tags
                    )
                ]

            async def list_tools(self, tools):
                return self._filter_components(tools)

            async def get_tool(self, name, call_next, *, version=None):
                tool = await call_next(name, version=version)
                if tool is None:
                    return None
                filtered = self._filter_components([tool])
                return filtered[0] if filtered else None

            async def list_resources(self, resources):
                return self._filter_components(resources)

            async def get_resource(self, uri, call_next, *, version=None):
                res = await call_next(uri, version=version)
                if res is None:
                    return None
                filtered = self._filter_components([res])
                return filtered[0] if filtered else None

            async def list_resource_templates(self, templates):
                return self._filter_components(templates)

            async def get_resource_template(self, uri, call_next, *, version=None):
                tmpl = await call_next(uri, version=version)
                if tmpl is None:
                    return None
                filtered = self._filter_components([tmpl])
                return filtered[0] if filtered else None

            async def list_prompts(self, prompts):
                return self._filter_components(prompts)

            async def get_prompt(self, name, call_next, *, version=None):
                prompt = await call_next(name, version=version)
                if prompt is None:
                    return None
                filtered = self._filter_components([prompt])
                return filtered[0] if filtered else None

        mcp.add_transform(DynamicVisibilityTransform())
    except Exception as exc:
        logger.warning(
            "Could not register dynamic visibility transform (exception_type=%s)",
            type(exc).__name__,
        )

    _register_skill_providers(mcp)
    _register_prompt_providers(mcp)

    return args, mcp, middlewares


def _register_skill_providers(mcp: Any) -> None:
    """Expose this server's skills as ``skill://`` MCP resources (CONCEPT:AU-ECO.mcp.skills-over-mcp-provider).

    Wires FastMCP-4's Skills-over-MCP ``SkillProvider`` onto the just-built
    server for every directory :func:`resolve_skill_provider_dirs` already
    resolves (fleet-contributed + this package's own skills) — the SAME
    discovery the in-loop ``SkillsToolset``/``agent-utilities install`` use, so
    a skill has one discovery path with two projections (in-loop execution vs
    wire distribution), not a second registry to keep in sync.

    Every server built here gains ``skill://{name}/SKILL.md``,
    ``skill://{name}/_manifest``, and ``skill://{name}/{path*}`` resources that
    an mcp/fastmcp-3 client can already read.

    LIVE ON THE DEFAULT INSTALL. The ``[mcp]`` extra floors on
    ``fastmcp>=4.0.0b1``, so ``SkillProvider``/``add_provider`` are always
    present and this registration always runs — the earlier
    ``hasattr(mcp, "add_provider")`` gate that made the whole server-side half
    inert (D-W15-7/D-W15-8 in ``reports/deferred/waves1-5-gate.md``) is gone
    with the fastmcp-3 default it guarded. Exercised end to end by
    ``tests/integration/mcp/test_skill_provider_live_path.py``.

    Never raises: a single unreadable provider directory logs a ``WARNING`` and
    is skipped, and any other failure degrades to one ``WARNING`` — serving
    skills over the wire must never stop a server being built.
    """
    try:
        from fastmcp.server.providers.skills import SkillProvider

        from agent_utilities.core.providers import resolve_skill_provider_dirs

        registered = 0
        for provider_name, root_dir in resolve_skill_provider_dirs():
            try:
                mcp.add_provider(SkillProvider(root_dir))
                registered += 1
            except Exception as exc:  # noqa: BLE001 - one unreadable provider
                # directory must not sink the whole sweep. The exception TYPE
                # is logged so a systematically broken provider is diagnosable.
                logger.warning(
                    "Could not register skill provider %s: %s",
                    provider_name,
                    type(exc).__name__,
                )
        logger.info(
            "Registered %d skill-over-MCP provider(s) as skill:// resources",
            registered,
        )
    except Exception as exc:  # noqa: BLE001 - server-side skill:// support is
        # optional (see the INERT note above); its absence must never stop a
        # server being built. The exception TYPE is logged.
        logger.warning(
            "Could not register skill-over-MCP providers: %s", type(exc).__name__
        )


def _register_prompt_providers(mcp: Any) -> None:
    """Expose this server's own prompts as ``prompt://`` MCP resources
    (CONCEPT:AU-ECO.mcp.cross-process-prompt-harvest — the ``prompt://``
    sibling of :func:`_register_skill_providers`'s ``skill://`` wiring).

    Wires one static :class:`~fastmcp.resources.FileResource` per
    ``*.json`` file under every directory :func:`resolve_prompt_provider_dirs`
    resolves **in this server's own process** — which, run inside a fleet
    child's own venv, is exactly that child's own ``prompts/`` directory
    (its own package is always installed in its own venv, even though it is
    deliberately NOT co-installed in graph-os's). graph-os cannot see these
    files by importing this package (``AGENTS.md`` "Dependency discipline"),
    but it already holds a live probe session to this server, so it reads
    each ``prompt://{provider}/{stem}`` resource body back over that session
    (:meth:`~agent_utilities.mcp.multiplexer.MCPMultiplexer._harvest_prompt_bodies`)
    and promotes it through the SAME ``PromptNode`` primitive the packaged
    base sweep uses
    (:func:`agent_utilities.knowledge_graph.ingestion.fleet_prompt_harvest.promote_harvested_prompts`).
    One discovery-and-write path server-side, two projections (local
    ``ingest_prompts_to_graph`` for what's co-installed, cross-process
    harvest for what's not).

    Never raises: a single unreadable provider directory or prompt file logs
    a ``WARNING`` and is skipped, and any other failure degrades to one
    ``WARNING`` — serving prompts over the wire must never stop a server
    being built.
    """
    try:
        from fastmcp.resources import FileResource

        from agent_utilities.core.providers import resolve_prompt_provider_dirs

        registered = 0
        for provider_name, root_dir in resolve_prompt_provider_dirs():
            try:
                json_files = sorted(root_dir.glob("*.json"))
            except OSError as exc:  # noqa: BLE001 - one unreadable provider
                # directory must not sink the whole sweep. The exception TYPE
                # is logged so a systematically broken provider is diagnosable.
                logger.warning(
                    "Could not list prompt provider %s: %s",
                    provider_name,
                    type(exc).__name__,
                )
                continue
            for json_file in json_files:
                if json_file.name.startswith("_"):
                    continue
                try:
                    mcp.add_resource(
                        FileResource(
                            uri=f"prompt://{provider_name}/{json_file.stem}",
                            path=json_file,
                            name=json_file.stem,
                            mime_type="application/json",
                        )
                    )
                    registered += 1
                except Exception as exc:  # noqa: BLE001 - one unreadable
                    # prompt file must not sink the whole sweep. The exception
                    # TYPE is logged so a systematically broken file is
                    # diagnosable.
                    logger.warning(
                        "Could not register prompt resource %s/%s: %s",
                        provider_name,
                        json_file.stem,
                        type(exc).__name__,
                    )
        logger.info(
            "Registered %d prompt-over-MCP resource(s) as prompt:// resources",
            registered,
        )
    except Exception as exc:  # noqa: BLE001 - server-side prompt:// support
        # is optional; its absence must never stop a server being built. The
        # exception TYPE is logged.
        logger.warning(
            "Could not register prompt-over-MCP providers: %s", type(exc).__name__
        )
