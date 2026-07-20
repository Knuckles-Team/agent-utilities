"""Native, fail-closed Eunomia authorization for FastMCP.

The implementation deliberately depends only on ``eunomia-core`` schemas.  It
does not pull the legacy Eunomia HTTP SDK (and its unpatched ``python-jose`` /
``ecdsa`` chain), place bearer credentials in policy attributes, or trust a
client-supplied identity when the MCP listener is authenticated.

Embedded policies are evaluated in process with Eunomia-compatible semantics.
Remote checks use the shared bounded, DNS-pinned HTTP boundary and the runtime
``EUNOMIA`` TLS profile, so CA bundles, mTLS, proxies, private-host exceptions,
and credentials remain operator configuration rather than source constants.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from eunomia_core import enums, schemas
from fastmcp.exceptions import ToolError
from fastmcp.prompts.base import Prompt
from fastmcp.resources.base import Resource
from fastmcp.resources.template import ResourceTemplate
from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.base import Tool
from fastmcp.utilities.components import FastMCPComponent
from mcp import types

logger = logging.getLogger(__name__)

EUNOMIA_BULK_CHECK_MAX = 100
_MAX_POLICY_BYTES = 1024 * 1024
_MAX_AUDIT_VALUE = 256
_MAX_COMPONENTS_PER_LIST = 10_000
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:/-]{1,256}$")


def _safe_text(value: Any, *, default: str = "unknown") -> str:
    """Return a bounded, single-line value suitable for an identifier or log."""
    rendered = str(value or "").strip()
    if not rendered or len(rendered) > _MAX_AUDIT_VALUE:
        return default
    if any(ord(character) < 32 or ord(character) == 127 for character in rendered):
        return default
    return rendered


def _safe_identifier(value: Any, *, default: str = "unknown") -> str:
    rendered = _safe_text(value, default=default)
    return rendered if _SAFE_ID.fullmatch(rendered) else default


def _attribute_value(obj: Any, path: str) -> Any:
    """Resolve Eunomia's dot-path syntax without evaluating user code."""
    current = obj
    for component in path.split("."):
        if isinstance(current, Mapping) and component in current:
            current = current[component]
        elif isinstance(current, list) and component.isdigit():
            index = int(component)
            if index >= len(current):
                return None
            current = current[index]
        elif hasattr(current, component):
            current = getattr(current, component)
        else:
            return None
        if current is None:
            return None
    return current


def _apply_operator(
    operator: enums.ConditionOperator, expected: Any, actual: Any
) -> bool:
    """Apply the operator ordering used by Eunomia's reference evaluator."""
    if expected is None or actual is None:
        return False
    if operator == enums.ConditionOperator.EQUALS:
        return expected == actual
    if operator == enums.ConditionOperator.NOT_EQUALS:
        return expected != actual
    if isinstance(expected, str) and isinstance(actual, str):
        if operator == enums.ConditionOperator.CONTAINS:
            return expected in actual
        if operator == enums.ConditionOperator.NOT_CONTAINS:
            return expected not in actual
        if operator == enums.ConditionOperator.STARTS_WITH:
            return actual.startswith(expected)
        if operator == enums.ConditionOperator.ENDS_WITH:
            return actual.endswith(expected)
    elif (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        if operator == enums.ConditionOperator.GREATER:
            return expected > actual
        if operator == enums.ConditionOperator.GREATER_OR_EQUAL:
            return expected >= actual
        if operator == enums.ConditionOperator.LESS:
            return expected < actual
        if operator == enums.ConditionOperator.LESS_OR_EQUAL:
            return expected <= actual
    elif isinstance(expected, list):
        if operator == enums.ConditionOperator.IN:
            return actual in expected
        if operator == enums.ConditionOperator.NOT_IN:
            return actual not in expected
    return False


def _condition_matches(condition: schemas.Condition, obj: Any) -> bool:
    return _apply_operator(
        condition.operator,
        condition.value,
        _attribute_value(obj, condition.path),
    )


def _rule_matches(rule: schemas.Rule, request: schemas.CheckRequest) -> bool:
    return (
        request.action in rule.actions
        and all(
            _condition_matches(condition, request.principal)
            for condition in rule.principal_conditions
        )
        and all(
            _condition_matches(condition, request.resource)
            for condition in rule.resource_conditions
        )
    )


def _evaluate_policies(
    policies: Sequence[schemas.Policy], request: schemas.CheckRequest
) -> schemas.CheckResponse:
    """Evaluate policies with explicit-deny > explicit-allow > default-deny."""
    explicit_allow: tuple[str, str] | None = None
    default_deny = False
    for policy in policies:
        matched = next(
            (rule for rule in policy.rules if _rule_matches(rule, request)), None
        )
        if matched is not None:
            if matched.effect == enums.PolicyEffect.DENY:
                return schemas.CheckResponse(
                    allowed=False,
                    reason=f"rule {matched.name} denied the action",
                )
            if matched.effect == enums.PolicyEffect.ALLOW:
                explicit_allow = (policy.name, matched.name)
        elif policy.default_effect == enums.PolicyEffect.DENY:
            default_deny = True
    if explicit_allow is not None:
        return schemas.CheckResponse(
            allowed=True,
            reason=f"rule {explicit_allow[1]} allowed the action",
        )
    if default_deny:
        return schemas.CheckResponse(
            allowed=False, reason="action denied by default effect"
        )
    return schemas.CheckResponse(
        allowed=False, reason="action denied because no policy explicitly allowed it"
    )


def _load_policy(path: str) -> schemas.Policy:
    """Read one bounded operator-selected policy file without leaking its path."""
    candidate = Path(path).expanduser()
    try:
        stat = candidate.stat()
        if not candidate.is_file() or stat.st_size > _MAX_POLICY_BYTES:
            raise ValueError("policy is not a bounded regular file")
        raw = candidate.read_bytes()
    except OSError as exc:
        raise ValueError("Eunomia policy file is unavailable") from exc
    if len(raw) > _MAX_POLICY_BYTES:
        raise ValueError("Eunomia policy file exceeds the configured limit")
    try:
        return schemas.Policy.model_validate_json(raw)
    except Exception as exc:
        raise ValueError("Eunomia policy file is invalid") from exc


class _EmbeddedPolicyBridge:
    """Async bridge over an immutable, schema-validated local policy set."""

    def __init__(self, policies: Sequence[schemas.Policy]) -> None:
        if not policies:
            raise ValueError("At least one Eunomia policy is required")
        self.policies = tuple(policies)
        self.mode = "embedded"

    async def check(self, request: schemas.CheckRequest) -> schemas.CheckResponse:
        return _evaluate_policies(self.policies, request)

    async def bulk_check(
        self, requests: Sequence[schemas.CheckRequest]
    ) -> list[schemas.CheckResponse]:
        return [_evaluate_policies(self.policies, request) for request in requests]


def _endpoint_url(endpoint: str, path: str) -> str:
    """Append a fixed API path to a bounded URL without inheriting query data."""
    if not isinstance(endpoint, str) or len(endpoint) > 8_192:
        raise ValueError("Eunomia endpoint is invalid")
    parsed = urlsplit(endpoint.strip())
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Eunomia endpoint is invalid")
    hostname = str(parsed.hostname).lower().rstrip(".")
    if (
        parsed.scheme == "http"
        and hostname not in {"localhost", "127.0.0.1", "::1"}
    ):
        raise ValueError("Eunomia requires HTTPS outside loopback")
    base_path = parsed.path.rstrip("/")
    return urlunsplit(
        (parsed.scheme, parsed.netloc, f"{base_path}/{path.lstrip('/')}", "", "")
    )


def _resolve_api_key(reference: str | None) -> str | None:
    """Resolve a runtime secret reference; raw secrets are never configuration."""
    if not reference:
        return None
    try:
        if reference.startswith("env://"):
            from agent_utilities.core.config import setting

            value = setting(reference[len("env://") :])
        else:
            from agent_utilities.security.secrets_client import create_secrets_client

            value = create_secrets_client().resolve_ref(reference)
    except Exception as exc:
        raise ValueError("Eunomia API key reference could not be resolved") from exc
    rendered = str(value or "")
    if not 16 <= len(rendered) <= 4_096 or any(
        character in rendered for character in "\r\n\x00"
    ):
        raise ValueError("Eunomia API key reference resolved to an invalid value")
    return rendered


class _RemotePolicyBridge:
    """Fail-closed remote PDP bridge using the canonical safe HTTP boundary."""

    def __init__(
        self,
        endpoint: str,
        *,
        api_key_ref: str | None = None,
        allowed_private_hosts: Sequence[str] = (),
        timeout: float = 10.0,
        max_response_bytes: int = 1024 * 1024,
        transport: Any = None,
    ) -> None:
        self._check_url = _endpoint_url(endpoint, "check")
        self._bulk_url = _endpoint_url(endpoint, "check/bulk")
        self._api_key = _resolve_api_key(api_key_ref)
        self._allowed_private_hosts = tuple(allowed_private_hosts)
        self._timeout = timeout
        self._max_response_bytes = max_response_bytes
        self._transport = transport
        self.mode = "remote"

    @property
    def _headers(self) -> dict[str, str]:
        return {"WAY-API-KEY": self._api_key} if self._api_key else {}

    async def _post(self, url: str, payload: Any) -> Any:
        from agent_utilities.protocols.source_connectors.http_safety import (
            safe_post_json_async,
        )

        return await safe_post_json_async(
            url,
            payload,
            headers=self._headers,
            timeout=self._timeout,
            max_bytes=self._max_response_bytes,
            max_request_bytes=self._max_response_bytes,
            allowed_private_hosts=self._allowed_private_hosts,
            transport=self._transport,
            tls_service="eunomia",
        )

    async def check(self, request: schemas.CheckRequest) -> schemas.CheckResponse:
        try:
            response = await self._post(
                self._check_url, request.model_dump(mode="json")
            )
            return schemas.CheckResponse.model_validate(response)
        except Exception as exc:
            logger.warning(
                "Remote authorization check failed closed (exception_type=%s)",
                type(exc).__name__,
            )
            return schemas.CheckResponse(
                allowed=False, reason="authorization service unavailable"
            )

    async def bulk_check(
        self, requests: Sequence[schemas.CheckRequest]
    ) -> list[schemas.CheckResponse]:
        selected = list(requests)
        if not selected:
            return []
        try:
            response = await self._post(
                self._bulk_url,
                [request.model_dump(mode="json") for request in selected],
            )
            if not isinstance(response, list) or len(response) != len(selected):
                raise ValueError("remote authorization response was misaligned")
            return [schemas.CheckResponse.model_validate(item) for item in response]
        except Exception as exc:
            logger.warning(
                "Remote bulk authorization check failed closed (exception_type=%s)",
                type(exc).__name__,
            )
            return [
                schemas.CheckResponse(
                    allowed=False, reason="authorization service unavailable"
                )
                for _ in selected
            ]


class _ChunkingBulkCheckBridge:
    """Bound remote bulk requests and preserve positional result alignment."""

    def __init__(self, bridge: Any, max_batch: int = EUNOMIA_BULK_CHECK_MAX) -> None:
        if not 1 <= max_batch <= EUNOMIA_BULK_CHECK_MAX:
            raise ValueError(
                f"max_batch must be between 1 and {EUNOMIA_BULK_CHECK_MAX}"
            )
        self._bridge = bridge
        self._max_batch = max_batch

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bridge, name)

    async def bulk_check(
        self, requests: Sequence[schemas.CheckRequest]
    ) -> list[schemas.CheckResponse]:
        selected = list(requests)
        merged: list[schemas.CheckResponse] = []
        for start in range(0, len(selected), self._max_batch):
            chunk = selected[start : start + self._max_batch]
            merged.extend(await self._bridge.bulk_check(chunk))
        return merged


def apply_bulk_check_chunking(
    middleware: Any, max_batch: int = EUNOMIA_BULK_CHECK_MAX
) -> Any:
    """Idempotently apply the server's bounded bulk-check contract."""
    bridge = getattr(middleware, "_eunomia", None)
    if bridge is None or isinstance(bridge, _ChunkingBulkCheckBridge):
        return middleware
    middleware._eunomia = _ChunkingBulkCheckBridge(bridge, max_batch=max_batch)
    return middleware


class JwtPrincipalEunomiaMiddleware(Middleware):
    """Authorize every MCP discovery and execution surface with a verified ID."""

    def __init__(
        self,
        bridge: Any,
        *,
        require_verified_principal: bool = False,
        enable_audit_logging: bool = True,
    ) -> None:
        self._eunomia = bridge
        self._require_verified_principal = require_verified_principal
        self._enable_audit_logging = enable_audit_logging

    def _extract_principal(self) -> schemas.PrincipalCheck:
        try:
            headers = dict(get_http_headers())
        except RuntimeError:
            headers = {}

        token = None
        try:
            from fastmcp.server.dependencies import get_access_token

            token = get_access_token()
        except RuntimeError:
            pass

        claims = getattr(token, "claims", None)
        claims = claims if isinstance(claims, Mapping) else {}
        client_id = getattr(token, "client_id", None) or claims.get("azp")
        verified = bool(token is not None and _safe_identifier(client_id) != "unknown")
        if verified:
            agent_id = _safe_identifier(client_id)
            user_id = _safe_identifier(claims.get("sub"))
        elif self._require_verified_principal:
            agent_id = "unknown"
            user_id = "unknown"
        else:
            # Header identity exists solely for local/stdio deployments without an
            # authentication provider. Network listeners are protected elsewhere.
            agent_id = _safe_identifier(headers.get("x-agent-id"))
            user_id = _safe_identifier(headers.get("x-user-id"))

        attributes: dict[str, Any] = {
            "agent_id": agent_id,
            "user_id": user_id,
            "user_agent": _safe_text(headers.get("user-agent"), default="undefined"),
        }
        if verified:
            attributes["jwt_verified"] = True
        return schemas.PrincipalCheck(uri=f"agent:{agent_id}", attributes=attributes)

    def _extract_resource(
        self, context: MiddlewareContext[Any], component: FastMCPComponent
    ) -> schemas.ResourceCheck:
        method = _safe_identifier(context.method)
        component_type = _safe_identifier(method.split("/", 1)[0])
        name = _safe_identifier(getattr(component, "name", None))
        if name == "unknown" or component_type == "unknown":
            raise ToolError("Access denied: component identity is invalid")
        uri = f"mcp:{component_type}:{name}"
        raw_arguments = getattr(context.message, "arguments", None)
        argument_names: list[str] = []
        if isinstance(raw_arguments, Mapping):
            if len(raw_arguments) > 256:
                raise ToolError("Access denied: too many tool arguments")
            argument_names = sorted(
                {
                    identifier
                    for key in raw_arguments
                    if (identifier := _safe_identifier(key)) != "unknown"
                }
            )
        attributes: dict[str, Any] = {
            "method": method,
            "component_type": component_type,
            "name": name,
            "uri": uri,
        }
        if argument_names:
            # Values can contain credentials or personal content. Policies receive
            # names only; authorization must not become a secret-exfiltration path.
            attributes["argument_names"] = argument_names
        return schemas.ResourceCheck(uri=uri, attributes=attributes)

    def _audit(
        self,
        principal: schemas.PrincipalCheck,
        resource: schemas.ResourceCheck,
        method: str | None,
        response: schemas.CheckResponse,
    ) -> None:
        if not self._enable_audit_logging:
            return
        reason = _safe_text(response.reason, default="policy decision")
        fields = (
            _safe_identifier(method),
            _safe_identifier(resource.uri),
            _safe_text((principal.attributes or {}).get("user_agent")),
        )
        log = logger.info if response.allowed else logger.warning
        log(
            "MCP authorization decision allowed=%s method=%s resource=%s "
            "user_agent=%s reason=%s",
            response.allowed,
            *fields,
            reason,
        )

    async def _authorize_execution(
        self, context: MiddlewareContext[Any], component: FastMCPComponent
    ) -> None:
        if getattr(component, "enabled", True) is False:
            raise ToolError("Access denied: component is disabled")
        principal = self._extract_principal()
        resource = self._extract_resource(context, component)
        response = await self._eunomia.check(
            schemas.CheckRequest(
                principal=principal, resource=resource, action="execute"
            )
        )
        self._audit(principal, resource, context.method, response)
        if not response.allowed:
            raise ToolError(f"Access denied: {_safe_text(response.reason)}")

    async def _authorize_listing(
        self,
        context: MiddlewareContext[Any],
        components: Sequence[FastMCPComponent],
    ) -> list[FastMCPComponent]:
        selected = list(components)
        if not selected:
            return []
        if len(selected) > _MAX_COMPONENTS_PER_LIST:
            logger.warning("MCP component listing exceeded authorization bound")
            return []
        principal = self._extract_principal()
        resources = [self._extract_resource(context, item) for item in selected]
        responses = await self._eunomia.bulk_check(
            [
                schemas.CheckRequest(
                    principal=principal, resource=resource, action="list"
                )
                for resource in resources
            ]
        )
        if len(responses) != len(selected):
            logger.warning("MCP authorization response was misaligned")
            return []
        allowed: list[FastMCPComponent] = []
        for response, component, resource in zip(
            responses, selected, resources, strict=True
        ):
            self._audit(principal, resource, context.method, response)
            if response.allowed:
                allowed.append(component)
        return allowed

    @staticmethod
    def _server(context: MiddlewareContext[Any]) -> Any:
        if context.fastmcp_context is None:
            raise ToolError("Access denied: MCP context is unavailable")
        return context.fastmcp_context.fastmcp

    async def on_call_tool(
        self,
        context: MiddlewareContext[types.CallToolRequestParams],
        call_next: CallNext[types.CallToolRequestParams, Any],
    ) -> Any:
        tool = await self._server(context).get_tool(context.message.name)
        await self._authorize_execution(context, tool)
        return await call_next(context)

    async def on_read_resource(
        self,
        context: MiddlewareContext[types.ReadResourceRequestParams],
        call_next: CallNext[types.ReadResourceRequestParams, Any],
    ) -> Any:
        resource = await self._server(context).get_resource(str(context.message.uri))
        await self._authorize_execution(context, resource)
        return await call_next(context)

    async def on_get_prompt(
        self,
        context: MiddlewareContext[types.GetPromptRequestParams],
        call_next: CallNext[types.GetPromptRequestParams, Any],
    ) -> Any:
        prompt = await self._server(context).get_prompt(context.message.name)
        await self._authorize_execution(context, prompt)
        return await call_next(context)

    async def on_list_tools(
        self,
        context: MiddlewareContext[types.ListToolsRequest],
        call_next: CallNext[types.ListToolsRequest, Sequence[Tool]],
    ) -> Sequence[Tool]:
        return await self._authorize_listing(context, await call_next(context))

    async def on_list_resources(
        self,
        context: MiddlewareContext[types.ListResourcesRequest],
        call_next: CallNext[types.ListResourcesRequest, Sequence[Resource]],
    ) -> Sequence[Resource]:
        return await self._authorize_listing(context, await call_next(context))

    async def on_list_resource_templates(
        self,
        context: MiddlewareContext[types.ListResourceTemplatesRequest],
        call_next: CallNext[
            types.ListResourceTemplatesRequest, Sequence[ResourceTemplate]
        ],
    ) -> Sequence[ResourceTemplate]:
        return await self._authorize_listing(context, await call_next(context))

    async def on_list_prompts(
        self,
        context: MiddlewareContext[types.ListPromptsRequest],
        call_next: CallNext[types.ListPromptsRequest, Sequence[Prompt]],
    ) -> Sequence[Prompt]:
        return await self._authorize_listing(context, await call_next(context))


def create_eunomia_middleware(
    policy_file: str | None = None,
    use_remote_eunomia: bool = False,
    eunomia_endpoint: str | None = None,
    enable_audit_logging: bool = True,
    *,
    require_verified_principal: bool = False,
    api_key_ref: str | None = None,
    allowed_private_hosts: Sequence[str] | None = None,
    timeout: float | None = None,
    max_response_bytes: int | None = None,
    max_batch: int | None = None,
    transport: Any = None,
) -> JwtPrincipalEunomiaMiddleware:
    """Create an embedded or remote native Eunomia middleware.

    Omitted network controls are read from :class:`AgentConfig`, whose values
    may come from XDG config or environment.  No endpoint or trust material is
    embedded in the package.
    """
    from agent_utilities.core.config import config

    if use_remote_eunomia:
        if policy_file is not None or not eunomia_endpoint:
            raise ValueError("Remote Eunomia requires only a configured endpoint")
        bridge: Any = _RemotePolicyBridge(
            eunomia_endpoint,
            api_key_ref=api_key_ref or config.eunomia_api_key_ref,
            allowed_private_hosts=(
                tuple(allowed_private_hosts)
                if allowed_private_hosts is not None
                else tuple(config.eunomia_allowed_private_hosts)
            ),
            timeout=timeout or config.eunomia_timeout_seconds,
            max_response_bytes=(
                max_response_bytes or config.eunomia_max_response_bytes
            ),
            transport=transport,
        )
    else:
        bridge = _EmbeddedPolicyBridge(
            [_load_policy(policy_file or config.eunomia_policy_file or "mcp_policies.json")]
        )
    middleware = JwtPrincipalEunomiaMiddleware(
        bridge,
        require_verified_principal=require_verified_principal,
        enable_audit_logging=enable_audit_logging,
    )
    return apply_bulk_check_chunking(
        middleware, max_batch=max_batch or config.eunomia_bulk_check_max
    )


def create_jwt_eunomia_middleware(
    policy_file: str,
    enable_audit_logging: bool = True,
) -> JwtPrincipalEunomiaMiddleware:
    """Compatibility constructor for a verified-principal embedded policy."""
    return create_eunomia_middleware(
        policy_file=policy_file,
        enable_audit_logging=enable_audit_logging,
        require_verified_principal=True,
    )


__all__ = [
    "EUNOMIA_BULK_CHECK_MAX",
    "JwtPrincipalEunomiaMiddleware",
    "apply_bulk_check_chunking",
    "create_eunomia_middleware",
    "create_jwt_eunomia_middleware",
]
