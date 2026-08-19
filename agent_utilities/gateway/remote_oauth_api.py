#!/usr/bin/python
from __future__ import annotations

"""Gateway callback + authorize routes for the remote browser-OAuth broker.

CONCEPT:AU-ECO.mcp.remote-oauth-broker — Deliverable 2 of NE-008 (GOC-85
follow-through, closing U-11/U-41/U-43/U-44/U-45's gateway gap): before this
module, :mod:`agent_utilities.mcp.remote_oauth_broker` shipped a complete,
tested broker CORE with no HTTP surface reachable from a browser — no route
could ever call ``begin()``/``callback()``. This module is that surface.

Mirrors :mod:`agent_utilities.gateway.research_api` /
:mod:`agent_utilities.gateway.ontology_api`: a plain ``fastapi.APIRouter``
mounted by one call, :func:`register_remote_oauth_routes`, with the same
FastAPI-``include_router``-else-plain-``add_route`` fallback those modules use.
Per the gateway's own identity convention
(:class:`agent_utilities.security.request_identity.ActorIdentityMiddleware`,
mounted by :func:`agent_utilities.gateway.graph_api.register_graph_routes`
BEFORE any route is reachable), every route here runs behind the same
server-minted, JWT-verified :class:`~agent_utilities.security.brain_context.ActorContext`
every other gateway route uses — resolved via
:func:`agent_utilities.security.brain_context.current_actor`, NEVER from
request JSON or query parameters. ``code``/``state`` are handled as secrets:
this module never logs them, never echoes them back in a response body, and
never places them in a redirect target — the ONE redirect this module ever
issues goes to a fixed, administrator-configured URL with no query string
appended (see :func:`_success_redirect_url`; an open-redirect surface is
structurally impossible here because the target is never derived from
caller/request input at all).

Three routes make up the browser surface:

* ``POST /providers/{provider_id}/authorize`` — begins a flow for the
  VERIFIED session's tenant/principal (never a caller-supplied one), with the
  caller's optionally-requested scopes filtered through
  :data:`ScopePolicy` (bounded to the provider's own configured ceiling by
  :meth:`RemoteOAuthBroker.begin` regardless of policy), returning the
  authorization URL to redirect the browser to.
* ``GET /oauth/callback`` — ONE shared callback for every provider (the
  broker's ``state`` alone disambiguates which provider/transaction a
  callback belongs to — see ``TransactionStore.consume_once``); completes the
  flow and redirects to the allowlisted success URL, or returns a plain JSON
  success body when no success URL is configured.
* ``POST /providers/{provider_id}/revoke`` — writes a fail-closed tombstone
  for the verified session's provider grant; provider resource and audience
  are resolved from the administrator-owned registry.

All handlers are typed FastAPI handlers with no possible caller-supplied
tenant/principal field on their request models — that is enforced by the
absence of such a field, not by a runtime check that could be bypassed.

Mounting: :func:`agent_utilities.gateway.graph_api.register_graph_routes`
invokes ``register_remote_oauth_routes(app, prefix=prefix)`` alongside the
ontology/research gateway routes. The optional broker and scope-policy
arguments are the startup injection seam.
"""

import logging
from collections.abc import Callable
from typing import Any, NoReturn

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from starlette.routing import Route

logger = logging.getLogger(__name__)

remote_oauth_router = APIRouter(tags=["remote-oauth"])

# ---------------------------------------------------------------------------
# Broker singleton — lazily built, overridable (tests / a deployment that
# wants to construct + inject its own registry rather than go through the
# AgentConfig provider declaration below).
# ---------------------------------------------------------------------------
_BROKER: Any | None = None


def _default_provider_registry() -> Any:
    """Build the administrator-populated registry from typed ``AgentConfig``.

    The configuration field accepts the JSON-array environment representation
    through its canonical alias, then each descriptor is validated by the real
    provider model. Unset input yields an empty registry; invalid typed
    configuration disables the registry with a bounded diagnostic — never a
    silently-accepted malformed provider.
    """
    from agent_utilities.mcp.remote_oauth_broker import (
        ProviderDescriptor,
        ProviderRegistry,
    )

    registry = ProviderRegistry()
    try:
        from agent_utilities.core.config import config

        entries = config.remote_oauth_providers
    except Exception as exc:  # noqa: BLE001 - disable the feature, not the gateway
        logger.error(
            "Remote OAuth provider configuration is invalid; registry disabled (%s)",
            type(exc).__name__,
        )
        return registry
    if entries is None:
        return registry
    for entry in entries:
        try:
            descriptor = ProviderDescriptor(**entry)
        except Exception:  # noqa: BLE001 - one bad entry must not break the rest
            logger.error(
                "Skipping an invalid REMOTE_OAUTH_PROVIDERS_JSON entry (%s)",
                entry.get("provider_id", "<no provider_id>"),
            )
            continue
        registry.register(descriptor)
    return registry


def _build_default_broker() -> Any:
    from agent_utilities.mcp.remote_oauth_broker import RemoteOAuthBroker

    return RemoteOAuthBroker(registry=_default_provider_registry())


def _get_broker() -> Any:
    global _BROKER
    if _BROKER is None:
        _BROKER = _build_default_broker()
    return _BROKER


def _set_broker(broker: Any | None) -> None:
    """Private scoped override for startup injection and test cleanup."""
    global _BROKER
    _BROKER = broker


# ---------------------------------------------------------------------------
# Scope policy — a pluggable seam, not a full policy engine. The provider's
# own ``scopes`` remains the hard ceiling (enforced inside
# ``RemoteOAuthBroker.begin`` itself, independent of this policy) — this hook
# only ever NARROWS a request, never widens one.
# ---------------------------------------------------------------------------
ScopePolicy = Callable[[Any, Any, "tuple[str, ...] | None"], "tuple[str, ...] | None"]


def _default_scope_policy(
    actor: Any, provider: Any, requested: tuple[str, ...] | None
) -> tuple[str, ...] | None:
    """Identity policy: pass the caller's requested scopes straight through.
    ``RemoteOAuthBroker.begin`` still rejects anything outside the provider's
    configured ``scopes`` — this default applies no ADDITIONAL narrowing."""
    return requested


_SCOPE_POLICY: ScopePolicy = _default_scope_policy


def _set_scope_policy(policy: ScopePolicy | None) -> None:
    """Override (or reset, with ``None``) the scope-filtering policy hook."""
    global _SCOPE_POLICY
    _SCOPE_POLICY = policy or _default_scope_policy


def _success_redirect_url() -> str | None:
    """The ONE administrator-configured post-authorization redirect target.

    Never derived from request input — there is no query param, header, or
    body field anywhere in this module that can influence it — so there is no
    open-redirect surface to close. Unset means the callback route returns a
    plain JSON success body instead of redirecting.
    """
    try:
        from agent_utilities.core.config import config

        return config.remote_oauth_success_redirect_url
    except Exception as exc:  # noqa: BLE001 - fail closed to a JSON success body
        logger.error(
            "Remote OAuth redirect configuration is invalid; redirect disabled (%s)",
            type(exc).__name__,
        )
        return None


# ---------------------------------------------------------------------------
# Request/response models — deliberately carry no tenant/principal/actor
# field. Identity comes ONLY from ``current_actor()``.
# ---------------------------------------------------------------------------
class AuthorizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    browser_session_id: str = Field(min_length=1, max_length=256)
    scopes: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "Optional caller-requested scope subset. Must be a subset of the "
            "provider's administrator-configured scopes; anything wider is "
            "rejected, never silently narrowed to the ceiling."
        ),
    )


class AuthorizeResponse(BaseModel):
    authorization_url: str


def _oauth_error_response(
    exc: BaseException, *, code: str, status_code: int
) -> NoReturn:
    """Raise an HTTPException with a sanitized public payload.

    Uses this repo's ONE public-failure boundary
    (:func:`agent_utilities.security.error_surface.public_error_payload`) so a
    caller never sees more than a generic code/message while the real cause is
    still server-logged for correlation. None of this module's own exception
    messages ever interpolate ``code``/``state``/a token value in the first
    place (checked against every raise site in
    :mod:`agent_utilities.mcp.remote_oauth_broker`), so this is defense in
    depth, not the only thing standing between a secret and a response body.
    """
    from agent_utilities.security.error_surface import public_error_payload

    payload = public_error_payload(exc, logger=logger, code=code)
    raise HTTPException(status_code=status_code, detail=payload) from None


def _resolve_verified_actor() -> Any:
    """The server-minted actor for THIS request, or a 401."""
    from agent_utilities.security.brain_context import (
        IdentityRequiredError,
        current_actor,
    )

    try:
        actor = current_actor()
    except IdentityRequiredError:
        raise HTTPException(
            status_code=401, detail="Verified Bearer identity required"
        ) from None
    if not actor.authenticated:
        raise HTTPException(status_code=401, detail="Verified Bearer identity required")
    return actor


async def authorize(provider_id: str, body: AuthorizeRequest) -> AuthorizeResponse:
    """Begin a remote-OAuth flow for the verified caller and return the
    authorization URL to send their browser to."""
    import asyncio

    from agent_utilities.mcp.remote_oauth_broker import (
        OAuthBindingError,
        OAuthDiscoveryError,
        OAuthProviderError,
        OAuthScopeError,
    )

    actor = _resolve_verified_actor()
    broker = _get_broker()
    provider = broker.registry.get(provider_id)
    if provider is None or not provider.enabled:
        raise HTTPException(status_code=404, detail="Unknown provider")
    requested_scopes = _SCOPE_POLICY(actor, provider, body.scopes)
    try:
        authorization_url = await asyncio.to_thread(
            broker.begin,
            provider_id=provider_id,
            actor=actor,
            browser_session_id=body.browser_session_id,
            requested_scopes=requested_scopes,
        )
    except (OAuthProviderError, OAuthBindingError, OAuthScopeError) as exc:
        _oauth_error_response(exc, code="oauth_authorize_rejected", status_code=400)
    except OAuthDiscoveryError as exc:
        _oauth_error_response(exc, code="oauth_provider_unreachable", status_code=502)
    except Exception as exc:  # noqa: BLE001 - the ONE public-failure boundary
        _oauth_error_response(exc, code="oauth_authorize_failed", status_code=500)
    return AuthorizeResponse(authorization_url=authorization_url)


async def oauth_callback(code: str, state: str, browser_session_id: str) -> Any:
    """Complete a remote-OAuth flow and redirect to the allowlisted success URL.

    ``code``/``state`` arrive as query parameters (an unavoidable property of
    the OAuth 2 redirect-based authorization-code flow — the identity provider
    itself puts them there) but this handler never logs, echoes, or forwards
    either value anywhere beyond the one ``broker.callback()`` call that
    consumes them.
    """
    import asyncio

    from starlette.responses import JSONResponse, RedirectResponse

    from agent_utilities.mcp.remote_oauth_broker import (
        OAuthBindingError,
        OAuthDiscoveryError,
        OAuthRevokedError,
        OAuthScopeError,
        OAuthStateError,
    )

    actor = _resolve_verified_actor()
    broker = _get_broker()
    try:
        await asyncio.to_thread(
            broker.callback,
            code=code,
            state=state,
            actor=actor,
            browser_session_id=browser_session_id,
        )
    except (
        OAuthStateError,
        OAuthBindingError,
        OAuthScopeError,
        OAuthRevokedError,
    ) as exc:
        _oauth_error_response(exc, code="oauth_callback_rejected", status_code=400)
    except OAuthDiscoveryError as exc:
        _oauth_error_response(exc, code="oauth_provider_unreachable", status_code=502)
    except Exception as exc:  # noqa: BLE001 - the ONE public-failure boundary
        _oauth_error_response(exc, code="oauth_callback_failed", status_code=500)

    target = _success_redirect_url()
    if target is None:
        return JSONResponse({"status": "success"})
    return RedirectResponse(url=target, status_code=302)


async def _revoke(provider_id: str) -> Any:
    """Revoke the verified caller's grant for one enabled provider."""
    import asyncio

    from starlette.responses import Response

    from agent_utilities.mcp.remote_oauth_broker import OAuthProviderError

    actor = _resolve_verified_actor()
    broker = _get_broker()
    provider = broker.registry.get(provider_id)
    if provider is None or not provider.enabled:
        raise HTTPException(status_code=404, detail="Unknown provider")
    try:
        await asyncio.to_thread(
            broker.revoke,
            actor=actor,
            provider_id=provider_id,
        )
    except OAuthProviderError as exc:
        _oauth_error_response(exc, code="oauth_revoke_rejected", status_code=400)
    except Exception as exc:  # noqa: BLE001 - the ONE public-failure boundary
        _oauth_error_response(exc, code="oauth_revoke_failed", status_code=500)
    return Response(status_code=204)


# Explicit registration makes the production call graph visible to the
# repository's source-reachability gate.
remote_oauth_router.add_api_route(
    "/providers/{provider_id}/authorize",
    authorize,
    methods=["POST"],
    response_model=AuthorizeResponse,
)
remote_oauth_router.add_api_route("/oauth/callback", oauth_callback, methods=["GET"])
remote_oauth_router.add_api_route(
    "/providers/{provider_id}/revoke", _revoke, methods=["POST"]
)


def register_remote_oauth_routes(
    app: Any,
    prefix: str = "",
    *,
    broker: Any | None = None,
    scope_policy: ScopePolicy | None = None,
) -> None:
    """Mount the remote-OAuth gateway surface onto ``app``.

    ``prefix`` defaults to EMPTY, unlike this gateway's other
    ``register_*_routes`` functions (which default to ``/api``): an OAuth
    ``redirect_uri`` must be an exact, stable, administrator-registered URL
    with the identity provider, so ``/oauth/callback`` is deliberately not
    nested under a namespace that could shift. Pass an explicit ``prefix`` if
    this deployment's provider registrations were made against a prefixed
    callback URL instead.

    Same FastAPI-``include_router``-else-plain-``add_route`` fallback as
    :func:`agent_utilities.gateway.research_api.register_research_routes`.
    Optional dependencies are injected once before route mounting.
    """
    if broker is not None:
        _set_broker(broker)
    if scope_policy is not None:
        _set_scope_policy(scope_policy)
    if hasattr(app, "include_router"):  # FastAPI
        app.include_router(remote_oauth_router, prefix=prefix)
        return

    from starlette.responses import JSONResponse

    def _bridge(endpoint, param_names):
        async def _handler(request):  # noqa: ANN001
            kwargs = {p: request.path_params.get(p) for p in param_names}
            kwargs.update(dict(request.query_params))
            try:
                result = await endpoint(**kwargs)
            except HTTPException as e:  # noqa: PERF203
                return JSONResponse(
                    {"status": "error", "message": e.detail}, status_code=e.status_code
                )
            if isinstance(result, BaseModel):
                return JSONResponse(result.model_dump())
            return result

        return _handler

    for route in remote_oauth_router.routes:
        if not isinstance(route, Route):
            continue  # pragma: no cover - APIRouter only emits Route entries here
        param_names = list(getattr(route, "param_convertors", {}) or {})
        app.add_route(
            prefix + route.path,
            _bridge(route.endpoint, param_names),
            methods=list(route.methods or ["GET"]),
        )


__all__ = [
    "AuthorizeRequest",
    "AuthorizeResponse",
    "ScopePolicy",
    "authorize",
    "oauth_callback",
    "register_remote_oauth_routes",
    "remote_oauth_router",
]
