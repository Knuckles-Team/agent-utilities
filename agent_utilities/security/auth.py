#!/usr/bin/python
from __future__ import annotations

"""JWT bearer authentication for HTTP and WebSocket boundaries.

CONCEPT:AU-OS.config.secrets-authentication — Secrets & Authentication

Validates bearer tokens against a configured JWKS endpoint. Compatible with
OIDC providers including Azure AD, Okta, Keycloak, and Auth0. Loopback-only
listeners may run without authentication; non-loopback listeners require this
JWT boundary.
"""


import json
import logging
import time
from typing import Any, cast
from urllib.parse import urlsplit

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)

# In-memory JWKS cache keyed by endpoint. A single global value can otherwise
# validate against the wrong issuer after a runtime configuration change.
_jwks_cache: dict[str, tuple[dict[str, Any], float]] = {}
_JWKS_CACHE_TTL: float = 300.0  # 5 minutes
_JWKS_MAX_BYTES = 1024 * 1024
_MAX_CREDENTIAL_BYTES = 16_384


def parse_bearer_authorization(authorization: list[bytes]) -> str | None:
    """Parse one unambiguous ASGI ``Authorization: Bearer`` credential.

    The raw-header boundary is intentionally shared by every HTTP/MCP caller.
    Duplicate, oversized, non-ASCII, unsupported, and whitespace-ambiguous
    credentials fail closed before any prevalidated identity can be reused.
    """
    if len(authorization) > 1:
        raise PermissionError("ambiguous credentials")
    if not authorization:
        return None
    value = authorization[0]
    if not isinstance(value, bytes) or len(value) > _MAX_CREDENTIAL_BYTES:
        raise PermissionError("invalid credential size")
    try:
        raw = value.decode("ascii")
    except UnicodeDecodeError:
        raise PermissionError("malformed authorization") from None
    scheme, separator, token = raw.partition(" ")
    if (
        scheme.lower() != "bearer"
        or not separator
        or not token
        or any(ord(character) < 33 or ord(character) == 127 for character in token)
    ):
        raise PermissionError("malformed authorization")
    return token


# ---------------------------------------------------------------------------
# JWKS Fetching & Caching
# ---------------------------------------------------------------------------


async def _fetch_jwks(jwks_uri: str) -> Any:
    """Fetch and cache JSON Web Key Set from a remote URI.

    Args:
        jwks_uri: URL of the JWKS endpoint.

    Returns:
        The parsed JWKS dict.
    """
    now = time.monotonic()
    cached = _jwks_cache.get(jwks_uri)
    if cached is not None and now < cached[1]:
        return cached[0]

    parsed = urlsplit(jwks_uri)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RuntimeError("AUTH_JWT_JWKS_URI must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.fragment:
        raise RuntimeError("AUTH_JWT_JWKS_URI cannot contain credentials or a fragment")
    if parsed.scheme != "https" and parsed.hostname.lower() not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise RuntimeError("JWKS transport requires HTTPS outside loopback")

    try:
        from agent_utilities.security.oidc_discovery import oidc_async_http_client

        async with oidc_async_http_client(timeout=10) as client:
            body = bytearray()
            async with client.stream("GET", jwks_uri) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > _JWKS_MAX_BYTES:
                        raise RuntimeError(
                            "JWKS response exceeds the configured safety bound"
                        )
            document = json.loads(body)
            keys = document.get("keys") if isinstance(document, dict) else None
            if (
                not isinstance(keys, list)
                or not 1 <= len(keys) <= 100
                or not all(isinstance(key, dict) for key in keys)
            ):
                raise RuntimeError("JWKS response has an invalid shape")
            _jwks_cache[jwks_uri] = (document, now + _JWKS_CACHE_TTL)
            logger.debug("JWKS refreshed (%d keys)", len(document["keys"]))
            return document
    except Exception:
        # Endpoint details and response bodies can contain deployment-specific
        # identifiers. Keep logs stable and fail closed once a cache entry expires.
        logger.warning("Failed to refresh the configured JWKS endpoint")
        raise


def _decode_jwt(
    token: str, jwks: dict, *, issuer: str | None, audience: str | None
) -> dict:
    """Decode and validate a JWT token using authlib.

    CONCEPT:AU-OS.config.secrets-authentication — Secrets & Authentication

    Args:
        token: The raw JWT string.
        jwks: The JWKS dict containing public keys.
        issuer: Expected issuer claim (optional).
        audience: Expected audience claim (optional).

    Returns:
        The validated claims dict.

    Raises:
        HTTPException: If the token is invalid, expired, or claims fail validation.
    """
    try:
        from joserfc import jwt
        from joserfc.errors import (
            BadSignatureError,
            DecodeError,
            ExpiredTokenError,
            InvalidClaimError,
        )
        from joserfc.jwk import KeySet
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="joserfc is required for JWT authentication. "
            "Install it with: pip install agent-utilities[auth]",
        ) from exc

    try:
        # Import the keyset
        key_set = KeySet.import_key_set(cast(Any, jwks))

        # Decode with signature verification
        from agent_utilities.core.config import config

        token_obj = jwt.decode(  # type: ignore
            token,
            key_set,
            algorithms=config.auth_jwt_algorithms,
        )
        claims = token_obj.claims

        # Build validation options
        options: dict[str, Any] = {"exp": {"essential": True}}
        if issuer:
            options["iss"] = {"essential": True, "value": issuer}
        if audience:
            options["aud"] = {"essential": True, "value": audience}

        registry = jwt.JWTClaimsRegistry(leeway=30, **options)
        registry.validate(claims)

        # Manual issuer/audience check (joserfc validation handles this, but keep manual as fallback/strict check)
        if issuer and claims.get("iss") != issuer:
            raise InvalidClaimError("iss")
        if audience:
            aud = claims.get("aud")
            if isinstance(aud, list):
                if audience not in aud:
                    raise InvalidClaimError("aud")
            elif aud != audience:
                raise InvalidClaimError("aud")

        return dict(claims)

    except ExpiredTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None
    except (BadSignatureError, DecodeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token signature",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None
    except InvalidClaimError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token claims",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None
    except Exception as e:
        logger.warning("JWT validation failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token validation failed",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


# ---------------------------------------------------------------------------
# FastAPI Security Dependencies
# ---------------------------------------------------------------------------


async def verify_jwt_only(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, Any] | None:
    """Verify a JWT Bearer token from the ``Authorization`` header.

    Returns ``None`` if JWT auth is not configured or no token is present.
    Raises ``HTTPException(401)`` if a token is present but invalid.

    CONCEPT:AU-OS.config.secrets-authentication — Secrets & Authentication
    """
    from agent_utilities.core.config import config

    if not config.auth_jwt_jwks_uri:
        return None  # JWT auth not configured

    if not credentials:
        return None

    jwks = await _fetch_jwks(config.auth_jwt_jwks_uri)
    claims = _decode_jwt(
        credentials.credentials,
        jwks,
        issuer=config.auth_jwt_issuer,
        audience=config.auth_jwt_audience,
    )

    logger.info("JWT authentication successful")
    return {**claims, "auth_type": "jwt"}


async def verify_credentials(
    request: Request,
    bearer: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, Any] | None:
    """Authenticate one request with the configured JWT bearer policy.

    This is the primary authentication dependency for agent server endpoints.
    Authentication is enforced when ``AUTH_JWT_JWKS_URI`` is configured.

    CONCEPT:AU-OS.config.secrets-authentication — Secrets & Authentication
    """
    authorization = [f"Bearer {bearer.credentials}".encode()] if bearer else []
    try:
        result = await authenticate_header_values(authorization=authorization)
    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None
    if result is not None:
        request.state.user_claims = result
    return result


async def authenticate_header_values(
    *,
    authorization: list[bytes],
) -> dict[str, Any] | None:
    """Authenticate raw ASGI credential headers without route dependencies.

    This is shared by the FastAPI dependency and the outer ASGI boundary that
    also protects mounted A2A/ACP/custom applications and WebSockets. Duplicate,
    oversized, malformed, or unsupported credential headers fail closed.
    """
    from agent_utilities.core.config import config

    token = parse_bearer_authorization(authorization)
    if not config.auth_jwt_jwks_uri:
        return None

    if token is not None:
        jwks = await _fetch_jwks(config.auth_jwt_jwks_uri)
        try:
            claims = _decode_jwt(
                token,
                jwks,
                issuer=config.auth_jwt_issuer,
                audience=config.auth_jwt_audience,
            )
        except HTTPException:
            raise PermissionError("invalid bearer token") from None
        return {**claims, "auth_type": "jwt"}

    raise PermissionError("authentication required")
