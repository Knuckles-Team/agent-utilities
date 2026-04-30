#!/usr/bin/python
"""JWT and API Key Authentication Module.

CONCEPT:AU-011 — Secrets & Authentication

Provides FastAPI security dependencies for authenticating requests to the
agent server.  Supports two modes that can be used independently or combined:

- **API Key** (legacy): Static shared secret via ``X-API-Key`` header.
- **JWT Bearer** (recommended): Validates tokens against a JWKS endpoint or
  static public key.  Compatible with any OIDC provider (Azure AD, Okta,
  Keycloak, Auth0, etc.).

When both are configured, a request is accepted if *either* credential is
valid (logical OR), allowing gradual migration from API keys to JWT.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

# In-memory JWKS cache with TTL (seconds)
_jwks_cache: dict[str, Any] = {}
_jwks_cache_expiry: float = 0.0
_JWKS_CACHE_TTL: float = 300.0  # 5 minutes


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
    global _jwks_cache, _jwks_cache_expiry  # noqa: PLW0603

    now = time.monotonic()
    if _jwks_cache and now < _jwks_cache_expiry:
        return _jwks_cache

    try:
        import httpx

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(jwks_uri)
            resp.raise_for_status()
            _jwks_cache = resp.json()
            _jwks_cache_expiry = now + _JWKS_CACHE_TTL
            logger.debug(
                "JWKS refreshed from %s (%d keys)",
                jwks_uri,
                len(_jwks_cache.get("keys", [])),
            )
            return _jwks_cache
    except Exception:
        logger.warning("Failed to fetch JWKS from %s", jwks_uri, exc_info=True)
        # Return stale cache if available rather than failing hard
        if _jwks_cache:
            return _jwks_cache
        raise


def _decode_jwt(
    token: str, jwks: dict, *, issuer: str | None, audience: str | None
) -> dict:
    """Decode and validate a JWT token using authlib.

    CONCEPT:AU-011 — Secrets & Authentication

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
        from authlib.jose import JsonWebKey
        from authlib.jose import jwt as authlib_jwt
        from authlib.jose.errors import (
            BadSignatureError,
            DecodeError,
            ExpiredTokenError,
            InvalidClaimError,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="authlib is required for JWT authentication. "
            "Install it with: pip install agent-utilities[auth]",
        ) from exc

    try:
        # Import the keyset
        key_set = JsonWebKey.import_key_set(jwks)

        # Decode with signature verification
        claims = authlib_jwt.decode(token, key_set)

        # Build validation options
        options: dict[str, Any] = {}
        if issuer:
            options["iss"] = {"essential": True, "value": issuer}
        if audience:
            options["aud"] = {"essential": True, "value": audience}

        claims.validate(leeway=30)

        # Manual issuer/audience check (authlib validate doesn't always enforce)
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
    except InvalidClaimError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token claim: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    except Exception as e:
        logger.warning("JWT validation failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token validation failed",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


# ---------------------------------------------------------------------------
# FastAPI Security Dependencies
# ---------------------------------------------------------------------------


async def verify_api_key_only(
    api_key: str | None = Depends(api_key_header),
) -> dict[str, Any] | None:
    """Verify a static API key from the ``X-API-Key`` header.

    Returns ``None`` if API key auth is not configured.  Raises
    ``HTTPException(403)`` if the key is invalid.

    CONCEPT:AU-011 — Secrets & Authentication
    """
    from agent_utilities.core.config import config

    if not config.enable_api_auth or not config.agent_api_key:
        return None  # API key auth not configured
    if api_key and api_key == config.agent_api_key:
        return {"auth_type": "api_key", "sub": "api-key-user"}
    return None  # Not authenticated via API key (may still pass via JWT)


async def verify_jwt_only(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, Any] | None:
    """Verify a JWT Bearer token from the ``Authorization`` header.

    Returns ``None`` if JWT auth is not configured or no token is present.
    Raises ``HTTPException(401)`` if a token is present but invalid.

    CONCEPT:AU-011 — Secrets & Authentication
    """
    from agent_utilities.core.config import config

    if not config.auth_jwt_jwks_uri:
        return None  # JWT auth not configured

    if not credentials:
        return None  # No token present (may still pass via API key)

    jwks = await _fetch_jwks(config.auth_jwt_jwks_uri)
    claims = _decode_jwt(
        credentials.credentials,
        jwks,
        issuer=config.auth_jwt_issuer,
        audience=config.auth_jwt_audience,
    )

    logger.info(
        "JWT authentication successful",
        extra={"sub": claims.get("sub"), "iss": claims.get("iss")},
    )
    return {"auth_type": "jwt", **claims}


async def verify_credentials(
    request: Request,
    api_key: str | None = Depends(api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, Any] | None:
    """Combined security dependency: accepts API key OR JWT Bearer token.

    This is the primary authentication dependency for agent server endpoints.
    Authentication is enforced only when at least one auth mechanism is
    configured (``ENABLE_API_AUTH=true`` or ``AUTH_JWT_JWKS_URI`` is set).

    Order of evaluation:
    1. JWT Bearer token (if configured and present)
    2. API key (if configured and present)
    3. If neither auth mechanism is configured → allow (open mode)
    4. If auth is configured but no valid credential → reject

    CONCEPT:AU-011 — Secrets & Authentication
    """
    from agent_utilities.core.config import config

    auth_configured = config.enable_api_auth or config.auth_jwt_jwks_uri

    if not auth_configured:
        return None  # No auth configured — open mode

    # Try JWT first (preferred)
    if bearer and config.auth_jwt_jwks_uri:
        jwt_result = await verify_jwt_only(bearer)
        if jwt_result:
            # Store claims on request state for downstream use
            request.state.user_claims = jwt_result
            return jwt_result

    # Try API key
    if api_key:
        api_result = await verify_api_key_only(api_key)
        if api_result:
            return api_result

    # Auth is configured but no valid credential provided
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide a valid API key or JWT Bearer token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
