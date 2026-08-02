from __future__ import annotations

"""OpenAI model catalogue discovery/verification (CONCEPT:AU-ORCH.adapter.openai-catalog-verification).

A configured OpenAI ``ModelDefinition`` names a ``model_id`` (e.g. ``gpt-4.1-mini``) but
nothing in this repo previously checked that the id actually exists on the account it
authenticates to — the router simply trusted the config. This module calls the live
``GET /v1/models/{id}`` catalogue endpoint (via ``openai.AsyncOpenAI``, already a core
dependency through ``pydantic-ai-slim[openai]`` — no new dependency added) to confirm
existence before a deployment relies on it.

OpenAI's models endpoint returns identity metadata only (``id``, ``created``,
``owned_by``) — it carries no capability/cost/latency data, so this module never
infers or fabricates those; :mod:`agent_utilities.models.model_profile` records them
as unsourced when no other source exists.

Never raises: a network/auth failure returns ``exists=False`` with a class-name-only
``error`` (never the exception's message/args, which can echo request headers/URLs
that embed the credential) so a failed verification can be logged/persisted without
risk of leaking the API key.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

__all__ = ["OpenAIModelVerification", "verify_openai_model"]


@dataclass(slots=True)
class OpenAIModelVerification:
    """Result of checking one model id against the live OpenAI catalogue."""

    model_id: str
    exists: bool
    owned_by: str | None = None
    created: int | None = None
    checked_at: str = ""
    error: str | None = None  # exception CLASS NAME only — never str(exc)


async def verify_openai_model(
    model_id: str,
    *,
    api_key: str | None,
    base_url: str | None = None,
) -> OpenAIModelVerification:
    """Confirm ``model_id`` exists in the live OpenAI (or OpenAI-compatible) catalogue.

    Best-effort and side-effect-free: a missing key, an unavailable client, or a
    request failure all return ``exists=False`` with a class-name-only ``error`` —
    never a raised exception, and never the exception's message (which can embed
    the request URL/headers and therefore the credential).
    """
    checked_at = datetime.now(UTC).isoformat()
    if not model_id:
        return OpenAIModelVerification(
            model_id=model_id,
            exists=False,
            checked_at=checked_at,
            error="empty model_id",
        )
    if not api_key:
        return OpenAIModelVerification(
            model_id=model_id,
            exists=False,
            checked_at=checked_at,
            error="no API key configured",
        )
    try:
        from openai import AsyncOpenAI
    except Exception:  # pragma: no cover - openai ships via pydantic-ai-slim[openai]
        return OpenAIModelVerification(
            model_id=model_id,
            exists=False,
            checked_at=checked_at,
            error="openai client unavailable",
        )

    if base_url:
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    else:
        # Preserve the SDK's default endpoint for both ``None`` and an empty
        # configuration value. Passing ``base_url=""`` creates a relative,
        # unusable client instead of selecting the OpenAI default.
        client = AsyncOpenAI(api_key=api_key)
    try:
        record = await client.models.retrieve(model_id)
    except Exception as exc:  # noqa: BLE001 - DELIBERATELY type-name-only: an
        # OpenAI SDK error message can embed the API key that was sent (proven by
        # test_verify_openai_model_missing_never_leaks_the_api_key), so logging
        # `exc` here would leak the credential. The failure is still reported
        # honestly as exists=False plus the exception class, never a fabricated
        # "exists".
        logger.debug(
            "OpenAI catalogue verification failed for a configured model "
            "(exception_type=%s)",
            type(exc).__name__,
        )
        return OpenAIModelVerification(
            model_id=model_id,
            exists=False,
            checked_at=checked_at,
            error=type(exc).__name__,
        )
    finally:
        try:
            await client.close()
        except Exception:  # noqa: BLE001 - closing the throwaway client is
            # cleanup AFTER the verification result is already determined; a
            # close failure must not turn a completed verification into an
            # error, and there is nothing actionable to report.
            pass
    return OpenAIModelVerification(
        model_id=model_id,
        exists=True,
        owned_by=getattr(record, "owned_by", None),
        created=getattr(record, "created", None),
        checked_at=checked_at,
    )
