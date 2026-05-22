#!/usr/bin/python
"""X (formerly Twitter) Search and Browsing Tools.

CONCEPT:OS-5.2 — Social Search Integration
"""

import json
import logging
import os
import re
import time
from datetime import UTC, date, datetime
from typing import Any

import httpx
from pydantic_ai import RunContext

from agent_utilities.harness.tracing import trace
from agent_utilities.models import AgentDeps
from agent_utilities.security.xai_auth import XaiAuthManager

from .versioning import tool_version

logger = logging.getLogger(__name__)

# Constants
DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_X_SEARCH_MODEL = "grok-4.20-reasoning"
DEFAULT_X_SEARCH_TIMEOUT_SECONDS = 180
DEFAULT_X_SEARCH_RETRIES = 2
MAX_HANDLES = 10


def _normalize_handles(handles: list[Any] | None, field_name: str) -> list[str]:
    """Clean and validate X handles list."""
    if not handles:
        return []
    cleaned: list[str] = []
    for handle in handles:
        normalized = (handle or "").strip().lstrip("@")
        if normalized:
            cleaned.append(normalized)
    if len(cleaned) > MAX_HANDLES:
        raise ValueError(f"{field_name} supports at most {MAX_HANDLES} handles")
    return cleaned


def _parse_iso_date(value: str, field_name: str) -> date:
    """Parse a strict YYYY-MM-DD string into a ``date``."""
    raw = value.strip()
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{field_name} must be YYYY-MM-DD (got {raw!r})") from exc


def _validate_date_range(from_date: str, to_date: str) -> None:
    """Validate date ranges before sending to xAI.

    Rules:
      * Either field, if non-empty, must parse as ``YYYY-MM-DD``.
      * When both are set, ``from_date <= to_date``.
      * ``from_date`` must not be in the future.
    """
    parsed_from: date | None = None
    parsed_to: date | None = None
    if from_date.strip():
        parsed_from = _parse_iso_date(from_date, "from_date")
    if to_date.strip():
        parsed_to = _parse_iso_date(to_date, "to_date")
    if parsed_from and parsed_to and parsed_from > parsed_to:
        raise ValueError(
            f"from_date ({parsed_from.isoformat()}) must be on or before "
            f"to_date ({parsed_to.isoformat()})"
        )
    if parsed_from is not None:
        today_utc = datetime.now(UTC).date()
        if parsed_from > today_utc:
            raise ValueError(
                f"from_date ({parsed_from.isoformat()}) is in the future; "
                f"X Search only indexes past posts (today UTC is "
                f"{today_utc.isoformat()})"
            )


def _extract_response_text(payload: dict[str, Any]) -> str:
    """Extract full consolidated text response from xAI Responses payload."""
    output_text = str(payload.get("output_text") or "").strip()
    if output_text:
        return output_text

    parts: list[str] = []
    for item in payload.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            ctype = content.get("type")
            if ctype in {"output_text", "text"}:
                text = str(content.get("text") or "").strip()
                if text:
                    parts.append(text)
    return "\n\n".join(parts).strip()


def _extract_inline_citations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract inline URL citations from xAI Responses payload."""
    citations: list[dict[str, Any]] = []
    for item in payload.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            for annotation in content.get("annotations", []) or []:
                if annotation.get("type") != "url_citation":
                    continue
                citations.append(
                    {
                        "url": annotation.get("url", ""),
                        "title": annotation.get("title", ""),
                        "start_index": annotation.get("start_index"),
                        "end_index": annotation.get("end_index"),
                    }
                )
    return citations


@trace(name="x_search", trace_type="TOOL")
@tool_version("1.0.0")
async def x_search(
    ctx: RunContext[AgentDeps],
    query: str,
    allowed_x_handles: list[str] | None = None,
    excluded_x_handles: list[str] | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
) -> str:
    """Search X (formerly Twitter) posts, profiles, and threads using xAI's native Grok index.

    This leverages the live X search tool via xAI, returning detailed summaries and citations.
    Exactly one of allowed_x_handles or excluded_x_handles may be specified, but not both.

    Args:
        ctx: Run context containing agent dependencies and configuration.
        query: The search keywords, hashtags, or phrases to look for.
        allowed_x_handles: Optional list of specific X handles to restrict search results to.
        excluded_x_handles: Optional list of X handles to filter out of the results.
        from_date: Optional start date filter in YYYY-MM-DD format (UTC).
        to_date: Optional end date filter in YYYY-MM-DD format (UTC).

    Returns:
        A JSON string containing success status, resolved answer text, and citation sources.
    """
    if not query or not query.strip():
        return json.dumps({"success": False, "error": "query is required for x_search"})

    # 1. Resolve Credentials with Auto-Login enabled
    auth_manager = XaiAuthManager()
    api_key = auth_manager.resolve_credentials(auto_login=True)  # type: ignore[call-arg]
    if not api_key:
        return json.dumps(
            {
                "success": False,
                "error": (
                    "xAI credentials are not configured. Please authorize by running "
                    "xai login flow or set XAI_API_KEY environment variable."
                ),
            }
        )

    # 2. Parse overrides from config or env
    config = getattr(ctx.deps, "config", {}) if ctx and ctx.deps else {}
    if not isinstance(config, dict):
        config = {}
    xai_config = config.get("xai", {}) or {}

    base_url = (
        str(
            xai_config.get("base_url")
            or os.environ.get("XAI_BASE_URL", DEFAULT_XAI_BASE_URL)
        )
        .strip()
        .rstrip("/")
    )
    model = str(
        xai_config.get("model")
        or os.environ.get("XAI_SEARCH_MODEL", DEFAULT_X_SEARCH_MODEL)
    ).strip()
    timeout_val = xai_config.get("timeout_seconds") or os.environ.get(
        "XAI_SEARCH_TIMEOUT_SECONDS", DEFAULT_X_SEARCH_TIMEOUT_SECONDS
    )
    timeout = max(30, int(str(timeout_val)))

    retries_val = xai_config.get("retries") or os.environ.get(
        "XAI_SEARCH_RETRIES", DEFAULT_X_SEARCH_RETRIES
    )
    max_retries = max(0, int(str(retries_val)))

    # 3. Normalize & Validate Filters
    try:
        allowed = _normalize_handles(allowed_x_handles, "allowed_x_handles")
        excluded = _normalize_handles(excluded_x_handles, "excluded_x_handles")
        if allowed and excluded:
            return json.dumps(
                {
                    "success": False,
                    "error": "allowed_x_handles and excluded_x_handles cannot be used together",
                }
            )

        _validate_date_range(from_date or "", to_date or "")
    except ValueError as exc:
        return json.dumps({"success": False, "error": str(exc)})

    # 4. Construct Tool Constraint
    tool_def: dict[str, Any] = {"type": "x_search"}
    if allowed:
        tool_def["allowed_x_handles"] = allowed
    if excluded:
        tool_def["excluded_x_handles"] = excluded
    if from_date and from_date.strip():
        tool_def["from_date"] = from_date.strip()
    if to_date and to_date.strip():
        tool_def["to_date"] = to_date.strip()

    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": query.strip(),
            }
        ],
        "tools": [tool_def],
        "store": False,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "agent-utilities/x_search_tool",
    }

    # 5. Execute HTTP Request with Retries
    response_data: dict[str, Any] | None = None
    last_error: str | None = None

    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(timeout=float(timeout)) as client:
                resp = client.post(
                    f"{base_url}/responses", headers=headers, json=payload
                )
                resp.raise_for_status()
                response_data = resp.json()
                break
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            last_error = f"HTTP {status}: {exc.response.text}"
            if status < 500 or attempt >= max_retries:
                break
            time.sleep(min(5.0, 1.5 * (attempt + 1)))
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            last_error = f"Connection error: {exc}"
            if attempt >= max_retries:
                break
            time.sleep(min(5.0, 1.5 * (attempt + 1)))

    if not response_data:
        return json.dumps(
            {
                "success": False,
                "error": f"Request failed: {last_error or 'Unknown error'}",
            }
        )

    # 6. Extract Answers and Citations
    answer = _extract_response_text(response_data)
    citations = list(response_data.get("citations") or [])
    inline_citations = _extract_inline_citations(response_data)

    active_filters = []
    if allowed:
        active_filters.append("allowed_x_handles")
    if excluded:
        active_filters.append("excluded_x_handles")
    if from_date and from_date.strip():
        active_filters.append("from_date")
    if to_date and to_date.strip():
        active_filters.append("to_date")

    degraded = bool(active_filters) and not citations and not inline_citations
    degraded_reason = (
        f"No citations returned despite active filters: {', '.join(active_filters)}"
        if degraded
        else None
    )

    return json.dumps(
        {
            "success": True,
            "provider": "xai",
            "tool": "x_search",
            "model": model,
            "query": query.strip(),
            "answer": answer,
            "citations": citations,
            "inline_citations": inline_citations,
            "degraded": degraded,
            "degraded_reason": degraded_reason,
        },
        ensure_ascii=False,
    )


@trace(name="browse_x_post", trace_type="TOOL")
@tool_version("1.0.0")
async def browse_x_post(ctx: RunContext[AgentDeps], url: str) -> str:
    """Retrieve and display the contents of a specific X (formerly Twitter) post.

    Accepts any X.com or Twitter.com status or post URLs and resolves the exact contents
    natively using the live X index query system.

    Args:
        ctx: Run context containing agent dependencies and configuration.
        url: The full web URL of the X post to retrieve.

    Returns:
        A JSON string containing the detailed content, author, timestamps, engagement metrics,
        and citations extracted from the post.
    """
    if not url or not url.strip():
        return json.dumps(
            {"success": False, "error": "url is required for browse_x_post"}
        )

    # 1. Validate & Parse X / Twitter URL
    # Support:
    #   * https://x.com/username/status/12345
    #   * https://twitter.com/username/status/12345
    #   * https://x.com/i/status/12345
    cleaned_url = url.strip()
    status_match = re.search(r"/status/(\d+)", cleaned_url)
    if not status_match:
        return json.dumps(
            {
                "success": False,
                "error": f"Invalid X post URL format: {url!r}. Must contain '/status/<id>'.",
            }
        )

    post_id = status_match.group(1)

    # 2. Extract Username/Handle if present
    handle_match = re.search(r"(?:x|twitter)\.com/([A-Za-z0-9_]+)/status/", cleaned_url)
    username = handle_match.group(1) if handle_match else None
    if username == "i":
        username = None

    # 3. Construct specific query optimized to retrieve the target post from the X index
    # We query the exact status link, the post ID, and username to guide Grok's live index lookup.
    query_parts = [
        f"Retrieve the exact text, author, timestamp, and engagement metrics of the X post status ID {post_id}"
    ]
    if username:
        query_parts.append(f"by user @{username}")
    query_parts.append(f"canonical URL {cleaned_url}.")
    query_parts.append(
        "CRITICAL: You MUST return the actual text content of the post, the author's handle, the date/time, and any metrics you can find. Do NOT return a generic summary."
    )

    query = " ".join(query_parts)

    # 4. Delegate to x_search to perform the actual lookup and synthesis
    logger.info("Executing browse_x_post lookup for status ID %s", post_id)
    search_result_str = await x_search(
        ctx=ctx,
        query=query,
        allowed_x_handles=[username] if username else None,
    )

    try:
        res = json.loads(search_result_str)
        if not res.get("success"):
            return search_result_str

        # Add direct metadata fields to the output response for first-class usability
        res["tool"] = "browse_x_post"
        res["post_id"] = post_id
        if username:
            res["username"] = username
        res["url"] = cleaned_url
        return json.dumps(res, ensure_ascii=False)

    except Exception as exc:
        return json.dumps(
            {
                "success": False,
                "error": f"Failed to parse browse result: {exc}",
            }
        )


x_tools = [x_search, browse_x_post]
