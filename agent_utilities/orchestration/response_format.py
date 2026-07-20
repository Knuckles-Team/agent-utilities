"""Closed response-format contract for governed GraphOS delegation."""

from __future__ import annotations

from typing import Literal, cast

ResponseFormat = Literal["text", "json"]

_RESPONSE_FORMATS = frozenset({"text", "json"})


def validate_response_format(value: str) -> ResponseFormat:
    """Return a supported response format or reject the request.

    The boundary is intentionally closed: callers must opt into structured JSON
    explicitly, and misspellings never degrade silently to free-form text.
    """

    if value not in _RESPONSE_FORMATS:
        raise ValueError("response_format must be one of: text, json")
    return cast(ResponseFormat, value)


__all__ = ["ResponseFormat", "validate_response_format"]
