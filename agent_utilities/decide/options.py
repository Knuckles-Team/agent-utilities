"""Caller-declared options for EG ``Decide`` -- owned by the connector SDK.

``Option``, ``q32``, ``declared_source``, ``text_param`` and
``unique_sorted`` live in :mod:`agent_connector_sdk.decide.options` (the SDK
cannot import AU; AU depends on the SDK), so there is exactly one definition
of the declared-option wire shape. AU re-exports them and adds only what the
SDK does not need.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from agent_connector_sdk.decide.options import (
    Q32_ONE,
    Option,
    declared_source,
    q32,
    text_param,
    unique_sorted,
)


def iri_list_param(name: str, iris: Iterable[str]) -> dict[str, Any]:
    """One typed ``iri_list`` parameter (AU-only; the SDK declares none)."""
    return {"name": name, "value": {"type": "iri_list", "value": list(iris)}}


__all__ = [
    "Q32_ONE",
    "Option",
    "declared_source",
    "iri_list_param",
    "q32",
    "text_param",
    "unique_sorted",
]
