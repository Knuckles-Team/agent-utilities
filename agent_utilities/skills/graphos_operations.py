"""Read-only vocabulary of graph-os's served operation names.

graph-os owns the operation registry. The harness validates that its bundled
skill and the fleet skills name real graph-os operations, so it reads this
packaged JSON snapshot of the registry's names, features and verbs. It holds no
handlers, registration or serving code.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType

_DATA = Path(__file__).with_name("graphos_operations.json")


@dataclass(frozen=True)
class OperationName:
    """One graph-os operation family as named by the registry snapshot."""

    name: str
    feature: str | None
    surface: str


@cache
def _load() -> dict:
    return json.loads(_DATA.read_text(encoding="utf-8"))


def intent_verbs() -> tuple[str, ...]:
    return tuple(_load()["intent_verbs"])


def tool_verbs() -> Mapping[str, tuple[str, ...]]:
    return MappingProxyType({k: tuple(v) for k, v in _load()["tool_verbs"].items()})


def supported_features() -> frozenset[str]:
    return frozenset(_load()["supported_features"])


def operations_by_name() -> Mapping[str, OperationName]:
    return MappingProxyType(
        {
            op["name"]: OperationName(op["name"], op["feature"], op["surface"])
            for op in _load()["operations"]
        }
    )


def canonical_operation_names(
    *, features: frozenset[str] = frozenset(), include_intent: bool = True
) -> frozenset[str]:
    """Operation names for one explicit feature profile."""
    unknown = features - supported_features()
    if unknown:
        raise ValueError(f"unknown Graph-OS features: {sorted(unknown)}")
    return frozenset(
        op.name
        for op in operations_by_name().values()
        if (op.feature is None or op.feature in features)
        and (include_intent or op.surface != "intent")
    )
