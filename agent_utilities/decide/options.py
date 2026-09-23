"""Caller-declared options for EG ``Decide`` (``CandidateSource::Declared``).

A declared option is the caller's own claim: EG records every one as a claim
premise and keeps the record visible to the declaring principal only. Facts
travel on EG's ``Q32`` fixed-point scale; option ids and fact keys are sorted
here so the matrix order EG records is the declared order.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

Q32_ONE = 1 << 32
_Q32_LIMIT = (1 << 63) - 1


def q32(value: float) -> int:
    """``value`` on the Q32 scale, clamped into ``i64`` (never a float on the wire)."""
    scaled = round(float(value) * Q32_ONE)
    return max(-_Q32_LIMIT, min(_Q32_LIMIT, scaled))


@dataclass(frozen=True, slots=True)
class Option:
    """One option a decision point offers EG."""

    option_id: str
    numbers: Mapping[str, float] = field(default_factory=dict)
    texts: Mapping[str, str] = field(default_factory=dict)
    classification: tuple[str, ...] = ()

    def wire(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "classification": list(self.classification),
            "numbers": [
                {"key": key, "q32": q32(self.numbers[key])}
                for key in sorted(self.numbers)
            ],
            "texts": [
                {"key": key, "text": self.texts[key]} for key in sorted(self.texts)
            ],
        }


def unique_sorted(options: Iterable[Option]) -> list[Option]:
    """Options sorted by id; a repeated id keeps its first declaration."""
    seen: dict[str, Option] = {}
    for option in options:
        seen.setdefault(option.option_id, option)
    return [seen[key] for key in sorted(seen)]


def declared_source(options: Iterable[Option]) -> dict[str, Any]:
    """The ``CandidateSource::Declared`` wire value for ``options``."""
    return {
        "source": "declared",
        "options": [option.wire() for option in unique_sorted(options)],
    }


def text_param(name: str, value: str) -> dict[str, Any]:
    """One typed ``text`` parameter (never substituted into query text)."""
    return {"name": name, "value": {"type": "text", "value": value}}


def iri_list_param(name: str, iris: Iterable[str]) -> dict[str, Any]:
    """One typed ``iri_list`` parameter."""
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
