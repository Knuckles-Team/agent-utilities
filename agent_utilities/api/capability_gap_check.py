"""Typed capability-gap determination for AU-SEMANTIC-R027.

``AU-SEMANTIC-R027`` requires that, before concluding a capability is
missing from the generated EG client, the check searches EG's own public
surface (its method catalog, generated contract, or documentation) under
EG's own naming -- never a grep over AU's internal vocabulary treated as
proof of absence. This module is the ``.1`` slice: the typed record of an
EG-side search, plus the refusal behavior when a "missing" verdict is not
backed by at least one such recorded search. The call sites that produce
migration-ready verdicts land separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EGSurface(StrEnum):
    """The EG-side surfaces a capability-gap search may target.

    Deliberately excludes "AU source grep" -- that is the exact proof this
    requirement forbids.
    """

    METHOD_CATALOG = "method_catalog"
    GENERATED_CONTRACT = "generated_contract"
    DOCUMENTATION = "documentation"


class CapabilityGapUnverified(RuntimeError):
    """Raised when a "missing capability" verdict has no recorded EG-side search."""


@dataclass(frozen=True, slots=True)
class EGSurfaceSearch:
    """One search of an EG-side surface, under EG's own naming."""

    surface: EGSurface
    query: str

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise CapabilityGapUnverified(
                "an EG-side surface search record needs a non-empty query"
            )


@dataclass(frozen=True, slots=True)
class CapabilityGapResult:
    """A capability-gap determination, with the EG-side searches that back it."""

    capability: str
    searches: tuple[EGSurfaceSearch, ...]
    is_missing: bool

    @classmethod
    def determine(
        cls,
        capability: str,
        searches: tuple[EGSurfaceSearch, ...],
        is_missing: bool,
    ) -> CapabilityGapResult:
        if is_missing and not searches:
            raise CapabilityGapUnverified(
                f"cannot conclude '{capability}' is missing from EG without at "
                "least one recorded search of EG's own public surface "
                "(method catalog, generated contract, or documentation) under "
                "EG's own naming"
            )
        return cls(capability=capability, searches=searches, is_missing=is_missing)
