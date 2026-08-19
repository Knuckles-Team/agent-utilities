"""Fail-closed errors raised by the governed retrieval domain."""

from __future__ import annotations

__all__ = [
    "CleanupIncompleteError",
    "GenerationConflictError",
    "GenerationNotFoundError",
    "RetrievalAuthorizationError",
    "RetrievalDomainError",
    "RetrievalReplayError",
    "StaleGenerationError",
]


class RetrievalDomainError(ValueError):
    """Base fail-closed retrieval-domain error."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(code if not detail else f"{code}: {detail}")


class RetrievalAuthorizationError(RetrievalDomainError):
    """ACL authority was missing, expired, or returned unusable evidence."""


class StaleGenerationError(RetrievalDomainError):
    """A request/vector/index referenced a non-current generation."""


class GenerationConflictError(RetrievalDomainError):
    """An immutable generation or CAS current pointer conflicted."""


class GenerationNotFoundError(RetrievalDomainError):
    """An exact generation identity could not be resolved."""


class CleanupIncompleteError(RetrievalDomainError):
    """Cleanup was partial or its checkpoint could not prove completion."""


class RetrievalReplayError(RetrievalDomainError):
    """An exact request/cache identity was replayed with drift."""
