"""Typed seams between governed retrieval and its authorities."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    AuthorizationEvidence,
    CleanupCheckpoint,
    EngineCandidate,
    GenerationManifest,
    GenerationRecord,
    RetrievalRequest,
    RetrievalResult,
)

__all__ = [
    "GenerationCatalog",
    "RetrievalAuthorizationAdapter",
    "RetrievalCache",
    "RetrievalEngineAdapter",
]


@runtime_checkable
class RetrievalAuthorizationAdapter(Protocol):
    """ACL authority invoked before any retrieval ranking or cache read."""

    def authorize(self, request: RetrievalRequest) -> AuthorizationEvidence:
        """Return exact, bounded authorization evidence or raise."""


@runtime_checkable
class RetrievalEngineAdapter(Protocol):
    """Engine authority for vector search and generation cleanup."""

    def search(
        self,
        request: RetrievalRequest,
        authorization: AuthorizationEvidence,
    ) -> tuple[EngineCandidate, ...]:
        """Return unranked engine candidates after receiving ACL evidence."""

    def cleanup_generation(self, generation: GenerationManifest) -> CleanupCheckpoint:
        """Return a deterministic checkpoint for the requested generation."""


@runtime_checkable
class GenerationCatalog(Protocol):
    """Authoritative catalog for current generation and cleanup state."""

    def current(self, tenant_ref: str, graph_ref: str) -> GenerationRecord | None:
        """Return the exact current generation, or None when unresolved."""

    def get(
        self, tenant_ref: str, graph_ref: str, generation_id: str
    ) -> GenerationRecord | None:
        """Return one exact generation record; aliases are not resolved."""

    def publish(
        self,
        manifest: GenerationManifest,
        *,
        expected_current: str | None,
    ) -> GenerationRecord:
        """CAS-publish one active generation."""

    def retire(self, generation: GenerationManifest) -> GenerationRecord:
        """Transition one active generation to retiring."""

    def finish_cleanup(
        self,
        generation: GenerationManifest,
        checkpoint: CleanupCheckpoint,
    ) -> GenerationRecord:
        """Record only a complete, exact cleanup checkpoint."""

    def fail_cleanup(
        self,
        generation: GenerationManifest,
        checkpoint: CleanupCheckpoint | None,
    ) -> GenerationRecord:
        """Record a failed/partial cleanup without making it current."""


@runtime_checkable
class RetrievalCache(Protocol):
    """Cache authority that stores only already-authorized results."""

    def get(self, request_digest: str) -> RetrievalResult | None:
        """Return an exact-key result, never a fuzzy or cross-tenant match."""

    def put(self, result: RetrievalResult) -> None:
        """Persist one authorized result under its exact request digest."""

    def delete(self, request_digest: str) -> None:
        """Remove a stale or invalid exact-key entry."""
