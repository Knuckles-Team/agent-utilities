"""Persistence-independent repository seams for source reconciliation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    CheckpointMutation,
    SourceAuthority,
    SourceCatalogEntry,
    SourceCheckpoint,
    SourceManifest,
    SourceReconciliation,
    SourceScope,
)


class RepositoryUnavailable(RuntimeError):
    """The authoritative source repository could not complete an operation."""


class RepositoryContractError(RuntimeError):
    """An adapter returned data outside the source scope or immutability contract."""


@runtime_checkable
class SourceRepository(Protocol):
    """Typed authority for source catalogs, evidence, and checkpoints.

    Adapters must enforce tenant/principal/grant scope before reads, must never
    replace an immutable authority, entry, manifest, or reconciliation, and may
    advance a checkpoint only through the revision/version-fenced CAS method.
    """

    def put_authority(self, authority: SourceAuthority) -> None:
        """Create or idempotently retain one immutable source authority."""

    def put_catalog_entry(self, entry: SourceCatalogEntry) -> None:
        """Create or idempotently retain one immutable catalog entry version."""

    def put_manifest(self, manifest: SourceManifest) -> None:
        """Create or idempotently retain one immutable source snapshot."""

    def put_reconciliation(self, reconciliation: SourceReconciliation) -> None:
        """Persist evidence idempotently; this never advances a checkpoint."""

    def get_authority(
        self,
        scope: SourceScope,
        authority_id: str,
        authority_digest: str | None = None,
    ) -> SourceAuthority | None:
        """Read one authority/version inside the caller's verified scope."""

    def get_manifest(
        self, scope: SourceScope, manifest_id: str
    ) -> SourceManifest | None:
        """Read one immutable manifest inside the verified tenant scope."""

    def get_catalog_entry(
        self,
        scope: SourceScope,
        entry_id: str,
        entry_revision: int | None = None,
    ) -> SourceCatalogEntry | None:
        """Read one or the latest immutable catalog version for tombstone proof."""

    def get_reconciliation(
        self, scope: SourceScope, reconciliation_id: str
    ) -> SourceReconciliation | None:
        """Read prior evidence for deterministic restart/replay."""

    def get_checkpoint(
        self, scope: SourceScope, authority_id: str
    ) -> SourceCheckpoint | None:
        """Read the sole durable checkpoint for an authority."""

    def compare_and_swap_checkpoint(
        self,
        scope: SourceScope,
        mutation: CheckpointMutation,
        checkpoint: SourceCheckpoint,
    ) -> SourceCheckpoint:
        """Apply one revision-fenced checkpoint advance atomically."""
