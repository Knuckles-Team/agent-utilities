"""EH-280 Phase 1: adapter over ``agent_connector_sdk.repository``'s
branch-aware transport, for the blob/branch delta-tracking half of codebase
structural ingest (CONCEPT:AU-KG.ingest.branch-aware-repository-delta).

Full context: ``/var/tmp/l9/finish/au-deletion/EH-280-CODEBASE-INGEST-CUTOVER-DESIGN.md``.

Short version: ``IngestionEngine._run_codebase_structural`` currently does its
own git-diff-and-content-hash delta tracking in Python
(``_load_file_hashes``/``_persist_file_hashes``/``_structural_delta_files``) to
decide which files changed since the last ingest. ``agent_connector_sdk``'s
new ``feat/branch-aware-transport`` branch provides the same responsibility as
a standard, blob-content-deduplicated, multi-ref-aware SDK primitive
(``LocalGitRepositoryProvider`` + ``index_repository(provider, client, prior=)``).
This module is the AU-side seam for that swap.

**Not wired into the live default path yet** — see the design doc's §5 for
why (EG's ``IndexRepository`` capability is compute-only today,
``DurabilityDomain::None``; the SDK branch is not merged; this venv's
``epistemic_graph`` wheel predates ``generated/``). Every call here fails
closed (raises :class:`RepositoryTransportUnavailable`) rather than silently
degrading to an empty or stale delta plan when those dependencies are
missing — the same discipline EH-345's fleet-catalog cutover used for calls
against ``epistemic_graph.generated.fleet_catalog``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent_utilities.core.config import setting

from .manifest import DeltaManifest

logger = logging.getLogger(__name__)

__all__ = [
    "RepositoryTransportUnavailable",
    "codebase_delta_via_sdk_transport_enabled",
    "index_codebase_via_sdk_transport",
]

# Additive: alongside the existing ``codebase_file``/``codebase_git``
# categories ``DeltaManifest`` already serves — does not touch either.
_MANIFEST_CATEGORY = "codebase_repo_manifest"


class RepositoryTransportUnavailable(RuntimeError):
    """The SDK's branch-aware repository transport is not usable here.

    Raised when ``agent_connector_sdk.repository`` (or its
    ``epistemic_graph.generated`` dependency) is not importable in this
    environment. Callers MUST NOT catch this and fall back to an empty or
    full-repo delta plan silently — that would misreport "nothing changed"
    or force a needless full re-parse depending on which fallback direction
    is chosen; the caller doing the fallback owns the choice explicitly.
    """


def codebase_delta_via_sdk_transport_enabled() -> bool:
    """Whether the Phase-2 default-path flip is turned on for this process.

    Always ``False`` until the design doc's §5 blockers clear; exists now so
    the flag and its plumbing are exercised before Phase 2 needs them.
    """
    return bool(setting("KG_CODEBASE_DELTA_VIA_SDK_TRANSPORT", False))


def _load_prior_manifest(
    manifest: DeltaManifest, graph_name: str, repository_id: str
) -> Any | None:
    """The previous run's ``RepositoryIndexManifest``, or ``None`` on a first run.

    A malformed/unreadable stored manifest is treated as "no prior" (forces a
    full walk on the SDK side) — the same safe direction
    ``_load_file_hashes`` already takes for a degraded read, never the
    direction that could skip a changed file.
    """
    from agent_connector_sdk.repository import RepositoryIndexManifest

    try:
        raw = manifest.get(graph_name, _MANIFEST_CATEGORY, repository_id)
    except Exception:  # noqa: BLE001 — degraded read forces a full walk
        return None
    if not raw:
        return None
    try:
        return RepositoryIndexManifest.model_validate_json(raw)
    except Exception:  # noqa: BLE001 — malformed stored manifest, same direction
        logger.warning(
            "stored repository manifest for %s is unreadable; forcing a full walk",
            repository_id,
        )
        return None


def _store_manifest(
    manifest: DeltaManifest,
    graph_name: str,
    repository_id: str,
    index_manifest: Any,
) -> None:
    """Persist this run's manifest as the next run's ``prior`` (best-effort).

    Mirrors ``_persist_file_hashes``'s own best-effort discipline: a failed
    persist can only cost the NEXT run extra (re-walking blobs it already
    has), never make this run's already-accepted writes incorrect.
    """
    try:
        manifest.record(
            graph_name,
            _MANIFEST_CATEGORY,
            repository_id,
            index_manifest.model_dump_json(),
        )
    except Exception:  # noqa: BLE001
        logger.debug(
            "repository manifest persist failed for %s", repository_id, exc_info=True
        )


async def index_codebase_via_sdk_transport(
    manifest: DeltaManifest,
    async_client: Any,
    *,
    graph_name: str,
    source_path: str,
    repository_id: str,
    limits: Any | None = None,
) -> Any:
    """Run one branch-aware repository index through the SDK transport.

    Returns the SDK's ``RepositoryIndexReceipt`` (``.manifest`` is already
    persisted as the next ``prior`` before this returns; ``.batches[*].result``
    carries the same ``IndexResult`` shape
    ``enrichment/extractors/code_test.py``'s ``entities_from_index_result``
    already knows how to map — Phase 2 wires that mapping in, this module
    does not call it).

    Raises :class:`RepositoryTransportUnavailable` if the SDK's repository
    subpackage cannot be imported. Any other failure (a real git/transport
    error, a rejected batch) propagates as-is — this function does not add
    its own fallback-to-empty-result behavior.
    """
    try:
        from agent_connector_sdk.repository import (
            LocalGitRepositoryProvider,
            index_repository,
        )
    except ImportError as exc:
        raise RepositoryTransportUnavailable(
            "agent_connector_sdk.repository is not importable in this "
            "environment (feat/branch-aware-transport not merged, or this "
            "venv's epistemic_graph wheel predates generated/) — see "
            "EH-280-CODEBASE-INGEST-CUTOVER-DESIGN.md §5"
        ) from exc

    prior = _load_prior_manifest(manifest, graph_name, repository_id)
    provider = LocalGitRepositoryProvider(
        Path(source_path), repository_id=repository_id
    )
    receipt = await index_repository(provider, async_client, prior=prior, limits=limits)
    _store_manifest(manifest, graph_name, repository_id, receipt.manifest)
    return receipt
