"""Typed client contract for epistemic-graph's ``workflow_derivation`` op.

AU-SEMANTIC-R015.1: this module exists before EG ships the op. It declares
the typed request/response shape the SDK-hosted enterprise-source-sync
runner will call once EG's generated client exposes ``workflow_derivation``
on ``SourceIngest`` (tracked by EG-REPO-INGEST-R002's
``skill_workflow_ingest`` capability), and it fails closed -- raising, never
returning a stub or default value -- for as long as that op is absent from
the installed generated contract.

AU-SEMANTIC-R015.2 swaps :func:`resolve_workflow_derivation_client`'s body to
call the real generated method once EG publishes it; the typed
request/response/Protocol shapes here are expected to survive that swap
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = [
    "WorkflowDerivationClient",
    "WorkflowDerivationRequest",
    "WorkflowDerivationResponse",
    "WorkflowDerivationUnavailableError",
    "resolve_workflow_derivation_client",
]


@dataclass(frozen=True, slots=True)
class WorkflowDerivationRequest:
    """Typed request for EG's not-yet-shipped ``workflow_derivation`` op."""

    source_envelope_id: str
    tenant_id: str
    manifest_preset: str


@dataclass(frozen=True, slots=True)
class WorkflowDerivationResponse:
    """Typed response mirroring the shape EG's generated client will return."""

    workflow_id: str
    derivation_receipt_id: str


@runtime_checkable
class WorkflowDerivationClient(Protocol):
    """Contract the SDK runner calls once EG's generated client exposes the op."""

    def workflow_derivation(
        self, request: WorkflowDerivationRequest
    ) -> WorkflowDerivationResponse: ...


class WorkflowDerivationUnavailableError(RuntimeError):
    """Raised fail-closed while EG has not yet shipped ``workflow_derivation``.

    This is never converted into a default/empty response. A caller that
    catches this error must treat workflow derivation as unavailable, not
    as "no workflow was derivable".
    """


def resolve_workflow_derivation_client() -> WorkflowDerivationClient:
    """Resolve the generated EG client's ``workflow_derivation`` op.

    Fails closed with :class:`WorkflowDerivationUnavailableError` until
    EG-REPO-INGEST-R002 ships the op on the generated ``SourceIngest``
    client (AU-SEMANTIC-R015.2 wires the real call here, replacing the
    local ``kg/core/source_sync.py`` derivation path).
    """
    try:
        from epistemic_graph.generated.source_ingest import (  # type: ignore[import-not-found]
            SourceIngestClient,
        )
    except ImportError as exc:
        raise WorkflowDerivationUnavailableError(
            "epistemic-graph generated client is not installed; "
            "workflow_derivation is unavailable (AU-SEMANTIC-R015.1 "
            "fail-closed refusal)"
        ) from exc

    if not hasattr(SourceIngestClient, "workflow_derivation"):
        raise WorkflowDerivationUnavailableError(
            "epistemic-graph's generated SourceIngestClient does not yet "
            "expose workflow_derivation (tracked by EG-REPO-INGEST-R002); "
            "refusing rather than deriving locally via source_sync.py "
            "(AU-SEMANTIC-R015.1 fail-closed refusal)"
        )

    raise WorkflowDerivationUnavailableError(
        "workflow_derivation was detected on the generated client but "
        "AU-SEMANTIC-R015.2 has not yet wired the real call; refusing "
        "rather than guessing at call semantics"
    )
