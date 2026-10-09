"""Typed client contract for epistemic-graph's ``runnable_skill_derivation`` op.

AU-SEMANTIC-R009.1: this module exists before EG ships the op. It declares
the typed request/response shape AU will call once EG's generated client
exposes ``runnable_skill_derivation`` on ``SourceIngest`` (tracked by
EG-REPO-INGEST-R002's ``skill_workflow_ingest`` capability), and it fails
closed -- raising, never returning a stub or default value -- for as long as
that op is absent from the installed generated contract.

AU-SEMANTIC-R009.2 swaps :func:`resolve_runnable_skill_derivation_client`'s
body to call the real generated method once EG publishes it; the typed
request/response/Protocol shapes here are expected to survive that swap
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = [
    "RunnableSkillDerivationClient",
    "RunnableSkillDerivationRequest",
    "RunnableSkillDerivationResponse",
    "RunnableSkillDerivationUnavailableError",
    "resolve_runnable_skill_derivation_client",
]


@dataclass(frozen=True, slots=True)
class RunnableSkillDerivationRequest:
    """Typed request for EG's not-yet-shipped ``runnable_skill_derivation`` op."""

    source_envelope_id: str
    tenant_id: str


@dataclass(frozen=True, slots=True)
class RunnableSkillDerivationResponse:
    """Typed response mirroring the shape EG's generated client will return."""

    skill_id: str
    derivation_receipt_id: str


@runtime_checkable
class RunnableSkillDerivationClient(Protocol):
    """Contract AU calls through once EG's generated client exposes the op."""

    def runnable_skill_derivation(
        self, request: RunnableSkillDerivationRequest
    ) -> RunnableSkillDerivationResponse: ...


class RunnableSkillDerivationUnavailableError(RuntimeError):
    """Raised fail-closed while EG has not yet shipped ``runnable_skill_derivation``.

    This is never converted into a default/empty response. A caller that
    catches this error must treat skill derivation as unavailable, not as
    "no skill was derivable".
    """


def resolve_runnable_skill_derivation_client() -> RunnableSkillDerivationClient:
    """Resolve the generated EG client's ``runnable_skill_derivation`` op.

    Fails closed with :class:`RunnableSkillDerivationUnavailableError` until
    EG-REPO-INGEST-R002 ships the op on the generated ``SourceIngest``
    client (AU-SEMANTIC-R009.2 wires the real call here).
    """
    try:
        from epistemic_graph.generated.source_ingest import (  # type: ignore[import-not-found]
            SourceIngestClient,
        )
    except ImportError as exc:
        raise RunnableSkillDerivationUnavailableError(
            "epistemic-graph generated client is not installed; "
            "runnable_skill_derivation is unavailable (AU-SEMANTIC-R009.1 "
            "fail-closed refusal)"
        ) from exc

    if not hasattr(SourceIngestClient, "runnable_skill_derivation"):
        raise RunnableSkillDerivationUnavailableError(
            "epistemic-graph's generated SourceIngestClient does not yet "
            "expose runnable_skill_derivation (tracked by "
            "EG-REPO-INGEST-R002); refusing rather than deriving locally "
            "(AU-SEMANTIC-R009.1 fail-closed refusal)"
        )

    raise RunnableSkillDerivationUnavailableError(
        "runnable_skill_derivation was detected on the generated client but "
        "AU-SEMANTIC-R009.2 has not yet wired the real call; refusing "
        "rather than guessing at call semantics"
    )
