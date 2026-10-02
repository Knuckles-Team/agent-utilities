"""Residual fail-closed types with no equivalent in the ``epistemic_graph`` client.

AU-BOUNDARY-R012 deleted the Epistemic Operations Protocol catalog and its
generated client projection: every DTO that mirrored a durable engine
operation (``RequestContext``, ``WorkItem``, ``ChangeEnvelope``,
``ClaimWorkItem``, ``EvidenceBundle``, the ``ResourceReservation*``/
``ResourceHostUpdate*`` families, the ``DevelopmentLane*`` operation request
and result types, and the JSON Schema catalog that generated them) had a
direct equivalent in the engine's own generated client
(``epistemic_graph.generated.models``) and was deleted; every caller that used
one re-points to that client directly, so ``epistemic_graph`` is the sole
Python-side graph projection.

What remains here is a small, hand-written set of types that are *not* a
second graph projection: each adds behavior of its own that the engine client
does not provide, and each is still depended on by retained Agent Utilities
code with no destination of its own yet:

* ``ProtocolModel`` — a reusable fail-closed Pydantic base (``extra="forbid",
  frozen=True, strict=True``) that ``data_prep/**`` and
  ``orchestration/resource_pool_authority.py`` /
  ``orchestration/service_scale_units.py`` use for their own, non-engine DTOs.
  The engine client has no equivalent reusable base to import — each of its
  generated classes inlines its own ``model_config`` — so there is nothing to
  re-point these callers to.
* ``OperationResult`` / ``OperationError`` / ``OperationRedirect`` — AU's own
  generic public-failure envelope for REST/MCP/streaming adapters
  (``security/error_surface.py``). The engine client has no generic operation
  envelope (it returns a typed result per method instead), so this is added
  AU-side behavior, not a projection of engine state.
* ``PlacementRoute`` — ``knowledge_graph/core/placement_catalog.py`` validates
  the engine's placement answer against this narrower, ``extra="forbid"``
  schema and deliberately strips the wire's additive ``endpoints`` field
  before validating (see that module's ``_validate_answer`` docstring). The
  engine client's ``epistemic_graph.generated.models.PlacementRouteWire`` is a
  superset that includes ``endpoints`` as a required field; adopting it here
  would change this module's deliberate fail-closed behavior, not merely
  rename a symbol, so it is out of scope for this deletion.
* ``DevelopmentLaneCleanupIntent`` — a small internal intent value
  (``orchestration/repository_work_item.py``) with no matching request/result
  type in ``epistemic_graph.generated.models`` (that module's
  ``DevelopmentLaneCleanupCompleteRequest`` is a different, much larger wire
  shape for a different operation).

None of these are reachable as a second durable-graph-state authority: they
carry no network call, no catalog, and no generator. A future requirement can
retire each individually once its caller moves or the engine client grows an
equivalent.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class ProtocolModel(BaseModel):
    """Fail-closed base for AU's own non-engine strict DTOs."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PlacementRoute(ProtocolModel):
    """AU's narrower, additive-``endpoints``-rejecting placement route view."""

    schema_version: Literal["1"]
    route_id: Annotated[str, Field(min_length=1)]
    tenant_ref: Annotated[str, Field(min_length=1)]
    partition_ref: Annotated[str, Field(min_length=1)]
    authoritative: Literal[True]
    placed: bool
    group: Annotated[int, Field(ge=0)]
    epoch: Annotated[int, Field(ge=0)]
    fencing_token: Annotated[int, Field(ge=0)]
    stale: bool
    leader_ref: Annotated[str, Field(min_length=1)] | None


class OperationRedirect(ProtocolModel):
    """A placement redirect attached to a failed :class:`OperationResult`."""

    kind: Literal["placement"]
    target_ref: Annotated[str, Field(min_length=1)]
    group: Annotated[int, Field(ge=0)]
    epoch: Annotated[int, Field(ge=0)]
    fencing_token: Annotated[int, Field(ge=0)]
    leader_ref: Annotated[str, Field(min_length=1)] | None


class OperationError(ProtocolModel):
    """The stable, privacy-safe error shape inside an :class:`OperationResult`."""

    code: Annotated[str, Field(min_length=1)]
    retryable: bool
    correlation_id: Annotated[str, Field(min_length=1)]
    detail_ref: Annotated[str, Field(min_length=1)] | None


class OperationResult(ProtocolModel):
    """AU's generic success/failure/redirect envelope for public surfaces."""

    schema_version: Literal["1"]
    operation_id: Annotated[str, Field(min_length=1)]
    status: Literal["succeeded", "failed", "redirected"]
    result_kind: Annotated[str, Field(min_length=1)] | None
    result_ref: Annotated[str, Field(min_length=1)] | None
    error: OperationError | None
    redirect: OperationRedirect | None


class DevelopmentLaneCleanupIntent(ProtocolModel):
    """A lane-cleanup intent with no matching engine-client request type."""

    schema_version: Literal["1"]
    hold_id: Annotated[str, Field(min_length=1, max_length=256)]
    lane_id: Annotated[str, Field(min_length=1, max_length=256)]
    expected_hold_revision: Annotated[int, Field(ge=0)]


__all__ = [
    "DevelopmentLaneCleanupIntent",
    "OperationError",
    "OperationRedirect",
    "OperationResult",
    "PlacementRoute",
    "ProtocolModel",
]
