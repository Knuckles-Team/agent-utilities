#!/usr/bin/python
from __future__ import annotations

"""AU-INTEGRATION-R014.1 — serving-path benchmark-comparison model.

Typed shape for comparing a proposed AI predicate-grouping, prompt/KV-reuse,
or GPU-fairness improvement against AU's existing typed query/model-serving
path, plus the integration-authorization refusal rule the requirement names
directly: a candidate whose access control relies on a single shared token
rather than a verified per-tenant, per-purpose identity does not meet the
bar for integration, regardless of any latency or throughput gain it shows.

This is the `.1` slice: the typed model, its field validation, and the
authorization-gate refusal case. The benchmark driver that runs identical
workloads through the existing serving path and a concrete candidate is a
later slice, scoped once that candidate exists (see tasks.md).
"""

from typing import Literal

from pydantic import BaseModel, Field

AuthorizationMode = Literal["per_tenant_purpose_identity", "shared_token"]


class ServingPathBenchmarkRow(BaseModel):
    """One side (existing path or candidate) of a serving-path comparison."""

    name: str = Field(min_length=1)
    authorization_mode: AuthorizationMode
    tenant_id: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    throughput_rps: float = Field(ge=0.0)
    latency_p50_ms: float = Field(ge=0.0)
    latency_p95_ms: float = Field(ge=0.0)

    def model_post_init(self, __context: object) -> None:
        if self.latency_p95_ms < self.latency_p50_ms:
            raise ValueError("latency_p95_ms cannot be below latency_p50_ms")


class ServingPathIntegrationDecision(BaseModel):
    """Outcome of comparing a candidate improvement to the existing path."""

    integrate: bool
    reason: str = Field(min_length=1)


def evaluate_serving_path_candidate(
    candidate: ServingPathBenchmarkRow,
    existing: ServingPathBenchmarkRow,
) -> ServingPathIntegrationDecision:
    """Apply the AU-INTEGRATION-R014 authorization-gate refusal rule.

    A shared-token candidate is refused outright -- it is not a verified
    per-tenant, per-purpose identity -- even when it beats the existing
    serving path on throughput or latency. Only once authorization is a
    verified per-tenant, per-purpose identity is a throughput/latency gain
    considered for integration as an optimization inside the existing path.
    """
    if candidate.authorization_mode != "per_tenant_purpose_identity":
        return ServingPathIntegrationDecision(
            integrate=False,
            reason=(
                "refused: candidate relies on a single shared token, not a "
                "verified per-tenant, per-purpose identity"
            ),
        )
    gained = (
        candidate.throughput_rps > existing.throughput_rps
        or candidate.latency_p95_ms < existing.latency_p95_ms
    )
    if not gained:
        return ServingPathIntegrationDecision(
            integrate=False,
            reason="refused: no throughput or latency gain over the existing path",
        )
    return ServingPathIntegrationDecision(
        integrate=True,
        reason=(
            "gain over the existing path with a verified per-tenant, "
            "per-purpose identity"
        ),
    )
