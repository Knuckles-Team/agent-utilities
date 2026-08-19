"""Shared typed profiling and read-only engine-client seams.

The Arrow kernel and a future epistemic-graph analytics client return the same
``ProfileResult``.  This module owns only result validation, stable digest
calculation and the explicitly authorized evidence-reference seam.  It never
writes, caches, persists or imports a dataframe/numeric runtime.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from typing import Any, Protocol

from .models import (
    ProfileEvidenceReference,
    ProfileRequest,
    ProfileResult,
    ProfileTarget,
)


class ProfileProtocolError(ValueError):
    """Raised when a profile client returns an unbounded or mismatched result."""


class ProfileEvidenceDenied(PermissionError):
    """Raised when native evidence authority is absent or returns an invalid ref."""


class ProfileClient(Protocol):
    """Read-only client protocol for a future engine-native profile operation."""

    def profile(self, request: ProfileRequest) -> ProfileResult:
        """Return the profile for exactly ``request.target``."""


class ProfileEvidenceAuthority(Protocol):
    """Separate native authority for profile-as-evidence references."""

    def authorize_profile_evidence(
        self,
        *,
        profile_digest: str,
        target: ProfileTarget,
    ) -> ProfileEvidenceReference:
        """Create a ChangeEnvelope-backed reference after policy admission."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def profile_digest(profile: ProfileResult) -> str:
    """Return a stable digest over the complete typed profile result."""

    payload = _canonical_json(profile.model_dump(mode="json"))
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _validate_bounds(result: ProfileResult, request: ProfileRequest) -> None:
    limits = request.limits
    if result.columns > limits.max_columns:
        raise ProfileProtocolError("engine profile exceeds the requested column limit")
    if len(result.column_profiles) > limits.max_columns:
        raise ProfileProtocolError("engine profile returned too many columns")
    if result.bytes_in_memory > limits.max_bytes:
        raise ProfileProtocolError("engine profile exceeds the requested byte limit")
    if result.disclosure_threshold != limits.disclosure_threshold:
        raise ProfileProtocolError("engine profile disclosure threshold differs")
    if len(result.warnings) > limits.max_warnings:
        raise ProfileProtocolError("engine profile returned too many warnings")
    if len(result.column_profiles) != result.columns:
        raise ProfileProtocolError("engine profile column count is inconsistent")
    expected_ordinals = (
        set(request.selector.ordinals) if request.selector is not None else None
    )
    actual_ordinals = {column.ordinal for column in result.column_profiles}
    if expected_ordinals is not None and actual_ordinals != expected_ordinals:
        raise ProfileProtocolError("engine profile columns differ from selector")
    for column in result.column_profiles:
        if column.distinct_count is not None and (
            column.distinct_count > limits.max_cardinality
        ):
            raise ProfileProtocolError("engine profile cardinality exceeds the limit")
        if len(column.top_k) > limits.max_top_k:
            raise ProfileProtocolError("engine profile returned too many top-k values")
        if len(column.quantiles) > limits.max_quantiles:
            raise ProfileProtocolError("engine profile returned too many quantiles")
        if any(item.count < limits.disclosure_threshold for item in column.top_k):
            raise ProfileProtocolError(
                "engine profile disclosed a below-threshold value"
            )


def validate_engine_profile(
    result: ProfileResult,
    request: ProfileRequest,
) -> ProfileResult:
    """Enforce exact target identity and every caller-declared result bound."""

    if not isinstance(result, ProfileResult):
        raise ProfileProtocolError("profile client must return ProfileResult")
    if result.algorithm != "epistemic-graph-profile":
        raise ProfileProtocolError("profile client returned the wrong authority")
    if result.target != request.target:
        raise ProfileProtocolError("profile target does not match the request")
    if result.schema_digest != request.target.schema_digest:
        raise ProfileProtocolError("profile schema digest does not match the request")
    if result.as_of_lsn != request.target.as_of_lsn:
        raise ProfileProtocolError("profile LSN does not match the request")
    _validate_bounds(result, request)
    return result


def profile_with_client(
    client: ProfileClient,
    request: ProfileRequest,
) -> ProfileResult:
    """Call one read-only engine client and validate its response exactly once."""

    started = time.monotonic()
    response = client.profile(request)
    if time.monotonic() - started > request.limits.deadline_ms / 1000:
        raise ProfileProtocolError("profile client exceeded the requested deadline")
    if inspect.isawaitable(response):
        raise ProfileProtocolError("profile client must expose a synchronous seam")
    return validate_engine_profile(response, request)


def build_profile_evidence_reference(
    profile: ProfileResult,
    *,
    authority: ProfileEvidenceAuthority | None,
) -> ProfileEvidenceReference:
    """Request a native ChangeEnvelope reference only from explicit authority."""

    if authority is None or not callable(
        getattr(authority, "authorize_profile_evidence", None)
    ):
        raise ProfileEvidenceDenied("native profile evidence authority is required")
    digest = profile_digest(profile)
    try:
        reference = authority.authorize_profile_evidence(
            profile_digest=digest,
            target=profile.target,
        )
    except Exception as exc:  # pragma: no cover - authority-specific failures
        raise ProfileEvidenceDenied(
            "native profile evidence authorization failed"
        ) from exc
    if not isinstance(reference, ProfileEvidenceReference):
        raise ProfileEvidenceDenied("native authority returned an invalid evidence ref")
    if reference.authority != "native_change_envelope":
        raise ProfileEvidenceDenied("profile evidence ref is not native")
    if reference.profile_digest != digest:
        raise ProfileEvidenceDenied("profile evidence digest does not match")
    return reference


__all__ = [
    "ProfileClient",
    "ProfileEvidenceAuthority",
    "ProfileEvidenceDenied",
    "ProfileProtocolError",
    "build_profile_evidence_reference",
    "profile_digest",
    "profile_with_client",
    "validate_engine_profile",
]
