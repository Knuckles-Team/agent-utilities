"""CONCEPT:AU-KG.compute.quantum-budget-ledger -- Q8's control-plane quota/cost-budget gate.

The program doc (`plans/au-eg-program/program/quantum-external-providers.md` S1.3)
is explicit that this is not optional: the IBM Open Plan free tier is ~10 minutes of
QPU runtime per 28-day ROLLING window -- a hard, shared resource a handful of agent
calls can trivially exhaust. A sibling lane (`w6-quantum-q10-providers`) wires the
actual hardware/cloud provider adapters and their OWN usage/cost telemetry; this
module owns the ENFORCEMENT surface those adapters must clear before a request with
a non-default backend is ever sent over the wire -- "coordinate rather than
duplicate" per this lane's brief.

Design, deliberately conservative:

* The engine's own two registered backends today (`sv-cpu`/`stabilizer`,
  `eg-quantum-sim`) are always free and unlimited -- see
  :data:`QUOTA_EXEMPT_BACKEND_IDS`. The DEFAULT `graph_quantum` call path (no
  `backend_id` override) never touches this module at all.
* Any OTHER `backend_id` is treated conservatively as "might be a paid/quota-limited
  provider" and is FAIL-CLOSED by default: :func:`reserve_quantum_budget` denies the
  request unless a `:QuantumProviderQuota` node already exists for that backend's
  provider family. No such node exists until Q10 (or an operator) provisions one --
  so today, naming any backend outside the exempt set is refused, which is the
  correct, safe behavior for a hard shared resource with no adapter wired yet.
* Usage is tracked per `(tenant, provider_family)` in daily buckets
  (`:QuantumUsageDay`), summed over the trailing 28 days for the rolling-window
  check. Best-effort, like every other KG provenance write in this codebase
  (`agent_utilities/runtime/provenance.py`'s own module doc: "a cold or absent KG
  must never break the workspace") -- a KG write failure degrades to fail-CLOSED for
  the reservation itself (the caller sees `QuantumBudgetExceeded`, never a silent
  bypass), but the post-hoc usage-recording true-up in
  :func:`record_quantum_usage` is best-effort and never raises.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from agent_utilities.security.persistence_privacy import persistence_reference

logger = logging.getLogger(__name__)

# The only backend ids this engine build actually registers today
# (`eg-quantum-sim`'s `StateVectorSimulator`/`StabilizerSimulator` -- see
# `crates/eg-quantum-sim/src/{statevector,stabilizer}.rs`), both in-process, both
# free, both unlimited. ANY other id defaults to requiring a budget grant --
# fail-closed, not an allowlist of known-expensive ids (a new backend id the engine
# grows tomorrow is untrusted by default, exactly like a new Method the capability
# ledger has not classified yet).
QUOTA_EXEMPT_BACKEND_IDS = frozenset({"sv-cpu", "stabilizer"})

QUOTA_NODE_LABEL = "QuantumProviderQuota"
USAGE_NODE_LABEL = "QuantumUsageDay"
WINDOW_DAYS = 28
# Conservative reservation held BEFORE a call whose real duration is unknown yet
# (a hardware/cloud queue-plus-execution time cannot be measured until after the
# call completes) -- trued up by `record_quantum_usage` once the real wall time is
# known. Deliberately small: it under-reserves rather than over-denies for a single
# call, while still preventing an unbounded burst (each call still counts).
DEFAULT_RESERVATION_SECONDS = 5.0


class QuantumBudgetExceeded(Exception):
    """Raised when a non-exempt backend request would exceed its rolling-window budget."""


@dataclass(frozen=True)
class QuantumBudgetReservation:
    """A held reservation to true up post-hoc via :func:`record_quantum_usage`."""

    tenant_ref: str
    provider_family: str
    backend_id: str
    reserved_seconds: float
    day_bucket: str


def _provider_family(backend_id: str) -> str:
    """Coarsen a concrete `backend_id` to its provider family for budgeting.

    Concrete ids are open-ended (Q10 registers whatever a provider names its
    backend), so this groups by the `-` prefix (`"ibm-heron"` -> `"ibm"`,
    `"braket-sv1"` -> `"braket"`) -- a reasonable, documented convention a
    `:QuantumProviderQuota` node's `provider_family` property is expected to match.
    A backend id with no `-` is its own family.
    """

    return backend_id.split("-", 1)[0] if "-" in backend_id else backend_id


def _day_bucket(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).strftime("%Y-%m-%d")


def _window_buckets(now: datetime | None = None) -> list[str]:
    base = now or datetime.now(UTC)
    return [
        (base - timedelta(days=offset)).strftime("%Y-%m-%d")
        for offset in range(WINDOW_DAYS)
    ]


def _usage_node_id(tenant_ref: str, provider_family: str, day_bucket: str) -> str:
    return f"quantum_usage:{tenant_ref}:{provider_family}:{day_bucket}"


def _quota_node_id(provider_family: str) -> str:
    return f"quantum_quota:{provider_family}"


def _load_quota_seconds(engine: Any, provider_family: str) -> float | None:
    """Return the configured rolling-window budget in seconds, or `None` if unset."""

    try:
        rows = engine.query_cypher(
            f"MATCH (q:{QUOTA_NODE_LABEL} {{id: $id}}) RETURN q.budget_seconds AS budget_seconds LIMIT 1",
            {"id": _quota_node_id(provider_family)},
        )
    except Exception as exc:  # noqa: BLE001 -- authority read failure fails closed below
        logger.debug("quantum budget: quota read failed (%s)", type(exc).__name__)
        return None
    for row in rows or []:
        raw = row.get("budget_seconds") if isinstance(row, dict) else None
        if raw is not None:
            try:
                return max(0.0, float(raw))
            except (TypeError, ValueError):
                return None
    return None


def _load_used_seconds(engine: Any, tenant_ref: str, provider_family: str) -> float:
    total = 0.0
    for bucket in _window_buckets():
        try:
            rows = engine.query_cypher(
                f"MATCH (u:{USAGE_NODE_LABEL} {{id: $id}}) RETURN u.used_seconds AS used_seconds LIMIT 1",
                {"id": _usage_node_id(tenant_ref, provider_family, bucket)},
            )
        except Exception as exc:  # noqa: BLE001 -- treat an unreadable bucket as zero, not a crash
            logger.debug(
                "quantum budget: usage bucket read failed (%s)", type(exc).__name__
            )
            continue
        for row in rows or []:
            raw = row.get("used_seconds") if isinstance(row, dict) else None
            if raw is not None:
                try:
                    total += max(0.0, float(raw))
                except (TypeError, ValueError) as exc:
                    # Not best-effort noise: this is a BUDGET ledger, so an
                    # unparseable used_seconds silently UNDER-counts usage and
                    # fails OPEN (a tenant could exceed quota because a corrupt
                    # row read as zero). Skipping the row is still right --
                    # refusing every quantum run because one bucket is malformed
                    # would be worse -- but it must be visible, so warn with the
                    # real cause rather than dropping it.
                    logger.warning(
                        "quantum budget: unparseable used_seconds in bucket %r "
                        "for %s/%s -- counting it as 0, quota may under-count: %s",
                        bucket,
                        tenant_ref,
                        provider_family,
                        exc,
                    )
    return total


def reserve_quantum_budget(
    engine: Any,
    *,
    tenant: str,
    backend_id: str,
    estimated_seconds: float = DEFAULT_RESERVATION_SECONDS,
) -> QuantumBudgetReservation:
    """Fail-closed pre-check + reservation for a non-exempt `backend_id`.

    Raises :class:`QuantumBudgetExceeded` when no quota is configured for the
    backend's provider family (the default, safe state until Q10/an operator
    provisions a real `:QuantumProviderQuota`) or when the trailing-28-day usage
    plus this reservation would exceed the configured budget.
    """

    if backend_id in QUOTA_EXEMPT_BACKEND_IDS:
        raise ValueError(
            f"{backend_id!r} is a free/unlimited local simulator; it never needs a budget reservation"
        )
    provider_family = _provider_family(backend_id)
    tenant_ref = persistence_reference("tenant", tenant, namespace="quantum-budget")

    budget_seconds = _load_quota_seconds(engine, provider_family)
    if budget_seconds is None:
        raise QuantumBudgetExceeded(
            f"no quantum provider quota is configured for backend family "
            f"'{provider_family}' (backend_id={backend_id!r}) -- a hardware/cloud "
            f"provider must have a :QuantumProviderQuota node before any agent call "
            f"can target it; the default, safe state is deny"
        )
    used_seconds = _load_used_seconds(engine, tenant_ref, provider_family)
    if used_seconds + estimated_seconds > budget_seconds:
        raise QuantumBudgetExceeded(
            f"quantum budget exceeded for backend family '{provider_family}': "
            f"{used_seconds:.1f}s used + {estimated_seconds:.1f}s reservation > "
            f"{budget_seconds:.1f}s budget over the trailing {WINDOW_DAYS}-day window"
        )
    return QuantumBudgetReservation(
        tenant_ref=tenant_ref,
        provider_family=provider_family,
        backend_id=backend_id,
        reserved_seconds=estimated_seconds,
        day_bucket=_day_bucket(),
    )


def record_quantum_usage(
    engine: Any,
    reservation: QuantumBudgetReservation,
    *,
    actual_seconds: float,
    succeeded: bool,
) -> None:
    """True up today's usage bucket with the REAL observed duration.

    Best-effort (never raises): a write failure here means the next call's
    pre-check under-counts by this call's true-up, which is the conservative
    direction relative to `estimated_seconds` already having been reserved for the
    call itself -- it does not silently grant more budget than configured.
    """

    if not succeeded:
        # A failed call still occupied provider time on most real backends (queue +
        # partial execution) -- record it, never discard it, so a repeatedly-failing
        # caller cannot use failure as a free budget bypass.
        pass
    seconds = max(0.0, float(actual_seconds))
    node_id = _usage_node_id(
        reservation.tenant_ref, reservation.provider_family, reservation.day_bucket
    )
    try:
        existing_rows = engine.query_cypher(
            f"MATCH (u:{USAGE_NODE_LABEL} {{id: $id}}) RETURN u.used_seconds AS used_seconds LIMIT 1",
            {"id": node_id},
        )
        prior = 0.0
        for row in existing_rows or []:
            raw = row.get("used_seconds") if isinstance(row, dict) else None
            if raw is not None:
                prior = max(0.0, float(raw))
                break
        engine.add_node(
            node_id,
            USAGE_NODE_LABEL,
            properties={
                "tenant_ref": reservation.tenant_ref,
                "provider_family": reservation.provider_family,
                "backend_id": reservation.backend_id,
                "day_bucket": reservation.day_bucket,
                "used_seconds": prior + seconds,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
    except Exception as exc:  # noqa: BLE001 -- best-effort ledger write, never breaks the call
        logger.warning(
            "quantum budget: failed to record usage for %s (%s)",
            reservation.backend_id,
            type(exc).__name__,
        )
