"""Durable single-owner leases for messaging inbound intake.

Credentials make a backend usable for governed outbound sends; they do not, by
themselves, grant permission to open an inbound poller.  An embedded intake
owner must acquire the deterministic WorkItem for each ``(platform, bot)``
identity.  The engine-native claim/renew/reclaim verbs remain the only lease
authority, so another process can take over after an expired owner lease.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from typing import Any

from agent_utilities.orchestration.work_item import (
    NativeWorkItemRequired,
    WorkItemBackendUnavailable,
    claim_specific,
    defer_work_item,
    get_work_item,
    heartbeat,
    submit_work_item_atomic,
)

logger = logging.getLogger(__name__)

# The lease is intentionally short enough to make a crashed poller fail over in
# bounded time, while the renewal loop keeps a healthy long-poll owner alive.
DEFAULT_LEASE_TTL_S = 90.0
_RENEW_INTERVAL_MIN_S = 0.01
_RENEW_INTERVAL_MAX_S = 30.0
_LEASE_KIND = "messaging_intake_lease"
_LEASE_QUEUE = "messaging-intake"
_LEASE_RESOURCE_CLASS = "messaging-intake"
# A lease generation is not a completed unit of work.  Keep the durable row
# reclaimable for the life of the deployment instead of dead-lettering it after
# the generic WorkItem default of three attempts.
_MAX_LEASE_ATTEMPTS = 2**31 - 1


@dataclasses.dataclass(frozen=True)
class IntakeLease:
    """One engine-issued claim authorizing a platform's inbound poller."""

    platform: str
    item_id: str
    claim: dict[str, Any]
    lease_ttl_s: float


def _session_tenant(session: Any) -> str:
    tenant = str(getattr(session, "tenant", "") or "").strip()
    if not tenant:
        raise NativeWorkItemRequired(
            "messaging intake lease requires a verified non-empty session tenant"
        )
    return tenant


def _identity_digest(platform: str) -> str:
    """Return a stable, non-secret identity reference for one backend config."""
    from agent_utilities.messaging.registry import MessagingRegistry

    config = MessagingRegistry.instance()._auto_config(platform)
    token = str(getattr(config, "token", "") or "").strip()
    app_id = str(getattr(config, "app_id", "") or "").strip()
    webhook_url = str(getattr(config, "webhook_url", "") or "").strip()

    # Telegram's bot id is the stable identity before the colon; hashing it
    # keeps even that identifier out of the durable WorkItem metadata.  Other
    # platforms use the configured app id or a hash of the bot token.  A
    # platform-only fallback is useful for webhook/test backends with no token.
    if platform == "telegram" and token:
        material = token.split(":", 1)[0]
    else:
        material = app_id or token or webhook_url or platform
    return hashlib.sha256(f"{platform}:{material}".encode()).hexdigest()


def _work_item_id(platform: str, identity_digest: str) -> str:
    key = f"messaging-intake:{platform}:{identity_digest}"
    return f"workitem:messaging-intake:{hashlib.sha256(key.encode()).hexdigest()}"


def _validate_lease_row(
    engine: Any,
    *,
    item_id: str,
    platform: str,
    identity_digest: str,
    tenant: str,
) -> None:
    item = get_work_item(engine, item_id)
    metadata = item.get("metadata") if item is not None else None
    if (
        item is None
        or item.get("kind") != _LEASE_KIND
        or item.get("queue") != _LEASE_QUEUE
        or item.get("tenant") != tenant
        or not isinstance(metadata, dict)
        or metadata.get("platform") != platform
        or metadata.get("identity_digest") != identity_digest
    ):
        # Never claim a row whose immutable identity does not match this
        # platform/configuration.  A collision or operator-created row fails
        # closed instead of becoming an alternate ownership authority.
        raise WorkItemBackendUnavailable(
            f"messaging intake lease row {item_id!r} failed identity validation"
        )


def acquire_intake_lease(
    engine: Any,
    platform: str,
    session: Any,
    *,
    lease_ttl_s: float = DEFAULT_LEASE_TTL_S,
) -> IntakeLease | None:
    """Submit and atomically claim the one durable lease for ``platform``.

    ``None`` is an authoritative non-owner result from the engine.  Missing
    native WorkItem capabilities raise so the caller can log a fail-closed
    deployment error rather than accidentally starting an un-fenced poller.
    """
    if lease_ttl_s <= 0:
        raise ValueError("lease_ttl_s must be positive")
    platform = str(platform).strip()
    if not platform:
        raise ValueError("platform must be non-empty")

    tenant = _session_tenant(session)
    identity_digest = _identity_digest(platform)
    item_id = _work_item_id(platform, identity_digest)
    submit_work_item_atomic(
        engine,
        kind=_LEASE_KIND,
        queue=_LEASE_QUEUE,
        tenant=tenant,
        work_item_id=item_id,
        idempotency_key=item_id,
        resource_class=_LEASE_RESOURCE_CLASS,
        fairness_group=f"messaging:{platform}",
        max_attempts=_MAX_LEASE_ATTEMPTS,
        description="messaging inbound intake ownership lease",
        created_by="messaging-intake",
        metadata={
            "platform": platform,
            "identity_digest": identity_digest,
        },
    )
    _validate_lease_row(
        engine,
        item_id=item_id,
        platform=platform,
        identity_digest=identity_digest,
        tenant=tenant,
    )

    owner_ref = f"messaging-intake:{platform}:{uuid.uuid4().hex}"
    claim = claim_specific(
        engine,
        item_id,
        token=owner_ref,
        lease_ttl_s=lease_ttl_s,
    )
    if claim is None:
        logger.info(
            "messaging intake lease is held by another owner: platform=%s", platform
        )
        return None
    if not claim.get("_native") or claim.get("tenant") != tenant:
        raise NativeWorkItemRequired(
            f"messaging intake lease claim for {platform!r} lacked native tenant authority"
        )
    return IntakeLease(
        platform=platform,
        item_id=item_id,
        claim=claim,
        lease_ttl_s=lease_ttl_s,
    )


def acquire_intake_leases(
    engine: Any,
    platforms: Iterable[str],
    session: Any,
    *,
    lease_ttl_s: float = DEFAULT_LEASE_TTL_S,
) -> tuple[IntakeLease, ...]:
    """Acquire every platform lease that this process may safely poll.

    A platform with no native lease capability or a live competing owner is
    omitted.  The caller must pass only the returned platforms to the listener;
    it must never fall back to the configured list.
    """
    leases: list[IntakeLease] = []
    for platform in dict.fromkeys(str(value).strip() for value in platforms):
        if not platform:
            continue
        try:
            lease = acquire_intake_lease(
                engine,
                platform,
                session,
                lease_ttl_s=lease_ttl_s,
            )
        except Exception as exc:  # noqa: BLE001 — intake must fail closed on any lease error
            logger.error(
                "messaging intake lease unavailable: platform=%s error_type=%s; "
                "refusing inbound polling",
                platform,
                type(exc).__name__,
            )
            continue
        if lease is not None:
            leases.append(lease)
    return tuple(leases)


def _release_intake_lease(engine: Any, lease: IntakeLease) -> None:
    """Return a healthy lease to ``ready``; expiry remains the crash fallback."""
    try:
        defer_work_item(
            engine,
            lease.item_id,
            lease.claim,
            next_retry_at=time.time(),
            reason_ref="messaging_intake_stopped",
        )
    except Exception as exc:  # noqa: BLE001 — expiry is the durable crash fallback
        logger.debug(
            "messaging intake lease release deferred to expiry: platform=%s error_type=%s",
            lease.platform,
            type(exc).__name__,
        )


def run_with_intake_leases(
    engine: Any,
    leases: tuple[IntakeLease, ...],
    stop_event: threading.Event,
    serve: Callable[[list[str], threading.Event], None],
) -> None:
    """Renew ``leases`` while ``serve`` owns the corresponding pollers."""
    if not leases:
        stop_event.set()
        return

    renew_stop = threading.Event()

    def _renew() -> None:
        interval = min(
            _RENEW_INTERVAL_MAX_S,
            max(_RENEW_INTERVAL_MIN_S, min(lease.lease_ttl_s for lease in leases) / 3),
        )
        while not renew_stop.wait(interval):
            if stop_event.is_set():
                return
            for lease in leases:
                try:
                    renewed = heartbeat(
                        engine,
                        lease.item_id,
                        lease.claim,
                        lease_ttl_s=lease.lease_ttl_s,
                    )
                except Exception as exc:  # noqa: BLE001 — losing a lease must stop polling
                    logger.error(
                        "messaging intake lease renewal failed: platform=%s error_type=%s",
                        lease.platform,
                        type(exc).__name__,
                    )
                    renewed = False
                if not renewed:
                    logger.error(
                        "messaging intake lease lost: platform=%s; stopping inbound polling",
                        lease.platform,
                    )
                    stop_event.set()
                    return

    renew_thread = threading.Thread(
        target=_renew,
        daemon=True,
        name="MessagingIntakeLeaseRenewal",
    )
    renew_thread.start()
    try:
        serve([lease.platform for lease in leases], stop_event)
    finally:
        renew_stop.set()
        renew_thread.join(
            timeout=max(
                1.0,
                min(_RENEW_INTERVAL_MAX_S, min(lease.lease_ttl_s for lease in leases)),
            )
        )
        for lease in leases:
            _release_intake_lease(engine, lease)


def run_owned_intake(
    engine: Any,
    platforms: Iterable[str],
    session: Any,
    stop_event: threading.Event,
    serve: Callable[[list[str], threading.Event], None],
    *,
    lease_ttl_s: float = DEFAULT_LEASE_TTL_S,
) -> None:
    """Run ``serve`` only after native leases admit this caller.

    This is the one synchronous boundary shared by the standalone messaging
    daemon and the embedded graph-os co-service. Callers must provide the
    deployment's explicit, verified session; credentials/platform discovery
    alone never reaches ``serve``. A missing lease (including an unavailable
    native WorkItem backend) stops the caller without opening a poller.
    """
    leases = acquire_intake_leases(
        engine,
        platforms,
        session,
        lease_ttl_s=lease_ttl_s,
    )
    if not leases:
        logger.error("messaging inbound intake has no native lease; refusing to poll")
        stop_event.set()
        return
    run_with_intake_leases(engine, leases, stop_event, serve)


__all__ = [
    "DEFAULT_LEASE_TTL_S",
    "IntakeLease",
    "acquire_intake_lease",
    "acquire_intake_leases",
    "run_owned_intake",
    "run_with_intake_leases",
]
