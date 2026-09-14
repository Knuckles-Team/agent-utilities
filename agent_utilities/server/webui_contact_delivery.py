"""Governed agent-webui contact delivery through native AU authorities.

Only the atomic WorkItem creator may invoke messaging. Contact PII stays in
the in-memory provider payload; graph state contains only HMAC references and
an opaque receipt.

CONCEPT:AU-ECO.messaging.messaging-reach-service-governed
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent_utilities.knowledge_graph.core.work_durability import (
    claim_specific,
    commit_result,
    get_work_item,
    submit_work_item_atomic,
)
from agent_utilities.messaging.service import MessagingService
from agent_utilities.security.persistence_privacy import persistence_reference
from agent_utilities.server.webui_contact_governance import (
    ContactDeliveryContext,
    allow_shared_attempt,
    destination_parts,
    digest,
    existing_receipt,
    prepare_contact,
    work_item_submission,
)

if TYPE_CHECKING:
    from agent_webui.contact_delivery import (
        ContactDeliveryRequest,
        ContactDeliveryResult,
    )

logger = logging.getLogger(__name__)

SyncRunner = Callable[[Callable[[], Any]], Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class _Admission:
    claim: dict[str, Any] | None = None
    result: ContactDeliveryResult | None = None


def _result(
    receipt: str | None = None, *, error: BaseException | None = None
) -> ContactDeliveryResult:
    from agent_webui.contact_delivery import ContactDeliveryResult

    if error is not None:
        logger.warning(
            "agent-webui contact delivery outcome unknown (%s)",
            type(error).__name__,
        )
    return ContactDeliveryResult(success=receipt is not None, receipt=receipt)


def _replay_admission(
    row: dict[str, Any] | None, context: ContactDeliveryContext
) -> _Admission:
    receipt = existing_receipt(row, context.request_digest)
    return _Admission(result=_result(receipt))


class WebUIContactDelivery:
    """Host adapter satisfying agent-webui's governed contact port."""

    def __init__(
        self,
        engine: Any,
        *,
        fixed_destination: str,
        sync_runner: SyncRunner,
        messaging_service: MessagingService | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not callable(sync_runner):
            raise TypeError("sync_runner must be callable")
        if destination_parts(fixed_destination) is None:
            raise ValueError("fixed_destination must be a valid platform:channel pair")
        if self._authority(engine) is None:
            raise ValueError("engine must expose the complete native contact authority")
        self._engine = engine
        self._fixed_destination = fixed_destination
        self._sync_runner = sync_runner
        self._messaging_service = messaging_service
        self._clock = clock

    @property
    def supports_atomic_idempotency(self) -> bool:
        """Declare support only while the active native authority is present."""
        return self._authority(self._engine) is not None

    @property
    def supports_shared_rate_limit(self) -> bool:
        """The same control authority owns limiter create/read/CAS."""
        return self._authority(self._engine) is not None

    async def __call__(self, request: ContactDeliveryRequest) -> ContactDeliveryResult:
        """Return success only after provider confirmation and durable commit."""
        preparation_error: BaseException | None = None
        try:
            context = prepare_contact(
                request, fixed_destination=self._fixed_destination
            )
        except Exception as exc:  # noqa: BLE001 — authority/HMAC failure is unknown
            preparation_error = exc
            context = None
        if context is None:
            return _result(error=preparation_error)

        try:
            admission = await self._sync_runner(
                lambda: self._admit(self._engine, context)
            )
        except Exception as exc:  # noqa: BLE001 — graph ambiguity is unknown
            return _result(error=exc)
        if admission.result is not None:
            return admission.result

        outcome = _result()
        if admission.claim is not None:
            try:
                outcome = await self._send_created(
                    self._engine, request, context, admission.claim
                )
            except Exception as exc:  # noqa: BLE001 — provider/commit ambiguity
                outcome = _result(error=exc)
        return outcome

    def _admit(self, engine: Any, context: ContactDeliveryContext) -> _Admission:
        authority = self._authority(engine)
        if authority is None:
            return _Admission(result=_result())

        existing = get_work_item(engine, context.item_id)
        if existing is not None:
            return _replay_admission(existing, context)
        if not allow_shared_attempt(authority, context, now=self._clock()):
            return _Admission(result=_result())

        _, created = submit_work_item_atomic(
            engine,
            **work_item_submission(context),
        )
        if not created:
            replay = get_work_item(engine, context.item_id)
            return _replay_admission(replay, context)
        claim = claim_specific(
            engine,
            context.item_id,
            token=f"webui-contact:{digest(context.item_id)[:32]}",
            lease_ttl_s=30.0,
        )
        return _Admission(claim=claim)

    @staticmethod
    def _authority(engine: Any) -> Any | None:
        authority = getattr(engine, "_work_item_engine", None)
        capabilities = (
            "query_cypher",
            "create_node_if_absent",
            "compare_and_set_node_fields",
            "claim_work_item",
            "commit_work_item_result",
        )
        fully_wired = authority is not None and all(
            callable(getattr(authority, name, None)) for name in capabilities
        )
        return authority if fully_wired else None

    async def _send_created(
        self,
        engine: Any,
        request: ContactDeliveryRequest,
        context: ContactDeliveryContext,
        claim: dict[str, Any],
    ) -> ContactDeliveryResult:
        submission = request.submission
        text = (
            "New agent-webui contact submission\n"
            f"Name: {submission.name}\nEmail: {submission.email}\n"
            f"Subject: {submission.subject}\n\n{submission.message}"
        )
        service = self._messaging_service or MessagingService.instance(engine=engine)
        send_result = await service.send(
            *context.destination,
            text,
            source="webui_contact",
            reason="deliver authenticated website contact submission",
            persist_outbound=False,
            policy_runner=self._sync_runner,
        )
        if not send_result.success:
            await self._sync_runner(
                lambda: commit_result(
                    engine,
                    context.item_id,
                    claim,
                    outcome="failed",
                    error_ref="contact_delivery_unconfirmed",
                    retryable=False,
                )
            )
            return _result()

        receipt_digest = persistence_reference(
            "contact_receipt",
            f"{context.item_id}\x00{context.request_digest}",
            namespace="webui-contact",
        ).rsplit("_", 1)[-1]
        receipt = f"contact_{receipt_digest[:32]}"
        committed = await self._sync_runner(
            lambda: commit_result(
                engine,
                context.item_id,
                claim,
                outcome="succeeded",
                result_ref=receipt,
                retryable=False,
            )
        )
        if committed == "committed":
            return _result(receipt)
        if committed == "noop":
            replay = await self._sync_runner(
                lambda: get_work_item(engine, context.item_id)
            )
            return _result(existing_receipt(replay, context.request_digest))
        return _result()


def build_webui_contact_delivery(
    sync_runner: SyncRunner,
) -> WebUIContactDelivery | None:
    """Build only when the server has a valid delivery-only contact policy."""
    from agent_webui.contact_delivery import load_contact_delivery_config

    config = load_contact_delivery_config()
    if (
        config is None
        or config.retention_days != 0
        or destination_parts(config.destination) is None
    ):
        return None
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if WebUIContactDelivery._authority(engine) is None:
        return None
    return WebUIContactDelivery(
        engine, fixed_destination=config.destination, sync_runner=sync_runner
    )
