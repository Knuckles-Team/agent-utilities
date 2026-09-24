"""Run one decision point: EG ``Decide`` first, the deterministic fallback on abstention.

The contract every consumer shares (lane decide-consumers, DESIGN §1):

1. an unbound point never calls EG -- the fallback answers (``unbound``);
2. otherwise EG's ``Decide`` is asked over the offered options; an executed,
   offered option is the answer, anything else leaves the fallback answering;
3. the record EG returned (executed or abstained) is made durable through
   ``DecisionLog.commit`` per the point's :class:`LogMode` -- EG re-runs the
   decision on the stored inputs before it accepts it;
4. when the fallback is an escalation (an LLM or a human, not AU's own
   deterministic rule) its answer re-enters EG as ``DecisionLog.resolve`` --
   a claim for a model, an observation for a human -- and never as a decision.

A transport failure costs exactly the fallback: EG being down or slow must
never stop a call site, and the choice says so in ``reason``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent_connector_sdk.ports.decide_runner import Fallback as SdkFallback

from agent_utilities.decide.options import Option, declared_source
from agent_utilities.decide.outcome import (
    Choice,
    Reading,
    read_batch,
    request_for,
    sampled,
)
from agent_utilities.decide.points import POINTS, Bindings, DecisionPoint, LogMode

logger = logging.getLogger(__name__)


class Escalated(str):
    """A fallback answer produced by an escalation rather than AU's own rule.

    It IS the chosen option id (a ``str``), so it satisfies the SDK's
    ``Fallback = Callable[[], str]`` contract unchanged; it also carries EG's
    ``AbstentionResolver`` wire value (``{"resolver": "human"}`` or
    ``{"resolver": "model", "producer": ...}``) and the resolution id.
    """

    resolver: Mapping[str, Any]
    resolution_id: str

    def __new__(
        cls, option_id: str, resolver: Mapping[str, Any], resolution_id: str
    ) -> Escalated:
        answer = super().__new__(cls, option_id)
        answer.resolver = dict(resolver)
        answer.resolution_id = resolution_id
        return answer

    @property
    def option_id(self) -> str:
        return str(self)


#: The deterministic (or escalated) answer when EG does not decide -- the SDK's type.
Fallback = SdkFallback


@dataclass(frozen=True, slots=True)
class _Prepared:
    point: DecisionPoint
    request: dict[str, Any] | None
    offered: frozenset[str]


def _loggable(point: DecisionPoint, record: Mapping[str, Any]) -> bool:
    if point.log_mode is LogMode.NEVER:
        return False
    return point.log_mode is LogMode.ALWAYS or sampled(point, record)


def _resolve_op(tenant: str, record_id: str, answer: Escalated) -> dict[str, Any]:
    return {
        "op": "resolve",
        "tenant_id": tenant,
        "resolution": {
            "record_id": record_id,
            "resolution_id": answer.resolution_id,
            "option_id": answer.option_id,
            "resolver": dict(answer.resolver),
        },
    }


def _settle(reading: Reading, logged: bool, fallback: Fallback) -> tuple[Choice, Any]:
    """The choice, plus the raw fallback answer (an :class:`Escalated` to record)."""
    if reading.option_id is not None:
        decided = Choice(
            reading.option_id,
            True,
            reading.reason,
            record=reading.record,
            logged=logged,
        )
        return decided, None
    answer = fallback()
    choice = Choice(
        str(answer),
        False,
        reading.reason,
        advisory=reading.advisory,
        record=reading.record,
        logged=logged,
    )
    return choice, answer


def _needs_resolution(choice: Choice, answer: Any) -> bool:
    return (
        isinstance(answer, Escalated) and choice.logged and choice.record_id is not None
    )


def _unavailable(exc: BaseException) -> Reading:
    return Reading(None, f"unavailable: {type(exc).__name__}: {exc}", None, {})


_UNBOUND = Reading(None, "unbound", None, {})


@dataclass
class DecisionRunner:
    """Decision points bound to one tenant, transport and binding source."""

    transport: Any
    bindings: Bindings
    tenant: str
    points: Mapping[str, DecisionPoint] = field(default_factory=lambda: POINTS)

    def _prepare(
        self,
        question_id: str,
        options: Sequence[Option],
        context: Mapping[str, Any],
    ) -> _Prepared:
        point = self.points[question_id]
        offered = frozenset(o.option_id for o in options)
        binding = self.bindings.binding_for(point)
        if binding is None or not offered:
            return _Prepared(point, None, offered)
        request = request_for(
            point,
            binding,
            tenant=self.tenant,
            candidates=context.get("candidates") or declared_source(options),
            params=context.get("params") or (),
        )
        return _Prepared(point, request, offered)

    async def _log(
        self, point: DecisionPoint, record: Mapping[str, Any] | None
    ) -> bool:
        if record is None or not _loggable(point, record):
            return False
        try:
            await self.transport.log({"op": "commit", "record": dict(record)})
        except Exception as exc:  # noqa: BLE001 — a log failure keeps the answer; the cause is logged
            logger.warning("decision %s not logged: %s", point.question_id, exc)
            return False
        return True

    async def _consult(self, prepared: _Prepared) -> tuple[Reading, bool]:
        """Ask EG and log its record; any failure is a reading, never a raise."""
        if prepared.request is None:
            return _UNBOUND, False
        try:
            batch = await self.transport.decide(prepared.request)
        except Exception as exc:  # noqa: BLE001 — EG unavailable costs only the fallback; cause kept in reason
            logger.warning(
                "decision %s unavailable: %s", prepared.point.question_id, exc
            )
            return _unavailable(exc), False
        reading = read_batch(batch, prepared.offered)
        return reading, await self._log(prepared.point, reading.record)

    async def _resolve(self, choice: Choice, answer: Any) -> None:
        if not _needs_resolution(choice, answer):
            return
        op = _resolve_op(self.tenant, str(choice.record_id), answer)
        try:
            await self.transport.log(op)
        except Exception as exc:  # noqa: BLE001 — the escalated answer stands; the cause is logged
            logger.warning("abstention %s not resolved: %s", choice.record_id, exc)

    def _run(self, call: Any, default: Any) -> Any:
        """Drive ``call`` from a sync site; a transport that cannot is ``default``."""
        try:
            return self.transport.run(call)
        except Exception as exc:  # noqa: BLE001 — no loop / timeout costs only the fallback; cause logged
            logger.warning("decision transport unavailable: %s", exc)
            return default(exc)

    def choose(
        self,
        question_id: str,
        options: Sequence[Option],
        fallback: Fallback,
        **context: Any,
    ) -> Choice:
        """Decide from a sync call site (EG is driven on the engine loop).

        ``context``: the SDK port's ``params`` / ``candidates`` keywords.
        """
        prepared = self._prepare(question_id, options, context)
        reading, logged = (
            self._run(self._consult(prepared), lambda exc: (_unavailable(exc), False))
            if prepared.request is not None
            else (_UNBOUND, False)
        )
        choice, answer = _settle(reading, logged, fallback)
        if _needs_resolution(choice, answer):
            self._run(self._resolve(choice, answer), lambda exc: None)
        return choice

    async def achoose(
        self,
        question_id: str,
        options: Sequence[Option],
        fallback: Fallback,
        **context: Any,
    ) -> Choice:
        """Decide from an async call site (same ``context`` as :meth:`choose`)."""
        prepared = self._prepare(question_id, options, context)
        reading, logged = await self._consult(prepared)
        choice, answer = _settle(reading, logged, fallback)
        await self._resolve(choice, answer)
        return choice


__all__ = ["DecisionRunner", "Escalated", "Fallback"]
