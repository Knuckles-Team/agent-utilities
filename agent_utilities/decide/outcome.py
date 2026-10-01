"""Pure halves of a decision: the ``Decide`` request and the reading of its answer.

Nothing here performs I/O, so the sync and async runners share it byte for
byte. The reading enforces the one safety rule AU owns: EG may only pick an
option the caller offered. Anything else -- an unknown option, an advisory
score, an abstention -- leaves the call site on its deterministic fallback.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from agent_connector_sdk.decide.outcome import Choice as SdkChoice

from agent_utilities.decide.points import Binding, DecisionPoint

#: EG outcomes that execute an option.
_EXECUTING = frozenset({"acted", "explored"})


@dataclass(frozen=True, slots=True)
class Choice(SdkChoice):
    """The connector SDK's :class:`~agent_connector_sdk.decide.outcome.Choice`
    (``option_id``, ``decided``, ``reason``, ``advisory``) plus what AU records:
    the EG ``record`` (executed or abstained) and whether it was ``logged``.
    One choice type answers both sides, so an AU runner installed into the SDK
    satisfies the SDK's ``DecisionRunner`` port as-is.
    """

    record: Mapping[str, Any] | None = None
    logged: bool = False

    @property
    def record_id(self) -> str | None:
        return None if self.record is None else str(self.record.get("record_id"))


@dataclass(frozen=True, slots=True)
class Reading:
    """EG's answer, before the fallback fills any gap."""

    option_id: str | None
    reason: str
    record: Mapping[str, Any] | None
    advisory: Mapping[str, int]


def request_for(
    point: DecisionPoint,
    binding: Binding,
    *,
    tenant: str,
    candidates: Mapping[str, Any],
    params: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """The ``Decide`` request for one point (params sorted by name, as EG requires)."""
    return {
        "tenant_id": tenant,
        "question": {
            "question_id": point.question_id,
            "kind": point.kind,
            "safety": point.safety,
        },
        "candidates": dict(candidates),
        "feature_schema": dict(binding.feature_schema),
        "head": None if binding.head is None else dict(binding.head),
        "policy": dict(binding.policy),
        "params": sorted((dict(p) for p in params), key=lambda p: str(p["name"])),
        "max_records": 1,
    }


def _evaluator(
    binding: Binding | None, role: str | None, now_ms: int
) -> dict[str, Any] | None:
    """Who the commit names to evaluate: the binding's principal, else the
    point's declared policy role, else nobody."""
    principal = None if binding is None else binding.evaluator
    if principal is None and role is None:
        return None
    ttl_s = 3600 if binding is None else binding.evaluator_ttl_s
    return {
        "principal": principal,
        "role": None if principal is not None else role,
        "expires_at_ms": now_ms + 1000 * ttl_s,
    }


def commit_op(
    record: Mapping[str, Any],
    binding: Binding | None,
    now_ms: int,
    role: str | None = None,
) -> dict[str, Any]:
    """The ``DecisionLog.commit`` of ``record``, naming its evaluator -- the
    binding's principal or the point's policy ``role`` (EG issues an expiring,
    record-scoped, evaluation-only grant)."""
    return {
        "op": "commit",
        "record": dict(record),
        "evaluator": _evaluator(binding, role, now_ms),
    }


def _record_of(batch: Any) -> Mapping[str, Any] | None:
    records = batch.get("records") if isinstance(batch, Mapping) else None
    if not records or not isinstance(records[0], Mapping):
        return None
    return records[0]


def _reasons(outcome: Mapping[str, Any]) -> str:
    reasons = outcome.get("reasons") or []
    names = [str(r.get("reason")) for r in reasons if isinstance(r, Mapping)]
    return ",".join(names) or "unspecified"


def _advisory(outcome: Mapping[str, Any]) -> dict[str, int]:
    scores = outcome.get("scores") or []
    return {
        str(s["option_id"]): int(s["score"]["value"])
        for s in scores
        if isinstance(s, Mapping) and isinstance(s.get("score"), Mapping)
    }


def read_batch(batch: Any, offered: frozenset[str]) -> Reading:
    """Read one ``DecisionBatch``: an executed, offered option, or why not."""
    record = _record_of(batch)
    if record is None:
        return Reading(None, "unavailable: empty decision batch", None, {})
    outcome = record.get("outcome")
    if not isinstance(outcome, Mapping):
        return Reading(None, "unavailable: malformed outcome", record, {})
    kind = str(outcome.get("outcome"))
    if kind in _EXECUTING:
        chosen = str(outcome.get("option_id"))
        if chosen in offered:
            return Reading(chosen, kind, record, {})
        return Reading(None, f"foreign_option: {chosen}", record, {})
    if kind == "advisory":
        return Reading(None, "advisory", record, _advisory(outcome))
    return Reading(None, f"abstained: {_reasons(outcome)}", record, {})


def sampled(point: DecisionPoint, record: Mapping[str, Any]) -> bool:
    """Whether ``record`` falls in the point's reproducible log sample."""
    every = max(1, point.sample_every)
    digest = str(record.get("record_digest") or "")
    tail = digest.rsplit(":", 1)[-1][-8:] or "0"
    try:
        return int(tail, 16) % every == 0
    except ValueError:
        return False


__all__ = ["Choice", "Reading", "commit_op", "read_batch", "request_for", "sampled"]
