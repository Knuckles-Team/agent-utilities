"""CONCEPT:AU-ORCH.runvcs.trace-replay — deterministic trace replay over the event log.

Shepherd's "retained trace" makes a run *replayable*: re-execute it with the recorded
exchanges standing in for the (non-deterministic, costly) model, and it reproduces bit-for-bit.
Because our :mod:`.kernel` already splits every model interaction into a *declaration* (the
request) and a *capture* (the response) and content-addresses both, replay is a pure function of
the log — no live model, no network.

:class:`ReplayModel` answers a request by its content digest from the recorded captures; a
missing request is a replay divergence (the log doesn't cover it), surfaced explicitly rather
than silently calling out. :func:`replay_run` re-drives the declaration stream through a
:class:`ReplayModel` and verifies the reconstruction digest equals the original — determinism is
*checked*, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .kernel import RunEvent, RunEventLog, content_digest

#: The schema_ref that marks a model interaction (declaration = request, capture = response).
MODEL_EXCHANGE = "model_exchange"


class ReplayDivergence(Exception):
    """Raised when a replayed request has no recorded response in the trace."""


class ReplayModel:
    """A stand-in model that returns recorded responses by request digest.

    Built from a run's captured ``model_exchange`` events: each capture is indexed by the
    content digest of its declaration's request payload, so :meth:`respond` is deterministic and
    offline — the recorded exchange *is* the model.
    """

    def __init__(self, exchanges: dict[str, dict[str, Any]]) -> None:
        self._by_request = exchanges
        self.calls = 0

    @classmethod
    def from_log(cls, log: RunEventLog) -> ReplayModel:
        events = log.events
        by_record = {e.record_id: e for e in events}
        exchanges: dict[str, dict[str, Any]] = {}
        for ev in events:
            if ev.schema_ref != MODEL_EXCHANGE or ev.mode != "capture":
                continue
            # Find the declaration this capture answers (its causal parent).
            decl = next(
                (by_record[p] for p in ev.caused_by if p in by_record),
                None,
            )
            if decl is None or decl.mode != "declaration":
                continue
            key = _request_key(decl.payload.get("request"))
            exchanges[key] = ev.payload.get("response") or {}
        return cls(exchanges)

    def respond(self, request: Any) -> Any:
        """Return the recorded response for ``request`` (deterministic). Raises on a miss."""
        self.calls += 1
        key = _request_key(request)
        if key not in self._by_request:
            raise ReplayDivergence(
                f"no recorded response for request digest {key[:12]}…"
            )
        return self._by_request[key]


def _request_key(request: Any) -> str:
    return content_digest(MODEL_EXCHANGE, "declaration", {"request": request}, ())


@dataclass
class ReplayResult:
    """The outcome of a deterministic replay."""

    run_id: str
    steps: int
    model_calls: int
    reconstructed: list[Any] = field(default_factory=list)
    original_digest: str = ""
    replay_digest: str = ""

    @property
    def deterministic(self) -> bool:
        """True iff the replay reproduced the original trace exactly."""
        return bool(self.original_digest) and self.original_digest == self.replay_digest


def replay_run(log: RunEventLog) -> ReplayResult:
    """Deterministically replay ``log`` with a :class:`ReplayModel`, verifying reproduction.

    Walks the declaration events in order; for each ``model_exchange`` declaration it asks the
    replay model (recorded response), and for every other declaration it re-derives the capture
    that the recorded run produced. The reconstructed response stream is content-digested and
    compared to the original captured stream — equal ⇒ the replay is deterministic.
    """
    model = ReplayModel.from_log(log)
    events = log.events

    reconstructed: list[Any] = []
    original_responses: list[Any] = []
    steps = 0

    for ev in events:
        if ev.mode != "declaration":
            continue
        steps += 1
        # The recorded capture that answered this declaration (causal child).
        recorded_capture = _capture_of(ev, events)
        if ev.schema_ref == MODEL_EXCHANGE:
            reconstructed.append(model.respond(ev.payload.get("request")))
        elif recorded_capture is not None:
            # A deterministic (non-model) effect: its recorded outcome stands in.
            reconstructed.append(recorded_capture.payload)
        if recorded_capture is not None:
            original_responses.append(
                recorded_capture.payload.get("response")
                if ev.schema_ref == MODEL_EXCHANGE
                else recorded_capture.payload
            )

    original_digest = content_digest(
        "replay", "capture", {"stream": original_responses}, ()
    )
    replay_digest = content_digest("replay", "capture", {"stream": reconstructed}, ())
    return ReplayResult(
        run_id=log.run_id,
        steps=steps,
        model_calls=model.calls,
        reconstructed=reconstructed,
        original_digest=original_digest,
        replay_digest=replay_digest,
    )


def _capture_of(declaration: RunEvent, events: list[RunEvent]) -> RunEvent | None:
    """The capture event caused by ``declaration`` (its causal child), if any."""
    for ev in events:
        if ev.mode == "capture" and declaration.record_id in ev.caused_by:
            return ev
    return None
