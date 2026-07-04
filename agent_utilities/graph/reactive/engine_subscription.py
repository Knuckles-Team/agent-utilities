#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:AU-KG.compute.change-feed-subscription — Engine change-feed subscription primitive for poll→push reactivity.

The ONE reusable primitive that turns a wasteful daemon poll-loop into a reactive
consumer of the engine's *committed-change* feed. The Rust ``epistemic-graph``
engine (engine concepts KG-2.229/230, ``client.streaming``) already pushes an ordered,
cursor-addressable record of every durable mutation per graph; today agent-utilities
only fans writes INTO the engine (the in-process :class:`~agent_utilities.graph.reactive`
event-sourcing layer) and never consumes that feed back. This module closes the loop.

Two delivery surfaces over the SAME cursor, both built on
``streaming.cdc_read`` / ``streaming.watch`` (no side-channel socket — the engine
serves them over the existing framed-MessagePack transport):

* **cold-start catch-up** — :meth:`EngineSubscription.catch_up` reads the bounded
  tail of the CDC feed via ``cdc_read`` (up to ``catch_up_limit`` events) so a
  freshly-started daemon converges to "caught up" without re-scanning the whole
  graph history. After catch-up the cursor sits at ``last_seq + 1``.
* **incremental push** — :meth:`EngineSubscription.poll` does ONE ``watch``
  long-poll from the current cursor (filtered by ``label``), delivering only the
  changes since last time and advancing the cursor. With ``block_ms=0`` (the
  daemon-tick default) it returns immediately with whatever is pending — so a tick
  does O(new-changes) work, not O(whole-history); with ``block_ms>0`` it blocks up
  to that long for the first change (a dedicated reactive thread).

Both deliver each change to the registered ``handler(event)`` (a ``CdcEvent`` dict:
``seq``/``kind``/``node_id``/``label``/``before``/``after``). The handler decides
what to do — re-specialize a world model, fire an autoscale evaluation, etc. The
subscription owns only *cursor management + change delivery*; it is the single
well-tested seam the reactive consumers share instead of three ad-hoc loops.

Resolution: :func:`resolve_streaming` extracts ``(streaming_client, graph_name)``
from a :class:`~agent_utilities.knowledge_graph.core.graph_compute.GraphComputeEngine`
(or an ``EpistemicGraphBackend`` / anything exposing ``._client.streaming`` +
``.graph_name``). On a backend without the engine streaming feature (a non-engine
mirror, or an engine built without ``streaming``) it returns ``None`` and the
subscription degrades to a no-op — the caller's periodic reconcile keeps it correct.
"""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

#: A delivered change is the engine's ``CdcEvent`` dict (seq/kind/node_id/label/…).
ChangeEvent = dict[str, Any]
ChangeHandler = Callable[[ChangeEvent], None]


def resolve_streaming(source: Any) -> tuple[Any, str] | None:
    """Resolve ``(streaming_client, graph_name)`` from an engine/backend/compute.

    Accepts a :class:`GraphComputeEngine` (``._client.streaming`` + ``.graph_name``),
    an ``EpistemicGraphBackend`` (``.graph`` → the compute engine), or any object
    that already exposes ``._client.streaming`` and ``.graph_name``. Returns
    ``None`` when no engine streaming surface is reachable (non-engine backend, or
    an engine build without the ``streaming`` feature) so callers can degrade to
    their periodic reconcile path instead of crashing.
    """
    if source is None:
        return None
    # An EpistemicGraphBackend exposes the underlying compute engine via ``.graph``.
    compute = getattr(source, "graph", None)
    if compute is not None and hasattr(compute, "_client"):
        source = compute
    client = getattr(source, "_client", None)
    graph_name = getattr(source, "graph_name", None)
    if client is None or not graph_name:
        return None
    streaming = getattr(client, "streaming", None)
    if streaming is None:
        return None
    return streaming, str(graph_name)


class EngineSubscription:
    """A cursor over one graph's engine change-feed, delivered to a handler.

    CONCEPT:AU-KG.compute.change-feed-subscription. Construct from an engine/backend/compute (``source``), a
    node ``label`` to filter on (``""`` ⇒ all), and a ``handler`` invoked once per
    delivered change. The subscription tracks ``cursor`` (the next CDC seq to
    read). ``available`` is ``False`` when no engine streaming surface was
    resolvable — every method is then a safe no-op (returns 0), so a caller wires
    it unconditionally and keeps its periodic reconcile as the safety net.
    """

    def __init__(
        self,
        source: Any,
        *,
        label: str = "",
        handler: ChangeHandler | None = None,
        catch_up_limit: int = 4096,
    ) -> None:
        self.label = label
        self.handler = handler
        self.catch_up_limit = max(1, int(catch_up_limit))
        self.cursor = 0
        self._caught_up = False
        resolved = resolve_streaming(source)
        if resolved is None:
            self._streaming = None
            self.graph_name = ""
        else:
            self._streaming, self.graph_name = resolved

    @property
    def available(self) -> bool:
        """True when an engine streaming surface was resolved (else no-op mode)."""
        return self._streaming is not None

    def _matches(self, event: ChangeEvent) -> bool:
        """Server-side ``watch``/``cdc_read`` already filters by label when set;
        ``cdc_read`` does not, so re-apply the label filter for the catch-up path."""
        if not self.label:
            return True
        return str(event.get("label") or "") == self.label

    def _deliver(self, events: list[ChangeEvent]) -> int:
        delivered = 0
        for event in events:
            if not self._matches(event):
                continue
            if self.handler is not None:
                try:
                    self.handler(event)
                except Exception as exc:  # noqa: BLE001 — one bad handler call never wedges the feed
                    logger.debug("engine subscription handler error: %s", exc)
            delivered += 1
        return delivered

    def catch_up(self) -> int:
        """Bounded cold-start catch-up over the CDC tail (``cdc_read``).

        Reads up to ``catch_up_limit`` events from the current cursor and delivers
        the label-matching ones, advancing the cursor past the tail. Idempotent:
        a second call from a converged cursor reads nothing. Returns the number of
        changes delivered. No-op (0) when streaming is unavailable.
        """
        if self._streaming is None:
            self._caught_up = True
            return 0
        try:
            events = self._streaming.cdc_read(
                self.graph_name, self.cursor, limit=self.catch_up_limit
            )
        except Exception as exc:  # noqa: BLE001 — feed unavailable ⇒ stay on periodic reconcile
            logger.debug("engine subscription cdc_read failed: %s", exc)
            self._caught_up = True
            return 0
        events = list(events or [])
        delivered = self._deliver(events)
        if events:
            self.cursor = int(events[-1].get("seq", self.cursor)) + 1
        self._caught_up = True
        return delivered

    def poll(self, *, block_ms: int = 0) -> int:
        """Deliver changes since the cursor via one ``watch`` long-poll.

        On the first call this runs :meth:`catch_up` first so cold start can't miss
        the backlog. Then issues ONE ``watch`` from the cursor (filtered by
        ``label``); ``block_ms=0`` returns immediately with whatever is pending (the
        daemon-tick default — O(new-changes) work, not a full re-scan), ``block_ms>0``
        blocks up to that long for the first change (a dedicated reactive thread).
        Advances the cursor to ``next_seq``. Returns the number of changes
        delivered. No-op (0) when streaming is unavailable.
        """
        if self._streaming is None:
            return 0
        delivered = 0
        if not self._caught_up:
            delivered += self.catch_up()
        try:
            result = self._streaming.watch(
                self.graph_name,
                self.cursor,
                label=self.label,
                timeout_ms=int(block_ms),
            )
        except Exception as exc:  # noqa: BLE001 — feed hiccup ⇒ stay on periodic reconcile
            logger.debug("engine subscription watch failed: %s", exc)
            return delivered
        events = list((result or {}).get("events") or [])
        delivered += self._deliver(events)
        next_seq = (result or {}).get("next_seq")
        if next_seq is not None:
            self.cursor = int(next_seq)
        elif events:
            self.cursor = int(events[-1].get("seq", self.cursor)) + 1
        return delivered


def subscribe(
    source: Any,
    label: str,
    handler: ChangeHandler,
    *,
    catch_up_limit: int = 4096,
) -> EngineSubscription:
    """Build an :class:`EngineSubscription` over ``source`` for ``label`` changes.

    The one-call front door the reactive consumers use: ``subscribe(engine, "Task",
    on_change)``. Returns the subscription; the caller drives it with
    :meth:`EngineSubscription.poll` on its tick (or a blocking ``poll(block_ms=…)``
    in a reactive thread). Always returns an object — when no engine streaming
    surface exists, ``.available`` is ``False`` and every poll is a no-op.
    """
    return EngineSubscription(
        source, label=label, handler=handler, catch_up_limit=catch_up_limit
    )
