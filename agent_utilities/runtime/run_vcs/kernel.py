"""CONCEPT:AU-ORCH.runvcs.event-kernel — content-addressed, typed, projectable run-event log.

Today ``:RunTrace``/``:ToolCall`` (``runtime/provenance.py``) are *observe-only*: a run
mirrors its actions into the KG as a flat, append-only ``WorkspaceAction`` chain keyed by
``(run_id, step)``. That is enough to *look at* a run but not to *reconstruct* one — the step
index is positional, not semantic, so two structurally-identical runs share no identity and a
replay has nothing content-stable to key against.

This module elevates that log into shepherd2's **Fact kernel** shape while staying idiomatic to
our stack (frozen pydantic-free dataclasses, stdlib-only, KG-optional):

* **Typed event** — every :class:`RunEvent` carries a ``schema_ref`` (the kind: ``cmd_run``,
  ``file_edit``, ``model_exchange`` …) and a ``mode``: a *declaration* (an intent, e.g. the
  request sent to the model / the action proposed) or a *capture* (the observed outcome, e.g.
  the model's reply / the command's exit). This is shepherd's ``RecordMode`` split and it is
  what makes deterministic replay possible — the recorded *capture* stands in for the model
  when a *declaration* is re-executed (:mod:`.replay`).
* **Content-addressed identity** — ``record_id == digest`` over ``(schema_ref, mode, payload,
  caused_by)``. Identical content ⇒ identical id: a re-run producing the same event reuses the
  same fact (the CoW/dedup property), and ``caused_by`` threads causality so a real event chain
  gets unique ids without leaning on a positional counter.
* **Projectable** — a :class:`RunCut` is an immutable frontier "the run *through* ordinal N";
  :meth:`RunEventLog.project` returns the causal prefix at a cut (shepherd's ``TraceSlice`` /
  "projection → Execution"), which is exactly what a fork/revert restores to.

Every event optionally mirrors into the KG as a ``:RunEvent`` node under its ``:RunTrace``
(``HAS_EVENT`` / ``NEXT`` / ``CAUSED_BY`` edges) so the whole event graph is queryable — the
surpassing move over shepherd's opaque store. All KG writes are best-effort: a cold graph never
breaks a run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

#: A record is either a *declaration* (intent, pre-effect) or a *capture* (observed outcome).
RecordMode = Literal["declaration", "capture"]


def _canonical(payload: Any) -> str:
    """Deterministic JSON for content addressing (sorted keys, compact, str-coerced)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def content_digest(
    schema_ref: str,
    mode: RecordMode,
    payload: dict[str, Any],
    caused_by: tuple[str, ...],
) -> str:
    """The content hash that IS a record's identity (shepherd ``record_id == digest``).

    Deliberately excludes ordinal/timestamp/run_id: identity is *semantic content plus
    causality*, so identical events dedup and replay is reproducible. Position is carried
    separately by :attr:`RunEvent.ordinal` (shepherd's ``RecordView``), not by identity.
    """
    material = _canonical(
        {
            "schema_ref": schema_ref,
            "mode": mode,
            "payload": payload,
            "caused_by": list(caused_by),
        }
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunEvent:
    """One retained, content-addressed run event (shepherd ``Fact``).

    ``record_id`` is the :func:`content_digest` — the stable semantic identity. ``ordinal`` is
    the append position in the owning run's path (shepherd ``owner_ordinal``): view metadata,
    not identity. ``caused_by`` are the record-ids of the events that causally precede this one
    (the previous event by default, so a linear run threads a causal chain).
    """

    run_id: str
    schema_ref: str
    mode: RecordMode
    payload: dict[str, Any]
    ordinal: int
    record_id: str
    caused_by: tuple[str, ...] = ()
    ts: float = 0.0

    @property
    def node_id(self) -> str:
        """The KG node id for this event (``runevent:<digest>``)."""
        return f"runevent:{self.record_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "schema_ref": self.schema_ref,
            "mode": self.mode,
            "payload": self.payload,
            "ordinal": self.ordinal,
            "record_id": self.record_id,
            "caused_by": list(self.caused_by),
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunEvent:
        return cls(
            run_id=str(data["run_id"]),
            schema_ref=str(data["schema_ref"]),
            mode=data["mode"],
            payload=dict(data.get("payload") or {}),
            ordinal=int(data["ordinal"]),
            record_id=str(data["record_id"]),
            caused_by=tuple(data.get("caused_by") or ()),
            ts=float(data.get("ts") or 0.0),
        )


@dataclass(frozen=True)
class RunCut:
    """An immutable frontier over a run's event log: "the run *through* ordinal N".

    Shepherd's ``Cut``: a published read selector. A commit pins one; a fork/revert restores the
    causal prefix it selects. ``through_record_id`` content-addresses the frontier so two runs
    that reach the same state share the same cut id.
    """

    run_id: str
    through_ordinal: int
    through_record_id: str

    @property
    def frontier_id(self) -> str:
        return (
            f"runcut:{self.run_id}:{self.through_ordinal}:{self.through_record_id[:12]}"
        )


class RunEventLog:
    """An append-only, content-addressed, projectable event log for ONE run.

    The in-process source of truth for a run's :class:`RunEvent` chain, with best-effort KG
    mirroring. Append yields a fully-formed content-addressed event; :meth:`cut` publishes a
    frontier; :meth:`project` returns the causal prefix at a frontier — the trio a
    commit/fork/revert/replay is built from.
    """

    def __init__(self, run_id: str, *, engine: Any | None = None) -> None:
        self.run_id = run_id
        self._events: list[RunEvent] = []
        self._engine = engine

    # ── append ───────────────────────────────────────────────────────────────
    def append(
        self,
        schema_ref: str,
        payload: dict[str, Any],
        *,
        mode: RecordMode = "capture",
        caused_by: tuple[str, ...] | None = None,
    ) -> RunEvent:
        """Append a typed event; assign its ordinal and content-addressed id.

        ``caused_by`` defaults to the previous event's ``record_id`` (a linear causal chain);
        pass an explicit tuple to model a DAG (e.g. a capture caused by its declaration).
        """
        if caused_by is None:
            caused_by = (self._events[-1].record_id,) if self._events else ()
        digest = content_digest(schema_ref, mode, payload, caused_by)
        event = RunEvent(
            run_id=self.run_id,
            schema_ref=schema_ref,
            mode=mode,
            payload=dict(payload),
            ordinal=len(self._events),
            record_id=digest,
            caused_by=caused_by,
            ts=time.time(),
        )
        self._events.append(event)
        self._mirror(event)
        return event

    def declare(self, schema_ref: str, payload: dict[str, Any]) -> RunEvent:
        """Append a *declaration* (an intent, pre-effect) — sugar for ``mode='declaration'``."""
        return self.append(schema_ref, payload, mode="declaration")

    def capture(
        self, schema_ref: str, payload: dict[str, Any], *, of: RunEvent | None = None
    ) -> RunEvent:
        """Append a *capture* (an observed outcome). ``of`` links it causally to its declaration."""
        caused_by = (of.record_id,) if of is not None else None
        return self.append(schema_ref, payload, mode="capture", caused_by=caused_by)

    # ── read / project ───────────────────────────────────────────────────────
    @property
    def events(self) -> list[RunEvent]:
        return list(self._events)

    @property
    def head(self) -> RunEvent | None:
        return self._events[-1] if self._events else None

    def cut(self, through_ordinal: int | None = None) -> RunCut:
        """Publish a :class:`RunCut` frontier at ``through_ordinal`` (default: the head)."""
        if not self._events:
            return RunCut(self.run_id, -1, "")
        if through_ordinal is None:
            through_ordinal = self._events[-1].ordinal
        through_ordinal = max(-1, min(through_ordinal, self._events[-1].ordinal))
        rec = "" if through_ordinal < 0 else self._events[through_ordinal].record_id
        return RunCut(self.run_id, through_ordinal, rec)

    def project(self, cut: RunCut | None = None) -> list[RunEvent]:
        """Return the causal prefix of events at ``cut`` (default: the whole log).

        Shepherd's "projection → Execution": the ordered event slice a fork/revert restores to.
        """
        if cut is None:
            return list(self._events)
        return [e for e in self._events if e.ordinal <= cut.through_ordinal]

    def digest_at(self, cut: RunCut | None = None) -> str:
        """A single content digest of the projected prefix — the log's identity at a cut."""
        projected = self.project(cut)
        return hashlib.sha256(
            _canonical([e.record_id for e in projected]).encode("utf-8")
        ).hexdigest()

    def truncate_to(self, cut: RunCut) -> None:
        """Drop every event after ``cut`` (in-place rewind to the frontier)."""
        keep = [e for e in self._events if e.ordinal <= cut.through_ordinal]
        self._events = keep

    # ── seeding (fork) ─────────────────────────────────────────────────────────
    @classmethod
    def from_events(
        cls, run_id: str, events: list[RunEvent], *, engine: Any | None = None
    ) -> RunEventLog:
        """Build a log seeded from an existing event slice (used by fork), re-basing ordinals.

        The seeded events keep their content-addressed ``record_id`` (identity is content, not
        position) but are re-numbered onto the new run's path so appends continue cleanly.
        """
        log = cls(run_id, engine=engine)
        for i, ev in enumerate(sorted(events, key=lambda e: e.ordinal)):
            log._events.append(
                RunEvent(
                    run_id=run_id,
                    schema_ref=ev.schema_ref,
                    mode=ev.mode,
                    payload=dict(ev.payload),
                    ordinal=i,
                    record_id=ev.record_id,
                    caused_by=ev.caused_by,
                    ts=ev.ts,
                )
            )
        return log

    # ── KG mirror (best-effort) ────────────────────────────────────────────────
    def _mirror(self, event: RunEvent) -> None:
        engine = self._engine
        if engine is None:
            return
        try:
            engine.add_node(
                event.node_id,
                "RunEvent",
                properties={
                    "run_id": event.run_id,
                    "schema_ref": event.schema_ref,
                    "mode": event.mode,
                    "ordinal": event.ordinal,
                    "record_id": event.record_id,
                    "payload_json": _canonical(event.payload)[:4000],
                    "ts": event.ts,
                },
            )
            engine.add_edge(f"trace:{event.run_id}", event.node_id, "HAS_EVENT")
            if event.ordinal > 0:
                prev = self._events[event.ordinal - 1]
                engine.add_edge(prev.node_id, event.node_id, "NEXT")
            for parent in event.caused_by:
                engine.add_edge(f"runevent:{parent}", event.node_id, "CAUSED_BY")
        except Exception as exc:  # noqa: BLE001 — KG mirror is best-effort
            logger.debug("run-vcs: event mirror failed: %s", exc)
