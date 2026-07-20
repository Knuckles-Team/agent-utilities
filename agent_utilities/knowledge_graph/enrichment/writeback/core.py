"""Unified write-back core: result, resolver, gate, sink registry (CONCEPT:EG-KG.storage.nonblocking-checkpoint/2.9).

Bitemporal ``as_of`` on writeback (CONCEPT:AU-KG.temporal.bi-temporal-memory-layers): read
paths (``engine_query.py``, ``hybrid_retriever.py``, ``context_compiler.py``) have long
accepted an ``as_of`` instant and filtered rows via
:func:`~agent_utilities.knowledge_graph.core.bitemporal.filter_as_of`; nothing on the
write side stamped the valid-time a KG-derived fact was mirrored under. ``run_writeback``'s
optional ``as_of`` (default ``None`` → wall-clock "now", so every existing caller is
unaffected byte-for-byte) is threaded onto :class:`WritebackContext` and applied by
:meth:`WritebackContext.stamp_valid_time` — call it from a sink alongside
:meth:`WritebackContext.stamp_external_id` to stamp the SAME
``storage_time``/``event_time``/``valid_from``/``valid_to`` quadruple the read path already
understands onto a mirrored record, so a later ``as_of`` query can answer "what did we mirror
as valid at time T", not only "what is the current value."
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agent_utilities.core.config import setting
from agent_utilities.knowledge_graph.core.bitemporal import stamp_bitemporal

logger = logging.getLogger(__name__)

#: provenance tag stamped on agent-written (inferred) upstream relations/records.
PROVENANCE_TAG = "agent-utilities:inferred"


@dataclass
class WritebackResult:
    """One result type for every write-back target (counts + dry-run proposals)."""

    target: str = ""
    created: int = 0
    enriched: int = 0
    relations_written: int = 0
    retired: int = 0
    skipped: int = 0
    errors: int = 0
    proposals: list[dict[str, Any]] = field(default_factory=list)

    def merge(self, other: WritebackResult) -> WritebackResult:
        self.created += other.created
        self.enriched += other.enriched
        self.relations_written += other.relations_written
        self.retired += other.retired
        self.skipped += other.skipped
        self.errors += other.errors
        self.proposals.extend(other.proposals)
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "created": self.created,
            "enriched": self.enriched,
            "relations_written": self.relations_written,
            "retired": self.retired,
            "skipped": self.skipped,
            "errors": self.errors,
            "proposals": self.proposals,
        }


@dataclass
class WritebackContext:
    """What a sink needs to do its work — the live backend + engine."""

    backend: Any = None
    engine: Any = None
    #: bitemporal valid-time (ISO-8601) this writeback pass asserts its writes
    #: are valid as of — CONCEPT:AU-KG.temporal.bi-temporal-memory-layers.
    #: ``None`` (the default) means "now": :meth:`stamp_valid_time` then
    #: stamps wall-clock storage/event/valid_from, matching every caller's
    #: pre-existing behavior byte-for-byte.
    as_of: str | None = None

    def resolver(self, domain: str) -> Callable[[str], str | None]:
        return resolve_external_id(self.backend, domain)

    def stamp_valid_time(self, props: dict[str, Any]) -> dict[str, Any]:
        """Inject this context's bitemporal validity onto ``props`` (in place, returned).

        Thin wrapper over :func:`~agent_utilities.knowledge_graph.core.bitemporal.
        stamp_bitemporal` using ``self.as_of`` as the asserted ``event_time`` —
        ``as_of=None`` (default) resolves to wall-clock "now", identical to every
        writeback call made before this context carried an ``as_of``.
        """
        return stamp_bitemporal(props, event_time=self.as_of)

    def stamp_external_id(
        self,
        node_id: str | None,
        target: str,
        external_id: str | None,
        *,
        node_type: str = "",
    ) -> bool:
        """Round-trip the SoR's returned CI id back onto the source KG node.

        After a sink creates a CI/asset/record it calls this with the returned id
        so the node carries three stamps: ``<target>_ci_id`` (the per-sink identity
        used to dedupe re-runs — idempotency across ALL sinks), and the shared
        ``domain`` / ``externalToolId`` federation key the resolver reads. Writing
        the stamp turns the next :func:`~.inventory.collect_inventory_creations`
        pass into a skip/update instead of a duplicate create. Also stamps this
        context's bitemporal validity (:meth:`stamp_valid_time`) onto the SAME
        props, so the mirrored record carries a ``valid_from``/``valid_to`` an
        ``as_of`` read can filter on (CONCEPT:AU-KG.temporal.bi-temporal-memory-layers).

        Best-effort (fail-closed): a stamp failure never breaks the sink write —
        it only means the node may be re-proposed next pass.
        """
        if not (node_id and external_id) or self.engine is None:
            return False
        eid = str(external_id)
        props = self.stamp_valid_time(
            {
                f"{(target or '').lower().strip()}_ci_id": eid,
                "externalToolId": eid,
                "domain": (target or "").lower().strip(),
            }
        )
        add_node = getattr(self.engine, "add_node", None)
        if add_node is None:
            return False
        label = node_type or "ConfigurationItem"
        try:
            # IntelligenceGraphEngine.add_node(node_id, node_type, properties) — merge-upsert.
            add_node(node_id, label, props)
            return True
        except TypeError:
            try:
                # Alternate signature: add_node(node_id, label="", **properties).
                add_node(node_id, label=label, **props)
                return True
            except Exception:  # noqa: BLE001 - stamping is best-effort
                logger.debug("stamp_external_id fallback failed", exc_info=True)
                return False
        except Exception:  # noqa: BLE001 - stamping is best-effort
            logger.debug("stamp_external_id failed for %s", node_id, exc_info=True)
            return False


@runtime_checkable
class WritebackSink(Protocol):
    """A target system's write-back adapter.

    ``run`` applies the requested ``ops`` (inferences/enrichments/creations/
    retirements) when ``dry_run`` is False, else returns the *proposed* writes.
    """

    domain: str
    enable_flag: str
    # NOTE: an optional ``risk_tier`` attribute ("standard" default, or "high_stakes"
    # — never auto-execute, queued for approval) is read defensively via
    # ``getattr(sink, "risk_tier", "standard")`` below, so it is intentionally NOT a
    # required Protocol member (most sinks omit it).

    def run(
        self, ctx: WritebackContext, ops: dict[str, Any], *, dry_run: bool
    ) -> WritebackResult:
        ...


def resolve_external_id(backend: Any, domain: str) -> Callable[[str], str | None]:
    """KG node id → upstream external id, via the ``externalToolId``/``domain`` key.

    The shared federation resolver (generalized from the LeanIX-specific one): one
    query per domain, cached in a closure.
    """
    mapping: dict[str, str] = {}
    if backend is not None:
        try:
            rows = backend.execute(
                "MATCH (n) WHERE n.domain = $d AND n.externalToolId IS NOT NULL "
                "RETURN n.id AS id, n.externalToolId AS guid",
                {"d": domain},
            )
            for r in rows or []:
                if isinstance(r, dict) and r.get("id") and r.get("guid"):
                    mapping[str(r["id"])] = str(r["guid"])
        except Exception:  # noqa: BLE001 - resolution is best-effort
            logger.debug(
                "external-id resolver query failed for %s", domain, exc_info=True
            )

    def _resolve(node_id: str) -> str | None:
        return mapping.get(node_id)

    return _resolve


# ── sink registry (plugin pattern) ───────────────────────────────────────────

_SINKS: dict[str, WritebackSink] = {}


def register_sink(sink: WritebackSink) -> WritebackSink:
    """Register a write-back sink under its ``domain`` (idempotent)."""
    _SINKS[sink.domain] = sink
    return sink


def get_sink(domain: str) -> WritebackSink | None:
    return _SINKS.get((domain or "").lower().strip())


def list_sinks() -> list[str]:
    return sorted(_SINKS)


def run_writeback(
    target: str,
    *,
    backend: Any = None,
    engine: Any = None,
    dry_run: bool = True,
    as_of: str | None = None,
    **ops: Any,
) -> dict[str, Any]:
    """Single fail-closed, dry-run-first write-back entrypoint (MCP + REST core).

    Resolves the ``target`` sink, enforces its ``enable_flag`` for live writes,
    and returns a uniform manifest. ``ops`` carries target-specific payloads
    (inferences / enrichments / creations / retirements / process_ids / …).

    ``as_of`` (ISO-8601, CONCEPT:AU-KG.temporal.bi-temporal-memory-layers) is the bitemporal
    valid-time this pass asserts its writes are valid as of — threaded onto the
    :class:`WritebackContext` a sink receives so it can stamp
    ``storage_time``/``event_time``/``valid_from``/``valid_to`` (via
    :meth:`WritebackContext.stamp_valid_time`, and automatically for any
    :meth:`WritebackContext.stamp_external_id` call) on every mirrored record.
    Defaults to ``None`` ("now" at stamp time) so a caller that doesn't pass it
    behaves byte-for-byte as before this parameter existed.
    """
    sink = get_sink(target)
    if sink is None:
        return {
            "status": "error",
            "error": f"unknown write-back target {target!r}",
            "available": list_sinks(),
        }
    write_enabled = bool(setting(sink.enable_flag, False, cast=bool))
    if not dry_run and not write_enabled:
        return {
            "status": "refused",
            "target": target,
            "reason": f"{sink.enable_flag} not set; refusing live write to the system-of-record",
            "hint": "run with dry_run=true to preview the proposed writes",
        }
    ctx = WritebackContext(backend=backend, engine=engine, as_of=as_of)

    # High-stakes sinks NEVER auto-execute: a live request (enabled, not dry-run,
    # not carrying an approval token) is previewed and queued for approval instead.
    risk_tier = getattr(sink, "risk_tier", "standard")
    if risk_tier == "high_stakes" and not dry_run and not ops.get("_approved"):
        try:
            preview = sink.run(ctx, ops, dry_run=True)
        except Exception as e:  # noqa: BLE001
            logger.debug("writeback sink %s preview failed", target, exc_info=True)
            return {"status": "error", "target": target, "error": str(e)}
        from .approval import ProposalQueue

        pid = ProposalQueue(backend=backend).enqueue(target, ops, preview.proposals)
        out = preview.as_dict()
        out.update(
            {
                "status": "queued",
                "target": target,
                "risk_tier": risk_tier,
                "proposal_id": pid,
                "hint": "high-stakes write queued; approve via graph_writeback action=approve",
            }
        )
        return out

    try:
        result = sink.run(ctx, ops, dry_run=dry_run)
    except Exception as e:  # noqa: BLE001 - never let one sink crash the surface
        logger.debug("writeback sink %s failed", target, exc_info=True)
        return {"status": "error", "target": target, "error": str(e)}
    out = result.as_dict()
    out["status"] = "completed"
    out["target"] = target
    out["dry_run"] = dry_run
    out["write_enabled"] = write_enabled
    out["risk_tier"] = risk_tier
    out["as_of"] = ctx.as_of
    return out
