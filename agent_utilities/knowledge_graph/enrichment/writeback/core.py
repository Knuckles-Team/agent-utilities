"""Unified write-back core: result, resolver, gate, sink registry (CONCEPT:KG-2.8/2.9)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agent_utilities.core.config import setting

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

    def resolver(self, domain: str) -> Callable[[str], str | None]:
        return resolve_external_id(self.backend, domain)


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
    ) -> WritebackResult: ...


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
    **ops: Any,
) -> dict[str, Any]:
    """Single fail-closed, dry-run-first write-back entrypoint (MCP + REST core).

    Resolves the ``target`` sink, enforces its ``enable_flag`` for live writes,
    and returns a uniform manifest. ``ops`` carries target-specific payloads
    (inferences / enrichments / creations / retirements / process_ids / …).
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
    ctx = WritebackContext(backend=backend, engine=engine)

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
    return out
