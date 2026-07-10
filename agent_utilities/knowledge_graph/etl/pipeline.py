#!/usr/bin/python
from __future__ import annotations

"""Unified ETL pipeline — one source→transform→sink interface (CONCEPT:AU-KG.ontology.one-source).

A thin orchestrator that collapses the KG's existing bidirectional machinery into a
single "move data between systems" entrypoint. It writes no transport of its own —
it composes the parts that already exist:

* **Extract + transform + load (inbound):** ``core.source_sync.sync_source`` runs the
  registered extractor/hydrator for ``source`` (LeanIX, ServiceNow, Egeria, Camunda,
  …); the ontology layer (interfaces / links / OWL bridge) is the transform, turning
  each external schema into the KG's canonical model. The KG is the canonical hub.
* **Load (outbound):** dispatch ``sink`` by kind —
  - a **WritebackSink** domain (leanix/servicenow/egeria/…) → ``run_writeback`` (the KG
    pushes intelligence back to the system of record; dry-run-first + ProposalQueue);
  - a **graph store** (Stardog/Neo4j/AGE/…, passed as a resolved ``sink_backend``) →
    full-data load: ``stardog_sync.push_to_stardog`` for a SPARQL store (partitioned
    into ``urn:source:<system>`` named graphs), else ``migration.copy_graph``.
* **Lineage:** every run is recorded via :mod:`.lineage` for impact analysis.

So ``run_etl(source="servicenow", sink="leanix")`` is ServiceNow → (ontological
normalization in the KG) → LeanIX; ``source="leanix", sink="stardog"`` mirrors LeanIX
into Stardog; either side may be omitted for a one-directional run. ``run_etl`` stays
pure (no MCP/registry import) — the caller resolves ``sink_backend``.

The returned manifest is built from :class:`.result.EtlResult` (CONCEPT:AU-KG.etl.result-contract)
— a validated pydantic contract (``status``/``counts``/``watermark``/``lineage``/…)
instead of a hand-assembled dict — then serialized back to a plain ``dict`` via
``.model_dump()`` so every existing consumer (REST/MCP surfaces, tests) keeps
indexing it exactly as before.
"""

import logging
from typing import Any

from .result import EtlResult

logger = logging.getLogger(__name__)


def run_etl(
    engine: Any,
    *,
    source: str | None = None,
    mode: str = "delta",
    ids: list[str] | None = None,
    sink: str | None = None,
    sink_backend: Any = None,
    sources: list[str] | None = None,
    dry_run: bool = True,
    ops: dict[str, Any] | None = None,
    record_lineage: bool = True,
) -> dict[str, Any]:
    """Run an ETL flow: optional inbound sync, optional outbound load, + lineage.

    Args:
        engine: the live KG engine/facade (the canonical hub).
        source: registered ingestion source to pull (omit for outbound-only).
        mode: inbound sync mode — ``delta`` | ``full`` | ``reconcile``.
        ids: optional record-id filter for the inbound sync.
        sink: a WritebackSink domain OR a graph-store name (for dispatch + lineage).
        sink_backend: a resolved ``GraphBackend`` when ``sink`` is a graph store
            (Stardog/Neo4j/…); the caller resolves it (registry / create_backend).
        sources: subset filter (source systems) for a graph-store push.
        dry_run: writeback dry-run (default True, fail-closed).
        ops: writeback payload (inferences/enrichments/creations/retirements).
        record_lineage: record an ETL-run lineage trail (default on).

    Returns a uniform manifest ``{status, inbound, outbound, lineage}``.
    """
    from .lineage import record_etl_run

    out: dict[str, Any] = {"status": "ok", "inbound": None, "outbound": None}

    # ── inbound: source → (ontological transform) → KG ──
    # The native SQL-table sink (KG-2.266) mirrors the source straight into an engine
    # table — the table is the destination, so skip the source→KG inbound hydrate.
    if source and sink != "table":
        from ..core.source_sync import sync_source

        try:
            out["inbound"] = sync_source(engine, source, mode=mode, ids=ids or None)
        except Exception as e:  # noqa: BLE001 - report, don't crash the surface
            out["inbound"] = {"status": "error", "error": str(e)}
            out["status"] = "partial"

    # ── outbound: KG → sink (writeback system-of-record, graph store, or SQL table) ──
    if sink == "table":
        # CONCEPT:AU-KG.ingest.mirror-inbound — mirror the inbound `source` connector's data into a native
        # engine SQL table (CREATE TABLE + bulk INSERT). `ops` carries optional
        # {table, config, limit, replace}. This is the ETL→table sink.
        from ..core.table_ingest import ingest_connector_to_table

        opts = ops or {}
        try:
            out["outbound"] = ingest_connector_to_table(
                engine,
                source or opts.get("source", ""),
                table=opts.get("table"),
                config=opts.get("config"),
                limit=int(opts.get("limit", 1000)),
                replace=bool(opts.get("replace", False)),
            )
            if out["outbound"].get("status") in ("error", "skipped"):
                out["status"] = "partial"
        except Exception as e:  # noqa: BLE001
            out["outbound"] = {"status": "error", "error": str(e)}
            out["status"] = "partial"
    elif sink:
        try:
            out["outbound"] = _run_outbound(
                engine,
                sink=sink,
                sink_backend=sink_backend,
                sources=sources,
                dry_run=dry_run,
                ops=ops or {},
            )
            if isinstance(out["outbound"], dict) and out["outbound"].get("status") in (
                "error",
                "refused",
            ):
                out["status"] = "partial"
        except Exception as e:  # noqa: BLE001
            out["outbound"] = {"status": "error", "error": str(e)}
            out["status"] = "partial"

    # ── lineage ──
    # Computed unconditionally (not just when lineage recording is on) so the
    # returned ``EtlResult.counts`` is always populated from the one place, even
    # when ``record_lineage=False``.
    counts = {
        "nodes": EtlResult.count_of(out["inbound"])
        or EtlResult.count_of(out["outbound"]),
        "edges": (out.get("outbound") or {}).get("edges", 0)
        if isinstance(out.get("outbound"), dict)
        else 0,
    }

    if record_lineage and (source or sink):
        direction = (
            "through" if (source and sink) else ("inbound" if source else "outbound")
        )
        run_id = record_etl_run(
            engine,
            source=source,
            sink=sink,
            direction=direction,
            counts=counts,
            status=out["status"],
        )
        out["lineage"] = {"run_id": run_id, "direction": direction}

    return EtlResult.coerce(
        out, source=source, sink=sink, mode=mode, counts=counts
    ).model_dump()


def _run_outbound(
    engine: Any,
    *,
    sink: str,
    sink_backend: Any,
    sources: list[str] | None,
    dry_run: bool,
    ops: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch the outbound half: a writeback sink, or a graph-store load."""
    from ..enrichment.writeback.core import get_sink, run_writeback

    # A registered system-of-record writeback sink (leanix/servicenow/egeria/…).
    if get_sink(sink) is not None:
        return run_writeback(
            sink,
            backend=getattr(engine, "backend", None),
            engine=engine,
            dry_run=dry_run,
            **ops,
        )

    # Otherwise a graph store: full-data load via the resolved backend.
    if sink_backend is None:
        from ..enrichment.writeback.core import list_sinks

        return {
            "status": "error",
            "error": f"unknown sink {sink!r} (not a writeback sink and no graph "
            "backend resolved)",
            "writeback_sinks": list_sinks(),
        }

    if getattr(sink_backend, "supports_sparql", False):
        from ..integrations.stardog_sync import push_to_stardog

        res = push_to_stardog(engine, sink_backend, sources=sources)
        res.setdefault("sink", sink)
        return res

    # Cypher-capable graph store → full cross-backend migration.
    from ..migration import copy_graph

    summary = copy_graph(engine, sink_backend)
    return {"status": "ok", "sink": sink, **summary}
