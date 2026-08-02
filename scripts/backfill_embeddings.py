#!/usr/bin/env python3
"""Operator-run backfill entrypoint for D-EMB / D-PERF-5 / D-EGD-5.

The ingest-time chokepoint (``agent_utilities/knowledge_graph/ingestion/
envelope_ingest.py``) now embeds every NEW typed-entity write going forward.
This script is the SEPARATE, explicit, operator-approved catch-up for the
graph's EXISTING under-embedded nodes (measured 2026-08-01: 26,807 total
nodes, 136 embedded — 0.5%).

Run this INSIDE the live graph-os pod (same pattern as
``scripts/perf/*.py`` / ``scripts/delegation_probe.py``):

    POD=$(kubectl -n platform get pods -l app=graph-os -o jsonpath='{.items[0].metadata.name}')
    kubectl -n platform exec -i $POD -c graph-os -- python3 - < scripts/backfill_embeddings.py -- --limit 200

Safety model:
  * DRY RUN BY DEFAULT. Nothing is written unless ``--execute`` is passed.
  * ``--limit`` bounds how many under-embedded nodes are scanned THIS RUN
    (default 200) — there is no "embed everything" default; a full-graph run
    requires an explicit, large ``--limit`` (or repeated invocations), which
    is deliberate: embedding ~26,000 nodes is an expensive, outward-facing
    operation against a shared, already latency-contended engine (D-PERF-2)
    and must be sized/approved by an operator, not run unattended by a script.
  * Persists each vector with the engine's atomic field compare-and-set, which
    preserves every existing classification/ACL/ownership property, then adds
    it to ANN/HNSW. It never re-upserts the entity through a new ChangeEnvelope.
    The CAS fences the exact entity-text snapshot, and every vector is validated
    before a vector property is written. The durable ``embedding`` property
    makes repeated bounded chunks advance; a textless node receives a separate
    ``no_text`` maintenance state, never a placeholder embedding. An ANN-side
    failure is recovered by the existing hydration loop.
  * Prints a cost estimate (measured throughput this run, extrapolated to the
    full remaining backlog) and the exact command to run the next chunk —
    it does NOT chain into further runs itself.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Any


def _cost_estimate(
    *,
    remaining: int,
    report: dict[str, int],
    elapsed: float,
) -> dict[str, float | int] | None:
    """Estimate from confirmed durable progress, never merely scanned rows."""
    embedded = max(0, int(report.get("embedded", 0)))
    if embedded == 0:
        return None
    seconds_per_embedded = elapsed / embedded
    remaining_after = max(0, int(remaining) - embedded)
    return {
        "seconds_per_embedded": seconds_per_embedded,
        "remaining_after": remaining_after,
        "eta_seconds": seconds_per_embedded * remaining_after,
    }


async def _run(limit: int, batch_size: int, execute: bool) -> None:
    print(f"sys.executable={sys.executable}", flush=True)

    from agent_utilities.knowledge_graph.core.session import set_session
    from agent_utilities.mcp.kg_server import _mint_process_session
    from agent_utilities.security.brain_context import set_actor

    session = await asyncio.to_thread(_mint_process_session, "auto")
    session.engine_verified_context()
    set_actor(session.actor)
    set_session(session)
    print(f"identity ok tenant={session.tenant} graph={session.graph!r}", flush=True)

    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    engine = (
        IntelligenceGraphEngine.get_active()
        or IntelligenceGraphEngine.get_or_create(defer_background_start=True)
    )
    backend = engine.backend
    print(f"backend: {type(backend).__name__}", flush=True)

    # ---- current population snapshot (same counts the diagnosis used) ----
    total = embedded = with_text = 0
    try:
        total = (backend.execute("MATCH (n) RETURN count(n) AS c") or [{}])[0].get(
            "c", 0
        )
        embedded = (
            backend.execute(
                "MATCH (n) WHERE n.embedding IS NOT NULL RETURN count(n) AS c"
            )
            or [{}]
        )[0].get("c", 0)
        with_text = (
            backend.execute("MATCH (n) WHERE n.text IS NOT NULL RETURN count(n) AS c")
            or [{}]
        )[0].get("c", 0)
    except Exception as e:  # noqa: BLE001 — diagnostic snapshot only
        print(f"population snapshot failed: {type(e).__name__}: {e}", flush=True)

    ratio = (embedded / total * 100) if total else 0.0
    remaining = max(0, total - embedded)
    print(
        f"\n=== CURRENT POPULATION ===\n"
        f"  total nodes:    {total}\n"
        f"  w/ embedding:   {embedded} ({ratio:.2f}%)\n"
        f"  w/ text:        {with_text}\n"
        f"  remaining:      {remaining}\n",
        flush=True,
    )

    if not execute:
        print(
            "DRY RUN (no --execute passed) — no writes will be made.\n"
            f"Would scan up to {limit} under-embedded node(s) this run "
            f"(batch_size={batch_size}).",
            flush=True,
        )

    from agent_utilities.knowledge_graph.core.maintainer import GraphMaintainer

    maintainer = GraphMaintainer(engine)

    if not execute:
        # Dry run still measures what WOULD be embedded (text extraction only,
        # no embed-endpoint call, no engine write) so the operator sees real
        # candidate counts before approving a real run.
        # DISCOVER ids by Cypher, then HYDRATE properties over the batch RPC.
        #
        # `properties(n)` is NOT in the native engine's supported Cypher subset —
        # it raises CypherEngineError("native Cypher authority rejected request",
        # error_type=RuntimeError), which is what made this script fail on its
        # very first real run. Selecting named fields instead is not an option
        # either: `derive_entity_text` deliberately falls back to ANY string-valued
        # leaf property when the known name/summary fields are absent, so it needs
        # the whole map, not a fixed projection.
        #
        # `_get_node_properties_batch` is the engine's own batched accessor and is
        # the same discovery-then-hydrate shape the delegation hot path already
        # uses (it cut `nodes.properties_batch` there from 78 calls to 4). One RPC
        # for the whole page, not one per node.
        ids = [
            row["id"]
            for row in (
                backend.execute(
                    "MATCH (n) WHERE n.embedding IS NULL "
                    "AND n._embedding_backfill_state IS NULL "
                    "RETURN n.id AS id ORDER BY n.id LIMIT $limit",
                    {"limit": limit},
                )
                or []
            )
            if row.get("id")
        ]
        graph = getattr(backend, "_graph", backend)
        props_by_id: dict[str, dict[str, Any]] = (
            graph._get_node_properties_batch(ids) or {} if ids else {}
        )
        rows = [
            {"id": node_id, "props": props_by_id.get(node_id) or {}} for node_id in ids
        ]
        from agent_utilities.knowledge_graph.enrichment.semantic import (
            derive_entity_text,
        )

        embeddable = sum(
            1 for row in rows if derive_entity_text(row.get("props") or {})
        )
        print(
            f"scanned={len(rows)} would_embed={embeddable} "
            f"skipped_no_text={len(rows) - embeddable}\n\n"
            "Re-run with --execute to actually generate + register embeddings.",
            flush=True,
        )
        return

    t0 = time.monotonic()
    report = maintainer.backfill_entity_embeddings(limit=limit, batch_size=batch_size)
    elapsed = time.monotonic() - t0

    print(f"\n=== BACKFILL RESULT (this run) ===\n{report}", flush=True)
    print(f"elapsed: {elapsed:.1f}s", flush=True)

    embedded_this_run = report.get("embedded", 0)
    indexed_this_run = report.get("indexed", 0)
    deferred_no_text = report.get("deferred_no_text", 0)
    conflicted = report.get("conflicted", 0)
    if indexed_this_run < embedded_this_run:
        print(
            f"ANN registration deferred for "
            f"{embedded_this_run - indexed_this_run} node(s); the background "
            "property-to-index hydrator will retry them.",
            flush=True,
        )
    if deferred_no_text or conflicted:
        print(
            f"durable non-vector progress: deferred_no_text={deferred_no_text}; "
            f"concurrent_conflicts={conflicted}",
            flush=True,
        )
    estimate = _cost_estimate(
        remaining=remaining,
        report=report,
        elapsed=elapsed,
    )
    if estimate is not None:
        per_node = float(estimate["seconds_per_embedded"])
        remaining_after = int(estimate["remaining_after"])
        eta_s = float(estimate["eta_seconds"])
        print(
            f"\n=== COST ESTIMATE (extrapolated) ===\n"
            f"  measured: {per_node:.3f}s/node this run "
            f"({embedded_this_run} embedded in {elapsed:.1f}s)\n"
            f"  remaining after this run: ~{remaining_after} node(s)\n"
            f"  estimated time for the rest: ~{eta_s / 60:.1f} min "
            f"({eta_s:.0f}s)\n\n"
            f"Next chunk:\n"
            f"  python3 scripts/backfill_embeddings.py --execute "
            f"--limit {limit} --batch-size {batch_size}\n",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max under-embedded nodes to scan THIS run (default 200). "
        "No unbounded/full-graph default on purpose.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Texts per embedding-endpoint batch call (default 64).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually write embeddings. Without this flag, only reports "
        "what WOULD be embedded (no engine writes, no embed-endpoint calls).",
    )
    args = parser.parse_args()
    asyncio.run(
        _run(limit=args.limit, batch_size=args.batch_size, execute=args.execute)
    )


if __name__ == "__main__":
    main()
