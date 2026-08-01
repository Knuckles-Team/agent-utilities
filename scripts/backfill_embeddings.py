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
  * Writes ONLY to the engine's ANN/HNSW index via ``backend.add_embedding``
    (GraphMaintainer.backfill_entity_embeddings) — never through the governed
    ChangeEnvelope path, which would silently reset an existing entity's
    classification/ACL to a default quarantined policy. See that method's
    docstring for the full reasoning.
  * Prints a cost estimate (measured throughput this run, extrapolated to the
    full remaining backlog) and the exact command to run the next chunk —
    it does NOT chain into further runs itself.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time


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
        rows = (
            backend.execute(
                "MATCH (n) WHERE n.embedding IS NULL "
                "RETURN n.id AS id, properties(n) AS props "
                "ORDER BY n.id LIMIT $limit",
                {"limit": limit},
            )
            or []
        )
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
    if embedded_this_run:
        per_node = elapsed / embedded_this_run
        remaining_after = max(0, remaining - report.get("scanned", 0))
        eta_s = per_node * remaining_after
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
