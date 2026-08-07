#!/usr/bin/env python3
"""Operator-run resync entrypoint for D-CDX-53.

The live graph can hold BOTH legacy ``Tool.relevance_score`` floats in
``[0, 1]`` and canonical integer points in ``[0, 100]`` at the same time.
Reads are already fixed at the model boundary (``ToolNode``/``MCPToolInfo``,
``agent_utilities/models/tool_score.py``) and the one query that ordered
directly on the raw stored value now ranks in Python instead
(``agent_utilities/graph/_router_impl.py::_rank_tool_rows_by_relevance``).
This script is the separate, explicit, operator-approved catch-up that
resyncs the PERSISTED values themselves, so database-side ``ORDER BY
t.relevance_score`` can eventually be trusted again fleet-wide instead of
depending on every caller re-normalizing at read time forever.

Run this INSIDE the live graph-os pod (same pattern as
``scripts/backfill_embeddings.py`` / ``scripts/delegation_probe.py``):

    POD=$(kubectl -n platform get pods -l app=graph-os -o jsonpath='{.items[0].metadata.name}')
    kubectl -n platform exec -i $POD -c graph-os -- python3 - \
        < scripts/resync_tool_relevance_scores.py -- --limit 500 --execute

Safety model (mirrors ``scripts/backfill_embeddings.py``):
  * DRY RUN BY DEFAULT. Nothing is written unless ``--execute`` is passed.
  * ``--limit`` bounds how many Tool rows are scanned THIS RUN (default 500).
  * A row that is neither already canonical nor a recognized legacy ``[0, 1]``
    float is QUARANTINED — reported, never silently coerced — so a corrupt
    value surfaces for manual attention instead of being guessed at.
  * One row's write failure is recorded in ``write_errors`` and does not
    abort the rest of the run.
"""

from __future__ import annotations

import argparse
import asyncio
import sys


async def _run(limit: int, execute: bool) -> None:
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
    from agent_utilities.knowledge_graph.core.tool_score_migration import (
        resync_tool_relevance_scores,
    )

    engine = (
        IntelligenceGraphEngine.get_active()
        or IntelligenceGraphEngine.get_or_create(defer_background_start=True)
    )
    report = await asyncio.to_thread(
        resync_tool_relevance_scores, engine, execute=execute, limit=limit
    )

    print(f"scanned={report['scanned']}", flush=True)
    print(f"already_canonical={report['already_canonical']}", flush=True)
    print(f"to_migrate={report['to_migrate']}", flush=True)
    print(f"migrated={report['migrated']}", flush=True)
    print(f"quarantined={len(report['quarantined'])}", flush=True)
    if report["quarantined"]:
        print(f"quarantined_rows={report['quarantined']}", flush=True)
    if report["write_errors"]:
        print(f"write_errors={report['write_errors']}", flush=True)

    if not execute:
        print(
            "DRY RUN — no writes performed. Re-run with --execute to migrate.",
            flush=True,
        )
    elif report["to_migrate"] == report["migrated"] and not report["write_errors"]:
        print("All planned rows migrated cleanly.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--execute", action="store_true", default=False)
    args, _unknown = parser.parse_known_args()
    asyncio.run(_run(limit=args.limit, execute=args.execute))


if __name__ == "__main__":
    main()
