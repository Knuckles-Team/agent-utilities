#!/usr/bin/python
"""CLI: ingest path(s)/URL(s) into the live Knowledge Graph.

CONCEPT:AU-KG.ingest.process-boundary-entrypoint — A thin process-boundary entry point over the standardized
``IngestionEngine`` so cross-package tools (e.g. the universal-skills
``web-crawler`` / ``skill-graph-builder``) can route content INTO the KG without
importing agent-utilities — they shell out to::

    python -m agent_utilities.knowledge_graph.ingestion <path-or-url> [...] \
        [--content-type document] [--curate] [--force]

Content type is auto-detected per source (``ContentType.classify``) unless
overridden. Documents flow through the standardized contract (verbatim
``Document`` + ``IdeaBlock`` chunks + ``Concept``), so what lands in the KG is
faithfully re-materialisable (e.g. distilled back into a skill-graph).
"""

import argparse
import asyncio
import json

from agent_utilities.core.config import setting

from .engine import ContentType, IngestionEngine, IngestionManifest


def _build_engine():
    """Construct an engine bound to the live graph (mirrors kg_server._get_engine).

    In a standalone process ``get_active()`` is ``None``, so a fresh engine is
    built; its backend connects to the running epistemic-graph daemon (role
    self-heals to client), writing to the same durable graph.
    """
    from agent_utilities.core.paths import ensure_dirs, kg_db_path
    from agent_utilities.knowledge_graph.backends import create_backend
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    active = IntelligenceGraphEngine.get_active()
    if active is not None:
        return active
    ensure_dirs()
    backend = create_backend(
        backend_type=setting("GRAPH_BACKEND"), db_path=str(kg_db_path())
    )
    return IntelligenceGraphEngine(backend=backend)


async def _amain(args: argparse.Namespace) -> int:
    engine = _build_engine()
    ie = IngestionEngine(kg_engine=engine)

    metadata: dict = {}
    if args.curate:
        metadata["curate"] = True
    if args.topic:
        metadata["topic"] = args.topic

    results = []
    for src in args.sources:
        ct = (
            ContentType(args.content_type)
            if args.content_type
            else ContentType.classify(src)
        )
        res = await ie.ingest(
            IngestionManifest(
                content_type=ct,
                source_uri=src,
                metadata=dict(metadata),
                force=args.force,
            )
        )
        results.append(
            {
                "source": src,
                "content_type": ct.value,
                "status": res.status,
                "nodes_created": res.nodes_created,
                "edges_created": res.edges_created,
                "error": res.error,
                "details": res.details,
            }
        )

    print(json.dumps({"results": results}, indent=2, default=str))
    # Non-zero exit only if every source failed (so partial success still passes).
    return 0 if any(r["status"] != "failed" for r in results) else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest path(s)/URL(s) into the live Knowledge Graph via the "
        "standardized IngestionEngine."
    )
    parser.add_argument("sources", nargs="+", help="Path(s) or URL(s) to ingest.")
    parser.add_argument(
        "--content-type",
        default=None,
        help="Override the auto-detected ContentType (e.g. document, codebase, kb).",
    )
    parser.add_argument(
        "--curate",
        action="store_true",
        help="Also run the opt-in LLM curation enrichment layer (Article nodes).",
    )
    parser.add_argument("--topic", default=None, help="Optional topic hint.")
    parser.add_argument(
        "--force", action="store_true", help="Re-ingest even if unchanged."
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
