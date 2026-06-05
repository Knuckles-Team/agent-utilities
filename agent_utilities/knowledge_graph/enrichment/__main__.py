"""Run KG enrichment over a path and report codebase insight (CONCEPT:KG-2.8).

Usage::

    python -m agent_utilities.knowledge_graph.enrichment <path> [options]

Options:
    --features         cluster the call graph into features (engine community detection)
    --cards            generate LLM capability cards ("how is it implemented")
    --pattern NAME     list code tagged with a design pattern
    --how NAME         explain how a symbol is implemented
    --limit N          cap list output
    --json             emit JSON

Uses the epistemic-graph Rust engine as the compute layer and ephemeral scratch
tenants (dropped on exit) so a one-shot analysis never pollutes the shared graph.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .features import make_community_fn
from .pipeline import EnrichmentPipeline, make_parse_fn
from .query import code_by_pattern, how_implemented, list_features, tests_needing_work

_PATTERNS = [
    "AbstractBaseClass",
    "Strategy",
    "Factory",
    "DataModel",
    "ContextManager",
    "Repository",
    "Adapter",
    "Singleton",
    "Manager",
    "Observer",
    "Iterator",
    "Enumeration",
    "Exception",
    "Mixin",
    "Property",
    "Memoized",
]


def main() -> int:
    ap = argparse.ArgumentParser(prog="kg-enrich")
    ap.add_argument("path")
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--cards", action="store_true")
    ap.add_argument("--pattern", default=None)
    ap.add_argument("--how", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from ..backends.epistemic_graph_backend import EpistemicGraphBackend
    from ..core.graph_compute import GraphComputeEngine

    gc = GraphComputeEngine(graph_name="kg_enrich_oneshot")
    backend = EpistemicGraphBackend()
    backend._graph = gc

    community_fn = None
    comm = None
    if args.features:
        comm = GraphComputeEngine(graph_name="kg_enrich_comm")
        community_fn = make_community_fn(comm)
    llm_fn = None
    if args.cards:
        from .cards import make_llm_fn

        llm_fn = make_llm_fn()

    from .capability_writeback import resolve_writeback_fn

    pipe = EnrichmentPipeline(
        backend,
        make_parse_fn(gc),
        community_fn=community_fn,
        llm_fn=llm_fn,
        writeback_fn=resolve_writeback_fn(
            backend
        ),  # CONCEPT:KG-2.8 (gated by KG_EA_WRITEBACK)
    )
    summary = pipe.enrich(args.path)

    report: dict[str, Any] = {"summary": summary.model_dump()}
    report["tests_needing_work"] = tests_needing_work(backend, limit=args.limit)
    report["patterns"] = {
        p: len(code_by_pattern(backend, p))
        for p in _PATTERNS
        if code_by_pattern(backend, p)
    }
    if args.features:
        report["features"] = list_features(backend, limit=args.limit)
    if args.pattern:
        report["by_pattern"] = code_by_pattern(backend, args.pattern, limit=args.limit)
    if args.how:
        report["how_implemented"] = how_implemented(backend, args.how)

    for tenant, eng in (("kg_enrich_oneshot", gc), ("kg_enrich_comm", comm)):
        if eng is not None:
            try:
                eng._client.tenants.delete(tenant)
            except Exception:
                pass

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    s = summary
    print(
        f"Enriched {s.files_parsed} files: {s.tests} tests, {s.code} code symbols, "
        f"{s.covers_edges} COVERS, {s.calls_edges} CALLS, {s.patterns_tagged} tagged, "
        f"{s.features} features, {s.cards_generated} cards."
    )
    print(f"\nDesign patterns: {report['patterns']}")
    print(f"\nTests needing work ({s.tests_needing_work}):")
    for r in report["tests_needing_work"][: args.limit or 15]:
        codes = ", ".join(i["code"] for i in r["issues"])
        print(
            f"  {r['name']:<46} mocks={r.get('mock_count')} asserts={r.get('assert_count')} -> {codes}"
        )
    if args.features:
        print("\nTop features (call-graph communities):")
        for f in report["features"][: args.limit or 10]:
            print(f"  {f['name']:<40} size={f['size']} patterns={f['patterns'][:4]}")
    if args.pattern:
        print(f"\nCode tagged '{args.pattern}':")
        for r in report["by_pattern"][: args.limit or 20]:
            print(f"  {r['kind']:<9} {r['name']:<40} [{r['file_path'].split('/')[-1]}]")
    if args.how:
        print(f"\nHow '{args.how}' is implemented:")
        for r in report["how_implemented"]:
            print(
                f"  {r['kind']} {r['name']} [{r['file_path'].split('/')[-1]}] patterns={r['patterns']}"
            )
            if r.get("summary"):
                print(f"    {r['summary']}")
            for resp in r.get("responsibilities", []):
                print(f"      - {resp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
