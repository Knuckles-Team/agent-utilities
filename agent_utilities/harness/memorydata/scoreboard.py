#!/usr/bin/python
from __future__ import annotations

"""Render the MemoryData bake-off as a markdown scoreboard (CONCEPT:AHE-3.74).

Turns a list of :class:`~agent_utilities.harness.memorydata.bakeoff.BakeoffResult` into a
report with three sections, modelled on :mod:`agent_utilities.rlm.benchmarks.scoreboard`:

1. **Measured results** — one row per ``family/task/config`` with EM, ROUGE-L, judge, and
   mean query latency.
2. **Best config per family** — the attribution table: which retrieval surface wins each
   family, and by what margin.
3. **Router vs best single** — present only when a router result is in the set, showing
   whether per-family routing beats the best single config.

:data:`MEMORYDATA_BASELINES` stubs the 22 published presets so a future Δ column can compare
our numbers to the paper's.
"""

from agent_utilities.harness.memorydata.bakeoff import BakeoffResult

__all__ = ["render_scoreboard", "MEMORYDATA_BASELINES"]


# Published MemoryData preset baselines (placeholder — fill with paper numbers for a Δ column).
# Keys are the 22 method presets MemoryData reports; ``None`` means "not yet staged".
MEMORYDATA_BASELINES: dict[str, float | None] = {
    name: None
    for name in (
        "long_context",
        "rag",
        "embedding_rag",
        "graph_rag",
        "bm25_rag",
        "letta",
        "mem0",
        "mem0_graph",
        "cognee",
        "memochat",
        "memoryos",
        "simplemem",
        "lightmem",
        "a_mem",
        "memtree",
        "everos",
        "zep",
        "zep_local",
        "memos",
        "memo_rag",
        "self_rag",
        "raptor",
    )
}


def render_scoreboard(
    results: list[BakeoffResult],
    baselines: dict[str, float | None] | None = None,
    *,
    title: str = "MemoryData Graph-OS Bake-off",
) -> str:
    """Render ``results`` as a markdown scoreboard (CONCEPT:AHE-3.74).

    ``baselines`` defaults to :data:`MEMORYDATA_BASELINES`; it is reserved for a future Δ
    column against the published presets.
    """
    baselines = baselines if baselines is not None else MEMORYDATA_BASELINES
    measured = [r for r in results if not r.is_router]
    router_rows = [r for r in results if r.is_router]

    lines: list[str] = [f"# {title}", ""]
    lines.append(
        "Each retrieval config is a graph-os surface (semantic/hybrid, bi-temporal as-of, "
        "context-plane synthesis, latent, memory-facts, graph-rerank) run over the same "
        "MemoryData families. EM is normalized exact-match; ROUGE-L is the LCS F1."
    )
    lines.append("")

    # 1. Measured results.
    lines.append("## Measured results")
    lines.append("")
    lines.append(
        "| Family | Task | Config | EM | ROUGE-L | Judge | Mean query (s) | n | Notes |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for r in sorted(measured, key=lambda x: (x.family, x.task, x.config)):
        lines.append(
            f"| {r.family} | {r.task} | {r.config} | {r.exact_match * 100:.1f}% | "
            f"{r.rouge_l:.3f} | {r.judge_score * 100:.1f}% | {r.mean_query_s:.4f} | {r.n} | {r.notes} |"
        )
    lines.append("")

    # 2. Best config per family (attribution).
    lines.append("## Best config per family")
    lines.append("")
    lines.append("| Family | Best config | EM | ROUGE-L | Runner-up | Δ EM |")
    lines.append("|---|---|---:|---:|---|---:|")
    best_by_family = _best_per_family(measured)
    for family in sorted(best_by_family):
        ranked = best_by_family[family]
        best = ranked[0]
        runner = ranked[1] if len(ranked) > 1 else None
        runner_name = runner.config if runner else "—"
        delta = (
            f"{(best.exact_match - runner.exact_match) * 100:+.1f}" if runner else "—"
        )
        lines.append(
            f"| {family} | {best.config} | {best.exact_match * 100:.1f}% | "
            f"{best.rouge_l:.3f} | {runner_name} | {delta} |"
        )
    lines.append("")

    # 3. Router vs best single config (only when a router result is present).
    if router_rows:
        lines.append("## Router vs best single config")
        lines.append("")
        lines.append(
            "| Family | Router EM | Best single EM | Best single config | Δ EM |"
        )
        lines.append("|---|---:|---:|---|---:|")
        router_by_family = _best_per_family(router_rows)
        for family in sorted(router_by_family):
            router_best = router_by_family[family][0]
            single = best_by_family.get(family)
            single_best = single[0] if single else None
            single_em = f"{single_best.exact_match * 100:.1f}%" if single_best else "—"
            single_cfg = single_best.config if single_best else "—"
            delta = (
                f"{(router_best.exact_match - single_best.exact_match) * 100:+.1f}"
                if single_best
                else "—"
            )
            lines.append(
                f"| {family} | {router_best.exact_match * 100:.1f}% | {single_em} | "
                f"{single_cfg} | {delta} |"
            )
        lines.append("")

    lines.append(
        "> Baselines for the 22 MemoryData presets are stubbed in `MEMORYDATA_BASELINES`; "
        "stage the published numbers there for a like-for-like Δ column."
    )
    lines.append("")
    return "\n".join(lines)


def _best_per_family(results: list[BakeoffResult]) -> dict[str, list[BakeoffResult]]:
    """Group results by family and rank each group by EM then ROUGE-L (descending)."""
    grouped: dict[str, list[BakeoffResult]] = {}
    for r in results:
        grouped.setdefault(r.family, []).append(r)
    for family, rows in grouped.items():
        rows.sort(key=lambda x: (x.exact_match, x.rouge_l), reverse=True)
    return grouped
