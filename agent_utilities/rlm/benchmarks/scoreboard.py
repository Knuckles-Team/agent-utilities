"""Render benchmark results next to the paper's published numbers (CONCEPT:AHE-3.32).

``PAPER_RESULTS`` holds the headline figures from Zhang et al. (2025, arXiv:2512.24601) so the
scoreboard shows our accuracy/cost beside theirs with an explicit delta — turning "we have the
mechanism" into a measured comparison. Synthetic rows are flagged so they are never read as a
like-for-like beat of the paper's real-dataset numbers.
"""

from __future__ import annotations

from .base import BenchResult

# Headline accuracies (%) and approx cost ($/query) from the paper, for reference rows.
# Source: arXiv:2512.24601v3, Table 1 + cost discussion. RLM = RLM(GPT-5, depth=1).
PAPER_RESULTS: dict[str, dict[str, float]] = {
    "oolong": {"rlm_acc": 56.0, "base_acc": 44.0, "rlm_cost": 0.99},
    "oolong_pairs": {"rlm_acc": 58.0, "base_acc": 0.1, "rlm_cost": 0.99},
    "browsecomp_plus": {"rlm_acc": 91.3, "compaction_acc": 70.5, "rlm_cost": 0.99},
    "longbench_codeqa": {"rlm_acc": 66.0, "base_acc": 20.0},
    "s_niah": {"rlm_acc": 100.0, "base_acc": 100.0},
}

# External scaffolds the paper reports but we do not run in-process.
EXTERNAL_BASELINES = ("codeact_subcalls", "claude_code")


def render_scoreboard(
    results: list[BenchResult], *, title: str = "RLM Benchmark Scoreboard"
) -> str:
    """Render ``results`` as a markdown report with a per-task paper-comparison section."""
    lines: list[str] = [f"# {title}", ""]
    lines.append(
        "Systems: **rlm** (agent-utilities), **vanilla** (single truncated call), "
        "**compaction** (chunk→summarize→answer). External scaffolds "
        f"({', '.join(EXTERNAL_BASELINES)}) are **not run** — paper figures shown for reference only."
    )
    lines.append("")

    # Measured results table.
    lines.append("## Measured results")
    lines.append("")
    lines.append(
        "| Task | Complexity | Mode | System | Scale (chars) | Accuracy | Cost ($/q) | Tokens | Wall (s) | Notes |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---|")
    for r in sorted(results, key=lambda x: (x.task, x.scale, x.system)):
        lines.append(
            f"| {r.task} | {r.complexity} | {r.mode} | {r.system} | {r.scale:,} | "
            f"{r.accuracy * 100:.1f}% | ${r.cost_usd:.4f} | {r.total_tokens:,} | "
            f"{r.wall_s:.2f} | {r.notes} |"
        )
    lines.append("")

    # Paper comparison: best RLM accuracy we measured per task vs the paper's RLM number.
    lines.append("## Comparison to the paper (arXiv:2512.24601)")
    lines.append("")
    lines.append(
        "| Task | Our RLM (best) | Paper RLM | Δ | Our mode | Paper baseline |"
    )
    lines.append("|---|---:|---:|---:|---|---|")
    rlm_by_task: dict[str, BenchResult] = {}
    for r in results:
        if r.system != "rlm":
            continue
        cur = rlm_by_task.get(r.task)
        if cur is None or r.accuracy > cur.accuracy:
            rlm_by_task[r.task] = r
    for task, paper in PAPER_RESULTS.items():
        ours = rlm_by_task.get(task)
        our_acc = f"{ours.accuracy * 100:.1f}%" if ours else "—"
        mode = ours.mode if ours else "—"
        paper_acc = paper.get("rlm_acc")
        delta = (
            f"{ours.accuracy * 100 - paper_acc:+.1f}"
            if ours and paper_acc is not None
            else "—"
        )
        base_keys = [k for k in paper if k.endswith("_acc") and k != "rlm_acc"]
        base = (
            f"{base_keys[0].replace('_acc', '')}={paper[base_keys[0]]:.1f}%"
            if base_keys
            else "—"
        )
        lines.append(
            f"| {task} | {our_acc} | {paper_acc:.1f}% | {delta} | {mode} | {base} |"
            if paper_acc is not None
            else f"| {task} | {our_acc} | — | — | {mode} | {base} |"
        )
    lines.append("")
    lines.append(
        "> Synthetic rows reproduce each task's *structure* (complexity class, hop count) but are "
        "not the paper's exact corpora; stage real datasets under `<data_dir>/rlm_benchmarks/` to "
        "compare like-for-like."
    )
    lines.append("")
    return "\n".join(lines)
