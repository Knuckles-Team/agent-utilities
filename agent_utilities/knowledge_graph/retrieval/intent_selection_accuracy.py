#!/usr/bin/python
from __future__ import annotations

"""Intent-surface selection-accuracy harness — CONCEPT:AU-ECO.mcp.intent-surface-selection-accuracy.

Seam 8 program-design §4 phase 4 ("A/B measurement — selection accuracy + task
success, condensed vs. intent"). The one number that actually matters for the
condensed-vs-intent trade-off: given ONLY a natural-language description of a
task (never the tool's own name), how often does :func:`~agent_utilities.mcp.
tools.intent_tools.resolve_intent` rank the SAME capability a caller who
already knew the tool name would have named directly?

"Naming the tool directly" is trivially 100% accurate by definition (you typed
the name) — that is not a competing ranking to measure, it is the baseline the
intent surface's convenience/context-savings is traded against. The real,
measurable question is this module's job: how good is the intent surface's
OWN routing, on its own, against a small hand-labelled corpus of realistic
phrasings? Reported as both **top-1** (would a single unattended dispatch
have called the right capability) and **top-3** (would it be one of the first
few candidates a human/agent could pick from, e.g. via `find`).

The corpus is intentionally small (15-25 cases) and hand-labelled — this is a
tripwire against a real regression in the resolver/CPD wiring, not a
statistically rigorous benchmark. Every measurement in this module is a LIVE
call into the real resolver (against the real, checked-in CPD set when
present) — never a fabricated/precomputed number.
"""

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "AccuracyCase",
    "AccuracyResult",
    "AccuracyReport",
    "CORPUS",
    "measure_selection_accuracy",
    "render_report",
]


@dataclass(frozen=True)
class AccuracyCase:
    """One labelled (verb, natural-language intent) -> expected capability case."""

    verb: str
    intent: str
    expected_tool: str


#: CONCEPT:AU-ECO.mcp.intent-surface-selection-accuracy — hand-labelled corpus, one case per
#: representative phrasing across all six intent verbs (``find`` is exercised
#: implicitly — every case here is also a valid ``find`` query since ``find``
#: ranks across all verbs). Intent wording deliberately avoids the tool's own
#: name (that would trivially route by string containment) so the measurement
#: reflects real natural-language routing, not name-matching.
CORPUS: tuple[AccuracyCase, ...] = (
    AccuracyCase(
        "ask", "run a read-only cypher query against the knowledge graph", "graph_query"
    ),
    AccuracyCase(
        "ask", "search the knowledge graph using hybrid semantic search", "graph_search"
    ),
    AccuracyCase("ask", "ask the graph a plain english question", "nl_query"),
    AccuracyCase(
        "ask", "answer a data analysis question with multi-step reasoning", "ask_data"
    ),
    AccuracyCase(
        "ask", "explain a belief or decision the system made", "graph_explain"
    ),
    AccuracyCase("ask", "show me the promql metrics query results", "graph_promql"),
    AccuracyCase("ask", "navigate the resolved code graph symbols", "graph_code_nav"),
    AccuracyCase(
        "ask", "run structural analysis over the knowledge graph", "graph_analyze"
    ),
    AccuracyCase("write", "write a new node into the knowledge graph", "graph_write"),
    AccuracyCase("write", "ingest a document into the knowledge graph", "graph_ingest"),
    AccuracyCase("write", "sync an external source into the graph", "source_sync"),
    AccuracyCase("write", "bulk write records back to the graph", "graph_writeback"),
    AccuracyCase("act", "orchestrate an agent workflow", "graph_orchestrate"),
    AccuracyCase("act", "run the loop engine cycle", "graph_loops"),
    AccuracyCase("act", "schedule a recurring goal", "graph_goals"),
    AccuracyCase("act", "execute code inside the sandbox", "graph_sandbox"),
    AccuracyCase("manage", "configure a backend connection", "graph_configure"),
    AccuracyCase("manage", "manage a secret credential", "graph_secret"),
    AccuracyCase("manage", "manage the kv cache layering policy", "graph_kvcache"),
    AccuracyCase("why", "explain why this decision was made", "graph_explain"),
    AccuracyCase("why", "evaluate the quality of this response", "graph_evaluate"),
)


@dataclass
class AccuracyResult:
    case: AccuracyCase
    ranked_tools: list[str] = field(default_factory=list)
    top1_hit: bool = False
    top3_hit: bool = False


@dataclass
class AccuracyReport:
    results: list[AccuracyResult]
    n: int
    top1_accuracy: float
    top3_accuracy: float


def measure_selection_accuracy(
    corpus: tuple[AccuracyCase, ...] = CORPUS, *, top_k: int = 5
) -> AccuracyReport:
    """Live-measure top-1/top-3 selection accuracy against ``corpus``.

    Calls the REAL :func:`~agent_utilities.mcp.tools.intent_tools.resolve_intent`
    for every case — never precomputed/fabricated. Lazy import so this module
    stays importable without the optional ``[mcp]`` extra until actually
    measured.
    """
    from agent_utilities.mcp.tools import intent_tools

    results: list[AccuracyResult] = []
    for case in corpus:
        candidates = intent_tools.resolve_intent(case.verb, case.intent, top_k=top_k)
        tools = [c.tool for c in candidates]
        results.append(
            AccuracyResult(
                case=case,
                ranked_tools=tools,
                top1_hit=tools[:1] == [case.expected_tool],
                top3_hit=case.expected_tool in tools[:3],
            )
        )
    n = len(results)
    top1 = sum(1 for r in results if r.top1_hit) / n if n else 0.0
    top3 = sum(1 for r in results if r.top3_hit) / n if n else 0.0
    return AccuracyReport(results=results, n=n, top1_accuracy=top1, top3_accuracy=top3)


def render_report(report: AccuracyReport) -> str:
    """A human-readable per-case + summary rendering (used by the CLI + test failures)."""
    lines: list[str] = [
        f"Intent-surface selection accuracy — {report.n} cases "
        f"(CONCEPT:AU-ECO.mcp.intent-surface-selection-accuracy)",
        f"  top-1 accuracy: {report.top1_accuracy:.2%}",
        f"  top-3 accuracy: {report.top3_accuracy:.2%}",
        "",
    ]
    for r in report.results:
        mark = "OK  " if r.top1_hit else ("~3~ " if r.top3_hit else "MISS")
        lines.append(
            f"  [{mark}] {r.case.verb:6s} {r.case.intent!r} "
            f"-> expected={r.case.expected_tool!r} got={r.ranked_tools[:3]!r}"
        )
    return "\n".join(lines)


def _to_jsonable(report: AccuracyReport) -> dict[str, Any]:
    return {
        "n": report.n,
        "top1_accuracy": round(report.top1_accuracy, 4),
        "top3_accuracy": round(report.top3_accuracy, 4),
        "cases": [
            {
                "verb": r.case.verb,
                "intent": r.case.intent,
                "expected_tool": r.case.expected_tool,
                "ranked_tools": r.ranked_tools,
                "top1_hit": r.top1_hit,
                "top3_hit": r.top3_hit,
            }
            for r in report.results
        ],
    }
