#!/usr/bin/env python3
"""Live measurement: does vLLM's prefix cache reuse a ContextCompiler bundle's KV?

CONCEPT:AU-KG.retrieval.context-compiler-kv-seam — Seam 6 deep half. This is the
documented, runnable proof for the serving-layer wire added in
``agent_utilities.knowledge_graph.retrieval.context_compiler_serving``: build a
real :class:`ContextBundle`, render it via
:meth:`ContextBundle.as_prompt_messages` (the bundle's ``as_text()`` as a
byte-stable system prefix), and send it to the LIVE ``vllm.arpa`` endpoint
several times through :func:`bundle_chat_completion` — the same wire a real
caller would use — while reading vLLM's own ``/metrics``
``vllm:prefix_cache_hits_total`` / ``vllm:prefix_cache_queries_total`` counters
(token-level ground truth, not just wall-clock) before/after each call.

Calls made (kept intentionally SHORT — ``max_tokens`` is tiny and there are
only 6 live requests total across THREE independent bundles, each with a
bounded timeout/retry, per the GB10-power-fault discipline — this must never
be a load test). For each of bundle A / B / C, in order:

  1. COLD  — turn 1.  First time this bundle's text is ever sent. Since each
     bundle is disjoint evidence, this also proves no reuse leaks in from a
     PRIOR bundle in the loop (no false reuse).
  2. WARM  — turn 2.  SAME stable prefix (system message) as the cold call,
     a DIFFERENT turn-specific suffix (user message). If vLLM's automatic
     prefix cache is doing its job, this call's prefill should reuse the KV
     blocks of this bundle's system-message tokens computed in the cold call.

Testing three independent bundles (not just one pair) turns "the warm call
was faster" from an anecdote into a reproducible pattern.

For each call this script reports: wall-clock latency, ``usage.prompt_tokens``
from the response, and the delta of vLLM's ``prefix_cache_hits_total`` /
``prefix_cache_queries_total`` counters (in tokens) scraped from ``/metrics``
immediately before and after the call. Run with::

    python3 scripts/measure_bundle_kv_reuse.py [--base-url http://vllm.arpa]

Never fabricates numbers — if ``/metrics`` is unreachable it still reports the
latency-based signal and says so explicitly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

# A fresh nonce per invocation so every run's bundles are GENUINELY novel to
# vLLM's persistent prefix cache (blocks from a prior run of this same script
# would otherwise still be warm server-side and make "cold" mislabeled) —
# without a nonce, only the FIRST-ever run of this script would show a true
# cold baseline; every subsequent run would already be warm from the last one.
_RUN_NONCE = uuid.uuid4().hex[:12]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_utilities.knowledge_graph.retrieval.context_compiler import (
    ContextCompiler,
)
from agent_utilities.knowledge_graph.retrieval.context_compiler_serving import (
    bundle_chat_completion,
)

_METRIC_RE = re.compile(
    r"^vllm:(prefix_cache_hits_total|prefix_cache_queries_total)\{[^}]*\}\s+([0-9.eE+]+)",
    re.MULTILINE,
)


class FakeRetriever:
    """Fixed candidate pool — same shape the unit tests use, no engine/network needed
    for the RETRIEVAL half; only the final chat call touches the live vLLM."""

    def __init__(self, nodes: list[dict]) -> None:
        self._nodes = nodes

    def retrieve_hybrid(self, query, context_window=10, **kwargs):
        return list(self._nodes)[:context_window]


def _make_bundle(compiler: ContextCompiler, query: str, top_k: int):
    return compiler.compile(query, top_k=top_k, candidate_pool=top_k, token_budget=4000)


def _long_claim(topic: str, idx: int, sentence: str) -> dict:
    # ~60-90 tokens of body text per item so a handful of items produces a
    # multi-hundred-token stable system prefix — enough for prefix-cache reuse
    # to be measurable over a handful of KV blocks. The run nonce is folded
    # into the text (not just the id) so the actual TOKEN CONTENT — and
    # therefore vLLM's prefix-cache key over it — is novel every run.
    sentence_with_nonce = f"{sentence} (run {_RUN_NONCE})"
    body = " ".join([sentence_with_nonce] * 12)
    return {
        "id": f"claim:{topic}:{idx}:{_RUN_NONCE}",
        "type": "Claim",
        "name": f"{topic} claim {idx}",
        "description": body,
        "score": 0.9 - idx * 0.01,
        "confidence": 0.85,
        "source_refs": [f"doc:{topic}:{idx}"],
    }


BUNDLE_A_NODES = [
    _long_claim(
        "alpha",
        i,
        "The alpha subsystem's failover threshold is governed by the "
        "quorum-replication policy documented in the epistemic ledger.",
    )
    for i in range(6)
]

BUNDLE_B_NODES = [
    _long_claim(
        "bravo",
        i,
        "The bravo ingestion pipeline's backpressure policy caps in-flight "
        "batches according to the resource-priority edict.",
    )
    for i in range(6)
]

BUNDLE_C_NODES = [
    _long_claim(
        "charlie",
        i,
        "The charlie escalation runbook requires two independent approvers "
        "before any production rollback is authorized.",
    )
    for i in range(6)
]

# (topic label, nodes, compile query, cold turn text) — one entry per
# independently-constructed bundle exercised below. Three bundles (not just
# one A/B pair) is what makes the "every repeat call hits exactly one
# block-size worth of tokens" pattern a reproducible finding rather than a
# single coincidence.
_BUNDLE_SPECS: list[tuple[str, list[dict], str, str]] = [
    (
        "A",
        BUNDLE_A_NODES,
        "alpha failover policy",
        "In one sentence, what governs the alpha failover threshold?",
    ),
    (
        "B",
        BUNDLE_B_NODES,
        "bravo backpressure policy",
        "In one sentence, what caps in-flight batches for bravo?",
    ),
    (
        "C",
        BUNDLE_C_NODES,
        "charlie rollback approval",
        "Who must approve a charlie rollback?",
    ),
]


@dataclass
class CallResult:
    label: str
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    metrics_before: dict[str, float] | None
    metrics_after: dict[str, float] | None
    error: str | None = None

    @property
    def hits_delta(self) -> float | None:
        if self.metrics_before is None or self.metrics_after is None:
            return None
        return self.metrics_after.get(
            "prefix_cache_hits_total", 0.0
        ) - self.metrics_before.get("prefix_cache_hits_total", 0.0)

    @property
    def queries_delta(self) -> float | None:
        if self.metrics_before is None or self.metrics_after is None:
            return None
        return self.metrics_after.get(
            "prefix_cache_queries_total", 0.0
        ) - self.metrics_before.get("prefix_cache_queries_total", 0.0)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "latency_s": round(self.latency_s, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "prefix_cache_hit_tokens_delta": self.hits_delta,
            "prefix_cache_query_tokens_delta": self.queries_delta,
            "error": self.error,
        }


def _scrape_metrics(
    metrics_url: str, timeout_s: float = 5.0
) -> dict[str, float] | None:
    try:
        with urllib.request.urlopen(metrics_url, timeout=timeout_s) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    out: dict[str, float] = {}
    for name, value in _METRIC_RE.findall(text):
        out[name] = out.get(name, 0.0) + float(value)
    return out or None


def _run_call(
    label: str,
    bundle,
    turn_text: str,
    *,
    base_url: str,
    metrics_url: str | None,
    max_tokens: int = 8,
) -> CallResult:
    before = _scrape_metrics(metrics_url) if metrics_url else None
    start = time.monotonic()
    error = None
    prompt_tokens = None
    completion_tokens = None
    try:
        resp = bundle_chat_completion(
            bundle,
            turn_text,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
    except Exception as exc:  # noqa: BLE001 — report, don't crash the measurement
        error = f"{type(exc).__name__}: {exc}"
    latency = time.monotonic() - start
    after = _scrape_metrics(metrics_url) if metrics_url else None
    return CallResult(
        label=label,
        latency_s=latency,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        metrics_before=before,
        metrics_after=after,
        error=error,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://vllm.arpa/v1")
    ap.add_argument(
        "--metrics-url",
        default=None,
        help="Defaults to <base-url without /v1>/metrics",
    )
    args = ap.parse_args()
    metrics_url = args.metrics_url or args.base_url.rsplit("/v1", 1)[0] + "/metrics"

    bundles = {}
    prefixes = {}
    for label, nodes, query, _ in _BUNDLE_SPECS:
        compiler = ContextCompiler(FakeRetriever(nodes))
        bundle = _make_bundle(compiler, query, top_k=len(nodes))
        bundles[label] = bundle
        prefixes[label] = bundle.as_prompt_messages("x")[0]["content"]
        print(
            f"bundle {label} stable-prefix chars: {len(prefixes[label])}",
            file=sys.stderr,
        )
    distinct = len(set(prefixes.values()))
    assert distinct == len(prefixes), "test bundles must each render a distinct prefix"

    # For every bundle: one COLD call (first-ever exposure — proves no false
    # reuse against whatever came before) then one WARM call (same bundle,
    # different turn_text — proves same-bundle reuse). 2 calls x 3 bundles = 6
    # total live requests, each capped at max_tokens=8 — short and bounded.
    results: list[CallResult] = []
    for label, _nodes, _query, cold_turn_text in _BUNDLE_SPECS:
        bundle = bundles[label]
        results.append(
            _run_call(
                f"cold_bundle{label}_turn1",
                bundle,
                cold_turn_text,
                base_url=args.base_url,
                metrics_url=metrics_url,
            )
        )
        results.append(
            _run_call(
                f"warm_bundle{label}_turn2_diff_suffix",
                bundle,
                "Restate that in different words.",
                base_url=args.base_url,
                metrics_url=metrics_url,
            )
        )

    report = {
        "base_url": args.base_url,
        "metrics_url": metrics_url,
        "metrics_reachable": any(r.metrics_before is not None for r in results),
        "bundle_prefix_chars": {label: len(p) for label, p in prefixes.items()},
        "calls": [r.to_dict() for r in results],
    }
    print(json.dumps(report, indent=2))

    print("\n--- summary ---", file=sys.stderr)
    for r in results:
        print(
            f"{r.label:35s} latency={r.latency_s:.3f}s  "
            f"prompt_tokens={r.prompt_tokens}  hit_tokens_delta={r.hits_delta}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
