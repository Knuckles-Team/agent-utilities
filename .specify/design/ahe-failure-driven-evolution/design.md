# Design Document: Production failures enter the SAME golden-loop intake as research gaps, via a signature-clustered gap-topic with no ADDRESSED_BY edge

CONCEPT:AU-AHE.harness.failure-evolution

> `agent_utilities/knowledge_graph/adaptation/failure_analyzer.py`.

## Decision — cluster Langfuse-observed failures into deterministic signatures and inject them as gap-topics the existing intake stage already watches for

The module docstring (`failure_analyzer.py:4-19`) states the prior gap
directly: "the research-driven golden loop never had" a way to learn from
**failures observed in production telemetry** — it only ingested papers and
unresolved research concepts. This module closes that: pull ERROR
observations, low-score traces, and cost/latency anomalies from Langfuse
(via a read-only `LangfuseTraceBackend`), cluster them into recurring
failure *signatures* deterministically (LLM-free), and materialize a
synthetic `failure_gap` Concept per pattern — **with no `ADDRESSED_BY`
edge**.

**The rejected alternative is building a separate, parallel remediation
pipeline specifically for production failures.** Instead, the absence of an
`ADDRESSED_BY` edge is the exact signal the golden loop's existing intake
stage (`topic_resolver.unresolved_topics`) already scans for, so a
production-failure gap "is picked up unchanged" by machinery that already
existed for research gaps, and synthesizes a remediation proposal through the
same path. The single shared gap-topic creation function
(`failure_analyzer.py:280-295`) is reused by three independent callers — the
`FailureAnalyzer._materialize` path, the fleet-event triage handler, and the
anomaly consumer — so all three failure-discovery sources funnel into one
gap representation rather than each minting its own.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/adaptation/failure_analyzer.py`,
  `agent_utilities/knowledge_graph/research/loop_controller.py` (the intake
  stage that consumes these gaps), `agent_utilities/graph/parallel_engine.py`,
  `agent_utilities/harness/trace_backend.py`.
- **Backward Compatible**: Yes — additive; the golden loop's intake stage
  behavior toward existing unresolved topics is unchanged, it simply now
  also sees failure-derived ones.
- **Known weak point**: signature clustering is deterministic and LLM-free by
  design (cheap, no added inference cost per pull), which means two failures
  that are semantically the same but produce different error-message
  surface text will not cluster into the same signature and will be
  double-counted as separate gaps.
