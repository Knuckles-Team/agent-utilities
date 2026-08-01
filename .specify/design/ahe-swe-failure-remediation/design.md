# Design Document: SWE-bench score becomes an actionable, attributed remediation queue, not a static leaderboard number

CONCEPT:AU-AHE.harness.swe-failure-remediation

> `agent_utilities/harness/swebench_remediation.py`.

## Decision — every unresolved instance becomes a clustered, attributed `failure_gap`, and promotion is gated on the exact instance re-resolving in isolation

The module's own opening line names the contrast directly: "OpenHands' SWE-bench
score is a static number." Here, every *unresolved* instance instead becomes
a `FailureRecord`, clustered, and filed as a `failure_gap` Concept through
the single shared AHE-3.18 path (`file_gap_topic`) — the same canonical gap
mechanism `failure-evolution` and `canonical-gap-lifecycle` already use. The
golden loop's `unresolved_topics()` intake picks those gaps up unchanged and
drives a remediation cycle; promotion is gated by a SWE-specific regression
check that **re-runs the exact failed instance in isolation** and only
passes when it now resolves.

**The rejected alternative is treating the SWE-bench resolved-rate as a
terminal score to report, with no mechanism turning an unresolved instance
into remediation work.** That's the baseline every SWE-bench harness
(including OpenHands') implicitly takes: run the suite, report the
percentage, move on. Because the workspace already mirrors every action to
the KG grounded on the `Code` symbols it mutated, a failure here is
attributable to specific code, not an opaque log line — which is what makes
"re-run the exact failed instance and check it now resolves" a meaningful
regression gate rather than a coincidental re-pass.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/swebench_remediation.py`,
  `agent_utilities/knowledge_graph/adaptation/failure_analyzer.py`
  (`file_gap_topic`, `cluster_failures`), `agent_utilities/harness/swebench_harness.py`.
- **Backward Compatible**: Yes — additive; the SWE-bench harness itself
  (evaluation, scoring) is unchanged, this only adds a remediation path for
  unresolved instances.
- **Known weak point**: the regression check re-runs the exact failed
  instance in isolation — a remediation that fixes that specific instance
  but regresses a *different*, previously-passing instance is not caught by
  this gate; it depends on the broader promotion pipeline's own regression
  checks to catch cross-instance regressions.
