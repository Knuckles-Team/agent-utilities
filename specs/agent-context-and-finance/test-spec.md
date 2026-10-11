# Test specification

| Path | Positive acceptance | Negative acceptance |
|---|---|---|
| Context | Exact tokenizer count, deterministic package, cited source/proof class, budget and schema digest | Stale epoch, cross-tenant candidate, absent citation, unknown tokenizer and over-budget package abstain |
| Retrieval feedback | Independent outcome joins to DecisionRecord and can propose a template | Self-reported success cannot promote a template. Wrong schema digest cannot reuse one |
| Re-embedding | Shadow generation passes PSI/quality threshold and EG atomic activation/rollback receipt | Failed lease, quality regression and partial generation leave old generation active |
| Finance math | Public golden fixture matches EG deterministic output. AU explanation cites strategy and inputs | LLM-only price/math, stale feed or missing calibration yields abstention |
| Paper/live boundary | Paper account changes simulation only; separately approved live intent reaches SDK once | Missing/expired lease, wrong account, unapproved scope and replay with changed payload cannot place order |

From a public checkout run `python3 scripts/uv_workspace.py doctor`, focused `python3 scripts/uv_workspace.py run --all-extras pytest <path> -q`, full `python3 scripts/uv_workspace.py run --all-extras pytest -q`, and `agent-utilities lane lease --resource precommit-all-files --operation gate -- python3 scripts/safe_precommit_all_files.py`. CI uses deterministic public EG/SDK contract fixtures for PRs. A live external broker is a separately provisioned release environment with a receipt. Record CCCC, jscpd, dupehound, KISS, Ruff/mypy and hosted CI deltas at the exact commit.

- **AU-CONTEXT-R004.1:** `tests/unit/knowledge_graph/retrieval/test_context_window_fit.py::test_window_fit_fields`.
- **AU-CONTEXT-R004.2:** `tests/unit/knowledge_graph/retrieval/test_context_window_fit.py::test_over_window_margin_default`.
- **AU-CONTEXT-R004.3:** `tests/unit/knowledge_graph/retrieval/test_context_window_fit.py::test_compile_model_context_checks_fit`.
- **AU-CONTEXT-R004.4:** `tests/unit/knowledge_graph/retrieval/test_context_window_fit.py::test_over_window_returns_typed_handoff`.
- **AU-CONTEXT-R004.5:** `tests/unit/knowledge_graph/retrieval/test_context_window_fit.py::test_fit_verdict_in_trace`.
- **AU-CONTEXT-R004.6:** `tests/integration/test_context_window_fit_served.py::test_served_over_window_prompt`.

## Producer contract coverage

AU-CONTEXT-R004 consumes EG-DECISION-ENGINE-R137.

Harness tests assert receipt and application of these contracts. They do not prove producer decisions, storage or admission.


### Planned context handoff acceptance

Extend `test_served_over_window_prompt` through the served context compiler with a scripted R137 decision. Use a prompt at 0.9 of the window and a received fraction of 0.8. Assert a typed handoff, preserved source bounds and no silent truncation. A fitting prompt compiles unchanged. Missing decision fields or an uncountable tokenizer abstain.

Pass the returned plan and limits to the R026 entry in `_router_impl.py`. Assert each limit survives unchanged, and a depth or cost breach stops execution. Inspect the trace for counts, verdict and tokenizer identity; it must contain no prompt text. This is planned consumer acceptance, not evidence that R137 or R026 is implemented or served-tested.
