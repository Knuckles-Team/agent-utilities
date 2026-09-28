# Test specification

| Path | Positive acceptance | Negative acceptance |
|---|---|---|
| Context | Exact tokenizer count, deterministic package, cited source/proof class, budget and schema digest | Stale epoch, cross-tenant candidate, absent citation, unknown tokenizer and over-budget package abstain |
| Retrieval feedback | Independent outcome joins to DecisionRecord and can propose a template | Self-reported success cannot promote a template; wrong schema digest cannot reuse one |
| Re-embedding | Shadow generation passes PSI/quality threshold and EG atomic activation/rollback receipt | Failed lease, quality regression and partial generation leave old generation active |
| Finance math | Public golden fixture matches EG deterministic output; AU explanation cites strategy and inputs | LLM-only price/math, stale feed or missing calibration yields abstention |
| Paper/live boundary | Paper account changes simulation only; separately approved live intent reaches SDK once | Missing/expired lease, wrong account, unapproved scope and replay with changed payload cannot place order |

From a public checkout run `python3 scripts/uv_workspace.py doctor`, focused `python3 scripts/uv_workspace.py run --all-extras pytest <path> -q`, full `python3 scripts/uv_workspace.py run --all-extras pytest -q`, and `agent-utilities lane lease --resource precommit-all-files --operation gate -- python3 scripts/safe_precommit_all_files.py`. CI uses deterministic public EG/SDK contract fixtures for PRs. A live external broker is a separately provisioned release environment with a receipt. Record CCCC, jscpd, dupehound, KISS, Ruff/mypy and hosted CI deltas at the exact commit.
