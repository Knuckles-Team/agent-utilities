# Test contract: harness evolution

| Scenario | Fixture | Expected result |
|---|---|---|
| All controls omitted; capture-only enabled | unit + provider fake | capture/train/promote default false; capture alone emits no trainer job or pointer change |
| Closed/unversioned provider, no chosen-token probabilities | capability negatives | typed refusal; no fabricated training row |
| Terminal valid capture and replay | fixed token ids, masks, frozen `log_q`, verifier, held blobs | exact digest match, masked non-policy tokens, immutable sampler identity |
| Incomplete/mismatched arrays, missing reward, dead blob holder | capture negatives | trace may persist, trainer eligibility false |
| External trainer absent/failed/cancelled | injected dispatch and A2A fake | no serving pointer movement; recoverable receipt, no duplicate job |
| Held-out regression or unsupported replay mass | promotion negative | promotion denied; old immutable version remains served |
| Duplicate evidence signal | served EG in-memory contract | one Gap and one WorkItem, stable idempotency key |
| Offer with open dependency, cooldown, over-budget or other tenant | decision contract | candidate excluded with reason; no AU fallback sort |
| Decide unavailable, commit replay mismatch, claim fence lost | integration fault | no execution and no second scheduler |
| Proposal missing approval, stale base, dirty tree, path escape, gate failure, uncertain remote result | graph-os/RM contract fakes | no unauthorized commit; pending proposal with typed reason |
| L4 conformance | native adapter with injected runner; Claude Code adapter with fake executable | both satisfy `HarnessPort`; outcome names the port, keeps the run identifier and uses a known status |
| Native envelope | injected `run_agent` returning the run-summary envelope | summary requested; JSON output parsed; degraded failure text kept; outcome marked recorded |
| Native timeout | runner slower than `timeout_s` | typed `timeout` outcome |
| Claude Code launch | fake executable records arguments, directory, input and environment | pinned flags first; MCP configuration last; prompt on input only; directory is the worktree; parent secrets absent |
| Claude Code result | fake success, error, unparseable and sleeping modes | typed usage, cost, model and transcript reference; `failed` with exit code; `timeout` with the child killed |
| Claude Code refusal | no worktree, no configuration file, no executable | `refused` outcome; child never starts |
| Diff stat | temporary Git repository with one changed file | exact files, insertions and deletions; no repository write |
| Selection | registry with fake ports | `native` by default; field selects the port; unknown name raises |
| L5 recording | patched RunTrace writer and usage recorder | non-native outcome written once with `harness:<name>` mode; native outcome not written again |
| Parallel engine dispatch | `AgentSpec(harness="claude-code")` with a fake port | node runs through the port and returns its result |
| Successful approved proposal | public contract integration | exact commit/gate/receipt digests; Gap resolves only after receipt re-ingest |

Fresh checkout: `python3 scripts/uv_workspace.py doctor`, then focused `python3 scripts/uv_workspace.py run --all-extras pytest tests/harness tests/unit -q`. New tests must inject fake EG, graph-os, provider, trainer and repository-manager boundaries; normal PR CI needs no live graph, vendor credentials, model server, GPU or private network. Optional live certification records exact receipts and is not a prerequisite for contribution PRs unless the changed contract requires it.

Quality: run the repository's configured CCCC, jscpd and Dupehound gates and `python3 scripts/safe_precommit_all_files.py` through the lease wrapper when available. Configured Dupehound is 0.1.2 with similarity 0.85 and 40-token minimum; jscpd is 5.0.16 with 50-token/5-line minimum and mild mode. Honor actual configured differential filters; require zero **new** duplicate findings. CCCC uses the configured hook threshold, not an invented score. Apply KISS by extending `SubstrateTrainer`, native WorkItem, EG Decide and existing Loop controller, and deleting replacement authority. Missing optional tooling is recorded as not run, never as a pass.
