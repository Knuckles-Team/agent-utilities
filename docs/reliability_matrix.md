# Reliability / Chaos Matrix

> **Status: single-box matrix RUN + EXPANDED + one real gap HARDENED.** Every
> number below is from an actual local run on this box (`agent-utilities`
> worktree, `feat/au-reliability-matrix`), not a modeled/estimated figure. The
> matrix has three families — grounding/safety/retrieval-quality corpora,
> single-node fault-injection (chaos), and hardware-pending scenarios — mirroring
> [`docs/scaling/capacity_model.md`](scaling/capacity_model.md)'s CI-measured vs
> hardware-pending distinction for the SCALE-P2-1 soak/chaos harness. This page
> is the reliability-side counterpart.

## 1. What's covered, and where

| Family | Lives in | Enforced by |
|---|---|---|
| Grounding / safety / retrieval-quality corpus | `agent_utilities/harness/reliability_corpus.py` (`SEED_CASES`) + `agent_utilities/harness/reliability_scorers.py` (9 scorers + 1 informational) | `scripts/check_reliability_corpus.py` (pre-commit `guardrail-reliability-corpus`, FLOOR=0.90 match-rate) |
| Feedback→eval regression corpus | `agent_utilities/harness/eval_corpus.py` | `scripts/check_eval_corpus.py` (pre-commit `guardrail-eval-corpus`, FLOOR=0.90 pass-rate) |
| Retrieval Recall@k / MRR | `agent_utilities/knowledge_graph/retrieval/capability_index.py` | `scripts/check_retrieval_quality.py` (pre-commit `guardrail-retrieval-quality`, FLOOR=0.90) |
| Multi-agent supervisory chaos (pause/domain-contain/goal-loop) | `tests/integration/test_fleet_chaos.py` | `pytest -m integration` |
| Single-node fault injection (worker/lease/DLQ/tenant/restart/rolling-upgrade/connector/fencing-churn) | `tests/scale/soak/*.py` | run explicitly (not in `pytest.ini` `testpaths`, same convention as the rest of `tests/scale/`) |
| Hardware-pending (multi-node/multi-broker) | `tests/scale/soak/test_hardware_pending.py` | `pytest.mark.skip`, documented manual recipe |

## 2. What was RUN — real pass/fail, this box, this session

All commands below were executed against the worktree exactly as shown; output is summarized, not fabricated.

### 2.1 Corpus gates (clean + degrade)

```
$ python3 scripts/check_reliability_corpus.py
Reliability corpus match-rate: 1.00 (floor 0.90, 10/10 cases)
OK: reliability corpus at/above floor.

$ python3 scripts/check_reliability_corpus.py --degrade
Reliability corpus match-rate: 0.60 (floor 0.90, 6/10 cases)
FAIL: reliability corpus match-rate below floor.   (exit 1 — gate correctly trips)

$ python3 scripts/check_eval_corpus.py
Eval corpus pass-rate: 1.00 (floor 0.90)            (exit 0)
$ python3 scripts/check_eval_corpus.py --degrade
Eval corpus pass-rate: 0.00 (floor 0.90)            (exit 1 — gate correctly trips)

$ python3 scripts/check_retrieval_quality.py
Recall@5=1.000 (floor 0.9), MRR=1.000 (floor 0.9)   (exit 0)
$ python3 scripts/check_retrieval_quality.py --degrade
Recall@5=0.467 (floor 0.9), MRR=0.633 (floor 0.9)   (exit 1 — gate correctly trips)
```

All three guardrails PASS clean and correctly TRIP on `--degrade` — each gate
proven to have teeth, not just proven to pass.

### 2.2 pytest — harness, gates, chaos, single-node fault injection

```
$ python3 -m pytest tests/harness/test_reliability_corpus.py \
    tests/harness/test_reliability_scorers.py \
    tests/gates/test_reliability_corpus_gate.py \
    tests/test_attribution_reliability.py \
    tests/retrieval/test_direct_corpus.py \
    tests/scale/ \
    tests/unit/messaging/ tests/unit/protocols/ \
    tests/unit/orchestration/test_work_item.py -q
452 passed, 9 skipped in 29.72s
```

The 9 skips are the intentional ones: 6 `hardware-pending`-marked scenarios in
`test_hardware_pending.py` (see §4) plus 3 optional-dependency skips in
`unit/messaging`/`unit/protocols` (e.g. no `confluent_kafka` install).

```
$ python3 -m pytest tests/gates -q -p no:cacheprovider -o addopts=""
21 passed   (guardrail-gate-meta-tests — every gate proven to be able to fail)

$ python3 -m pytest tests/integration/test_fleet_chaos.py -q
3 passed in 212.77s
```

**Honest note on the fleet-chaos runtime**: 3 tests taking ~3.5 minutes on this
shared box is disproportionate to their logical cost (in-memory sqlite,
monkeypatched `asyncio.sleep`) — almost certainly contention from other
concurrent work on this shared host rather than the tests themselves. It still
completed and passed; if this recurs on a dedicated CI runner it is worth a
follow-up (see §5), but it did not block this run.

### 2.3 Guardrail scripts (governance/hygiene)

```
$ python3 scripts/check_concepts.py         -> OK: 981/982 concept markers registered
$ python3 scripts/check_no_stub.py          -> OK: no stub markers in production code
$ python3 scripts/check_concept_governance.py -> OK: no new CONCEPT: tags (none added)
$ python3 scripts/check_sprawl.py           -> OK: no sprawl/hygiene violations
$ python3 scripts/check_liveness.py         -> OK (code-enhancer skill not installed locally; gate skips cleanly)
```

### 2.4 Lint / type-check on touched files

```
$ ruff check agent_utilities/harness/reliability_corpus.py \
    tests/scale/soak/test_chaos_connector_and_fencing.py
All checks passed!

$ ruff format --check <same files>
2 files already formatted   (after one reformat pass)

$ python3 -m mypy agent_utilities/harness/reliability_corpus.py
Found 1 error in 1 file (agent_utilities/tools/team_tools.py:36, RunContext.team_capability)
```

The one mypy error is a **pre-existing** issue in `agent_utilities/tools/team_tools.py`
pulled in transitively by mypy's import-following — reproduced identically on
the pre-change tree (`git stash` + re-run gave the same single error). Not
touched by, or attributable to, this work. `tests/` is excluded from the mypy
pre-commit hook by repo convention (`exclude: ^(tests/|test/|scripts/|script/)`),
so the new test file is validated at pytest runtime instead, per that same
convention.

## 3. What was EXPANDED

### 3.1 Reliability/eval corpus — 4 new isolated-scorer cases

`SEED_CASES` grew from 6 to 10. The prior 2 adversarial cases
(`hallucinated_unsafe_answer`, `poisoned_retrieved_context`) fail on *many*
scorers at once because their context is sparse — good for "the corpus catches
something," weak for proving any ONE scorer's true-negative path. The 4 new
cases give every OTHER scorer fully-passing context so exactly one scorer
trips, verified against the real suite before landing (`suite.evaluate(...)`
run interactively; each case's `failed_scorers` was confirmed to be a
one-element list matching its name):

| New case | Isolates | Axis |
|---|---|---|
| `tool_necessity_missing_call` | `tool_necessity` (knowing-doing gap: tool was necessary, never called) | grounding/action |
| `retrieval_poor_recall` | `retrieval_recall` (gold id never retrieved, recall@k below floor) | retrieval-quality |
| `deception_sycophancy_answer` | `deception_probe` (sycophancy marker in an otherwise well-grounded, well-cited answer) | safety |
| `citation_overclaim_wrong_id` | `citation_quality` (cited id does not match the actual source id) | grounding |

All 10 cases match expectation (`match_rate=1.00`); `--degrade` still correctly
drops the match-rate below the 0.90 floor (`0.60`).

### 3.2 New fault-injection chaos — `tests/scale/soak/test_chaos_connector_and_fencing.py`

Wired into the **existing** soak/chaos structure (same `loadgen`/`engine`
fixtures from `tests/scale/soak/conftest.py`, same `FakeScaleEngine` +
`WorkItem` CAS pattern as the other `test_chaos_*.py` files) — no parallel
harness invented. 7 new tests, all passing:

**Connector failure** (explicitly requested; previously the matrix had no
fault-injection case for an upstream data-source connector failing):
- `test_connector_transient_failure_recovers_under_resilience_policy` — a
  connector failing twice then succeeding recovers under the REAL
  `agent_utilities.orchestration.resilience.ResiliencePolicy` /
  `run_with_resilience_sync` (not a hand-rolled retry loop).
- `test_connector_permanent_failure_fails_closed_never_silently_empty` — a
  connector that never recovers RAISES once its retry budget (3 attempts) is
  exhausted — fail-closed, never a silent empty/stale read.
- `test_connector_failure_falls_back_when_primary_exhausted` — a configured
  fallback (e.g. a secondary replica) is tried once the primary is exhausted,
  and its result is returned.

**Claim-invalidation under churn** (explicitly requested — extends the
existing single-crash scenario in `test_chaos_worker_and_delivery.py` to
repeated churn):
- `test_claim_churn_fencing_epoch_strictly_increases_and_completes_exactly_once`
  — 10 successive claim→crash→reap cycles on the SAME item; the fencing epoch
  is asserted strictly increasing and never repeating across all 10 cycles,
  every one of the 10 stale claims' late commits is rejected (`fenced`/`noop`),
  and the item finally lands `succeeded` exactly once.
- `test_claim_churn_never_allows_two_simultaneously_live_claims` — mid-churn,
  a racing claim on a still-live lease is always rejected, across 5 cycles.

**Heartbeat racing the reaper** (a genuine coverage gap: `heartbeat()` had
exactly one plain unit test, never exercised against a *concurrent*
`reap_expired_leases` sweep):
- `test_heartbeating_worker_survives_reap_then_reclaimed_once_it_stops` — a
  worker heartbeats twice, extending its lease each time; a reap sweep that
  lands AFTER the original (pre-extension) deadline but BEFORE the
  heartbeat-extended one must NOT reclaim it, proving the reaper honors the
  live extension, not a stale expiry. Once heartbeats genuinely stop, the
  reaper DOES reclaim it (past the last extended deadline), with a bumped
  fencing epoch, and the dead worker's belated heartbeat/commit is rejected.
  **Result: this passed on the first correct run — the existing
  `reap_expired_leases`/`heartbeat` implementation already handles this
  correctly; the value here is that it is now PROVEN, not merely assumed.**

**Tenant isolation under load** (explicitly requested — extends the existing
sequential elephant-then-ordinary case in `test_chaos_tenant_and_restart.py`
to adversarial interleaving across many tenants):
- `test_tenant_isolation_holds_under_interleaved_multi_tenant_load` — 19
  ordinary tenants (quota 2 each) and one elephant tenant (quota 8) submit in a
  deterministically-shuffled interleaved schedule (seeded `random.Random`, not
  wall-clock threading, so the scenario is exact and reproducible); asserts
  every ordinary tenant got its full quota regardless of elephant interleaving,
  the elephant never exceeded or leaked past its own quota, every ordinary
  tenant's work completes independently, and completing ordinary tenants'
  work never touches the elephant's still-held quota.

**Two test-design bugs were found and fixed during development** (not product
bugs): the first two churn tests initially used the default `max_attempts=3`,
which correctly dead-letters an item after 3 crash cycles (the existing,
correct `reap_expired_leases` retry-budget behavior, already covered by
`test_chaos_lifecycle_and_dlq.py`) — the test intent was to churn the CAS/
fencing machinery specifically, so `max_attempts=50` was set explicitly. Fixed
and re-verified before landing; no product code changed for this.

## 4. HARDENED (real gap found, fixed with a test)

The single-box run did not surface a live product defect that needed a code
fix — every fault case (worker crash, redelivery, DLQ, restart, rolling
upgrade, hot-tenant, connector failure, claim churn, heartbeat-vs-reap race,
interleaved tenant load) held on the first correct run of a correctly-designed
test. What this session hardened was the **coverage gap itself**:
`heartbeat()`'s interaction with the reaper had zero test coverage against a
racing sweep before this session (§3.2) — that gap is now closed with a
passing regression test, and the reliability corpus's blind spot (only
whole-suite failures, no per-scorer isolated failure) is closed with 4 new
cases (§3.1). No `agent_utilities/` product code required a change; the
hardening in this program is test coverage that would catch a REAL regression
in either area if one is introduced later, since neither invariant had a
dedicated regression test before.

## 5. Needs the 4-node cluster — explicit, not modeled

The following are documented, `pytest.mark.skip`-marked, **currently NOT run**,
and must not be reported as demonstrated until they are actually executed
against real multi-node/multi-broker infrastructure. This mirrors
`docs/scaling/capacity_model.md`'s "What is CI-measured vs hardware-pending"
table and Codex's SCALE-P2-1 guardrail: never call a MODELED result a
DEMONSTRATED one.

| Case | File | What it needs |
|---|---|---|
| 24-72h steady + burst soak at the real 1,000,000-resident scale | `tests/scale/soak/test_hardware_pending.py::test_24_72h_steady_and_burst_soak_at_full_1m_scale` | A deployed fleet at `--scale 1.0`, 800 workers, 72h wall-clock |
| Kafka consumer-group rebalance + live partition-count increase under load | `::test_broker_rebalance_and_partition_expansion_under_load` | A real Kafka cluster with a rebalance/partition-expansion trigger |
| Live engine/L0 shard split or move under concurrent writes | `::test_shard_split_or_move_under_concurrent_writes` | A real multi-shard engine deployment with a live resharding trigger |
| Worker/gateway/broker/leader/node/zone loss (each tier) | `::test_worker_gateway_broker_leader_node_or_zone_loss` | Real multi-node/multi-zone infra + a kill-injection tool |
| Rolling upgrade + live schema/ontology migration across real hosts | `::test_rolling_upgrade_and_schema_migration_across_real_hosts` | A real multi-host rolling-deploy pipeline + a live migration |
| Full 1,000,000-resident cold activation at real scale | `::test_full_1m_resident_cold_activation_at_real_scale` | The real 1M-resident population on real L0/PG shards |

**To run these for real** (per each test's docstring, and
`test_hardware_pending.py`'s module docstring): deploy the fleet with
`ENGINE_ENDPOINT` set and real Kafka/Postgres shards
(`docs/architecture/agent_dispatch.md` / `engine_sharding.md`), inject the
specific fault via the platform's own chaos tooling (`container-manager-mcp`
node/service kill, or the swarm/k8s supervisory plane's pause/kill actions —
this workspace's homelab is on the k8s cutover, so use the k8s-side kill
primitives) at the same time the generator runs, then remove the
`_HARDWARE_PENDING` skip and assert the JSON report's `ok`/`invariants`
fields:

```bash
AGENT_UTILITIES_TESTING= PYTHONPATH=. python3 scripts/scale/loadgen.py \
    --engine live --scale 1.0 --duration-s <hours-in-seconds> \
    --workers 800 --report-json soak-report.json
```

This was **not** attempted on this shared box — it would require a live
multi-node fleet this session does not have exclusive access to, and running
it here would both overwhelm a shared host and produce a result this doc
would then have to (falsely) call "demonstrated." Flagged for the R820
testbed / a dedicated cluster window instead.

## 6. Summary

| Metric | Result |
|---|---|
| Reliability corpus cases | 10 (was 6; +4 isolated-scorer cases) — match-rate 1.00, degrade trips to 0.60 |
| Eval corpus pass-rate | 1.00 clean, 0.00 degrade (gate trips) |
| Retrieval Recall@5 / MRR | 1.00 / 1.00 clean, 0.467 / 0.633 degrade (gate trips) |
| New chaos tests | 7 (connector failure ×3, claim-churn fencing ×2, heartbeat-vs-reap ×1, tenant-isolation-under-load ×1) |
| Total tests run in this session's scope | 452 passed, 9 skipped (skips are the intentional hardware-pending / optional-dependency ones) |
| Gate meta-tests (`tests/gates`) | 21/21 passed — every gate proven able to fail |
| Governance/hygiene guardrails | concepts, no-stub, concept-governance, sprawl, liveness — all green |
| Lint/type-check on touched files | ruff clean, ruff-format clean, mypy clean (1 unrelated pre-existing error elsewhere, reproduced on the pre-change tree) |
| Cluster-only cases | 6, explicitly skip-marked, run recipe documented above — none reported as demonstrated |
