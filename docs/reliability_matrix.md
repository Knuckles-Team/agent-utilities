# Reliability validation matrix

This matrix distinguishes deterministic regression coverage from production
certification. It does not carry stale pass counts and does not convert unavailable
infrastructure into skipped success.

| Reliability property | Deterministic source/test asset | Exact-release production proof |
|---|---|---|
| lease reclaim and stale-writer fencing | `tests/scale/soak/test_chaos_worker_and_delivery.py` | worker-process-loss fault |
| idempotent redelivery/no duplicate effect | worker/delivery and lifecycle tests | worker, broker and commit-phase faults |
| timeout, retry, backoff and dead letter | `test_chaos_lifecycle_and_dlq.py` | sustained load plus worker/broker loss |
| cancellation never falsely completes | lifecycle tests | remote analytics/process cancellation |
| noisy-neighbor tenant isolation | `test_chaos_tenant_and_restart.py` | zone loss and policy/deletion scenarios |
| durable restart and fencing | tenant/restart tests | node loss, Raft leader loss, regional recovery |
| bounded steady/burst backlog | `test_steady_burst.py` | scale=1 queue/lag/latency telemetry |
| cross-shard atomicity and recovery | Epistemic Graph MutationBatch/2PC tests | five commit kill points, reshard and restore |
| schema/protocol coexistence | compatibility structural gates | rolling binary/protocol upgrade |
| index and ontology migration | release catalog gates | online index and ontology upgrade |
| backup portability | format-v3 coordinator-aware restore code | backup/restore and cross-cell recovery |
| privacy-safe evidence | evidence schema/privacy gate | external signing and independent verification |

The production campaign lives in
`deploy/release/certification-campaign.yml`; the opt-in pytest entry is
`tests/scale/soak/test_production_certification.py`. It requires scale=1, 24–72 hours,
actual action/probe commands, aggregate Prometheus samples, exact signed artifacts and
an external evidence signer/verifier. These authorities are typed `AgentConfig`
settings: `CERT_HOOK_COMMANDS`, `CERT_FAULT_ACTION_COMMANDS`, and
`CERT_FAULT_PROBE_COMMANDS` are exact scenario-to-JSON-argv maps;
`CERT_LOAD_COMMAND`, `CERT_METRICS_COMMAND`, `CERT_EVIDENCE_SIGNER_COMMAND`, and
`CERT_EVIDENCE_VERIFIER_COMMAND` are JSON argv arrays. Activation, input, and capacity
fields are `CERTIFICATION_MODE`, `CERT_RELEASE_MANIFEST`, `CERT_ARTIFACTS_DIR`, and
`CERT_HARDWARE_CLASS`.

Only a bounded 24–72 hour duration extension is configurable in the campaign document;
all other campaign semantics must exactly match the packaged policy. Certification
requires both monotonic elapsed time and normalized load-report real duration to reach
the configured duration. Metric coverage cannot shorten the soak. The pass predicate
also requires the complete canonical metric set, including bounded gateway error ratio
and zero recent pod restarts.
Adapter stdout/stderr and streamed Prometheus response bodies have fixed byte ceilings;
boundary failures retain no response content or environment-specific location data.

Prometheus uses HTTPS `CERT_PROMETHEUS_URL` plus the dedicated
`CERT_PROMETHEUS_TLS_PROFILE` or `CERT_PROMETHEUS_TLS_PROFILE_REF` selector. Optional
bearer authentication is only `CERT_PROMETHEUS_BEARER_TOKEN_REF`, resolved from a
runtime secret authority; the token, a token file, and a token path are never persisted
as configuration. Run `agent-utilities-doctor --only production_certification` before
starting the campaign. The doctor cryptographically verifies the signed release against
the packaged compatibility matrix with signature verification enabled. Missing,
unsigned, wrong-matrix, or otherwise invalid prerequisites fail the selected run.

Passing evidence must bind the release/configuration/component digests, non-identifying
hardware class, action/observation/metric digests, invariant results, sample coverage,
actual elapsed duration, SLOs and observed RPO/RTO. No endpoint, filesystem location, host/user identity, source
content or credential is permitted in the evidence document.

See [Compatibility and Production Certification](release/compatibility-and-certification.md)
and [Backup, Restore and Cross-Cell Recovery](operations/disaster-recovery.md).
