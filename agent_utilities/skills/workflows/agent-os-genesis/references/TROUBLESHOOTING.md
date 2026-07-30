# Troubleshooting

Start from the failed exit gate and preserve the trace/evidence bundle.

| Symptom | Check first | Safe response |
|---|---|---|
| Render fails | values schema, unresolved references, API versions | correct the plan; do not bypass validation |
| Cluster apply is forbidden | authority mode and exact RBAC denial | emit an administrator requirement or remove the cluster-scoped capability |
| Workload is pending | requests, quotas, taints, architecture, PVC binding | adjust the declared placement/resources or provide capacity |
| Readiness fails | application health, engine reachability, identity/secrets | inspect correlated logs/traces; do not restart-loop a dependency failure |
| Engine will not start | data format, lock owner, disk space/IO, encryption reference | stop extra writers, restore compatible state, or run the supported migration |
| Connector returns 401/403 | issuer/audience, service token, target policy, clock | repair identity routing or authorization; never disable auth as a test |
| External route fails | DNS, certificate names, ingress/Gateway backend, network policy | test each hop and keep the prior route available |
| Background work starves users | foreground priority, IO/model/GPU limits, checkpoints | pause/checkpoint lower lanes and verify foreground latency |
| Second apply changes resources | nondeterministic render/defaults or manual drift | identify the owner; reconcile through the declared GitOps/runtime path |
| Local model answers without a tool | tool allow-list, exact server/tool binding, prompt, model capability | inspect the run trace; do not count it as delegated execution |

For runtime-only Graph-OS incidents after a healthy deployment, use the
`graph-runtime-and-governance` or `epistemic-graph-troubleshooting` skill. Use Genesis
only when the fix changes the substrate contract.

Always retain: plan digest, source revision, rendered artifacts, runtime events,
redacted logs/traces, exact failing gate, remediation, and rollback result.
