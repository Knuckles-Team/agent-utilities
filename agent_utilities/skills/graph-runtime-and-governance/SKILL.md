---
name: graph-runtime-and-governance
skill_type: skill
description: >-
  Configure, observe, audit, secure, and troubleshoot an already-installed
  Graph-OS runtime. Use for health checks, incidents, compliance, sessions,
  traces, cache or secret operations, tool discovery and loading, intent routing,
  policy verification, coverage checks, or recovery within the deployed
  topology. For provisioning, profile changes, topology changes, connector
  rollout, or upgrades, use agent-utilities-deployment.
---

# Graph runtime and governance

Diagnose from observed state, preserve governance boundaries, and prove recovery
with the same public path users depend on.

Keep installation profile, manifest, topology, connector rollout, and upgrade
work in `agent-utilities-deployment`.

## Action reference

| Tool | Actions | Notes |
|---|---|---|
| `graph_config` | `describe`/`get`/`diff`/`reload` (read-only), `set` (ActionPolicy approval, provenance, awaited catalog reconciliation) | Sole generic `AgentConfig` mutation authority |
| `graph_configure` | `register_mcp`, `set_secret`/`vault_sync`, `add_connection`/`remove_connection`/`list_connections` (named external graph backends), `schema_pack`/`schema_candidates`, `harness_fence`, `install_hooks`, `get_config`/`list_config`, `system_doctor`/`config_doctor`/`preflight`, `setup_databases`/`verify_databases`, `generate_config` | Legacy generic config actions are read-only; use `graph_config action=set` for writes |
| `graph_governance` | `grant_approval` (atomically decide a pending ActionApproval), `submit_risk_veto` (record a governed veto against a target), `verify_action` (evaluate an ActionPolicy request without writing a decision), `ownership_report` (W2.8 read-only per-graph RBAC-ownership disposition pass over the live engine catalog), `ownership_apply` (preview — always dry-run — the UNAMBIGUOUS grant plan + its rollback for that report), `policy_status` (live PermissionPolicy/context_policy + KG-native audit-sink configuration, no engine required) | the live mutating ownership grant apply itself is a SEPARATE, human-gated CLI (`scripts/apply_graph_ownership_grants.py --apply-unambiguous`, HG-4) — never reachable from MCP/REST |
| `graph_audit` | `verify` (cryptographically walk the default graph's audit chain; returns `first_broken_seq` if tampered), `for_target` (every `:ToolCall` that acted on a `target_id`, in call order) | wraps the engine's mature hash-chained audit log (redb `AUDIT` table, SHA-256); degrades cleanly with `available: false` on a build without the `security` feature or a durable persist dir |
| `graph_compliance` | `posture` (joins `graph_audit.verify()` with a node-count/status rollup of ingested governance labels — Control/Policy/Risk/ComplianceGate/…), `export` (bulk-export a `disclosure_level`-redacted subgraph for `node_ids` or a `cypher` id-selector) | pure aggregation over primitives that already exist, no new compliance/redaction logic; `posture`'s counts reflect whatever CISO Assistant/Egeria/TRM connectors have already ingested |
| `graph_incident` | `correlate` (`window_s`/`days`; idempotent — an already-open incident with the same signature is deduped, not re-written), `list` (`status`-filtered, newest first), `get` (`incident_id`) | read-only — proposes/executes NO remediation; groups recent `:HealthAnomaly` rows across hardware/os/orchestration/service/network producers on the same host/window into one `:Incident`, estimating the deepest contributing layer as likely root cause |
| `graph_observe` | `trace_rootcause` (failed assertions + low scores joined to their trace's agent, `query`=agent/capability filter), `prompt_regression` (mean score per prompt version — which regressed), `failure_cluster` (failing traces clustered by failed assertion — systemic breaks across agents) | queries the KG-native trace/score subgraph — joins an opaque trace store can't do |
| `graph_secret` | `set` (governed by ActionPolicy `secret.set`), `get`, `list` (key names only, never values), `delete` (governed by `secret.delete`) | manages the `__secrets__` store — values sealed by encryption-at-rest, key names/metadata stay queryable; the enterprise OpenBao/Vault backend is used transparently when configured |
| `graph_sessions` | `list`, `get`, `delete`, `reply` (`user_reply` to a `session_id`), `cancel` | durable session management |
| `usage_query` | `summary`, `by_model`/`by_project`/`by_agent`, `tools`, `activity`, `sessions`/`session_detail`/`top_sessions`, `search`, `traces`, `series` (filter by date/project/agent/model) | usage/cost observability |
| `ingest_sessions` | `collect` (auto-detect local agents), `upload` (pre-parsed `bundles_json` to a remote engine), `paths` (explicit files) | agent chat/session-log ingestion |
| `graph_kvcache` | `get` (base64 block bytes or miss), `put`, `contains`/`exists`, `stats` (occupancy + dedup counters) | content-addressed shared KV-cache; every transport error degrades to a cache miss, never raises |
| `graph_kv_checkpoint` | `create` (checkpoint a KV-cache at an explicit run/point, keyed by model identity/quantization/serving-engine/version/prefix-digest/tenant/policy-version), `instantiate_agent` (fail-closed load + `initializedFrom` lineage edge from a new agent run), `restore_conversation` (fail-closed load + `restoredFrom` lineage edge; `allow_cold_start` for an explicit, traced cold-start fallback) | heavy KV blob stays behind the engine's content-addressed blob store, never inlined into the response; a tenant mismatch or stale policy version is ALWAYS refused |
| `graph_traces` | `search` (filter by `service`/`operation`/free-form `query`, capped by `limit`), `get` (single `trace_id`) | degrades cleanly with no trace surface |

### Argument evaluation (AIF)

`graph_argument` represents and evaluates structured arguments in the Argument
Interchange Format (AIFdb/arg-tech.org): I-nodes (claims — each one IS a `:Belief`)
linked through S-nodes — RA (inference), CA (conflict), PA (preference) — plus the
AIF+ TA (transition)/YA (illocutionary) dialogue extensions. Actions: `import_aif`
(validates node arity first — an RA/CA needs ≥1 premise + exactly 1 conclusion, a PA
needs ≥2 premises + exactly 1 conclusion — then writes through the shared
ChangeEnvelope path; an RA/CA-node also mints the derived `SUPPORTS`/`ATTACKS` edge),
`export_aif` (best-effort, degrades to an empty map rather than raising),
`evaluate` (projects the CA/PA structure to Dung form and hands the I-node ids to the
REAL engine argumentation solver — `eg-epistemic`'s `Method::ResolveConflict`,
grounded/preferred/stable semantics; no second solver is implemented here), and
`add_scheme` (register a new named AIF Scheme template, e.g. a Waltonian scheme).
`evaluate` requires the opt-in `epistemic-tms` engine feature. Honest limits:
RA-node support is written as a `SUPPORTS` edge for provenance, but classical Dung
semantics are attack-only — support doesn't feed the grounded/preferred/stable
computation itself (it DOES feed the always-on confidence propagation the epistemic
answer layer reads); PA-node preference filtering only ever discounts one side of a
MUTUAL (symmetric) CA-conflict; import is fail-loud, export is fail-soft.

## Workflow

### 1. Establish scope and safety

- Identify the affected tenant, graph, session, service, and time window.
- Separate read-only diagnosis from mutating remediation.
- Capture the current state before changing configuration or runtime data.
- Never request or display a secret value; work with secret identifiers and
  health status only.

Use the skill directly for a bounded health, trace, audit, or configuration
check. Delegate a multi-layer incident only when independent diagnosis and
dependency-ordered remediation will improve recovery.

### 2. Inspect the runtime

| Need | Primary operation |
|---|---|
| Configuration and system doctor | `graph_configure` |
| Health, metrics, or runtime state | `graph_observe` |
| Trace and session history | `graph_traces`, `graph_sessions`, `usage_query` |
| Incident workflow | `graph_incident` |
| Audit or compliance evidence | `graph_audit`, `graph_compliance` |
| Approval, veto, and policy verification | `graph_governance` |
| Cache operations | `graph_kvcache` |
| Secret metadata | `graph_secret` |

Trace a failure from the user-visible symptom to the first failing tool call,
then to its runtime dependency. Prefer evidence from the existing doctor,
metrics, traces, and audit surfaces over speculative configuration changes.

### 3. Manage tool visibility responsibly

After the GraphOS operation-API cutover, use the six resident intent verbs
`find`, `ask`, `why`, `write`, `act`, and `manage`. Their common input is an
exact `op` or natural-language `intent`, object-valued `params`, optional
`plan_ref` and `idempotency_key`, and `execute`. Discover a schema with
`find(op="<operation.id>")`, then use an exact operation ID for a mutation.
Natural-language mutation requests return a preview; destructive operations
need a plan, while approval and administration need human console confirmation
with fresh MFA. The server enforces the same authority checks on every surface.

The four other resident tools are `find_tools`, `load_tools`, `unload_tools`, and
`multiplexer_status`. `find_tools(browse=true)` replaces `list_catalog()`.
Loading an item exposes it to the current MCP session subject to exact scopes
and Eunomia policy; it does not confer authority. A loaded child tool also has
an `act(op="fleet.call", params={...})` fallback if the client misses
`list_changed`. Use `multiplexer_status` to inspect loaded items and delivery.

The action reference above records legacy `graph_*` entry points. Treat them
as migration clues, not operation IDs: resolve each behavior in the generated
GraphOS registry before invoking it. Until the cutover is served, inspect the
running server's actual tool list and use its current contract.

When onboarding a child MCP server on the pre-cutover surface, update the
`mcp_config*.json` entry, generated README example, and live registry together.
Verify with the currently served `list_catalog` or `find_tools` and
`multiplexer_status`, then load the specific discovered items. On the new
surface, use `find_tools(browse=true)` or resolve the corresponding `fleet.*`
operation in `find` first.

### 4. Remediate minimally

For diagnosis, audit, or review-only work, report the observed cause, supporting
evidence, and remediation plan, then stop before changing configuration or state.

- Change one causal condition at a time.
- Preserve tenant, role, approval, and audit context.
- Prefer idempotent recovery actions.
- Keep a rollback path for configuration and state changes.

### 5. Verify recovery

- Re-run the original failing operation through its normal entry point.
- Confirm health, traces, and audit records agree.
- Check that the fix did not weaken policy or expose additional tools.
- Record the root cause, evidence, action, and remaining risk.

## Coverage governance

Every registered Graph-OS verb must appear in exactly one retained domain
skill's `agents/graph-os.yaml` sidecar. Run the coverage command after changing
the tool surface or skill taxonomy:

```text
python -m agent_utilities.mcp.skill_coverage
pytest tests/unit/test_gateway_mcp_parity.py -q      # tool <-> REST-route <-> skill legs
```

This is the third parity leg (tool⇄REST-route are the other two): every verb in
`kg_server.REGISTERED_TOOLS` (minus a small, justified exemption list) is wrapped by a
domain skill's `agents/graph-os.yaml`, and no domain skill's coverage points at a dead
verb. A verb with no covering skill is **uncovered**; a coverage entry pointing at a
non-existent verb is an **orphan** — fix both before merge, or add the verb to the
exemption list with a written justification.

Keep coverage metadata out of `SKILL.md` frontmatter.

Use an economy model for health classification, trace filtering, and checklist
verification. Escalate security, policy, or ambiguous root-cause decisions.

## Guardrails

- Do not log credentials, tokens, raw secret values, or private unrelated data.
- Do not disable policy or audit controls to make a check pass.
- Do not treat a healthy process as proof that the user-visible operation works.
- Escalate when recovery needs new authority or an irreversible external action.
