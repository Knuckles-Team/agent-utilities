---
name: graph-runtime-and-governance
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

Use the intent verbs `ask`, `find`, `write`, `act`, `manage`, and `why` when a
small tool surface is appropriate. Use `find_tools` or `list_catalog` to discover
a capability, `load_tools` to expose only what the current task needs, and
`unload_tools` when finished. Pin an exact tool when ambiguity would be unsafe.

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
```

Keep coverage metadata out of `SKILL.md` frontmatter.

Use an economy model for health classification, trace filtering, and checklist
verification. Escalate security, policy, or ambiguous root-cause decisions.

## Guardrails

- Do not log credentials, tokens, raw secret values, or private unrelated data.
- Do not disable policy or audit controls to make a check pass.
- Do not treat a healthy process as proof that the user-visible operation works.
- Escalate when recovery needs new authority or an irreversible external action.
