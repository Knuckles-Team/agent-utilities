# Adjudication packet — AU-OS.deployment

27 live concepts. The deterministic pass already decided 1 pointer(s) and 2 retirement(s) from module locality, git archaeology and id shape alone. Confirm or correct the items below, then write the decisions into .specify/triage/AU-OS.deployment.yaml.

## Clusters — confirm ONE parent each; the members inherit it

### connector-coverage-check  (3 concepts)
    agent_utilities/deployment/doctor.py:3583 | """Assert every configured connector is ingesting + fresh (CONCEPT·AU-OS.deployment.connector-coverage-check).
    agent_utilities/knowledge_graph/ingestion/connector_coverage.py:4 | """Per-connector coverage + freshness assessment (CONCEPT·AU-OS.deployment.connector-coverage-check).
    members: connector-coverage-check, os-4, report-resolved-mode

## Proposed RETIRE — the id names nothing (confirm or rescue) (2)

### os-2
    why: the id is a bare legacy pillar reference (os-2) — a citation of the old KG-N.NN numbering, not a name anyone chose
    scripts/scaffold_graph_action.py:2 | """Scaffold a new graph_* tool action — kill the 6-file wiring drudgery (CONCEPT·AU-OS.deployment.os-2).
    scripts/validate_change.py:2 | """Diff-scoped change validator — the agent's fast inner loop (CONCEPT·AU-OS.deployment.os-2).

### os-4
    why: the id is a bare legacy pillar reference (os-4) — a citation of the old KG-N.NN numbering, not a name anyone chose
    agent_utilities/deployment/doctor.py:3649 | """Validate the ``workspace.yml`` repository manifest (CONCEPT·AU-OS.deployment.os-4).
    docs/guides/workspace-config.md:110 | The `agent-utilities-doctor` `workspace_config` check (CONCEPT·AU-OS.deployment.os-4) validates

## UNDECIDED — the cheap signals ran out here (5)

### blueprint-library
    why: the marker exists only in prose/doc files — nothing in the shipped tree realises it, which is usually a retirement but occasionally a real decision recorded only in prose
    docs/journey.md:159 | Finally, when deploying real software services or provisioning databases, the **Company Infrastructure Orchestration layer** (`CONCEPT·AU-OS.deployment.infra-orchestration`) pulls

### cold-boot-import-reentrancy
    why: the marker exists only in test files — nothing in the shipped tree realises it, which is usually a retirement but occasionally a real decision recorded only in prose
    .specify/design/cold-boot-import-reentrancy/design.md:29 | - **Proposed ID**: `CONCEPT·AU-OS.deployment.cold-boot-import-reentrancy`
    .specify/design/cold-boot-import-reentrancy/design.md:3 | CONCEPT·AU-OS.deployment.cold-boot-import-reentrancy

### infra-orchestration
    why: the marker exists only in prose/doc files — nothing in the shipped tree realises it, which is usually a retirement but occasionally a real decision recorded only in prose
    docs/journey.md:159 | Finally, when deploying real software services or provisioning databases, the **Company Infrastructure Orchestration layer** (`CONCEPT·AU-OS.deployment.infra-orchestration`) pulls

### platform-journey
    why: the marker exists only in prose/doc files — nothing in the shipped tree realises it, which is usually a retirement but occasionally a real decision recorded only in prose
    docs/journey.md:44 | To scale under load, the engine relies on its **Massive Scale Architecture & Sandbox** (`CONCEPT·AU-OS.host.homeostatic-recovery-daemon`). Using the **Distributed Replay & Complian

### standard-repo-templates
    why: the marker text is truncated by the grammar ('(`CONCEPT·AU-OS.deployment.standard-repo-templates/5.75`). → [Standard Private Repos + CI]') — the id itself reads like a real name, so the marker text needs cleaning either way; decide whether the concept survives that cleanup
    agent_utilities/deployment/repo_templates.py:534 | """Compact, generator-friendly view for genesis.yaml (CONCEPT·AU-OS.deployment.standard-repo-templates)."""
    scripts/gen_genesis_manifest.py:36 | # provisions per profile (CONCEPT·AU-OS.deployment.standard-repo-templates / OS-5.75). Sourced from

## Proposed OWN DOCUMENT — is this really a decision? (19)

### agent-factory-autoload
    why: a singleton: no sibling shares its source footprint or introducing commit (6 source file(s), 14 marker site(s))
    agent_utilities/cli/__init__.py:382 | """Install the agent-utilities skill toolkit into agent tool(s) (CONCEPT·AU-OS.deployment.agent-factory-autoload).
    scripts/retrofit_fleet_contribution.py:4 | CONCEPT·AU-OS.deployment.agent-factory-autoload / ORCH-1.80. Idempotently brings one ``agents/<pkg>`` repo up to

### airgap-mode
    why: a singleton: no sibling shares its source footprint or introducing commit (4 source file(s), 5 marker site(s))
    .env.example:95 | # [OPTIONAL] Sovereign/air-gap deployment gate (CONCEPT·AU-OS.deployment.airgap-mode).
    agent_utilities/core/model_factory.py:605 | # CONCEPT·AU-OS.deployment.airgap-mode — create_async_http_client applies the

### concept-2
    why: a singleton: no sibling shares its source footprint or introducing commit (2 source file(s), 6 marker site(s))
    agent_utilities/deployment/repo_templates.py:205 | # ── generalized, reusable GitLab CI templates (CONCEPT·AU-OS.deployment.concept-2) ─────────────
    agent_utilities/deployment/repo_templates.py:550 | "step": "agent-utilities-deployment enterprise workflow (CONCEPT·AU-OS.deployment.concept-2)",

### connector-coverage-check
    why: the head of a 3-concept cluster — the siblings are proposed as pointers at this one, so this is the decision that has to be written down
    agent_utilities/deployment/doctor.py:3583 | """Assert every configured connector is ingesting + fresh (CONCEPT·AU-OS.deployment.connector-coverage-check).
    agent_utilities/knowledge_graph/ingestion/connector_coverage.py:4 | """Per-connector coverage + freshness assessment (CONCEPT·AU-OS.deployment.connector-coverage-check).

### cross-platform-locks-plus
    why: a singleton: no sibling shares its source footprint or introducing commit (3 source file(s), 3 marker site(s))
    agent_utilities/knowledge_graph/core/file_lock.py:3 | CONCEPT·AU-OS.deployment.cross-platform-locks-plus — Cross-platform locks plus per-OS process spawn, endpoints and doctor hints.
    agent_utilities/knowledge_graph/core/shard_topology.py:50 | """The per-platform default local endpoint (CONCEPT·AU-OS.deployment.cross-platform-locks-plus).

### dynamic-two-fail-closed
    why: a singleton: no sibling shares its source footprint or introducing commit (5 source file(s), 6 marker site(s))
    agent_utilities/claude_harness/pretooluse_gate.py:6 | CONCEPT·AU-OS.deployment.dynamic-two-fail-closed — Dynamic two-layer fail-closed PreToolUse ActionPolicy permission gate
    agent_utilities/cli/__init__.py:85 | # CONCEPT·AU-OS.deployment.dynamic-two-fail-closed — the PreToolUse dynamic gate body (reads the event on stdin).

### embedded-auto-provision
    why: a singleton: no sibling shares its source footprint or introducing commit (2 source file(s), 3 marker site(s))
    agent_utilities/knowledge_graph/core/graph_compute.py:1159 | CONCEPT·AU-OS.deployment.embedded-auto-provision — embedded auto-provision. For a server that has no remote
    agent_utilities/knowledge_graph/core/graph_compute.py:1699 | ``coupled`` (CONCEPT·AU-OS.deployment.embedded-auto-provision — embedded auto-provision) selects the child

### engine-resolver-auto-provision
    why: a singleton: no sibling shares its source footprint or introducing commit (7 source file(s), 35 marker site(s))
    agent_utilities/knowledge_graph/core/engine_resolver.py:1 | # CONCEPT·AU-OS.deployment.engine-resolver-auto-provision - One engine resolver auto-provisioning every entrypoint by precedence remote then share-running-local then autostart-shar
    scripts/gen_genesis_manifest.py:330 | # The ONE multi-model store at every scale (CONCEPT·AU-OS.deployment.engine-resolver-auto-provision resolver). It is NOT

### flagging-repos
    why: a singleton: no sibling shares its source footprint or introducing commit (3 source file(s), 4 marker site(s))
    agent_utilities/deployment/doctor.py:3491 | """Assert the agent-packages repos are ingested + fresh (CONCEPT·AU-OS.deployment.flagging-repos).
    agent_utilities/knowledge_graph/ingestion/manifest.py:191 | doctor check to enforce a freshness SLA (CONCEPT·AU-OS.deployment.flagging-repos) — flagging repos

### fleet-lifecycle-control
    why: a singleton: no sibling shares its source footprint or introducing commit (10 source file(s), 22 marker site(s))
    agent_utilities/knowledge_graph/research/change_publisher.py:31 | (CONCEPT·AU-OS.deployment.fleet-lifecycle-control) under the reserved ``merge_promotion`` action kind before any
    agent_utilities/knowledge_graph/research/auto_merge.py:139 | #: Operational ActionPolicy verdict on the promotion itself (CONCEPT·AU-OS.deployment.fleet-lifecycle-control,

### governance-derived-claude-code
    why: a singleton: no sibling shares its source footprint or introducing commit (6 source file(s), 10 marker site(s))
    agent_utilities/deployment/cli.py:53 | help="Write a governance-derived Claude Code permission fence (CONCEPT·AU-OS.deployment.governance-derived-claude-code).",
    agent_utilities/claude_harness/claude_fence.py:6 | CONCEPT·AU-OS.deployment.governance-derived-claude-code — Governance-derived Claude Code permission-fence generator

### infrastructure-blueprint-library
    why: a singleton: no sibling shares its source footprint or introducing commit (2 source file(s), 2 marker site(s))
    agent_utilities/models/knowledge_graph.py:390 | # Company Infrastructure (CONCEPT·AU-ECO.ui.company-infrastructure-orchestration, CONCEPT·AU-OS.deployment.infrastructure-blueprint-library)
    agent_utilities/models/company.py:320 | CONCEPT·AU-OS.deployment.infrastructure-blueprint-library — Infrastructure Blueprint Library.

### liveness-vs-readiness-split
    why: a singleton: no sibling shares its source footprint or introducing commit (4 source file(s), 7 marker site(s))
    agent_utilities/mcp/kg_server.py:3811 | # Unauthenticated liveness + readiness for HTTP deployments (CONCEPT·AU-OS.deployment.liveness-vs-readiness-split).
    agent_utilities/observability/runtime_health.py:44 | Liveness vs readiness (CONCEPT·AU-OS.deployment.liveness-vs-readiness-split — see the callers):

### merge-triggered-venv-flip
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 5 marker site(s))
    agent_utilities/deployment/venv_autosync.py:392 | # CONCEPT·AU-OS.deployment.merge-triggered-venv-flip
    agent_utilities/deployment/venv_autosync.py:3 | CONCEPT·AU-OS.deployment.merge-triggered-venv-flip

### unified-install-tree
    why: a singleton: no sibling shares its source footprint or introducing commit (3 source file(s), 3 marker site(s))
    scripts/check_ontology.py:253 | Reuses the federation read-path resolver (XDG-first, CONCEPT·AU-OS.deployment.unified-install-tree) so the gate
    agent_utilities/knowledge_graph/core/ontology_federation.py:147 | """XDG-first provider-ontology resolution (CONCEPT·AU-OS.deployment.unified-install-tree).

### universal-outbound-credentialprovider
    why: a singleton: no sibling shares its source footprint or introducing commit (3 source file(s), 3 marker site(s))
    agent_utilities/security/__init__.py:68 | # credential_provider (CONCEPT·AU-OS.deployment.universal-outbound-credentialprovider) + source_credentials (CONCEPT·AU-OS.config.source-credential-registry)
    agent_utilities/security/source_credentials.py:8 | The companion to :mod:`credential_provider` (CONCEPT·AU-OS.deployment.universal-outbound-credentialprovider). Where the

### vault-first-routine-genesis
    why: a singleton: no sibling shares its source footprint or introducing commit (3 source file(s), 7 marker site(s))
    agent_utilities/deployment/doctor.py:1810 | # an operator on Okta isn't told they need Keycloak (CONCEPT·AU-OS.deployment.vault-first-routine-genesis genesis
    agent_utilities/mcp/kg_server.py:2034 | """REST twin of graph_configure action=vault_sync (CONCEPT·AU-OS.deployment.vault-first-routine-genesis)."""

### vault-seed-service
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 1 marker site(s))
    agent_utilities/mcp/tools/analysis_tools.py:2655 | # CONCEPT·AU-OS.deployment.vault-seed-service — read-existing + seed a service's secrets.

### workspace-venv-reconciler
    why: a singleton: no sibling shares its source footprint or introducing commit (3 source file(s), 9 marker site(s))
    agent_utilities/ecosystem/hook_installer.py:57 | # CONCEPT·AU-OS.deployment.workspace-venv-reconciler (D-VS-6) — the one piece of
    agent_utilities/deployment/venv_sync.py:2322 | CONCEPT·AU-OS.deployment.workspace-venv-reconciler (D-VS-6). ``detect_drift``
