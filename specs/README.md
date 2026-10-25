# Meristem specifications

These specs follow the ecosystem [spec standard](https://github.com/Knuckles-Team/pipelines/blob/main/reference/spec-standard.md): one owning spec per capability, extend an existing spec before creating one, audits and reviews land in the owning spec, one ID and title form, one file set. The `spec-standard` gate enforces it.

This tracked `specs/` directory is the public build contract for **AU** owned work. Every
spec must contain the behavior, architecture, interfaces, tests, quality gates, and acceptance
criteria needed to implement it from this repository. External program notes may inform a draft,
but no private document or local workspace path is required to build or verify a public spec.
Delivery status is recorded here with exact merged revision and test evidence.

## Structure

Create `specs/<stable-id>/` with `spec.md` (user outcome, requirements, acceptance), `plan.md`
(architecture, reuse, interfaces, live wiring, decisions), `test-spec.md` (positive, negative,
integration, quality and release proof), `tasks.md` (ordered implementation and verification),
`requirements.md` (the definition of every requirement ID this spec owns, with its closing proof),
and `status.json` (machine-readable delivery, acceptance, and public receipts, including a
`requirements` array with one entry per requirement ID carrying its own `delivery_state` and
evidence).
Start from [`_template/`](_template/). Keep status and evidence explicit; a planned or tested item
is not a landed item. Put durable evidence links in the spec directory, never local scratch output.
This follows a specify/plan/tasks flow with an explicit test contract.

Each spec must name its stable ID and sole owner. Cross-repository work links the other owners'
public specs by stable ID and GitHub URL; each repository documents the complete contracts and
acceptance it owns. Designs must inventory existing code and reuse the legal
owner and live wiring before proposing new components. Include CCCC, jscpd, dupehound, KISS,
language-native and repo release gates where applicable, with exact pass evidence and a
no-duplicate-authority check.

## States

States are `SPECIFIED`, `LANDED`, and `VERIFIED`. The generator sets them. See the [spec standard](https://github.com/Knuckles-Team/pipelines/blob/main/reference/spec-standard.md#states).

## Arbora owner map

| Prefix | Repository specs | Responsibility |
|---|---|---|
| `EG` | [epistemic-graph](https://github.com/Knuckles-Team/epistemic-graph/tree/main/specs) | Rust database, graph compute, ontology, governed ingestion, durable records and clients |
| `SDK` | [agent-connector-sdk](https://github.com/Knuckles-Team/agent-connector-sdk/tree/main/specs) | connector control, transport, manifests, certification, governed effects |
| `AU` | [Meristem](https://github.com/Knuckles-Team/agent-utilities/tree/main/specs) | agent orchestration and control workflows |
| `ARBORA` | [arbora](https://github.com/Knuckles-Team/arbora/tree/main/specs) | serving composition, fleet, gateway, A2A, deployment operations |
| `WEBUI` | [agent-webui](https://github.com/Knuckles-Team/agent-webui/tree/main/specs) | browser presentation and interaction |
| `RM` | [repository-manager](https://github.com/Knuckles-Team/repository-manager/tree/main/specs) | repository discovery, worktrees, source control and execution |

## Contributions

Read this repository's `AGENTS.md` and any contribution guide. Propose `spec.md` first, resolve
architecture and test details in `plan.md` and `test-spec.md`, then implement `tasks.md` in a
dedicated branch or worktree. Link the PR to spec IDs and update evidence and status only when the
corresponding gates actually pass. Use the [universal-skills spec-generator](https://github.com/Knuckles-Team/universal-skills/tree/main/universal_skills/development/spec-generator),
[spec-verifier](https://github.com/Knuckles-Team/universal-skills/tree/main/universal_skills/development/spec-verifier),
and [task-planner](https://github.com/Knuckles-Team/universal-skills/tree/main/universal_skills/development/task-planner)
and the [arbora-development](https://github.com/Knuckles-Team/arbora/blob/main/arbora/skills/arbora-development/SKILL.md)
bootstrap skill, together with the [SDD full lifecycle](https://github.com/Knuckles-Team/universal-skills/tree/main/universal_skills/development-workflows/sdd-full-lifecycle) workflow.

## Local specifications

- [`AU-CONTEXT`](agent-context-and-finance/spec.md) — agent context and finance orchestration.
- [`AU-CONTROL`](agent-control-plane/spec.md) — agent control plane and workflows.
- [`AU-SEC`](agent-security-policy/spec.md) — agent security and policy boundary.
- [`AU-BOUNDARY`](au-boundary-deconstruction/spec.md) — move non-agent ownership to the right repository.
- [`AU-QUAL`](au-boundary-quality/spec.md) — quality and acceptance of the AU boundary.
- [`AU-DEV`](au-developer-environment/spec.md) — developer environment and release setup.
- [`AU-INTEGRATION`](au-integration-reliability/spec.md) — reliability of live integrations.
- [`AU-SEMANTIC`](au-semantic-client/spec.md) — semantic client behavior.
- [`AU-HARNESS`](harness-evolution/spec.md) — harness improvement and evaluation.
- [`AU-RETIRE`](au-engine-duplicate-retirement/spec.md) — retire duplicate graph authority after per-operation parity.
- [`AU-RETRIEVAL`](retrieval-evaluation/spec.md) — public, reproducible structure-aware retrieval evaluation.
- [`AU-FREEZE`](release-evidence-freeze/spec.md) — exact-revision AU source and release evidence freeze.
- [`AU-BASELINE`](baseline-ingestion/spec.md) — baseline ingestion after boot and served ontology content.
- [`AU-HARNESS-CYCLE`](harness-cycle/spec.md) — harness evolution cycle: propose, probe, compose, persist, reflect.
- [`AU-SCM`](au-scm/spec.md) — development loop on changesets for the in-run coding harness.
