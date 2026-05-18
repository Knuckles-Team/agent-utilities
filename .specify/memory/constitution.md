# Project Constitution - agent-utilities

## Vision & Mission
**agent-utilities** is a protocol-first, framework-light agent core library. The mission is to provide production-grade agent orchestration with resilience and observability, bridging long-term agent memory with deep structural codebase awareness and cross-domain research knowledge.

## Core Principles
### Guiding Principles
- **Agents are protocol-native**: Agents communicate via open standards (ACP, A2A, MCP) not proprietary APIs.
- **Protocol logic is isolated**: Protocol adapters are separate from agent business logic.
- **Transport-agnostic**: Agents work over any transport (SSE, HTTP, stdio, WebRTC).
- **No framework lock-in**: Avoid opinionated orchestration frameworks like LangChain chains.
- **Explicit state over implicit context**: State is explicit and managed, not hidden in global variables.
- **Tools and transports are pluggable**: Any tool or transport can be swapped without changing agent code.
- **UI-agnostic**: No assumptions about user interface (terminal, web, mobile, voice).
- **JSON Prompting (Prompts-as-Code)**: Favor structured JSON blueprints over free-form Markdown for high-fidelity task specification.

### Normative Statements
- You MUST use Pydantic AI (`pydantic-ai-slim>=1.83.0,<1.84.0`) and Pydantic Graph (`pydantic-graph>=1.83.0,<1.84.0`).
- You MUST rely on the Unified Intelligence Graph for agent state and specialist discovery.
- You MUST ensure backwards compatibility and structural integrity of the `AGENTS.md` and `MEMORY.md` memory system.
- All code MUST be Python 3.11+ compliant and explicitly type-hinted.

## Governance
- **Unified Registry**: The Knowledge Graph acts as the unified registry for project governance (Policies) and operational workflows (Process Flows).
- **Policies**: Declarative constraints and guardrails (e.g., "Always use TDD") are grounded in Knowledge Base topics and applied based on the current context.
- **Process Flows**: Procedural step-by-step execution guides are retrieved from the KG and enforced.
- **Constitution Updates**: Changes to this constitution MUST be approved via pull request and validated to ensure no conflicts with existing policies.

## Quality Gates
- **Testing**:
  - All features MUST be implemented with corresponding **Pytests**.
  - All tests MUST complete within 60 seconds (strictly enforced via `pytest-timeout`). Tests that sleep or hang indefinitely violate CI/CD stability and will fail.
  - Integration tests pass for the Unified Intelligence Graph and all supported backends.
- **Verification Loop**:
  - After any code change, `pre-commit run --all-files` MUST be executed to verify integrity.
  - If issues are introduced, the implementation plan MUST be updated to address them, and the process repeated until all checks pass.
- **Prohibited Uses**:
  - Do NOT use agent-utilities for UI development (use agent-webui or agent-terminal-ui).
  - Do NOT use agent-utilities for SaaS-specific integrations (build MCP servers instead).

## Tech Stack & Standards
- **Language**: Python 3.11+
- **Core Frameworks**: Pydantic AI (`pydantic-ai-slim`), Pydantic Graph (`pydantic-graph`)
- **Key Tooling**: `requests`, `pydantic` (`>=2.13`), `pyyaml`, `python-dotenv`, `fastapi`, `httpx`
- **Graph Backends**: LadybugDB (default), FalkorDB, Neo4j
- **Document Backends**: SQLiteMemory (default), SQLite, PostgreSQL, MongoDB

If you append to the Knowledge Graph schema, you should always consider adding an OWL layer if it would be beneficial to do so.

We should always look at the existing ontology and try to notice some existing types that are in the schema definition but NOT yet in the OWL ontology. We want to highly consider adding those to OWL ontology.

## Concept Governance — Extend Before Invent

New functionality MUST first be expressed as an extension, augmentation, or composition
of an existing pillar/concept before a new CONCEPT: tag or domain is introduced.
The Knowledge Graph is the arbiter.

### Pre-Feature Gate
1. Query the KG for the 5 nearest semantic matches to the proposed feature.
2. If any match has similarity ≥ 0.7, the feature MUST extend that concept.
3. If no match, submit a New Concept Proposal with:
   - Target pillar assignment (ORCH / KG / AHE / ECO / OS)
   - C4 integration diagram showing how it wires into the pillar topology
   - 15-phase pipeline wiring point
4. New CONCEPT: tags require explicit approval in the design document.

### CI Enforcement
A GitHub Actions workflow validates that:
- No new `CONCEPT:` tags appear without a corresponding `.specify/design/` document.
- All new concepts reference an existing pillar.
- The KG graph integrity check (phase 15) passes.

## Development Pipeline — DSTDD (Design-Spec-Test Driven Development)

All features follow the DSTDD lifecycle:

1. **Design Phase** (first): Analyze the Knowledge Graph for nearest concepts,
   determine extension strategy, create C4 context diagram, assess risk.
   Artifacts: `.specify/design/<feature>/design.md`

2. **Spec Phase**: Decompose into user stories with acceptance criteria.
   Specs MUST reference the design document and pass the pre-flight checklist.
   Artifacts: `.specify/specs/<feature>/spec.md`

3. **Test Phase**: Generate TDD tests + validate against KG integrity (phase 15).
   Artifacts: `.specify/specs/<feature>/tasks.md`

## Post-Modification Artifact Mandate

After ANY code modification (whether driven by SDD, comparative analysis, or manual changes),
the following artifacts MUST be reviewed and updated as appropriate:

1. **`/docs`** — Update or create relevant documentation pages for changed functionality
2. **`AGENTS.md`** — Update agent capability descriptions, tool listings, and architecture notes
3. **`CHANGELOG.md`** — Add entry under the appropriate version section (Unreleased if pre-release)
4. **`README.md`** — Update feature lists, architecture descriptions, and usage examples
5. **`.specify/`** — Sync specs, tasks, and design docs; dual-write to KG via `kg_ingest`
6. **`.specify/reports/`** — Generate/update C4 architecture diagrams for changed components
7. **Pytests** — Add or update tests for ALL modified or new functionality

### Enforcement

- The `kg_inspect(view='constitution')` MCP tool MUST be consulted before SDD task execution
- These rules are persisted as `Policy` nodes in the KG with `enforcement: MANDATORY`
- SDD implementer MUST cross-check its generated plan against all MANDATORY policies
- A plan that omits any of these 7 artifacts is INVALID and must be revised
- The evolution pipeline (`agent-utilities-evolution` skill) auto-injects these into every SDD plan

## Assimilation Governance — Wire or Discard

When assimilating features from external codebases, open-source libraries, or research papers
into `agent-utilities`, the following rules are **MANDATORY**:

### Core Heuristic
- **Wire-First**: Every assimilated feature MUST connect to an existing hot path within ≤3 hops
  of an MCP tool, A2A skill, or API entry point. Features without a live call path are rejected.
- **Extend, Don't Duplicate**: If the feature overlaps with an existing concept (similarity ≥ 0.7
  via KG search), it MUST extend that concept — not create a parallel implementation.
- **No Dead Code**: Features that cannot demonstrate a live call path from an entry point to the
  new code are architectural debt and MUST NOT be merged.
- **Unified Downstream**: Assimilated features should be wired downstream from existing hot paths,
  not bolted on as independent silos. Prefer composition over addition.

### Constitution Preservation
- When ingesting an external codebase into the KG, its `constitution.md` / `CONSTITUTION.md`
  MUST be saved as `PolicyNode` entries via `PolicyIngestor.ingest_constitution()`.
- Cross-project rule synthesis: Ingested constitution rules are tagged with their originating
  project and used during comparative analysis to inform integration constraints.
- Conflicting rules between projects are flagged for human resolution during SDD plan review.

### Evolution Pipeline Integration
- The `agent-utilities-evolution` skill enforces this section automatically during SDD plan generation.
- Every recommendation in an evolution SDD plan MUST include:
  1. Which existing hot-path module it wires into
  2. Which entry point (MCP tool, A2A skill, API route) exposes it
  3. Which C4 component it belongs to
  4. Which existing CONCEPT:ID it extends (or a New Concept Proposal if none match)
- Plans that violate Wire-First or introduce dead code are **INVALID** and must be revised.

## Shared Memory Architecture

All agents in the `agent-packages` ecosystem share a unified Knowledge Graph:
- **Global KG**: `~/.local/share/agent-utilities/kg/knowledge_graph.db`
- **Config**: `~/.config/agent-utilities/` (XDG Base Directory Specification)
- **Cache**: `~/.cache/agent-utilities/`
- **Per-project specs**: `.specify/` (git-tracked, project-specific)
- **MCP Exposure**: The KG is accessible as an MCP server for cross-IDE integration
  (Antigravity, Claude Code, OpenCode, Devin). Read-only by default, write access
  requires `kg:write` scope. Every write carries provenance (agent_id, session_id).
