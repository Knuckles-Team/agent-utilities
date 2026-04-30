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
