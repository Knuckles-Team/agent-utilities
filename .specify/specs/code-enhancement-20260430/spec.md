# Code Enhancement: agent-utilities

> Automated code enhancement review for agent-utilities. Covers 4 analysis domains.

## User Stories

- As a **developer**, I want to **address Dependency Audit findings (grade: F, score: 46)**, so that **improve project dependency audit from F to at least B (80+)**.
- As a **developer**, I want to **address Codebase Optimization findings (grade: F, score: 42)**, so that **improve project codebase optimization from F to at least B (80+)**.
- As a **developer**, I want to **address Test Coverage findings (grade: C, score: 75)**, so that **improve project test coverage from C to at least B (80+)**.

## Functional Requirements

- **FR-001**: Minor update: pydantic-settings 2.13.0 (installed) -> 2.14.0
- **FR-002**: Minor update: urllib3 2.3.0 (installed) -> 2.6.3
- **FR-003**: Minor update: ladybug 0.15.3 (installed) -> 0.16.0
- **FR-004**: Minor update: pydantic-ai-slim 1.83.0 (installed) -> 1.88.0
- **FR-005**: Minor update: pydantic-graph 1.83.0 (installed) -> 1.88.0
- **FR-006**: Minor update: llama-index-embeddings-openai 0.5.1 (constraint — not installed) -> 0.6.0
- **FR-007**: Minor update: playwright 1.58.0 (installed) -> 1.59.0
- **FR-008**: Minor update: llama-index-embeddings-ollama 0.8.6 (constraint — not installed) -> 0.9.0
- **FR-009**: Minor update: llama-index-embeddings-huggingface 0.6.1 (constraint — not installed) -> 0.7.0
- **FR-010**: Minor update: falkordb 1.0.1 (constraint — not installed) -> 1.6.1
- **FR-011**: Minor update: hvac 2.3.0 (constraint — not installed) -> 2.4.0
- **FR-012**: Minor update: authlib 1.6.5 (installed) -> 1.7.0
- **FR-013**: Minor update: opentelemetry-instrumentation-starlette 0.60b1 (installed) -> 0.62b1
- **FR-014**: Minor update: opentelemetry-instrumentation-fastapi 0.60b1 (installed) -> 0.62b1
- **FR-015**: Minor update: opentelemetry-instrumentation-asgi 0.60b1 (installed) -> 0.62b1
- **FR-016**: 16 functions exceed 200 lines (actionable refactoring targets): build_agent_app (510L), _execute_dynamic_mcp_agent (457L), app_factory (440L), create_graph_agent (432L), create_agent (359L)
- **FR-017**: Monolithic: base_utilities.py (1045L) — 3 functions with high complexity (worst: retrieve_package_name at 95L, CC=22); Low cohesion: 36 distinct concepts in one file
- **FR-018**: Monolithic: workspace.py (657L) — 1 functions with high complexity (worst: resolve_mcp_config_path at 69L, CC=17); Low cohesion: 26 distinct concepts in one file
- **FR-019**: Monolithic: steps.py (2423L) — 5 functions with high complexity (worst: router_step at 316L, CC=35); Low cohesion: 61 distinct concepts in one file
- **FR-020**: Needs attention: factory.py (592L) — 1 functions with high complexity (worst: create_agent at 359L, CC=59)
- **FR-021**: Needs attention: app.py (567L) — 1 functions with high complexity (worst: build_agent_app at 510L, CC=66)
- **FR-022**: Needs attention: repl.py (590L) — 1 functions with high complexity (worst: RLMEnvironment.run_full_rlm at 140L, CC=16)
- **FR-023**: 65 functions with nesting depth >4
- **FR-024**: 1 flat directories with >15 Python files: agent_utilities/tools
- **FR-025**: 6 tests without assertions
- **FR-026**: 8 potential doc-test drift items

## Success Criteria

- Overall GPA: 1.5 → 3.0
- Domains at B or above: 1 → 4
- Actionable findings: 26 → 0
