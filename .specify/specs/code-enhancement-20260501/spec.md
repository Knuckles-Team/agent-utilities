# Code Enhancement: agent-utilities

> Automated code enhancement review for agent-utilities. Covers 16 analysis domains.

## User Stories

- As a **developer**, I want to **address Project Analysis findings (grade: D, score: 69)**, so that **improve project project analysis from D to at least B (80+)**.
- As a **developer**, I want to **address Codebase Optimization findings (grade: F, score: 42)**, so that **improve project codebase optimization from F to at least B (80+)**.
- As a **developer**, I want to **address Security Analysis findings (grade: F, score: 0)**, so that **improve project security analysis from F to at least B (80+)**.
- As a **developer**, I want to **address Test Coverage findings (grade: C, score: 70)**, so that **improve project test coverage from C to at least B (80+)**.
- As a **developer**, I want to **address Architecture & Design Patterns findings (grade: C, score: 75)**, so that **improve project architecture & design patterns from C to at least B (80+)**.
- As a **developer**, I want to **address Concept Traceability findings (grade: F, score: 30)**, so that **improve project concept traceability from F to at least B (80+)**.
- As a **developer**, I want to **address Pre-Commit Compliance findings (grade: F, score: 54)**, so that **improve project pre-commit compliance from F to at least B (80+)**.
- As a **developer**, I want to **address Test Execution findings (grade: F, score: 25)**, so that **improve project test execution from F to at least B (80+)**.
- As a **developer**, I want to **address Version Sync Analysis findings (grade: D, score: 60)**, so that **improve project version sync analysis from D to at least B (80+)**.
- As a **developer**, I want to **address Pytest Quality findings (grade: D, score: 61)**, so that **improve project pytest quality from D to at least B (80+)**.
- As a **developer**, I want to **address Environment Variables findings (grade: F, score: 55)**, so that **improve project environment variables from F to at least B (80+)**.

## Functional Requirements

- **FR-001**: 16 functions exceed 200 lines (actionable refactoring targets): build_agent_app (510L), _execute_dynamic_mcp_agent (457L), app_factory (440L), create_graph_agent (432L), create_agent (359L)
- **FR-002**: Monolithic: steps.py (2423L) — 5 functions with high complexity (worst: router_step at 316L, CC=35); Low cohesion: 61 distinct concepts in one file
- **FR-003**: Monolithic: base_utilities.py (1045L) — 3 functions with high complexity (worst: retrieve_package_name at 95L, CC=22); Low cohesion: 36 distinct concepts in one file
- **FR-004**: Monolithic: engine.py (1881L) — 2 functions with high complexity (worst: IntelligenceGraphEngine._query_nx_fallback at 64L, CC=19); God class: IntelligenceGraphEngine (65 methods) — consider mixins/composition
- **FR-005**: Needs attention: mcp_utilities.py (899L) — 2 functions with high complexity (worst: create_mcp_server at 305L, CC=57)
- **FR-006**: Needs attention: factory.py (592L) — 1 functions with high complexity (worst: create_agent at 359L, CC=59)
- **FR-007**: Needs attention: app.py (567L) — 1 functions with high complexity (worst: build_agent_app at 510L, CC=66)
- **FR-008**: 65 functions with nesting depth >4
- **FR-009**: 1 flat directories with >15 Python files: agent_utilities/tools
- **FR-010**: 10 HIGH severity vulnerabilities found
- **FR-011**: 734 MEDIUM severity vulnerabilities found
- **FR-012**: eval/exec usage detected: 20 instances
- **FR-013**: 6 tests without assertions
- **FR-014**: 8 potential doc-test drift items
- **FR-015**: SRP: 164 modules exceed 500 lines (god modules)
- **FR-016**: SRP: 30 classes have >15 methods
- **FR-017**: Low dependency injection ratio: 8%
- **FR-018**: 11 orphaned concepts (only in one source)
- **FR-019**: 12 concepts with drift (missing from one source)
- **FR-020**: 1306 test functions missing concept markers
- **FR-021**: 623 significant functions (>10 lines) missing concept markers in docstrings
- **FR-022**: Total lint findings: 2 (high/error: 0, medium/warning: 2, low: 0)
- **FR-023**: 2 hook(s) may be outdated: ruff-pre-commit, uv-pre-commit
- **FR-024**: 1 directories with >40 files: agent_utilities/prompts
- **FR-025**: 2 directories with >20 files: tests/unit/core, agent_utilities/tools
- **FR-026**: Found 2 file(s) with version '0.3.0' that are NOT tracked in .bumpversion.cfg:
- **FR-027**:   - .specify/reports/results.json
- **FR-028**:   - .specify/reports/code_enhancement_report.md
- **FR-029**: 15 test files exceed 500 lines — split into focused modules
- **FR-030**: 13 test files have >30 tests — too dense
- **FR-031**: 6 tests have no assertions
- **FR-032**: 280 tests use weak assertions (assert result is not None, assert True, etc.)
- **FR-033**: 29 tests have excessive mocking (>5 mocks) — test behavior, not implementation
- **FR-034**: Only 2% of env vars documented in README.md
- **FR-035**: Undocumented env vars: A2A_TOOLS, ACP_SESSION_ROOT, AGENT_SECRETS_MASTER_KEY, AGENT_URL, AGENT_USER_TOKEN, AGENT_UTILITIES_TESTING, AGENT_WORKSPACE, ANTHROPIC_API_KEY, AUDIENCE, BROWSER_TOOLS
- **FR-036**: 104 Python env vars not in .env.example: A2A_TOOLS, ACP_SESSION_ROOT, AGENT_SECRETS_MASTER_KEY, AGENT_URL, AGENT_USER_TOKEN
- **FR-037**: 45 env vars have no default value in code

## Success Criteria

- Overall GPA: 1.62 → 3.0
- Domains at B or above: 5 → 16
- Actionable findings: 37 → 0
