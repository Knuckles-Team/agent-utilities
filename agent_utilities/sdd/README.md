# Spec-Driven Development (SDD)

This directory contains the core logic for Spec-Driven Development (SDD) orchestration, mirroring the `spec-kit` methodology.

## Overview

The `sdd/` module manages the lifecycle of a feature from governance and requirements to implementation and verification. It integrates deeply with the `patterns/` module to provide a robust, self-healing development workflow.

## Components

### 1. SDD Manager (`__init__.py`)
- **Purpose**: Data management for SDD artifacts (Specs, Tasks, Plans).
- **Functionality**: Handles loading/saving of structured JSON and mirrored Markdown artifacts in the `.specify/` (or `agent_data/`) directory.

### 2. SDD Orchestrator (`orchestrator.py`)
- **Purpose**: High-level workflow orchestration.
- **Functionality**: Links together the various SDD phases:
    - **Governance**: Constitution generation.
    - **Specification**: Feature definition.
    - **Testing**: Baseline and TDD phases.
    - **Implementation**: Parallel execution dispatch.
    - **Verification**: Quality gate audit.

## Usage

The `SDDOrchestrator` is typically used within the `planner` or `dispatcher` nodes of the graph:

```python
from agent_utilities.sdd.orchestrator import SDDOrchestrator

orchestrator = SDDOrchestrator(deps)
await orchestrator.run_sdd("Implement a new login endpoint")
```

## Maintenance

- **Schema**: SDD models are defined in `agent_utilities/models/`. Ensure that any changes to these models are reflected in the manager's serialization logic.
- **Artifacts**: Artifacts are stored in the workspace. Ensure that the manager respects `.gitignore` and doesn't pollute the repository root.
