# Agentic Prompts

This directory contains the structured JSON blueprints that define the "souls" and specialized behaviors of agents in the ecosystem.

## Overview

The `prompts/` module uses **JSON-as-Code** for high-precision task specification. Moving away from free-form Markdown, these blueprints use a standardized Pydantic schema to ensure consistency, versioning, and programmatic manipulation.

## Blueprint Structure

Each `.json` file follows a structured schema (defined in `structured_prompts.py`):
- `name`: Unique identifier for the specialist.
- `topic`: Primary domain of expertise.
- `tone`: Personality and communication style (e.g. "Professional", "Concise").
- `structure`: Expected output format and structural requirements.
- `content`: The core system prompt or behavioral instructions.
- `tools`: (Optional) Curated list of tools bound to this specialist.

## Key Specialist Prompts

- **Planner (`planner.json`)**: High-level task decomposition and goal setting.
- **Router (`router.json`)**: Initial query analysis and topology selection.
- **Python Programmer (`python_programmer.json`)**: Domain-specific logic for Python development.
- **Verifier (`verifier.json`)**: Quality gate and result scoring logic.
- **Coordinator (`coordinator.json`)**: Multi-agent task management and barrier sync logic.

## Usage

Prompts are loaded and compiled at runtime by the `PromptBuilder`. The system automatically identifies the relevant blueprint based on the agent's name or domain tags.

## Maintenance

- **Template Variables**: Use `{{variable}}` syntax for dynamic injection (e.g. `{{workspace_context}}`).
- **Optimization**: Use the `self_improvement_tools` to evolve these prompts based on textual gradients and outcome rewards.
- **Consistency**: When adding a new specialist, ensure it follows the tone and structure established for similar domains.
