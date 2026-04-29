# Structured Prompts

> CONCEPT:AU-006 — Structured Prompting System

## Overview

The Structured Prompts system provides a **Pydantic-based schema** for defining agent system prompts as machine-parseable JSON documents. Prompt content is decomposed into typed, composable sections instead of monolithic markdown blobs.

## Schema

```json
{
    "task": "python_programmer",
    "type": "prompt",
    "metadata": {
        "description": "Python specialist agent",
        "topic": "Python Development",
        "tone": "technical and precise",
        "style": "professional assistant"
    },
    "identity": {
        "role": "Python Systems Wizard",
        "goal": "Craft production-ready Python solutions",
        "personality": ["enthusiastic", "pedantic about quality"]
    },
    "instructions": {
        "core_directive": "Write idiomatic, high-performance Python code.",
        "responsibilities": ["Idiomatic Development", "Ecosystem Expertise"],
        "capabilities": {
            "modern_python": ["decorators", "dataclasses", "protocols"],
            "testing": ["pytest", "fixtures", "coverage"]
        },
        "workflow": ["Analyze codebase", "Write code", "Test"],
        "quality_checklist": ["black", "mypy --strict", "ruff"]
    },
    "tools": ["agent-builder", "tdd-methodology"],
    "constraints": [],
    "deliverables": []
}
```

## Pydantic Models

### `StructuredPrompt`

The main model for loading and rendering prompt JSON files:

```python
from agent_utilities.structured_prompts import StructuredPrompt

# Load from JSON
prompt = StructuredPrompt.model_validate(json_data)

# Render to system prompt text
system_prompt = prompt.render()
```

### `PromptMetadata`

Descriptive metadata: `description`, `topic`, `tone`, `style`, `audience`.

### `PromptIdentity`

Agent persona: `role`, `goal`, `personality`.

### `PromptInstructions`

Behavioral instructions:
- `core_directive`: Primary instruction / full prompt body
- `responsibilities`: Key areas of focus
- `capabilities`: Categorized skill lists (dict of string → list)
- `workflow`: Ordered steps
- `quality_checklist`: Verification items
- `methodology`: Detailed approach description

## Loading from Files

```python
from agent_utilities.graph.config_helpers import load_specialized_prompts

# Load a prompt by name (searches prompts/ directory)
system_prompt_text = load_specialized_prompts("python_programmer")
```

## Prompt Catalog

All prompt JSON files are located in `agent_utilities/prompts/`. Currently includes 48 prompts across these categories:

| Category | Prompts |
|---|---|
| **Specialists** | python, javascript, typescript, c, cpp, java, golang, rust, mobile |
| **Council** | chairman, contrarian, executor, expansionist, first_principles, outsider, reviewer |
| **Infrastructure** | router, coordinator, planner, architect, verifier, researcher, critique |
| **Domain** | database, devops, cloud_architect, security_auditor, qa_expert, data_scientist |
| **System** | base_agent, main_agent, safety_guard, safety_policy, memory_instruction, memory_selection |
| **Content** | brand_strategy, code_generation, content_generation, document_specialist, browser_automation |
