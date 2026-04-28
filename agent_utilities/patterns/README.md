# Agentic Engineering Patterns

This directory contains first-class implementations of Simon Willison's Agentic Engineering Patterns and other advanced agentic workflows.

## Overview

The `patterns/` module provides standardized, reusable logic for complex engineering tasks that go beyond simple tool calls. These patterns are orchestrated by the `PatternManager` and are available to all agents via `AgentDeps`.

## Included Patterns

### 1. TDD Cycle (`tdd.py`)
- **What it is**: A full Red-Green-Refactor orchestration.
- **When to use**: When implementing new features or fixing bugs where high confidence is required.
- **How it works**: Spawns specialized subagents for each phase (Red, Green, Refactor) to ensure strict TDD compliance.

### 2. First Run Tests (`first_run_tests.py`)
- **What it is**: An automated baseline tester.
- **When to use**: At the start of every implementation task to ensure the current state is stable.
- **How it works**: Executes `pytest` (or configured command) and feeds results to the agent's context.

### 3. Agentic Manual Testing (`manual_testing.py`)
- **What it is**: Exploratory testing orchestrated by a subagent.
- **When to use**: For verifying behaviors that are hard to automate (e.g., CLI outputs, UI states, API integrations).
- **How it works**: Uses `ExecutionNotes` to record commands, observations, and artifacts (images) which are then streamed to the Knowledge Graph.

### 4. Code Walkthroughs (`walkthroughs.py`)
- **What it is**: Automated, linear codebase documentation generator.
- **When to use**: After completing a feature or when onboarding to a new codebase.
- **How it works**: Analyzes the implementation and generates a Markdown walkthrough with links to specific files and logic.

### 5. Interactive Explanations (`interactive_explanations.py`)
- **What it is**: Educational artifact generator.
- **When to use**: For explaining complex logic or system architectures to human users.
- **How it works**: Generates self-contained Vanilla HTML/JS artifacts with premium aesthetics.

## How to Use

Patterns are accessed via `ctx.deps.patterns` in any agent tool:

```python
async def my_tool(ctx: RunContext[AgentDeps]):
    # Run a walkthrough
    await ctx.deps.patterns.generate_walkthrough("./src")

    # Run manual testing
    await ctx.deps.patterns.manual_test("Verify the login flow")
```

## Maintenance

- **Subagents**: Patterns often use `dispatch_subagent`. Ensure that the subagent goals and system prompts are kept up to date with the latest LLM capabilities.
- **Artifacts**: Ensure that generated artifacts (Markdown, HTML) follow the latest design guidelines and remain self-contained.
- **Testing**: Add unit tests in `tests/patterns/` for any new pattern logic.
