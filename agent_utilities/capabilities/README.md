# Agentic Capabilities

This directory contains advanced operational primitives that enhance agent autonomy, resilience, and collaboration.

## Overview

Capabilities are higher-level features that interact with the graph orchestration layer and the Knowledge Graph to provide sophisticated behaviors like self-healing, memory eviction, and multi-agent coordination.

## Core Capabilities

- **Checkpointing (`checkpointing.py`)**: Full conversation snapshots at tool and turn boundaries. Enables cross-process session fork, rewind, and resumability.
- **Context Warnings (`context_warnings.py`)**: Proactive monitoring of token usage. Warns the model at 70% (URGENT) and 90% (CRITICAL) of the context window.
- **Output Eviction (`eviction.py`)**: Intercepts massive tool outputs (>80k chars), moves them to the Knowledge Base, and leaves a concise preview in the history.
- **Lifecycle Hooks (`hooks.py`)**: Unified `PRE_TOOL_USE`, `POST_TOOL_USE`, `BEFORE_RUN`, and `AFTER_RUN` hooks for auditing and telemetry.
- **Stuck Loop Detection (`stuck_loop.py`)**: Detects repetitive tool calls, alternating patterns, and no-op loops to prevent token waste and agent "insanity".
- **Agent Teams (`teams.py`)**: Shared task management and P2P messaging primitives for multi-agent collaboration.

## Integration

Capabilities are typically initialized in `agent_factory.py` or `runner.py` and are registered as listeners or interceptors within the graph execution loop.

## Maintenance

- **Thresholds**: Monitor and adjust default thresholds (e.g. eviction size, loop detection depth) based on model performance and cost.
- **Graph Nodes**: Ensure that capability events (e.g. `StuckLoopDetected`) are persisted to the Knowledge Graph as `SelfEvaluationNode` or `TelemetryNode`.
