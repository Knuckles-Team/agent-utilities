# Token Usage Tracker (CONCEPT:OS-5.1)

## Overview
4-bucket granular token analytics (prompt/response/thoughts/tool_use) with session aggregation, agent breakdown, and budget alerting. Ported from MATE's token_usage_service.py. OWL-inferred `highCostAgent` classification.

## Implementation Details
- **Source Code**: ``agent_utilities/observability/token_tracker.py``
- **Pillar**: OS

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Audit Logger (CONCEPT:OS-5.1)

## Overview
Append-only compliance audit trail with 30+ action constants, never-raise semantics, configurable retention, and query filtering. Ported from MATE's audit_service.py. OWL-inferred `escalationChain` temporal reasoning.

## Implementation Details
- **Source Code**: ``agent_utilities/observability/audit_logger.py``
- **Pillar**: OS

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Telemetry & Observability (CONCEPT:OS-5.1)

## Overview
Real-time Graph Streaming (SSE) and lifecycle events. Per-step state snapshots via `graph.iter()`. Early OTEL/logfire gate. Includes Native Langfuse Tracing hooks via `@trace` decorators and automated continuous improvement dataset promotion.

## Implementation Details
- **Source Code**: ``agent_utilities/observability/telemetry.py``, ``agent_utilities/harness/tracing.py``, ``agent_utilities/harness/evaluators.py``
- **Pillar**: OS

## Native Langfuse Integration
`agent-utilities` integrates directly with the Langfuse API client (`langfuse-agent`) to provide zero-overhead, batch-flushed tracing. By providing `LANGFUSE_SECRET_KEY` in the environment, agents automatically push traces, metrics, and LLM-as-a-judge scores. Traces that fall below `LANGFUSE_DATASET_CAPTURE_THRESHOLD` are automatically promoted to Langfuse Datasets to enable closed-loop continuous improvement.

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
