---
name: kg-report-persister
description: >
  Universal Knowledge Graph persistence atomic skill. Parses workflow summaries,
  extracts performance logs, and creates ExecutionSummary/PerformanceAnomaly nodes.
domain: core
license: MIT
tags: [graph-os, persistence, cypher, logging, monitoring]
metadata:
  author: Genius
  version: '0.1.0'
requires:
  - graph_write
---

# KG Report Persister Skill

Stateless atomic operation to parse workflow execution outputs, extract core execution metrics/logs, and persist them as typed `ExecutionSummary` and `PerformanceAnomaly` nodes/edges inside the Graph-OS Knowledge Graph.

## Prerequisites

- `graph_write` — for transactional Cypher writes and node/edge merging inside the unified Knowledge Graph.

## Steps

### Step 1: parse_workflow_output
Read input execution payload and extract workflow outcomes:
- Parse the input JSON payload which must contain:
  - `execution_id` (string, unique run identifier)
  - `workflow_name` (string, the name of the workflow/skill)
  - `start_time` (ISO-8601 string)
  - `end_time` (ISO-8601 string)
  - `status` (string, e.g. "SUCCESS", "FAILED")
  - `steps_executed` (list of step execution statuses and durations)
  - `raw_logs` (string, optional raw log execution outputs)
- Calculate total execution duration in seconds.
- Scan step outcomes to identify any failures or execution anomalies (e.g. step duration > 2x historical mean or standard threshold of 60 seconds).
- Extract clean error snippets or stack traces from the raw logs if the status is "FAILED".

### Step 2: structure_knowledge_nodes [depends_on: parse_workflow_output]
Map the extracted metrics and logs into exact graph schema node and relationship definitions:
- **ExecutionSummary Node**:
  - `id`: `exec_summary:<execution_id>`
  - `workflow_id`: `wf:<workflow_name>`
  - `status`: String ("SUCCESS" or "FAILED")
  - `duration_seconds`: Float (total runtime)
  - `timestamp`: String (ISO-8601 start time)
  - `error_message`: String (null if success, first 500 chars of stack trace if failed)
- **PerformanceAnomaly Node** (Created only if high latency or execution failures are detected):
  - `id`: `perf_anomaly:<execution_id>:<step_name>`
  - `step_id`: String (name of the failing/slow step)
  - `metric_type`: String ("latency" or "error")
  - `value`: Float/String (duration or error details)
  - `threshold`: Float/String (expected limit/sla)
  - `context`: String (additional diagnostics)

### Step 3: persist_to_graph [depends_on: structure_knowledge_nodes]
Execute transactional graph writes to persist the structured telemetry data and establish domain lineage relationships:
- Invoke the `graph_write` tool with action `bulk_ingest` or raw `add_node`/`add_edge` operations:
  1. Merge `ExecutionSummary` node with its properties.
  2. If an anomaly is present, merge `PerformanceAnomaly` node with its properties.
  3. Create relationship: `(wf:<workflow_name>)-[:HAS_EXECUTION]->(exec_summary:<execution_id>)`.
  4. If an anomaly is present, create relationship: `(exec_summary:<execution_id>)-[:HAS_ANOMALY]->(perf_anomaly:<execution_id>:<step_name>)`.
- Verify write validation metrics, logging any database insertion metrics.
- Return a summary object containing:
  - `status`: "SUCCESS" or "FAILED"
  - `persisted_nodes`: List of successfully written node IDs
  - `relationships_created`: List of created edge patterns
