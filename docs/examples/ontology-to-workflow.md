# Worked Example: BPMN Process to Executable Workflow

**What this demonstrates.** The full descriptive-to-executable bridge: a BPMN 2.0
process definition is lifted into the Knowledge Graph as step-level structure
(`BusinessProcess` / `BusinessTask` / `FLOWS_TO`, CONCEPT:KG-2.53), compiled into
an executable `WorkflowDefinition` with sequence-flow-derived dependencies and a
`REALIZES` bridge edge (`graph_orchestrate action=compile_process`,
CONCEPT:ORCH-1.41), executed through the ontology gate that SHACL-validates the
stored definition and applies permissioning before dispatch (CONCEPT:ORCH-1.42),
and closed out with run-level provenance — the run's `RunTrace` gets an
`EXECUTED_PROCESS` edge back to the `BusinessProcess` (CONCEPT:ORCH-1.43).

**Prerequisites (ladder rung).** Single-host rung or above from
[Deployment configurations](../guides/deployment-configurations.md): a running
`graph-os` MCP server (or gateway) backed by an engine you can write to. No
Camunda deployment is required — the fixture stands in for the engine's BPMN
XML. Deep dive: [Ontology system](../architecture/ontology_system.md).

All fixture files live in [`examples/ontology_workflow/`](https://github.com/Knuckles-Team/agent-utilities/tree/main/examples/ontology_workflow/).

---

## 1. The BPMN fixture

[`examples/ontology_workflow/sample_process.bpmn`](https://github.com/Knuckles-Team/agent-utilities/blob/main/examples/ontology_workflow/sample_process.bpmn)
is a small, valid BPMN 2.0 order-fulfillment process — four executable tasks and
one exclusive gateway:

```text
start -> Validate Order (serviceTask)
      -> In stock? (exclusiveGateway)
           -> [${inStock == true}]  Pack Order (userTask)        -> Ship Order
           -> [${inStock == false}] Create Backorder (serviceTask) -> Ship Order
      -> Ship Order (serviceTask) -> end
```

The `startEvent`/`endEvent` elements are deliberately present: the KG-2.53 lift
collapses pass-through elements so `FLOWS_TO` ordering between lifted tasks
survives them.

## 2. Lift the BPMN into the KG (KG-2.53)

The extractor is `agent_utilities/knowledge_graph/enrichment/extractors/camunda.py`.
It takes an injected, duck-typed Camunda client; when that client also exposes
`get_process_definition_xml` (the camunda-mcp `camunda_process action=xml`
surface), each definition's BPMN XML is parsed and the step-level structure is
lifted. The runnable script
[`examples/ontology_workflow/lift_sample_process.py`](https://github.com/Knuckles-Team/agent-utilities/blob/main/examples/ontology_workflow/lift_sample_process.py)
feeds it a file-backed client serving the fixture:

```bash
PYTHONPATH=. python3 examples/ontology_workflow/lift_sample_process.py
```

**Expected output** (extraction batch written through the engine):

```json
{
  "category": "camunda",
  "nodes_written": 6,
  "edges_written": 10,
  "process_id": "bpmn_process:order_fulfillment:1:demo"
}
```

What the batch actually contains (captured by running the extractor against
this exact fixture):

```text
NODE bpmn_process:order_fulfillment:1:demo            BusinessProcess {"name": "Order Fulfillment", "key": "order_fulfillment", "version": null}
NODE bpmn_task:order_fulfillment:1:demo:backorder     BusinessTask {"name": "Create Backorder", "element_id": "backorder", "task_type": "serviceTask", "is_gateway": false, ...}
NODE bpmn_task:order_fulfillment:1:demo:pack_order    BusinessTask {"name": "Pack Order", "element_id": "pack_order", "task_type": "userTask", "is_gateway": false, ...}
NODE bpmn_task:order_fulfillment:1:demo:ship_order    BusinessTask {"name": "Ship Order", "element_id": "ship_order", "task_type": "serviceTask", "is_gateway": false, ...}
NODE bpmn_task:order_fulfillment:1:demo:stock_check   BusinessTask {"name": "In stock?", "element_id": "stock_check", "task_type": "exclusiveGateway", "is_gateway": true, ...}
NODE bpmn_task:order_fulfillment:1:demo:validate_order BusinessTask {"name": "Validate Order", "element_id": "validate_order", "task_type": "serviceTask", "is_gateway": false, ...}

EDGE <each task> -[PART_OF]-> bpmn_process:order_fulfillment:1:demo            (5 edges)
EDGE validate_order -[FLOWS_TO]-> stock_check
EDGE stock_check    -[FLOWS_TO]-> pack_order   {"condition": "${inStock == true}"}
EDGE stock_check    -[FLOWS_TO]-> backorder    {"condition": "${inStock == false}"}
EDGE pack_order     -[FLOWS_TO]-> ship_order
EDGE backorder      -[FLOWS_TO]-> ship_order
```

Note the gateway is lifted as a `BusinessTask` typed `exclusiveGateway` with
`is_gateway: true`, and the start/end events were collapsed (no nodes, but the
`validate_order -> stock_check` ordering survived).

## 3. Compile the process into a workflow (ORCH-1.41)

MCP call (also in
[`examples/ontology_workflow/compile_process_call.json`](https://github.com/Knuckles-Team/agent-utilities/blob/main/examples/ontology_workflow/compile_process_call.json)) —
`task` carries the `BusinessProcess` node id, `agent_name` an optional workflow
name:

```json
{
  "tool": "graph_orchestrate",
  "arguments": {
    "action": "compile_process",
    "task": "bpmn_process:order_fulfillment:1:demo",
    "agent_name": "process_order_fulfillment"
  }
}
```

REST twin: `POST /api/graph/orchestrate/compile-process` with body
`{"process_id": "bpmn_process:order_fulfillment:1:demo", "name": "process_order_fulfillment"}`
(see `graph_orchestrate_compile_process_endpoint` in
`agent_utilities/mcp/kg_server.py`).

**Expected output** — the `compile_and_store` report plus `status` and the
stored topology diagram (`mermaid` is `null` when no diagram could be stored):

```json
{
  "workflow_id": "workflow:process_order_fulfillment:9eaa3116",
  "name": "process_order_fulfillment",
  "step_count": 4,
  "unresolved_tasks": [],
  "process_id": "bpmn_process:order_fulfillment:1:demo",
  "status": "compiled",
  "mermaid": "---\ntitle: process_order_fulfillment\n...\nflowchart TD\n  validate[...]\n  validate --> backorder\n  validate --> pack\n  backorder --> ship\n  pack --> ship\n  ..."
}
```

The compiled plan (captured by running `ProcessPlanCompiler` directly against
the lifted fixture, with KG-registered capabilities matching each task):

```json
[
  {"id": "validate",  "refined_subtask": "Validate Order",   "parallel": false, "depends_on": []},
  {"id": "backorder", "refined_subtask": "Create Backorder", "parallel": true,  "depends_on": ["validate"]},
  {"id": "pack",      "refined_subtask": "Pack Order",       "parallel": true,  "depends_on": ["validate"]},
  {"id": "ship",      "refined_subtask": "Ship Order",       "parallel": false, "depends_on": ["backorder", "pack"]}
]
```

Compilation semantics worth noticing:

- The `In stock?` gateway is **not** a step — it was collapsed, so `pack` and
  `backorder` become parallel branches both depending on `validate`, and `ship`
  joins both branches.
- The branch conditions survive in `plan.metadata.branch_conditions`
  (`{"from": ..., "to": ..., "condition": "${inStock == true}"}`).
- A task with no KG-registered agent/tool match stays in the plan as an explicit
  manual step (`manual:` id prefix, step metadata `unresolved: true`) and is
  listed in `unresolved_tasks`. Cycles (BPMN loop-backs) fail compilation with
  `ProcessCompilationError` naming the tasks on the cycle.
- The stored `WorkflowDefinition` gets a
  `(:WorkflowDefinition)-[:REALIZES]->(:BusinessProcess)` bridge edge.

## 4. Execute through the ontology gate (ORCH-1.42)

MCP call (also in
[`examples/ontology_workflow/execute_workflow_call.json`](https://github.com/Knuckles-Team/agent-utilities/blob/main/examples/ontology_workflow/execute_workflow_call.json)):

```json
{
  "tool": "graph_orchestrate",
  "arguments": {
    "action": "execute_workflow",
    "agent_name": "process_order_fulfillment",
    "task": "Fulfill order #1042 for tenant acme",
    "max_steps": 30
  }
}
```

Before any dispatch, `gate_workflow_execution`
(`agent_utilities/knowledge_graph/core/workflow_gate.py`) runs:

1. **Shape gate** (`KG_WORKFLOW_SHAPE_GATE`, default on): the stored
   `WorkflowDefinition` + `WorkflowStep` nodes are materialized into a focused
   RDF graph and validated against the bundled governance shapes. Violations
   refuse execution.
2. **Permission gate** (only when `KG_BRAIN_ENFORCE` is on): the ontology
   permissioning row gate is applied to the workflow node for the current
   actor; a denial raises `PermissionError` (fail-closed, CONCEPT:OS-5.14 — see
   [Identity JWT example](identity-jwt.md)).

**Expected output** (success path):

```json
{
  "result": "<workflow execution result string>",
  "mermaid": "---\ntitle: process_order_fulfillment\n..."
}
```

**Expected output** (shape-gate refusal — a malformed stored definition never
burns an agent run):

```json
{
  "error": "workflow definition failed ontology validation — execution refused",
  "workflow": "process_order_fulfillment",
  "workflow_id": "workflow:process_order_fulfillment:9eaa3116",
  "violations": [
    {"focus_node": "...", "path": "...", "message": "..."}
  ]
}
```

A workflow name with no stored definition passes the gate untouched — dynamic
execution paths are not gated.

## 5. Lineage close-out (ORCH-1.43)

When the executed workflow carries a `REALIZES` edge, `WorkflowRunner`
(`agent_utilities/workflows/runner.py`) closes the provenance loop on
completion: it upserts a workflow-level `RunTrace` node and links it
`EXECUTED_PROCESS` to the `BusinessProcess` (best-effort by design — lineage
never fails a run). "Which runs executed this harvested process?" is then a
graph query:

```json
{
  "tool": "graph_query",
  "arguments": {
    "cypher": "MATCH (r:RunTrace)-[:EXECUTED_PROCESS]->(p) WHERE p.id = 'bpmn_process:order_fulfillment:1:demo' RETURN r.id AS run, r.status AS status, r.duration_ms AS duration_ms, r.timestamp AS ts"
  }
}
```

**Expected output:**

```json
[
  {
    "run": "trace:exec-4f2a9c1e",
    "status": "completed",
    "duration_ms": 18423.5,
    "ts": "2026-06-11T03:12:09Z"
  }
]
```

Deployments that treat an external metadata server as the lineage system of
record wire `WorkflowRunner(lineage_sink=...)` — e.g. forwarding the normalized
close-out record to egeria-mcp's `assert_lineage`. The sink receives
`{process_id, process_external_id, workflow_id, workflow_name, run_id, status,
completed_steps, failed_steps, duration_ms, timestamp}` and is best-effort.

## What landed in the KG

```text
(:BusinessProcess {id: "bpmn_process:order_fulfillment:1:demo"})
  <-[:PART_OF]-      (:BusinessTask x5, incl. the gateway)   # KG-2.53 lift
  task -[:FLOWS_TO {condition?}]-> task                      # sequence flows
  <-[:REALIZES]-     (:WorkflowDefinition {name: "process_order_fulfillment"})  # ORCH-1.41
                       -[:HAS_STEP]-> (:WorkflowStep x4)
  <-[:EXECUTED_PROCESS]- (:RunTrace {status, duration_ms})   # ORCH-1.43
```

---

*Verification: smoke-run against this tree (2026-06-11). Executed:
`python3 -m pytest tests/unit/knowledge_graph/test_process_plan_compiler.py
tests/unit/knowledge_graph/enrichment/test_camunda_extractor.py -x -q` — 25
passed; plus a direct one-off run of the camunda extractor and
`ProcessPlanCompiler.compile`/`compile_and_store` against
`examples/ontology_workflow/sample_process.bpmn` (with an in-memory engine
double), from which the extraction batch, plan steps, metadata, report and
mermaid shown above were captured verbatim. The `execute_workflow` and lineage
outputs in steps 4–5 were reviewed against code and the passing suites
`tests/unit/knowledge_graph/test_workflow_gate.py` and
`tests/unit/test_workflow_lineage_closeout.py` (run here: 50 passed combined
with the dispatch suite), not captured from a live LLM-backed run.*
