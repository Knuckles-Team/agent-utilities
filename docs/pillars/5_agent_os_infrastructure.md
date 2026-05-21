# Pillar 5: Agent OS Infrastructure

## Overview

The **Agent OS Infrastructure** pillar provides the production-grade foundation, telemetry, and proactive security guardrails that elevate `agent-utilities` from a research prototype into an enterprise-ready Agentic Operating System.

## Why We Built This (Rationale)

Deploying autonomous systems in production introduces severe risks:
1. **Prompt & Command Injection**: Malicious or malformed inputs can hijack the agent's LLM context and execute arbitrary code.
2. **Infinite Loops (Doom Loops)**: Agents can get stuck repeatedly calling the same tool with the same failed arguments, burning thousands of dollars in LLM API credits.
3. **Lack of Auditability**: When an agent modifies a production database or deletes a file, tracking exactly *why* that decision was made is critical for compliance.

## How It Works (Implementation)

### Proactive Security & Jailbreak Defense (OS-5.4 & OS-5.12)
The **Threat Defense Engine (Injection)** intercepts all inputs and tool outputs, matching them against 25+ threat vectors (e.g., reverse shells, data exfiltration). The **Threat Defense Engine (Jailbreak)** module implements a 4-category taxonomy defending against advanced attacks like GCG adversarial suffixes and LLM context boundary confusion.

### Topological Vulnerability Scanning (OS-5.11)
Moving beyond text regex, the system scans the actual execution graph (the HTN planner output) for structural vulnerabilities. By leveraging the Analogy Engine (KG-2.15), it matches current execution trajectories against known "risk subgraphs" (e.g., untrusted data flow into a shell execution node) and halts execution.

### Doom-Loop Detectors & Tool Repetitions (OS-5.18 & OS-5.5)
The **Execution Stability Engine (Doom-Loop)** monitors tool call signatures and execution histories. If an agent repeats an identical sequence of actions without making state progress, the **Execution Stability Engine (Repetition Guard)** cuts the execution and dynamically injects a corrective prompt into the LLM context, forcing it to change strategies.

### Observability & Audit Logging (OS-5.7 & OS-5.9)
The OS implements a strict append-only **Audit Logger** capturing 30+ action constants. The **Token Usage Tracker** provides 4-bucket granular analytics (prompt, response, thoughts, tool_use) with per-session budgets and threshold alerting to enforce strict USD cost control.

## Benefits Introduced

- **Enterprise Compliance**: Immutable audit trails and structural vulnerability scans allow the agent ecosystem to be safely deployed in highly regulated environments.
- **Cost Guarantees**: Robust doom-loop detection and token tracking ensure the system will never silently drain API budgets.
- **Execution Stability**: Comprehensive guardrails and session concurrency management ensure the platform operates reliably 24/7.

## Key Concepts Leveraged
- **OS-5.0**: Agent OS Kernel
- **OS-5.4**: Threat Defense Engine (Injection)
- **OS-5.5**: Execution Stability Engine (Repetition Guard)
- **OS-5.7**: Audit Logger
- **OS-5.11**: Threat Defense Engine (Topological)
- **OS-5.12**: Threat Defense Engine (Jailbreak)
- **OS-5.18**: Execution Stability Engine (Doom-Loop)


### Human-in-the-Loop (Tool Approval & Elicitation)

`agent-utilities` provides true **pause-and-resume** human-in-the-loop for sensitive tool execution and MCP elicitation. When a specialist sub-agent calls a tool flagged with `requires_approval=True`, the graph suspends at that exact node, streams an approval request to the connected UI, and resumes only after the user responds.

**Key Components:**
- **`ApprovalManager`** (`approval_manager.py`) — asyncio.Future-based registry that pauses coroutines and resumes them when the UI responds
- **`run_with_approvals()`** — wraps pydantic-ai's two-call `DeferredToolRequests` → `DeferredToolResults` pattern into a single blocking call
- **`/api/approve`** endpoint — REST endpoint that both UIs POST to when the user approves/denies
- **`global_elicitation_callback()`** — MCP `ctx.elicit()` callback using the same pause/resume mechanism

**Protocol Support:**
| Protocol | Approval Mechanism |
|---|---|
| AG-UI (web + terminal) | Sideband SSE events + `POST /api/approve` |
| ACP | pydantic-acp's native `NativeApprovalBridge` (automatic) |
| SSE (`/stream`) | Same as AG-UI |
