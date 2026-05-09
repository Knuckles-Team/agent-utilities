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
The **Prompt Injection Scanner** intercepts all inputs and tool outputs, matching them against 25+ threat vectors (e.g., reverse shells, data exfiltration). The **Jailbreak Robustness Hardening** module implements a 4-category taxonomy defending against advanced attacks like GCG adversarial suffixes and LLM context boundary confusion.

### Topological Vulnerability Scanning (OS-5.11)
Moving beyond text regex, the system scans the actual execution graph (the HTN planner output) for structural vulnerabilities. By leveraging the Analogy Engine (KG-2.15), it matches current execution trajectories against known "risk subgraphs" (e.g., untrusted data flow into a shell execution node) and halts execution.

### Doom-Loop Detectors & Tool Repetitions (OS-5.18 & OS-5.5)
The **Enhanced Doom-Loop Detector** monitors tool call signatures and execution histories. If an agent repeats an identical sequence of actions without making state progress, the **Tool Repetition Guard** cuts the execution and dynamically injects a corrective prompt into the LLM context, forcing it to change strategies.

### Observability & Audit Logging (OS-5.7 & OS-5.9)
The OS implements a strict append-only **Audit Logger** capturing 30+ action constants. The **Token Usage Tracker** provides 4-bucket granular analytics (prompt, response, thoughts, tool_use) with per-session budgets and threshold alerting to enforce strict USD cost control.

## Benefits Introduced

- **Enterprise Compliance**: Immutable audit trails and structural vulnerability scans allow the agent ecosystem to be safely deployed in highly regulated environments.
- **Cost Guarantees**: Robust doom-loop detection and token tracking ensure the system will never silently drain API budgets.
- **Execution Stability**: Comprehensive guardrails and session concurrency management ensure the platform operates reliably 24/7.

## Key Concepts Leveraged
- **OS-5.0**: Agent OS Kernel
- **OS-5.4**: Prompt Injection Scanner
- **OS-5.5**: Tool Repetition Guard
- **OS-5.7**: Audit Logger
- **OS-5.11**: Topological Vulnerability Scanner
- **OS-5.12**: Jailbreak Robustness Hardening
- **OS-5.18**: Enhanced Doom-Loop Detector
