# Session Concurrency Management (CONCEPT:OS-5.1)

## Overview
Distributed request queuing, interrupt mapping, and double-texting concurrency control (enqueue/reject/interrupt/rollback).

## Implementation Details
- **Source Code**: ``agent_utilities/server/concurrency.py``
- **Pillar**: OS

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Prompt Injection Scanner (CONCEPT:OS-5.1)

## Overview
Pattern-based prompt injection and command injection scanner with 25+ threat vectors ported from Goose. Integrates with PolicyEngine and persists findings as `SecurityFindingNode` in the KG for OWL transitive risk propagation.

## Implementation Details
- **Source Code**: ``agent_utilities/security/prompt_scanner.py``
- **Pillar**: OS

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Tool Repetition Guard (CONCEPT:OS-5.3)

## Overview
Detects infinite tool call loops via consecutive call tracking and per-session budgets. Denied repetitions distill into `ExperienceNode` tactical rules (AHE-3.5) for cross-session loop avoidance.

## Implementation Details
- **Source Code**: ``agent_utilities/security/repetition_guard.py``
- **Pillar**: OS

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Guardrail Callback Engine (CONCEPT:OS-5.3)

## Overview
Push-based input/output guardrail interception with block/redact/warn actions, regex/keyword matching, and PolicyEngine adapter. Ported from MATE's guardrail_callback.py. OWL-inferred `correlatedThreat` detection.

## Implementation Details
- **Source Code**: ``agent_utilities/security/guardrail_engine.py``
- **Pillar**: OS

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Topological Vulnerability Scanner (CONCEPT:OS-5.1)

## Overview
Enhances security by scanning execution graphs for structural vulnerabilities (e.g., untrusted data flows, dependency deadlocks) by matching against known risk subgraphs using the Analogy Engine.

## Implementation Details
- **Source Code**: ``agent_utilities/security/topological_scanner.py``
- **Pillar**: OS

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Jailbreak Robustness Hardening (CONCEPT:OS-5.3)

## Overview
Extends Prompt Injection Scanner (OS-5.4) with 4-category jailbreak attack taxonomy from SoK research: template-based (DAN, AIM, UCAR, Grandma), optimization-based (GCG suffix, token smuggling), LLM-based (context confusion, multi-turn escalation), manual (role-play, authority override). 12 new threat patterns. Derived from SoK: Robustness against Jailbreak (arXiv:2605.05058v1).

## Implementation Details
- **Source Code**: ``agent_utilities/security/prompt_scanner.py``
- **Pillar**: OS

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Enhanced Doom-Loop Detector (CONCEPT:OS-5.0)

## Overview
Pattern-aware doom-loop detection with result-aware tool call signatures, repeating sequence detection (patterns 2-5), and corrective prompt generation. KG persistence via `DoomLoopIncidentNode`. Adapted from ml-intern's doom_loop.py.

## Implementation Details
- **Source Code**: ``agent_utilities/security/doom_loop_detector.py``
- **Pillar**: OS

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
