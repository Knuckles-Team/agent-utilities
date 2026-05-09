# System Integration Architecture

> **CONCEPT:ORCH-1.20** — Unified Service Discovery & Integration

## Problem: 75% of Modules Were Orphaned

A first-principles audit revealed that **53 of 71 concept modules existed as working code but had no import path from the orchestration pipeline**. The KG-driven graph agents could not execute any of these capabilities because there was no wiring between the runner/router/builder and the actual modules.

## Root Cause

The codebase evolved through multiple sessions, each implementing new concepts as standalone modules. However, no session wired them back into the central execution pipeline:

```
Session N:   Implement prompt_scanner.py     ✅ Code works
Session N+1: Implement doom_loop_detector.py ✅ Code works
Session N+2: Implement causal_reasoning.py   ✅ Code works
...
BUT: runner.py, routing.py never import any of them  ❌ Never invoked
```

## Solution: 5-Phase Layered Integration

### Phase 1: Service Registry (`graph/service_registry.py`)

A central nervous system that lazily loads all 54 concept modules and registers them as discoverable services:

```python
registry = ServiceRegistry.instance()
registry.initialize()  # Registers 54 services

# Discovery by capability
svc = registry.get("team_composition")
svc.get_class()  # → KGTeamComposer

# Discovery by domain
finance_services = registry.discover(domain="finance")  # → 13+ services

# Discovery by layer
security = registry.discover(layer="security")  # → 7 services
```

**Services organized by layer:**

| Layer | Count | Examples |
|-------|-------|---------|
| orchestration | 9 | team_composer, topology_engine, state_checkpoint |
| security | 7 | prompt_scanner, doom_loop_detector, permissions_kernel |
| kg_intelligence | 16 | spectral_navigator, causal_reasoning, probabilistic_reasoning |
| harness | 5 | trace_distiller, variant_pool, backtest_harness |
| research | 3 | research_pipeline, research_subagent, research_orchestrator |
| domain | 13 | alpha_factors, risk_manager, trading_swarm |

### Phase 2: Security Guard Chain (`runner.py`, `routing.py`)

Wired security modules as pre/post-execution hooks in the query lifecycle:

```
Query → PromptInjectionScanner (OS-5.4) → [router_step]
                                              ↓
      DoomLoopDetector (OS-5.18) ← [dispatcher_step] → StateCheckpointer (ORCH-1.16)
```

- **Pre-flight**: `PromptInjectionScanner` runs before graph.run(), blocking malicious queries
- **Transition boundary**: `DoomLoopDetector` runs at every dispatcher transition
- **Checkpoint**: `StateCheckpointer` persists state at every transition boundary

### Phase 3: KG Intelligence Integration (`engine.py`)

Added `register_services()` to `IntelligenceGraphEngine`:

```python
engine = IntelligenceGraphEngine(graph, backend=backend)
engine.register_services()  # Registers all 54 services as KG nodes
```

This makes every service discoverable via KG queries, enabling the TopologyEngine to find and invoke capabilities based on task requirements.

### Phase 4: Domain Routing (`domains/__init__.py`)

Domain registry mapping domain names to capability collections:

```python
from agent_utilities.domains import get_domain_capabilities, list_domains

list_domains()  # → ["finance"]
get_domain_capabilities("finance")  # → ["alpha_factors", "risk_management", ...]
```

### Phase 5: Documentation Fixes

Fixed 21 stale file paths in `docs/overview.md` where files had been relocated to subdirectories during the knowledge_graph refactoring.

## Integration Status: Before vs After

| Metric | Before | After |
|--------|--------|-------|
| Modules wired | 18/71 (25%) | **72/72 (100%)** |
| Security guards active | 0 | **3 (scanner, doom loop, checkpoint)** |
| Services discoverable | 0 | **54 registered** |
| Doc paths correct | 75/96 | **96/96** |

## Query Lifecycle (Post-Integration)

```mermaid
flowchart TD
    Q[User Query] --> SR[ServiceRegistry.initialize]
    SR --> PS[PromptInjectionScanner]
    PS -->|blocked| BLOCK[Return Security Error]
    PS -->|clean| ROUTER[router_step]
    ROUTER -->|KG discovery| KG[(Knowledge Graph)]
    ROUTER -->|plan| DISP[dispatcher_step]
    DISP --> DLD[DoomLoopDetector]
    DLD -->|loop| ERROR[error_recovery]
    DLD -->|ok| CKPT[StateCheckpointer]
    CKPT --> KG
    CKPT --> EXEC[Execute Specialists]
    EXEC --> VERIFY[verifier]
    VERIFY --> SYNTH[Synthesis]
```
