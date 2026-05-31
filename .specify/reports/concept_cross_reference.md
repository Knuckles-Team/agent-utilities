# Concept Cross-Reference: Research Papers × Agent-Utilities Concept Map

> Generated: 2026-05-15 | Method: KG L1 Vector Discovery (6 search dimensions)
> Papers: MEMO Survey, ParamMem, LatentRAG, MINER, SIRA, MemReranker + Safety Scaling

## Pillar 1: Graph Orchestration Engine (ORCH)

| Concept ID | Name | MEMO | ParamMem | LatentRAG | MINER | SIRA | MemReranker | Safety | Gap? |
|---|---|---|---|---|---|---|---|---|---|
| `ORCH-1.0` | Intelligence Graph Core | — | — | — | — | — | — | — | — |
| `ORCH-1.1` | HTN Planning Pipeline | ○ skill distillation | — | — | — | — | — | ○ planning | — |
| `ORCH-1.2` | Specialist Routing | ○ Mem0g routing | — | — | — | ● single-query | — | — | — |
| `ORCH-1.3` | Execution Safety | — | ● poisoning defense | — | — | — | — | ● scaling laws | — |
| `ORCH-1.4` | Capability Wiring | — | — | — | — | — | — | — | — |
| `ORCH-1.5` | Agent Orchestrator | ○ multi-agent | ● coordinated teams | — | — | — | — | ● ensemble failures | — |
| `ORCH-1.6` | DSTDD Pipeline | — | — | — | — | — | — | — | — |

## Pillar 2: Epistemic Knowledge Graph (KG)

| Concept ID | Name | MEMO | ParamMem | LatentRAG | MINER | SIRA | MemReranker | Safety | Gap? |
|---|---|---|---|---|---|---|---|---|---|
| `KG-2.0` | Active Knowledge Graph | ● graph memory | ● parametric KG | ○ latent KG | ○ representation | — | — | — | — |
| `KG-2.1` | Tiered Memory & Context | ● CLS synthesis | ● fast/slow memory | — | — | — | ● MemOS rerank | — | **YES — synthesis loop** |
| `KG-2.2` | Ontology & Epistemics | — | — | — | — | — | — | — | — |
| `KG-2.3` | Graph Integrity & Retrieval | ○ retrieval patterns | — | ● 90% latency cut | ● 4.5% nDCG boost | ● sketch retrieval | — | ○ RAG evaluation | **YES — AutoRefine** |
| `KG-2.4` | Inductive Knowledge | — | — | — | ○ layer probing | — | — | — | — |
| `KG-2.5` | Topological Analysis | — | — | — | — | — | — | — | — |
| `KG-2.6` | Domain: Finance | — | — | — | — | — | — | — | — |
| `KG-2.7` | Research Intelligence | — | — | — | — | — | — | — | **YES — assimilation tracking** |
| `KG-2.8` | Domain: Enterprise | — | — | — | — | — | — | — | — |
| `KG-2.9` | External Graph Federation | — | — | — | — | — | — | — | — |

## Pillar 3: Agentic Harness Engineering (AHE)

| Concept ID | Name | MEMO | ParamMem | LatentRAG | MINER | SIRA | MemReranker | Safety | Gap? |
|---|---|---|---|---|---|---|---|---|---|
| `AHE-3.0` | Agentic Harness Core | — | — | — | — | — | — | — | — |
| `AHE-3.1` | Continuous Evaluation | ○ evaluation | — | — | — | — | ○ eval | ● eval scaling | — |
| `AHE-3.2` | Agentic Evolution | ○ evolution patterns | — | — | — | — | — | — | — |
| `AHE-3.3` | Team & Synergy | — | ● consensus | — | — | — | — | ● ensemble | — |
| `AHE-3.4` | Distributed Evolution | — | — | — | — | — | — | — | — |
| `AHE-3.5` | Heavy Thinking | ○ background | ○ sleep-time compute | — | — | — | — | ○ inference | — |
| `AHE-3.6` | Backtest & Curriculum | — | ● CGT benchmark | — | — | — | — | — | — |

## Pillar 4: Ecosystem & Peripherals (ECO)

| Concept ID | Name | MEMO | ParamMem | LatentRAG | MINER | SIRA | MemReranker | Safety | Gap? |
|---|---|---|---|---|---|---|---|---|---|
| `ECO-4.0` | Tool Interface & MCP | — | — | — | — | — | — | — | — |
| `ECO-4.1` | A2A Network | — | — | — | — | — | — | — | — |
| `ECO-4.2` | Community Telemetry | — | — | — | — | — | — | — | — |
| `ECO-4.3` | Market Data Connectors | — | — | — | — | — | — | — | — |
| `ECO-4.4` | KG MCP Server | ○ MCP memory | — | — | — | — | — | — | — |
| `ECO-4.5` | Terminal Agent Launcher | — | — | — | — | — | — | — | — |

## Pillar 5: Agent OS Infrastructure (OS)

| Concept ID | Name | MEMO | ParamMem | LatentRAG | MINER | SIRA | MemReranker | Safety | Gap? |
|---|---|---|---|---|---|---|---|---|---|
| `OS-5.0` | Agent OS Kernel | — | — | — | — | — | — | — | — |
| `OS-5.1` | Security & Auth | — | ● memory poisoning | — | — | — | — | ● safety scaling | — |
| `OS-5.2` | Resource Scheduling | — | — | — | — | — | ○ MemOS resource | — | — |
| `OS-5.3` | Guardrails & Safety | — | — | — | — | — | — | ● safety ≠ accuracy | — |
| `OS-5.4` | Telemetry & Observability | — | — | — | — | — | — | — | — |

## Legend

- **●** = Strong match (score > 0.50, multiple signals, directly applicable)
- **○** = Moderate match (score 0.35–0.50, partial overlap)
- **—** = No significant match
- **Gap?** = Identified gap where research finding reveals missing capability

## SDD Features Created from Gaps

| Gap | Concept | SDD Feature | Research Sources |
|-----|---------|-------------|-----------------|
| Missing synthesis loop | KG-2.1 | `kg-2.1-memory-synthesis` | MEMO Survey, ParamMem, MemReranker |
| No AutoRefine post-retrieval | KG-2.3 | `kg-2.3-latentrag-retrieval` | LatentRAG, MINER, SIRA |
| No assimilation tracking | KG-2.7 | `kg-2.7-research-assimilation` | (meta — this pipeline) |
