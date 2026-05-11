# DSTDD-Pipeline: Design-Spec-Test Driven Development

## Overview
This specification formalizes the **Design-Spec-Test Driven Development (DSTDD)** cycle to enforce the "Extend Before Invent" governance across the `agent-utilities` ecosystem.

## Workflow Pipeline
1. **Design Phase**:
   - Any new feature or external code ingestion starts in the `.specify/design/` directory.
   - An agent must query the Knowledge Graph using the `kg_analogy_search` MCP tool to find analogous concepts.
   - The design artifact (Mermaid C4 / connection diagram) is generated and stored here.
2. **Spec Phase**:
   - Decompose the design into a Spec artifact.
   - The Spec MUST reference an existing Pillar (ORCH, KG, AHE, ECO, OS) or explicitly justify a new Concept tag.
3. **Test Phase**:
   - Auto-generate TDD tests against the Spec.
   - Ensure the implementation is validated by the integration test suite.

## Knowledge Graph Enforcement
The pipeline is governed by the Knowledge Graph (KG). No PR or major merge should occur if a new `CONCEPT:` tag is introduced without a matching node in the KG and Pillar reference.
