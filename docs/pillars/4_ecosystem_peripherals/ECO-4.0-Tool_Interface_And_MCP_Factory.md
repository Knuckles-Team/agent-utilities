# Provider Prompt Adaptation (CONCEPT:ECO-4.0)

## Overview
Abstracted-backend provider-aware prompt optimization with static and KG-backed rule storage. Built-in rules for OpenAI, Anthropic, Google with contextual activation. Based on Rosetta Prompt research.

## Implementation Details
- **Source Code**: ``agent_utilities/prompting/provider_adapter.py``
- **Pillar**: ECO

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Self-Describing Function Registry (CONCEPT:ECO-4.0)

## Overview
Runtime function registration with input/output JSON schemas and declarative trigger bindings (http/cron/event). Unified `discover_all_capabilities()` for AgentOS-style category collapse via KG.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/engine_registry.py``
- **Pillar**: ECO

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Dynamic Skill Evolution (CONCEPT:ECO-4.0)

## Overview
On-the-fly skill creation and synthesis to avoid catastrophic forgetting during continual learning. SkillNeologismDetector (identifies when existing skills don't cover a task), SkillFactory (creates new skills from execution traces), SkillMerger (detects overlapping skills via Jaccard similarity and consolidates). Derived from Skill Neologisms (arXiv:2605.04970v1).

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/skill_evolver.py``
- **Pillar**: ECO

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
