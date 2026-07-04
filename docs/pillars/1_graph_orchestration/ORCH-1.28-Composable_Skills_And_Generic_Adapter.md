# Composable Skills + Generic Environment Adapter (CONCEPT:AU-ORCH.adapter.composable-skills-environment)

## Overview

Upgrades the RLM's skill-as-SOP from raw source-mounting to **composable Skill units** and adds a
**generic environment adapter** so an external environment runs through a minimal tool surface while
its evaluator stays intact ("less harness"). Assimilated from predict-rlm (`Trampoline-AI/predict-rlm`)
and the AppWorld RLM-GEPA thesis. Extends **ORCH-1.12** (Predict-RLM Runtime).

## How it works

- **Composable `Skill`** (`rlm/skills.py`) bundles `instructions` + `packages` + `modules` + `tools`.
  `merge_skills` order-preserving-dedups packages, concatenates instructions under `## Skill:` headers,
  and **raises on module/tool name conflicts** (explicit composition, no silent last-wins).
  `PredictRLM.mount_skill_unit(skill)` mounts a unit and appends its instructions.
- **Generic adapter** (`RegistryEnvironmentAdapter`) exposes a small `list_items / describe / call /
  submit` surface over a callable registry; `submit` hands off to the **host-owned evaluator** — the
  RLM supplies only the policy (the optimizable skill), the host owns state + scoring.

## Key files / API

| Piece | Location |
|---|---|
| Skills + adapter | `rlm/skills.py` (`Skill`, `merge_skills`, `EnvironmentAdapter`, `RegistryEnvironmentAdapter`) |
| Mount | `rlm/predict_rlm.py` (`PredictRLM.mount_skill_unit`); `rlm/runner.py` (`run_rlm(skills=...)`) |

## Wiring (≤3 hops)
`graph_orchestrate(action="rlm_run", skills=...)` → `run_rlm` → `merge_skills` / `mount_skill_unit` (2 hops).

## Research provenance
predict-rlm `src/predict_rlm/rlm_skills.py` (`merge_skills`, conflict detection) — verified.
