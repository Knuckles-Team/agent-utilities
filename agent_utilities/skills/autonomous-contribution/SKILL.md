---
name: autonomous-contribution
description: >-
  Autonomous PR generator for contributing local evolutionary breakthroughs
  (TeamConfigs, Skills) back to the upstream agent-packages ecosystem.
license: MIT
tags: [evolution, pr, git, github, upstream]
metadata:
  author: Genius
  version: '0.47.0'
---

# Autonomous Contribution Skill

> **CONCEPT:AHE-3.4 Distributed Agentic Evolution**

This skill orchestrates the packaging and submission of locally evolved intelligence (new `SKILL.md` files or highly successful `TeamConfig` nodes) back to the central `agent-packages` repository.

## Triggers

This skill is invoked automatically by the `genius-agent` `--evolve` background daemon when a local `SelfImprovementCycle` yields a new artifact that passes the local verification threshold.

## Required Telemetry

When preparing a Pull Request, you MUST ensure that the payload contains the mandatory **ECO-4.3 Community Telemetry**:
1. `origin`: Set to `"community"`.
2. `timestamp`: The precise ISO-8601 timestamp of when the artifact was verified.
3. `author`: The deterministic origin hash of the agent that generated the artifact.
4. **Guardrail**: All new skills must include `Author: Autonomous` in the frontmatter of the generated `SKILL.md`.

## Workflow

1. **Format Artifact**: Serialize the `TeamConfigNode` to JSON or format the `CallableResourceNode` metadata into a standard `SKILL.md` package. Ensure telemetry fields are embedded.
2. **Branching**: Use `git_tools` to check out a new branch prefixed with `evolve/` (e.g., `evolve/team-config-12345` or `evolve/skill-new-feature`).
3. **Commit**: Add the files and commit with a standard semantic commit message (e.g., `feat(evolution): add autonomous skill <name>`).
4. **Push & PR**: Push the branch to the remote origin and use `github-tools` to open a Pull Request against the main branch. The PR body must clearly explain the performance metrics (e.g., `composite_score`) that justified the promotion.

## Human-in-the-Loop

Do not attempt to auto-merge the Pull Request. The central repository requires a human maintainer to review and approve all autonomous contributions before they are ingested globally via `engine_ingestion.py`.
