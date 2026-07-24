---
name: autonomous-contribution
skill_type: skill
description: >-
  Autonomous PR generator for contributing local evolutionary breakthroughs
  (TeamConfigs, Skills) back to the upstream agent-packages ecosystem.
---

# Autonomous Contribution Skill

> **CONCEPT:AU-AHE.evaluation.backtest-harness Distributed Agentic Evolution**

This skill orchestrates the packaging and submission of locally evolved intelligence (new `SKILL.md` files or highly successful `TeamConfig` nodes) back to the central `agent-packages` repository.

## Triggers

This skill is invoked automatically by the `genius-agent` `--evolve` background daemon when a local `SelfImprovementCycle` yields a new artifact that passes the local verification threshold.

## Required Telemetry

When preparing a Pull Request, you MUST ensure that the payload contains the mandatory **AU-ECO.ui.company-infrastructure-orchestration Community Telemetry**:
1. `origin`: Set to `"community"`.
2. `timestamp`: The precise ISO-8601 timestamp of when the artifact was verified.
3. `author`: The deterministic origin hash of the agent that generated the artifact.
4. **Guardrail**: All new skills must include `Author: Autonomous` in the frontmatter of the generated `SKILL.md`.

## Workflow

1. Create the artifact: serialize the `TeamConfigNode` to JSON or format the `CallableResourceNode` metadata into a standard `SKILL.md` package, embedding the telemetry fields above.
2. Start a new branch prefixed with `evolve/` (e.g., `evolve/team-config-12345` or `evolve/skill-new-feature`) using `git_tools`.
3. Add the files and commit with a standard semantic commit message (e.g., `feat(evolution): add autonomous skill <name>`).
4. Use `github-tools` to push the branch to the remote origin and open a Pull Request against the main branch. The PR body must clearly explain the performance metrics (e.g., `composite_score`) that justified the promotion.

Use the skill directly to package and open a single artifact's PR. Delegate a
batch of several evolved artifacts so each is branched, committed, and opened
independently, keeping every submission behind the same human-review boundary.

## Human-in-the-Loop

Do not attempt to auto-merge the Pull Request. The central repository requires a human maintainer to review and approve all autonomous contributions before they are ingested globally via `engine_ingestion.py`. Use an economy model for the routine
packaging and formatting steps above; escalate to a stronger model only when the
performance metrics justifying the promotion are ambiguous.
