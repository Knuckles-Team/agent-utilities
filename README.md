# Agent Utilities

[![GitHub Repo stars](https://img.shields.io/github/stars/Knuckles-Team/agent-utilities)](https://github.com/Knuckles-Team/agent-utilities/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Knuckles-Team/agent-utilities)](https://github.com/Knuckles-Team/agent-utilities/forks)
[![GitHub contributors](https://img.shields.io/github/contributors/Knuckles-Team/agent-utilities)](https://github.com/Knuckles-Team/agent-utilities/graphs/contributors)
[![GitHub license](https://img.shields.io/github/license/Knuckles-Team/agent-utilities)](LICENSE)
[![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/Knuckles-Team/agent-utilities)](https://github.com/Knuckles-Team/agent-utilities/commits/main)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/Knuckles-Team/agent-utilities)](https://github.com/Knuckles-Team/agent-utilities/pulls)
[![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/Knuckles-Team/agent-utilities)](https://github.com/Knuckles-Team/agent-utilities/pulls?q=is%3Apr+is%3Aclosed)
[![GitHub issues](https://img.shields.io/github/issues/Knuckles-Team/agent-utilities)](https://github.com/Knuckles-Team/agent-utilities/issues)
[![GitHub top language](https://img.shields.io/github/languages/top/Knuckles-Team/agent-utilities)](https://github.com/Knuckles-Team/agent-utilities)
[![GitHub language count](https://img.shields.io/github/languages/count/Knuckles-Team/agent-utilities)](https://github.com/Knuckles-Team/agent-utilities)
[![GitHub repo size](https://img.shields.io/github/repo-size/Knuckles-Team/agent-utilities)](https://github.com/Knuckles-Team/agent-utilities)
[![GitHub repo file count (file type)](https://img.shields.io/github/directory-file-count/Knuckles-Team/agent-utilities)](https://github.com/Knuckles-Team/agent-utilities)
[![PyPI - Version](https://img.shields.io/pypi/v/agent-utilities)](https://pypi.org/project/agent-utilities/)
[![PyPI - Downloads](https://img.shields.io/pypi/dd/agent-utilities)](https://pypi.org/project/agent-utilities/)
[![PyPI - License](https://img.shields.io/pypi/l/agent-utilities)](https://pypi.org/project/agent-utilities/)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/agent-utilities)](https://pypi.org/project/agent-utilities/)
[![PyPI - Implementation](https://img.shields.io/pypi/implementation/agent-utilities)](https://pypi.org/project/agent-utilities/)
[![Build](https://github.com/Knuckles-Team/agent-utilities/actions/workflows/release.yml/badge.svg)](https://github.com/Knuckles-Team/agent-utilities/actions/workflows/release.yml)
[![Documentation](https://github.com/Knuckles-Team/agent-utilities/actions/workflows/pages.yml/badge.svg)](https://knuckles-team.github.io/agent-utilities/)

## Overview

Agent Utilities is the Python control plane for building, coordinating, evaluating, and improving AI agents. It provides agent and workflow execution while relying on GraphOS for public service composition and epistemic-graph for durable knowledge.

*Version: 2.5.0*

## Key Capabilities

- Build agents with model providers, skills, tools, and structured outputs.
- Coordinate plans, teams, workflows, and durable execution.
- Carry identity, context, budgets, approvals, and policy through agent work.
- Evaluate outcomes and produce reviewable improvement proposals.

## Documentation

Start at the [Agent Utilities documentation](https://knuckles-team.github.io/agent-utilities/). It includes the [quick start](https://knuckles-team.github.io/agent-utilities/guides/quick-start/), [architecture](https://knuckles-team.github.io/agent-utilities/architecture/), and release-aware [capability status](https://knuckles-team.github.io/agent-utilities/status/).

## Architecture

GraphOS owns public MCP, REST, and A2A composition. Agent Utilities owns agent and workflow behavior. The connector SDK owns source transport, while epistemic-graph owns durable graph state, schemas, and reasoning.

## Quick Start

Python 3.12 or newer is required. Install the serving extra, generate and check a local profile, then launch the local MCP server:

```bash
uvx --from "agent-utilities[serving]" setup-config generate --profile tiny
uvx --from "agent-utilities[serving]" graph-os --transport stdio
```

The [quick start guide](https://knuckles-team.github.io/agent-utilities/guides/quick-start/) covers provider configuration and other deployment profiles.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md) for contribution and validation guidance.

## License

Agent Utilities is released under the [MIT License](LICENSE).
