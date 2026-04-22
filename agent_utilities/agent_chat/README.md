# Agent Chat Utilities

This directory contains utilities for processing and enhancing agent-user chat interactions.

## Overview

Chat utilities handle high-level prompt parsing, mention resolution, and history management. They act as a bridge between raw user input and the structured execution layer.

## Key Features

### 1. Codemap Mentions (`parser.py`)
- **Syntax**: Supports `@codemap{slug-or-id}` in user prompts.
- **Functionality**: Automatically resolves codemap mentions by querying the Knowledge Graph and injecting the structural artifact into the agent's context.

## Maintenance

- **Regex Patterns**: If adding new mention types (e.g. `@kb`, `@agent`), update the patterns in `parser.py`.
- **Performance**: Ensure that mention resolution is fast and cached where possible to prevent latency during chat initialization.
