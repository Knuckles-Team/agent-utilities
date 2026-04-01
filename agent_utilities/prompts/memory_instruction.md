# Memory Instruction (Agent.md System)

**Observed in**: Assistant internal architecture
**Variable:** `MEMORY_INSTRUCTION_PROMPT`

## Purpose

The meta-instruction that wraps all loaded Agent.md memory files in the system prompt. This single line establishes that user-provided instructions take absolute precedence over default behavior.

## Prompt

```
Codebase and user instructions are shown below. Be sure to adhere to these instructions.
IMPORTANT: These instructions OVERRIDE any default behavior and you MUST follow them
exactly as written.
```

## Memory File Loading Order

Files are loaded in reverse order of priority (latest = highest priority):

1. **Managed memory** (`/etc/Agent-code/Agent.md`) — Global instructions for all users
2. **User memory** (`~/.Agent/Agent.md`) — Private global instructions for all projects
3. **Project memory** (`Agent.md`, `.Agent/Agent.md`, `.Agent/rules/*.md` in project roots) — Checked into codebase
4. **Local memory** (`Agent.local.md` in project roots) — Private project-specific instructions

## File Discovery

- User memory is loaded from `~/.Agent/`
- Project and Local files are discovered by traversing from the current directory up to root
- Files closer to the current directory have higher priority (loaded later)
- `Agent.md`, `.Agent/Agent.md`, and all `.md` files in `.Agent/rules/` are checked in each directory

## @include Directive

Memory files support transitive file inclusion:

- Syntax: `@path`, `@./relative/path`, `@~/home/path`, or `@/absolute/path`
- Works in leaf text nodes only (not inside code blocks)
- Circular references are prevented by tracking processed files
- Non-existent files are silently ignored
- Maximum include depth: 5
- Only text file extensions are allowed (prevents loading images, PDFs, etc.)

## Frontmatter Support

Memory files support YAML frontmatter with a `paths` field for conditional injection:

```yaml
---
paths:
  - src/components/**
  - "*.tsx"
---
```

Files with `paths` frontmatter are only injected when the active file matches the glob patterns.

## Configuration

- `MAX_MEMORY_CHARACTER_COUNT`: 40000 characters (recommended maximum per file)
- `AgentMdExcludes` setting: Glob patterns to exclude specific Agent.md files
- `MAX_INCLUDE_DEPTH`: 5 levels of transitive inclusion
- HTML comments in memory files are stripped before injection
