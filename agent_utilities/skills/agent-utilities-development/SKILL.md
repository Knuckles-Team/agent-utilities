---
name: agent-utilities-development
skill_type: skill
description: >-
  Review or implement a concrete agent-utilities repository change. Use for
  read-only orientation, impact or diff review, concept-aware design, approved
  implementation, tests, wiring, REST and MCP parity, documentation, regression
  gates, and isolated worktree delivery. For evidence triage, gap proposals, or
  skill and prompt optimization before implementation approval, use
  agent-utilities-evolution.
---

# Agent Utilities development

Review a proposed change without mutation, or implement an approved change in an
isolated worktree and prove its live path and relevant gates.

## Workflow

### 1. Read the governing context

- Read every applicable `AGENTS.md` before editing.
- Inspect the owning architecture guide, specification, tests, and public entry
  points.
- Search existing concepts and implementations before adding a new abstraction.
- Preserve unrelated changes in a dirty worktree.

Use `graph-query-and-explanation` for code context and impact when the code graph
is available. Fall back to repository search when it is not.

### 2. Isolate and scope

- Translate the request into observable acceptance checks.
- Identify every owned consumer of any contract that will change.
- For orientation, design, impact, or diff review, inspect the current checkout
  read-only and report findings; stop before creating a branch, worktree, edit,
  commit, or other delivery artifact.
- For an authorized implementation, create a dedicated feature branch and
  worktree from the current main branch.
- Avoid compatibility aliases: update consumers atomically and delete the old
  path.

### 3. Implement at the owning seam

- Put shared behavior in the core, leaving transports as thin adapters.
- Keep REST and MCP entry points on the same action or service implementation.
- Make normal enhancements native to the existing flow unless their cost or
  risk requires an explicit control.
- Use Pydantic models for structured boundaries and existing dependency patterns.
- Keep examples synthetic and exclude credentials, private endpoints, personal
  data, and machine-specific paths.

### 4. Prove wiring

Trace from a real entry point to the changed behavior. Add a live-path test that
would fail if the new code were merely importable but never invoked. For dynamic
registration, verify the discovery call as well as the registered object.

### 5. Document the behavior

- Update the owning guide and Mermaid diagram.
- Update generated sources rather than hand-editing generated artifacts.
- Keep concept references, code, tests, and docs consistent.
- Update exact skill names and paths in prompts, fixtures, scripts, and docs.

### 6. Validate and deliver

Run the narrow tests first, then every gate touched by the change. Run the full
pre-commit suite before delivery. Inspect the final diff for generated churn,
stale names, sensitive data, and stray files. Commit with a neutral repository
identity; do not merge or push unless requested.

Use an economy model for inventory, search, mechanical edits, and deterministic
checks. Reserve stronger reasoning for ambiguous design, security review, and
cross-system synthesis.

## Skill changes

When editing bundled skills:

1. keep `SKILL.md` frontmatter to `name` and `description`;
2. generate `agents/openai.yaml` deterministically;
3. put Graph-OS coverage in `agents/graph-os.yaml`;
4. run the skill validator for every retained skill;
5. test both direct and delegated synthetic tasks;
6. update the coverage gate and current inventory documentation.

## Adding a new platform capability (engine → verb → route → skill)

Adding a capability end-to-end — reachable, documented, and discoverable with no
drift between layers — follows a fixed build order:

1. **Engine crate (Rust)**, when the capability needs native compute: implement it
   in the epistemic-graph engine and expose it as a wire `Method`
   (`crates/eg-types/src/protocol.rs`). The pure-Python `epistemic_graph` client
   mirrors the wire protocol 1:1, so a new method surfaces as a coroutine on a
   sub-client with no client-side hand-editing; that client is the source of truth
   for "what the engine can do."
2. **MCP verb + REST route.** A new engine method is auto-discovered by
   `engine_tools._discover_domains()` (client introspection) and appears under its
   domain's `engine_<domain>` action-routed tool automatically — a brand-new
   *domain* needs an entry in `_DOMAIN_CLASSES`/`_DOMAIN_BLURB`, with its REST twin
   `/engine/<domain>` registered in the same change (`ACTION_TOOL_ROUTES`). For a
   synthesized, agent-facing operation, add a curated `graph_*`/`ontology_*`/
   `object_*` tool and register its REST route in the SAME call so the
   surface-parity gate stays green (see *Two surfaces by default*).
3. **Wrapping skill.** Author or extend the domain skill covering the new verb so
   operators can discover it. The naming/coverage contract and the doctor that
   enforces it are documented in `graph-runtime-and-governance`'s "Coverage
   governance" section — run it as part of closing this out.

Verify the whole chain in one pass:

```bash
python -m agent_utilities.mcp.skill_coverage    # verb <-> skill coverage: 0 uncovered, 0 orphans
pytest tests/unit/test_gateway_mcp_parity.py    # tool <-> REST-route parity
python scripts/gen_graphos_manifest.py          # regenerate the action manifest from the client
```

A new verb shipped without covering documentation shows as **uncovered**; stale
coverage pointing at a removed verb shows as an **orphan** — fix both before merge,
or add the verb to the documented exemption list with a written justification.

## Guardrails

- Do not modify the shared main checkout for non-trivial work.
- Do not bypass failing gates or silently accept warnings.
- Do not create a second implementation for another entry point.
- Do not commit secrets, credential files, local inventories, or scratch output.
