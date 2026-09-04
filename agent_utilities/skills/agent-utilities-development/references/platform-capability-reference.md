# Adding a new platform capability — reference

Deep reference for `agent-utilities-development`: the fixed engine -> verb ->
route -> skill build order for a capability that must be reachable, documented
and discoverable with no drift between layers, plus the one-pass verification
commands and the uncovered/orphan rule. The parent [`SKILL.md`](../SKILL.md)
keeps the guardrails, the anti-sprawl checklist and the workflow; the
*Two surfaces by default* rule cited in step 2 is a section of the repository's
`AGENTS.md`.

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
R="python3 scripts/uv_workspace.py run --all-extras --"   # never bare `uv run`
$R python -m agent_utilities.mcp.skill_coverage  # verb <-> skill coverage: 0 uncovered, 0 orphans
$R pytest tests/unit/test_gateway_mcp_parity.py  # tool <-> REST-route parity
$R python scripts/gen_graphos_manifest.py        # regenerate the action manifest from the client
```

A new verb shipped without covering documentation shows as **uncovered**; stale
coverage pointing at a removed verb shows as an **orphan** — fix both before merge,
or add the verb to the documented exemption list with a written justification.
