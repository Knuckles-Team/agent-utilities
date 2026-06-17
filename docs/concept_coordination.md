# Multi-Session Concept-ID Coordination (CONCEPT:OS-5.42)

> How many Claude sessions/worktrees claim concept ids without colliding, and how
> the generated registries merge without overwriting each other.

## The problem

Concept ids — the `CONCEPT:<PILLAR>-<n>` markers in code (`KG-2.101`, `AHE-3.49`,
`OS-5.42`, and the per-package `KEY-001`/`GL-003` …) — are the **single contended
resource** when several sessions work the ecosystem in parallel. Each session runs
in its own git worktree under `/home/apps/worktrees/`, all merging to a shared
`main`. The marker is the source of truth and `docs/concepts.yaml` is generated
from it, so historically a session picked "the next number after the current max"
by eye. Two sessions reading the same max both pick `.101`, collide at merge, and
one is forced to renumber.

Prose ("please don't collide") cannot prevent a race. This protocol replaces it
with an **atomic claim** plus a **merge-safe registry**.

## How a claim is made atomic

A claim reserves the next free id in a *namespace* and is serialized so two
callers can never mint the same id.

* **The "taken" set is the union of three sources** — markers already in code,
  ids already in `docs/concepts.yaml`, and *open reservations*. Counting open
  reservations is what closes the cross-worktree race: an in-flight claim in
  another worktree is counted before its marker ever lands.
* **Reservations live in a committed, line-oriented ledger** —
  `docs/concept_reservations.yaml`, one reservation per physical line. Because
  the file is committed and line-oriented, concurrent worktrees reconcile through
  a `merge=union` git driver (see below) instead of overwriting each other.
* **Within a host the read-modify-write is serialized by an `fcntl.flock`** — the
  same primitive `KGCoordinator.spawn_server()` uses for KG-server election. The
  lock is per-repo (keyed by repo path) so distinct ledgers never block each other.

The ledger is **authoritative for claiming** — it works offline and across
worktrees. When the graph-os gateway (`:8100`) is healthy, a reservation is
*additionally* projected into the Knowledge Graph as a `ConceptReservation` node
for queryability; that projection is best-effort and never gates the claim.

Implementation: `agent_utilities/governance/concept_allocator.py`.

## Namespaces

| Kind | Example | Next-id form |
|------|---------|--------------|
| Pillar (agent-utilities) | `KG-2`, `OS-5`, `AHE-3` | dotted: `KG-2.102` |
| Package prefix | `KEY`, `GL`, `SNOW` | zero-padded: `KEY-004` |

Pass the **major number** for pillars (`KG-2`, not `KG`). Package prefixes are the
letters-only codes in `agent-packages/scripts/generate_concepts.py` (`PREFIX_MAP`).

## Claiming an id

**Before you write a new `CONCEPT:` marker, reserve it.**

CLI (offline-capable — no gateway needed, the right tool inside a worktree):

```bash
# Reserve the next KG-2 id, recording the design doc that justifies it:
agent-utilities --json concept reserve --ns KG-2 --design-doc .specify/design/my-feature/

# Inspect / free / reconcile:
agent-utilities --json concept list
agent-utilities --json concept release --id KG-2.150
agent-utilities --json concept reconcile

# A per-package repo's ledger (pass its repo root):
agent-utilities --json concept reserve --ns KEY --repo agent-packages/agents/keycloak-agent
```

MCP / REST (when driving through graph-os):

```
concept_registry(action="reserve", namespace="KG-2", design_doc="…")   # MCP tool
POST /concept/registry  {"action":"reserve","namespace":"KG-2"}        # REST twin
```

Then write the marker in code using the id you were given, and (optionally) run
`build_concepts_yaml.py` — it calls `reconcile()` and flips your reservation from
`reserved` to `landed` automatically.

## Reservation lifecycle

```
reserve ──► reserved ──(marker lands in code)──► landed
                │
                └──(TTL elapses, default 24h)──► expired  (id is freed for reuse)
```

`reconcile()` runs inside `build_concepts_yaml.main()` so the ledger self-cleans on
every registry regeneration. A `landed` reservation still counts as taken; an
`expired` or `released` one frees its id.

## Merging without overwriting

`.gitattributes` (in agent-utilities, and emitted into each package repo):

```
docs/concept_reservations.yaml  merge=union
docs/concepts.yaml              merge=concepts-regen   # regenerate, don't hand-merge
```

* **The ledger uses the built-in `union` driver.** Two worktrees that each append a
  distinct reservation merge with both lines intact and no conflict. (Proven by a
  divergent-merge test in `tests/`.)
* **`concepts.yaml` is generated, so it is regenerated on merge, not hand-merged.**
  The `concepts-regen` driver (registered by `scripts/install.sh` via
  `git config merge.concepts-regen.driver`) re-derives it from the merged code
  tree with `build_concepts_yaml.py`. The code markers are the real conflict
  surface and conflict visibly in `.py`/`.rs`. If the driver is not installed, git
  falls back to a normal merge and CI's `check_concepts.py` backstops staleness —
  so a stale hand-merge cannot land.

## Where the protocol is documented (one source, generated pointers)

This file is the **single canonical description**. It is not duplicated into the
~70 per-package `AGENTS.md` files (which are standalone and would drift). Instead a
short, marker-delimited pointer block (`<!-- BEGIN/END concept-coordination -->`)
is injected by the generators:

* `scripts/gen_agents_md.py` → agent-utilities `AGENTS.md`.
* `agent-packages/scripts/generate_ecosystem_docs.py` → each package `AGENTS.md`
  (and emits the per-repo `.gitattributes` block).
* `agent-packages/scripts/generate_concepts.py` → each package `docs/concepts.md`.

## CI governance

`.github/workflows/concept-governance.yml` (the Extend-Before-Invent gate) derives
its valid-pillar allowlist from `docs/concepts.yaml` (so it can never drift behind
the registry) and matches the canonical marker grammar. A new tag still requires a
design doc under `.specify/design/`. Pair that with a reservation so the id you put
in the design doc is one no other session can take.

## Related

* `docs/centralized_kg_coordination.md` (CONCEPT:KG-2.5) — runtime/data-plane
  coordination (gateway election, backpressure). This file is the *source-plane*
  (concept-id / git) counterpart.
* `AGENTS.md` → "Working with Git Worktrees" — the worktree workflow this builds on.
