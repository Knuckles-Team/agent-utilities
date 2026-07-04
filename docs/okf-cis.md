# OKF-CIS — the concept-ID standard (READ THIS if you have an in-flight branch)

**As of the OKF-CIS cutover, concept ids use ONE grammar across every repo:**

```
<SLUG>-<PILLAR>.<domain>.<concept>[.<facet>...]
   AU-KG.ingest.entropy-dedup
   EG-KG.storage.redb
   DS-AHE.trainer.gpu-slot
```

The old `KG-2.134` / `ORCH-1.98` / `EG-352` / `PREFIX-NNN` forms are **retired** — the
grammar is regex-incompatible with them, so a legacy `CONCEPT:KG-2.7` marker now **fails
CI** (`scripts/check_no_legacy_markers.py`). This page tells you (1) how to write new
markers and (2) how to migrate an in-flight branch that still has old ones.

> **Why this matters for parallel sessions:** a branch cut *before* the cutover still has
> legacy markers. Merging it into `main` reintroduces them and breaks the no-legacy gate.
> Migrate your branch (below) before merging.

---

## 1. Writing a NEW concept (new sessions)

| Level | Rule | Source of truth |
|-------|------|-----------------|
| `SLUG` | 2-letter code of the repo that OWNS the concept | `agent_utilities/governance/slug_registry.yaml` |
| `PILLAR` | one of **ORCH · KG · AHE · ECO · OS · GBOT** (closed set) | `concept_hierarchy.PILLARS` |
| `domain` | a domain from the **closed vocab for that pillar** — never invent one | `agent_utilities/governance/domain_vocab.yaml` |
| `concept` | short semantic kebab slug (2-4 words, **no numbers**) | you |
| `facet` | optional deeper segments | you |

- SLUG = provenance (which repo), PILLAR = which of the 6 capability domains. They are
  independent: a data-science training concept is `DS-AHE.trainer.<x>` (slug DS, pillar AHE).
- **Adding a new `domain`** is governed: propose it, add it to `domain_vocab.yaml` (with
  signal keywords), then use it. This is the anti-sprawl gate — breadth is bounded, depth
  (facets) is free.
- Write the marker exactly as before, new grammar:
  `# CONCEPT:AU-KG.ingest.entropy-dedup — entropy-gated dedup on ingest.`
- **Reserve before writing** (unchanged protocol, see `concept_coordination.md`):
  `agent-utilities --json concept reserve --ns AU-KG.ingest --slug entropy-dedup`
- Gates that must stay green: `check_concepts` (marker↔registry parity), `check_domain_vocab`
  (slug registered + domain in closed vocab), `check_no_legacy_markers` (zero legacy).

Every id maps deterministically to an OKF bundle path (`AU/KG/ingest/entropy-dedup.md`) and
a resolvable RDF IRI (`http://knuckles.team/kg/concept/AU/KG/ingest/entropy-dedup`), so the
6 pillars federate cross-repo. Helpers: `concept_hierarchy.parse_okf_id` / `concept_iri` /
`okf_id_to_path`.

## 2. Migrating an IN-FLIGHT branch (do this before you merge to main)

Your branch has legacy markers (`CONCEPT:KG-2.x` etc.). Rewrite them to OKF-CIS:

```bash
# from your worktree, with agent-utilities on PATH:
AU=/home/apps/workspace/agent-packages/agent-utilities
PLAN=/home/apps/workspace/reports/okf-cis/working_plan.yaml   # the curated old->new map

# a) auto-curate any brand-new local ids your branch added (not in the plan):
python3 $AU/scripts/autocurate_repo.py <repo-name> <your-worktree> /tmp/mysupp.yaml
python3 - <<'PY'   # merge your supplement into the plan (skips already-mapped ids)
import yaml
base=yaml.safe_load(open("/home/apps/workspace/reports/okf-cis/working_plan.yaml"))
seen={e["old_id"] for e in base["entries"]}
for e in (yaml.safe_load(open("/tmp/mysupp.yaml")) or {}).get("entries",[]):
    if e["old_id"] not in seen: base["entries"].append(e)
yaml.safe_dump(base, open("/tmp/myplan.yaml","w"), sort_keys=False, width=100)
PY

# b) rewrite every marker + code-ref string in your worktree (idempotent):
python3 $AU/scripts/apply_concept_migration.py --plan /tmp/myplan.yaml \
    --repo <repo-name>=<your-worktree>

# c) verify + (agent-utilities only) regenerate the registry:
python3 $AU/scripts/check_no_legacy_markers.py <your-worktree>
python3 $AU/scripts/build_concepts_yaml.py     # AU only — regenerates docs/concepts.yaml
```

The applier is **idempotent** (a no-op on already-OKF ids), disambiguates formerly-collided
ids by file (else routes references to the concept's primary meaning), and skips history
files (`CHANGELOG.md`, `concepts.yaml`). Anything it can't map lands in a `.rej` report.

**Resolving an OLD id you see in a doc/memory/commit:** `working_plan.yaml` and
`reports/okf-cis/legacy_map.yaml` are the old→new lookup. The RDF also carries every legacy
id as a `:flatId` on its new `:GovernedConcept`, so historical references still resolve.

## 3. The whole toolchain (all on `main`)

- `governance/concept_hierarchy.py` — the ONE grammar (`OKF_MARKER_RE`, `parse_okf_id`,
  `concept_iri`), the domain-vocab + slug-registry loaders.
- `governance/domain_vocab.yaml`, `governance/slug_registry.yaml` — the closed vocab + slugs.
- `scripts/apply_concept_migration.py` — the marker rewriter. `scripts/autocurate_repo.py` —
  deterministic id→OKF for a repo's remaining legacy. `scripts/build_concept_rdf.py` —
  emits `ontology_concepts.ttl`. Gates: `check_no_legacy_markers.py`, `check_domain_vocab.py`.
- Full design: `/home/genius/.claude/plans/noble-greeting-teapot.md`; curated map +
  legacy_map + ttl in `workspace/reports/okf-cis/`.
