# Design Document: Concept lineage — one document, N pointers

> The unit of documentation is the **decision**, not the marker. This doc covers
> both halves of that change: the relation that lets a marker point at the
> decision it realises, and the tool that proposes which markers should.

CONCEPT:AU-OS.governance.concept-lineage-parent-doc ·
CONCEPT:AU-OS.governance.concept-domain-triage

## The measurement this starts from

`check_concept_governance.py --audit-merged` (D-RG2-3) asks one question of every
live `CONCEPT:<id>` marker: does a design document under `.specify/design/**`
mention it? On 2026-07-31 the answer was **1,039 live concepts with no design
document**, frozen by id in `scripts/concept_design_doc_baseline.txt`. Those
1,039 fall into just **80 `PILLAR.domain` groups**.

Read literally, the gate is demanding 1,039 design documents. Auditing the
largest group by hand refutes that reading. `AU-KG.compute` has **80 live
concepts**; they resolve to:

| Disposition | Count | What it is |
|---|---:|---|
| own design document | 15 | a real trade-off with a rejected alternative |
| pointer at a parent | 52 | a marker on code realising one of those 15 |
| retired | 12 | an id that never named a decision at all |

Twelve of the eighty markers were not concepts in any sense. Six were ids
slugified out of the sentence they sat in — `when-exposes` came from "*when* the
engine *exposes* the native...", `data-is-private-its` from "Data is
private-to-its-owner by default", `same-semantics-as` from "the *same semantics
as*...". Two (`kg-2`, `kg-3`) were bare citations of the retired `KG-2.NN`
numbering that OKF-CIS replaced, glued on by the migration as if they were names.
Two were single generic nouns (`vector`, `resolve`). One existed only as a string
literal in a test fixture (`code-feature`). None of them names a choice anyone
made.

Of the remainder, the clustering is stark: **eighteen** markers are per-connector
entity-type declarations in `source_sync.py`/`owl_bridge.py`/`mcp_tool.py`
(`dockerhub-repositories`, `home-assistant-states`, `uptime-kuma-monitors`, …),
all realising one decision about how a connector's entity types reach the graph.
**Five** are per-surface markers in `numeric/` realising one decision to drop
numpy/scipy for a compiled kernel. Writing eighteen documents for the first
cluster would mean writing seventeen restatements.

**A restatement is worse than no document.** It satisfies the gate, it looks like
documentation, and it teaches nothing — the exact failure the constitution's
fabrication warning names. So the rule needs a second way to be satisfied.

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| `AU-OS.governance.merged-concept-visibility-audit` | the `--audit-merged` mode that produced this backlog; same gate, opposite direction (it *finds* debt, this *classifies* it) | 0.55 | OS |
| `AU-OS.governance.concept-hierarchy-standardization` | owns the id grammar and the OKF path/IRI projection; lineage is a relation *between* ids, not a change to them | 0.40 | OS |
| `AU-AHE.evaluation.swallow-baseline-stable-key` | the id-keyed-ratchet argument reused verbatim here (a count-keyed ratchet cannot tell a new regression from an old fix) | 0.35 | AHE |

### Extension Analysis

- **Primary Extension Point**: `scripts/check_concept_governance.py` (the gate)
  and `agent_utilities/governance/` (where `domain_vocab.yaml` and
  `slug_registry.yaml` already live as hand-authored, machine-validated
  registries).
- **Extension Strategy**: augment the gate's documentation predicate; add one
  sibling registry alongside the two that already exist.
- **New Concept Required?**: Yes — two, because the relation and the tool that
  proposes uses of it are independently falsifiable. The relation is wrong if a
  pointer can hide an undocumented decision; the tool is wrong if its proposals
  need more checking than doing the work by hand.

## The relation

`agent_utilities/governance/concept_lineage.yaml`, loaded and validated by
`agent_utilities/governance/concept_lineage.py`:

```yaml
parents:
  AU-KG.compute.dockerhub-repositories:
    parent: AU-KG.compute.mcp-backed-dedicated-trackers
    rationale: one of twelve per-source entity-type declarations realising the
      single decision that a tracker reaches its upstream only through its MCP
      server
retired:
  AU-KG.compute.when-exposes:
    reason: the id was slugified out of the sentence "when the engine exposes
      the native ..." — it never named a decision
```

The gate then treats `has_design_doc(child) or has_design_doc(parent)` as
documented. Everything else in the design exists to stop that from becoming a
way to *hide* a decision, which is the only way this can do harm:

- **The parent must genuinely own a document, and must be a live concept.** A
  link into an undocumented or dead parent is reported by name as a violation,
  and — unlike an undocumented concept — the accepted-debt baseline never
  excuses it. It is not pre-existing debt; it is a claim of coverage that the
  tree contradicts.
- **One hop only.** A parent may not itself be a child. A chain reads as
  documented at every intermediate step while only its terminus is ever checked.
- **The rationale may not restate the id.** It must add at least eight words
  beyond those already in the child and parent ids. This is the anti-filler rule
  the design-doc requirement already carries, applied to the one-line form.
- **Retirement is a ratchet, not a deletion.** A retired id that acquires a live
  marker again fails the gate. Without that, "retire the marker" is indis-
  tinguishable from "someone deleted a line".
- **`--update-baseline` refuses to run** while any link is broken, so the
  "accept as known debt" button cannot launder a false claim of coverage.

The id-keyed baseline from `--audit-merged` is untouched: it still cannot be
satisfied by fixing an unrelated entry.

## The tool

`scripts/concept_domain_triage.py` processes one `PILLAR.domain` at a time and is
built around one constraint: **the operator does not want 79 more agents**, and
model tokens are the scarce resource.

Everything cheap happens first, with **zero model tokens**:

- **Marker sites** — one `git grep` for the domain prefix.
- **Git archaeology** — the commit that first *added* each marker, and the other
  concepts that commit introduced. One history walk per domain
  (`git log -G<prefix> -p`), not one per concept: ~9 s for eighty concepts versus
  ~4.5 minutes at ~3.5 s each.
- **Clustering** — connected components over a *thresholded* similarity graph:
  Jaccard overlap of source-file footprints, plus a link for any introducing
  commit narrow enough to be about these concepts. Raw "shares a file" was tried
  first and is unusable: hub modules (`source_sync.py`, `graph_compute.py`,
  `core/config.py`) each carry a dozen unrelated markers, and transitive closure
  fused 70 of `AU-KG.compute`'s 79 into one meaningless blob. Jaccard asks the
  question that actually matters — *are these two markers on the same body of
  code?*
- **Id shape** — prose-fragment, bare legacy pillar citation, or single generic
  noun; plus whether the marker text in source is truncated by the grammar.

Only what those signals cannot settle is escalated. `propose` emits a per-concept
proposal file with evidence and a suggested disposition; `packet` emits a compact
adjudication digest containing **only** the undecided residue plus one entry per
cluster — so confirming a 18-member cluster is one judgement, not eighteen.

Measured on `AU-KG.compute`: `propose` is ~12 s and zero tokens; the packet is
**~6.4k tokens** for the whole eighty-concept domain. Reading every marker and
module directly, which is how this pilot's ground truth was first established,
cost roughly an order of magnitude more.

`apply` is dry-run by default and does the typing only for decisions a human
already wrote down. It validates the *resulting* registry against
`parse_lineage` before writing, so an invalid registry never reaches disk, and it
refuses a `parent` decision whose parent has no document yet. Retirement edits
delete the marker token and tidy the line, but any line that would be left as an
empty husk is reported as `MANUAL` rather than mangled.

Both `propose` and `apply` are idempotent and preserve human decisions across
re-runs, so a domain can be picked up and put down.

### One trap worth recording

The triage output quotes real source lines, which contain real `CONCEPT:`
markers — and `--audit-merged` scans `.md`/`.yaml` under `reports/`. The first
run therefore *minted a live concept out of its own evidence file*
(`AU-KG.compute.homelab-rss-reader`, from a marker its 100-character excerpt had
cut short). Every emitted quote now has the prefix's colon replaced by a middle
dot. This is the same trap the gate's own meta-test hit (commit `38a9b918`); it
is a property of any tool that writes about markers.

## Data Flow

1. **ORCH**: no runtime participation — this is a commit-time gate plus a CLI.
2. **KG**: `build_concepts_yaml.py` and the RDF generator are unchanged; lineage
   is a governance relation over ids, not an ontology edge. Promoting it to
   `skos:broader` is deliberately **not** done here (see Risk).
3. **AHE**: none.
4. **ECO**: none.
5. **OS**: `check_concept_governance.py` runs in `pre-commit` and in the
   `concept-governance.yml` CI workflow; both gain the parent relation and the
   retirement ratchet.

## Risk Assessment

- **Blast Radius**: `scripts/check_concept_governance.py`,
  `agent_utilities/governance/` (one new module + one new registry),
  `tests/gates/test_concept_governance_gate.py`. No runtime code path.
- **Backward Compatible**: Yes. An empty registry reproduces the previous
  behaviour exactly, and the baseline file format is unchanged.
- **Breaking Changes**: None. Two *new* ways to fail (broken link, revived
  retirement) that were previously not expressible.
- **Deliberate non-goal — no `skos:broader` projection.** A parent link is an
  *editorial* statement about where a decision is written up. It is not a
  subsumption claim, and emitting it as `skos:broader` would put an assertion
  into the ontology that nobody validated as taxonomy. If the two ever need to
  agree, that is its own decision with its own document.
- **Deliberate non-goal — no rename.** The pilot found a fourth shape the three
  dispositions do not cover: a *junk id on real code with no sibling to absorb
  it* (`AU-KG.compute.data-is-private-its` marks a genuine private-by-default
  tenant-sharing design across six modules). Retiring it would delete real
  traceability; keeping it enshrines a sentence fragment as a name. Renaming a
  concept id touches the reservation ledger, the OKF file tree, and every marker
  site, so it is tracked as its own item (D-CC-1) rather than smuggled in here.
