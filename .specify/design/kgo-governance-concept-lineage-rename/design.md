# Design Document: RENAME as a fourth, first-class concept-lineage disposition, not a hand source-edit + regen

CONCEPT:AU-OS.governance.concept-lineage-rename

> `agent_utilities/governance/concept_lineage.py:138-145` (`Renamed`
> dataclass), `:159` (`Lineage.renamed` field), `:166-172`
> (`Lineage.resolve`), `:283-330` (`parse_lineage`'s `renamed:` parsing,
> self-rename/chain/disjointness validation) and
> `scripts/check_concept_governance.py:315-326` (`reintroduced_renames`, the
> ratchet). `scripts/concept_domain_triage.py:748-780` (`_apply_rename`,
> the site-substitution writer) and `:861-926` (the `rename` decision branch
> in `cmd_apply`, including the target-flattening logic). D-CC-1, D-CC-9.

## Decision — a fourth disposition, `rename`, that rewrites every marker site in one step and records old-to-new provenance, so the mechanism itself does the typing instead of a hand source-edit plus a full regen

`concept_domain_triage.py` (the mechanised domain-triage tool built for the
`AU-KG.compute` pilot, D-CC-2) shipped with exactly three dispositions:
`document` (write a design doc), `parent` (point at one that already exists)
and `retire` (delete a marker that never named a decision). None of the three
fits a marker that names a REAL decision but simply needs a DIFFERENT id —
most often because its domain word is not, or is no longer, in the closed
vocabulary (`agent_utilities/governance/domain_vocab.yaml`,
`scripts/check_domain_vocab.py`). `document` is wrong because there is nothing
new to write — the decision is already recorded, just under a name that fails
a different gate. `retire` is wrong because the marker did not name nothing —
deleting it and recording a retirement would make `reintroduced_retirements`
permanently reject the (correct, renamed) marker's return under a legal id,
since retirement's own ratchet has no way to say "this concept still exists,
just elsewhere."

Before this fix, the gap was not hypothetical: `AU-KG.trace.canonical-id-non-idempotence`
(`trace` was never a registered `KG` domain) had to move to
`AU-KG.identity.canonical-id-non-idempotence` by hand — a source edit plus a
full `docs/concepts.yaml` regeneration via `build_concepts_yaml.py`, done
correctly (never a hand-edit of the generated view) but with no governance
record of the move at all. D-CC-9 hit the identical shape independently
(`AU-KG.history.scoped-conversation-search`, `history` unregistered) — two
lanes, same gap, confirming it recurs rather than being a one-off.

`rename` closes it. `concept_domain_triage.py apply` (`:861` onward) is the
only writer, mirroring how `retire` and `parent` are already applied:

1. **Validate the target**, not just its syntax: `rename_to` must parse as a
   legal OKF-CIS id (`parse_okf_id`) AND its domain must itself be in the
   closed vocab (`is_valid_domain`) — renaming out of one illegal domain into
   another is refused (`:882-889`), since that fixes nothing.
2. **Flatten, never chain** (`:891-897`): if the requested target is itself the
   OLD id of an existing rename entry, the tool follows to that entry's final
   `to` and uses THAT as the real target — so the table never grows a second
   hop no matter how many times a concept is renamed over its lifetime.
   Symmetrically (`:900-907`), anything already pointing INTO the id being
   renamed is rewritten to point at the final target directly
   (`lineage reflatten`), so a later rename of an already-renamed id never
   leaves a dangling one-hop-too-many chain behind it. `parse_lineage`
   (`concept_lineage.py:315-322`) additionally rejects any chain that slips
   through at load time, the same one-hop invariant `parent` links already
   enforce and for the same reason: "where does this id live now?" must stay
   a single lookup (`Lineage.resolve`, `:166-172`), never a graph traversal.
3. **Refuse a collision** (`:889` region, `final_target in all_live`): a
   rename target that already exists as an independent live concept is a
   MERGE, not a rename, and this mechanism deliberately does not attempt to
   merge two decisions' histories — that is a harder, separate problem.
4. **Rewrite every site in one step, verbatim substitution only**
   (`_apply_rename`, `:748-780`): unlike `_apply_retirement`, which deletes
   content and must tidy the resulting prose, a rename only swaps the id
   token (`CONCEPT:<old>` -> `CONCEPT:<new>`) — source, tests, and any design
   doc that quotes the id literally, since `collect_sites`'s full-tree
   `git grep` sweep already includes `.specify/design/**`. This is why a
   concept's own existing design doc (if it had one under the old name)
   keeps satisfying `has_design_doc(new_id)` for free, with no separate
   "also go update the doc" step.
5. **Record provenance, not just apply the edit**: `renamed[cid] = {"to":
   final_target, "reason": reason}` is validated (`parse_lineage`) BEFORE any
   site is touched — the same "validate before writing" ordering `retire`
   already used, so a run that would produce an invalid registry cannot have
   already mutated source. `check_concept_governance.py`'s
   `reintroduced_renames` (`:315-326`) then makes the move a ratchet: a live
   marker reappearing under the OLD id fails `--audit-merged` by name, the
   same property `reintroduced_retirements` gives retirement, so a rename
   cannot be silently undone by someone reintroducing the old spelling.

**The rejected alternative is the status quo this item was opened against**:
keep doing renames as an ad hoc source-edit plus a full `docs/concepts.yaml`
regeneration, with no registry entry at all. That was tried twice
independently (the live `AU-KG.trace` -> `AU-KG.identity` precedent, and this
document's own D-CC-9 fix before this mechanism existed) and both times left
zero machine-checkable record that the move happened — a stale baseline entry,
an old report, or a hand-written note holding the old id would have no way to
learn where the decision went, and nothing would stop the old id from quietly
coming back. A second rejected shape, folding rename into `retire` (`retire`
the old id, `document` the new one as if unrelated), was rejected because it
manufactures a duplicate "decision" out of one — the new marker would need its
own justification for why it names something new, when it does not.

## Risk Assessment

- **Blast Radius**: `agent_utilities/governance/concept_lineage.py`,
  `agent_utilities/governance/concept_lineage.yaml` (schema gains a third
  top-level key, `renamed:`, defaulting to empty — every existing
  `Lineage(parents=..., retired=...)` call site keeps working via the
  dataclass default), `scripts/check_concept_governance.py`,
  `scripts/concept_domain_triage.py`.
- **Backward Compatible**: Yes. `renamed:` is optional in the YAML (`data.get("renamed")
  or {}`); a registry written before this change parses unchanged, and
  `Lineage.renamed` defaults to `{}` for any caller that does not pass it.
- **Known weak point**: the collision check (`final_target in all_live`) reads
  the live-marker universe computed at the START of one `apply` invocation; it
  cannot see a target that another concurrent lane's rename or new marker
  introduces mid-run. This mirrors the same race every other `apply` decision
  already accepts (the tool is not transactional across lanes) and is bounded
  by the domain-triage workflow being per-domain and reviewed before
  `--write`, not by anything this disposition adds.
