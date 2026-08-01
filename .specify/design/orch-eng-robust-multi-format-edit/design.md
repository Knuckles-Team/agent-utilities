# Design Document: A graduated fuzzy-match ladder with a reflection loop replaces brittle exact-string edit application

CONCEPT:AU-ORCH.execution.robust-multi-format-edit

> `agent_utilities/harness/edit_engine.py` — the only implementation file (2
> source files total; the second, `agent_utilities/harness/__init__.py`, is
> just the export list).

## The real decision

`edit_engine.py`'s own docstring states the problem and the chosen fix as
directly as any marker site in this batch:

> *"An exact `str.replace(old, new, 1)` operation is brittle. Any whitespace
> drift between what the model emitted and what is on disk makes it fail, with
> no recovery. Production coding harnesses (e.g. aider) instead parse one of a
> few well-known edit formats and apply them with a ladder of
> increasingly-forgiving matchers, then *reflect* on failures by re-prompting
> the model with did-you-mean hints. This module brings that capability
> natively to agent-utilities so our own Claude and every spawned coding
> sub-agent get materially higher edit-success rates."* (`edit_engine.py:9-16`)

Two formats are auto-detected from the text (`edit_engine.py:19-22`):
search/replace blocks (`<<<<<<< SEARCH` / `=======` / `>>>>>>> REPLACE`, with
the target filename on the line above) and standard unified diffs (`--- a/f` /
`+++ b/f` / `@@` hunks).

The matching ladder for search/replace blocks is graduated, applied in order
of increasing tolerance (`edit_engine.py:25-27`):

```
exact → leading-whitespace-flexible → drop-spurious-blank-line
      → "..." elision → SequenceMatcher closest-window
```

When even the closest-window fuzzy match fails, `apply_with_reflection`
re-prompts the model with did-you-mean hints derived from the failure, rather
than surfacing a bare "edit failed" to the caller — closing the loop back to
the model instead of stopping at the first unmatched edit.

The docstring is explicit about provenance: *"The algorithms are our own
implementation; the laddered strategy is inspired by aider's
`editblock_coder`."* This is an assimilated pattern (comparative-analysis /
research-evolution pipeline territory), reimplemented rather than vendored,
with an independent `EditOutcome`/`EditResult` result type
(`agent_utilities/harness/__init__.py:163-169`) that fits agent-utilities'
own capability-wiring conventions.

## The rejected alternative

The rejected alternative is named in the same sentence as the motivation: a
bare, single-shot `str.replace(old, new, 1)`. It is the simplest possible
implementation and the one that predates this module — it fails outright on
any whitespace drift between the model's proposed edit and what is actually
on disk, with **no recovery path**: one failed exact match is a failed edit,
full stop. The chosen design accepts real implementation complexity (five
tiers of matcher plus a reflection loop) specifically to convert "any
whitespace drift → hard failure" into "whitespace drift → try progressively
more forgiving matches → if all fail, ask the model for a corrected edit
before giving up." The trade-off accepted deliberately: a fuzzy match at the
`SequenceMatcher` closest-window tier could, in principle, apply an edit to a
location the model did not intend if the target text is ambiguous enough —
tolerance for whitespace drift is bought at some risk of matching the wrong
occurrence when a file has near-duplicate blocks.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/edit_engine.py`,
  `agent_utilities/harness/__init__.py`.
- **Backward Compatible**: Yes — an additive capability; nothing in the
  codebase is forced onto this path.
- **Known weak point**: the ladder's most forgiving tier
  (`SequenceMatcher` closest-window) trades exactness for recall — a file
  with two near-identical blocks (a common shape in generated boilerplate or
  copy-pasted test fixtures) is exactly the case where a "closest window"
  match is most likely to silently pick the wrong one instead of failing
  loudly enough to trigger the reflection loop.
