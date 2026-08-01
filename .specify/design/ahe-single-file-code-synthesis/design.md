# Design Document: The evolution loop finally emits a real diff, bounded to one existing repo-relative file

CONCEPT:AU-AHE.harness.single-file-code-synthesis

> `agent_utilities/knowledge_graph/research/code_synthesis.py`.

## Decision — a promoted `kind="code"` proposal now produces an actual `{path, content}` edit, bounded to a single existing `.py` file

The module docstring (`code_synthesis.py:4-16`) states the gap this closes:
the deployed self-evolution loop could already promote, govern,
sandbox-validate, and branch a `kind="code"` change — but nothing on the
live path ever *emitted the diff*, so every real proposal fell back to the
`kind="sdd_plan"` prose skeleton and a human still wrote every line. This
module fills that hole: for a promoted proposal whose target file resolves,
it reads that file and produces a single-file edit, handed unchanged to the
existing `synthesize_change_set → validate_in_sandbox → change_publisher`
pipeline as `extra_files`.

**The rejected alternative is what existed before: code generation left
entirely to a human, with the loop only ever producing a prose plan.** A
second, deliberate constraint sits inside the same decision — the safety
envelope: **single, repo-relative, existing `.py` file only**, never
multi-file, never a new path. The rejected alternative here is unrestricted
multi-file or new-file generation from the first version of this
capability — a broader blast radius the sandbox-validation gate alone
wasn't judged sufficient to make safe by default. An unresolvable proposal
(the target file can't be located) yields `None` and falls back to the prose
skeleton exactly as before, so un-attributed proposals see zero behavior
change from this module existing.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/research/code_synthesis.py`,
  `agent_utilities/knowledge_graph/research/change_synthesis.py`,
  `agent_utilities/knowledge_graph/research/change_publisher.py`,
  `agent_utilities/knowledge_graph/enrichment/distill.py`.
- **Backward Compatible**: Yes — an unresolvable proposal degrades to the
  pre-existing prose-skeleton behavior; no existing path is removed.
- **Known weak point**: the generated file is sandbox-validated for syntax +
  import only — a diff that is syntactically valid and imports cleanly but
  is semantically wrong (doesn't actually fix the attributed issue) is not
  caught by this gate; it relies on the regression suite further down the
  pipeline to catch behavioral errors.
