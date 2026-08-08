# Design Document: One shared, training-free stopper terminates every iterative retrieval loop — no algorithm re-derives its own halting rule

CONCEPT:AU-KG.retrieval.adaptive-stopping-iterative-retrieval ·
CONCEPT:AU-KG.retrieval.iterative-expansion-adore ·
CONCEPT:AU-KG.retrieval.assimilated-from-mragent

> `agent_utilities/knowledge_graph/retrieval/adaptive_stopping.py` (the shared
> primitive), consumed by `agent_utilities/knowledge_graph/retrieval/
> iterative_expansion.py` (ADORE) and `agent_utilities/knowledge_graph/
> retrieval/active_reconstruction.py` (MRAgent).

## Decision — `IterativeStopper` is the ONE halting policy every multi-round retrieval loop delegates to

`CONCEPT:AU-KG.retrieval.adaptive-stopping-iterative-retrieval`

`adaptive_stopping.py:4-29` names the two alternatives it replaces directly:
"An iterative retrieve→answer loop normally runs a fixed number of rounds or
relies on an LLM self-judgement to decide it is 'done'; both waste rounds (and
tokens) and neither is grounded." TASR's (arXiv 2606.x) training-free,
deterministic observation is the decision: **halt the moment the model repeats
its previous answer** — once an extra round of evidence stops moving the
answer, further rounds are redundant.

**The rejected alternatives, named explicitly in the docstring:**
1. **A fixed round count** — wastes rounds when the loop converges early, and
   under-runs when it doesn't.
2. **LLM self-judgement of "done"** — an extra model call per round, and
   ungrounded (the model's opinion of its own progress, not a measured signal).

The module adds two complementary guards on top of the primary TASR rule —
**coverage saturation** (stop when a round surfaces fewer than
`min_new_evidence` new ids for `patience` consecutive rounds) and **max
rounds** (a hard cap so a pathological loop always terminates) — all as one
pure state machine (`IterativeStopper.update` → `StopDecision`), "no model, no
I/O, no environment," so termination logic is unit-testable independent of
whatever retrieval algorithm is looping.

### Pointer — `CONCEPT:AU-KG.retrieval.iterative-expansion-adore`

`iterative_expansion.py:1-19,27`. ADORE ("Iterative Query Expansion with
Retrieval-Grounded Relevance Feedback") is the first of two DIFFERENT loop
*mechanisms* that both delegate termination to `IterativeStopper` rather than
inventing their own. ADORE runs *reformulate → retrieve → judge*: each round a
reformulator emits pseudo-passages, an alpha-repetition query balances their
term frequency against the retriever, results are graded 0-3 (UMBRELA-style),
and the graded evidence conditions the next round's reformulation — with the
module itself **fully dependency-injected** (`retrieve_fn`/`judge_fn`/
`reformulate_fn` supplied by the caller) so "the whole policy is unit-testable
with no LLM and no network." The alternative this heads off is a
reformulation loop hard-wired to a specific LLM client, which would make the
policy itself untestable without a live model.

### Pointer — `CONCEPT:AU-KG.retrieval.assimilated-from-mragent`

`active_reconstruction.py:4-31`. MRAgent ("Memory is Reconstructed, Not
Retrieved," arXiv:2606.06036) is the second loop mechanism, and it is
qualitatively different from both ADORE and the pre-existing single-hop
typed-edge traversal: instead of one-shot top-k retrieval OR a fixed n-hop
expansion, an agent walks a **Cue → Tag → Content → (reverse) Cue** graph in
an evidence-conditioned loop, so "content is expanded only along the tags
most relevant to the query," pruning "the combinatorial neighbour blow-up
that a fixed n-hop expansion would incur." The docstring is explicit about
what was previously missing: agent-utilities already had ADORE's
iterative-reformulation loop (KG-2.88) and fixed single-hop traversal
(KG-2.34); MRAgent's evidence-conditioned multi-hop reconstruction with
tag-mediated frontier pruning was the gap. Reusing `IterativeStopper`
(explicitly cited: "this is one more control structure over the existing
retrieval primitives, not a new subsystem") is what keeps this from becoming a
third, independently-terminated loop implementation.

## Risk Assessment

- **Blast Radius**: `adaptive_stopping.py`, `iterative_expansion.py`,
  `active_reconstruction.py`, plus any future iterative retrieval loop that
  chooses to delegate termination here.
- **Backward Compatible**: Yes — both consumers are dependency-injected,
  network-free modules; adopting the shared stopper is additive.
- **Known weak point**: nothing mechanically prevents a *future* iterative loop
  from re-deriving its own ad-hoc halting rule instead of reusing
  `IterativeStopper` — the convention is enforced by code review / the concept
  marker, not a gate.
