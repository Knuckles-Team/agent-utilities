# Design Document: Implicit, inference-time latent refinement over Semantic IDs — not explicit rationale text, and not backbone training

CONCEPT:AU-KG.retrieval.pauserec-implicit-reasoning-generative

> `agent_utilities/knowledge_graph/retrieval/generative_recommender.py`.

## Decision — adopt PauseRec's mechanism at inference/agentic time over an existing SID encoder, with no decoded rationale and no backbone fine-tuning

`generative_recommender.py:4-48` (adapted from He et al., arXiv:2606.14142)
states PauseRec's finding directly: LLM-based generative recommenders
represent items as **Semantic IDs (SIDs)** — short tuples of discrete codes
outside the natural-language vocabulary — and *explicit* Chain-of-Thought
reasoning over those SIDs is brittle for three stated reasons: world
knowledge becomes hard to verbalize after reasoning fine-tuning; natural-
language and SID token embeddings drift apart (text↔SID misalignment); and
recommendation quality is fragile with respect to the exact rationale text.
PauseRec's remedy — trainable `<pause>` tokens giving the model latent
computation steps optimized only by the next-item objective, with no decoded
rationale — is adopted here as an inference-time analogue: a configurable
`pause_steps` budget of deliberate refinement steps over the already-produced
SIDs, each nudging a working representation toward the catalog items and
user history it is closest to, with **no explicit rationale string**.

**The rejected alternative, named directly, is training an LLM backbone**:
"we are an agentic framework; we do **not** train an LLM backbone here." This
is the adaptation decision that makes the module distinct from a literal port
of the paper — PauseRec's `<pause>` tokens are trained embeddings; this
module's `pause_steps` are a dedicated latent computation window applied at
inference time over an already-trained `TemporalSemanticIdEncoder`, "the
inference analogue of PauseRec's literal `<pause>` tokens," not a second
training objective. A second rejected alternative, implicit in the design
choice to keep explicit CoT off the table entirely: generating a visible
rationale before the SID recommendation, which the cited PauseRec finding
shows measurably hurts quality for SID-based generative recommendation
specifically (as opposed to natural-language generation, where CoT typically
helps). The `TextSidBridge` projection addresses the second cited failure
mode (text↔SID misalignment) directly: it routes a natural-language query
embedding through the encoder's shared codebooks so query and items occupy
one code space, rather than comparing across two spaces that have drifted
apart.

Everything is deterministic and dependency-injected (the encoder is passed
in, no LLM/training/network call is involved, stdlib + numpy only) — a pure
L2 retrieval helper built strictly on top of `TemporalSemanticIdEncoder`, with
no upward dependencies.

## Risk Assessment

- **Blast Radius**: `generative_recommender.py`,
  `temporal_semantic_id.py` (`TemporalSemanticIdEncoder`, consumed not
  modified).
- **Backward Compatible**: Yes — a new retrieval helper layered strictly on
  top of the existing encoder; it does not change how SIDs are produced.
- **Known weak point**: the paper's useful budget "saturates" after a couple
  of steps per the docstring's own framing — `pause_steps` is a caller-set
  constant, not adaptively determined per query, so a query that would
  benefit from more refinement steps than the configured budget allows gets
  no signal that it was under-refined.
