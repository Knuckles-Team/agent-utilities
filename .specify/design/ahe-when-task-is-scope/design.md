# Design Document: A third Guide role gates self-play conjectures on their WEAKEST dimension, denying the Conjecturer its reward hack

CONCEPT:AU-AHE.harness.when-task-is-scope

> `agent_utilities/harness/self_guided_play.py`.

## Decision — Self-Guided Self-Play's Guide scores relevance/conciseness/naturalness and takes the MINIMUM, not the average, as the accept/reject signal

The module docstring (`self_guided_play.py:1-11`) grounds this in a
documented failure mode from the source paper (SGS, arXiv:2604.20209):
asymmetric self-play plateaus because the Conjecturer learns to *hack its
reward* — collapsing over long runs to artificially complex, superficially-
related problems (disjunction spam, over-long conclusions, redundant
premises) that don't actually help the Solver improve. The fix is a third
LLM role, the **Guide**, scoring each generated problem on relevance,
conciseness, and naturalness; low-scoring conjectures are rejected before
they ever train the Solver. `GuideScore.overall` (`self_guided_play.py:92-107`)
is explicit about the aggregation: "Overall quality = the weakest dimension
(a single low score tanks it)."

**The rejected alternative is training on any generated problem that scores
reasonably on average** — the more permissive aggregation, and the one that
would still let the paper's documented collapse patterns through: a
conjecture that's superficially relevant (high relevance score) but
padded/gamed on conciseness or naturalness could still average out to a
passing score. Taking the minimum instead means "superficially relevant but
messy or gamed must be rejected" (stated directly in-code), closing exactly
the loophole an averaged score would leave open. Beyond the paper, this
implementation adds a curriculum + plateau breaker: accepted-and-solved
conjectures raise the target difficulty, rejected conjectures advance
nothing (denying the Conjecturer its reward hack directly), and a stalled
rolling solve-rate perturbs difficulty downward to escape a plateau — style
deliberately mirrored on `pre-emit-quality-gate`'s dataclass/pluggable-scorer
shape rather than inventing a new gate pattern.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/self_guided_play.py` only —
  a self-contained, dependency-injected loop.
- **Backward Compatible**: Yes — the module is unit-testable with no LLM via
  the deterministic heuristic `Guide` scorer; a real deployment injects an
  LLM-backed scorer without changing the loop's contract.
- **Known weak point**: the deterministic heuristic scorer (relevance =
  lexical/Jaccard overlap, conciseness/naturalness = pattern-based penalties)
  is explicitly a stand-in for the paper's LLM-based Guide — it mirrors the
  rubric's *shape* but not its judgment, so a conjecture that games the
  heuristic's specific lexical/length signals without genuinely being
  low-quality could pass the deterministic gate even though an LLM Guide
  would have caught it.
