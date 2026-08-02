# Design Document: Measure the intent surface's OWN routing accuracy against a small hand-labelled corpus, never against "the caller named the tool directly"

CONCEPT:AU-ECO.mcp.intent-surface-selection-accuracy

> `agent_utilities/knowledge_graph/retrieval/intent_selection_accuracy.py:1-70`
> (module docstring, `CORPUS`, `measure_selection_accuracy`).

## Decision — a small (15-25 case), hand-labelled corpus of natural-language phrasings that deliberately AVOID the target tool's own name, scored top-1/top-3 against the REAL `resolve_intent` resolver, as the tripwire metric for the condensed-vs-intent tool-selection trade-off

The intent surface (`AU-ECO.mcp.intent-surface-condensed-collapse`) trades a large
condensed tool list for six intent verbs plus a resolver. Whether that trade is
worth it hinges on one number: given only a natural-language description (never the
tool's own name), how often does the resolver rank the SAME capability a caller who
already knew the tool name would have picked. This module is exactly that
measurement. Two properties are load-bearing: (1) **every case is a LIVE call into
the real resolver against the real, checked-in CPD set** — never a fabricated or
precomputed number (`intent_selection_accuracy.py:20-23`) — and (2) the corpus
wording deliberately avoids the tool's own name (`intent_selection_accuracy.py:49-52`),
because a phrase containing the tool name would trivially resolve to 100% by string
containment and measure nothing about real natural-language routing.

## Rejected alternative — treat "naming the tool directly" as a competing ranking strategy to benchmark against, or use a large/statistically-rigorous corpus

Two related alternatives are named and rejected in the module's own docstring.
First: scoring "did the caller name the tool" as one of the strategies under test —
rejected because it is "trivially 100% accurate by definition (you typed the name)
— that is not a competing ranking to measure, it is the baseline the intent
surface's convenience/context-savings is traded against" (`intent_selection_accuracy.py:12-14`).
Including it as a scored strategy would inflate the reported accuracy without
measuring anything real. Second: building a large, statistically rigorous benchmark
corpus — rejected because "this is a tripwire against a real regression in the
resolver/CPD wiring, not a statistically rigorous benchmark"
(`intent_selection_accuracy.py:17-19`); a large corpus would cost much more to
hand-label and maintain for a question this module only needs to answer
approximately and repeatably (did routing quality regress), not with statistical
precision.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/retrieval/intent_selection_accuracy.py`,
  `agent_utilities/mcp/tools/intent_tools.py` (`resolve_intent`).
- **Backward Compatible**: Yes — a measurement harness with no runtime effect on
  the resolver itself.
- **Known weak point**: 15-25 hand-labelled cases cannot catch every regression
  shape; a resolver change that degrades accuracy on phrasings NOT in the corpus
  is invisible to this tripwire until someone extends the corpus.
