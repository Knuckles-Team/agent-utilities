# Design Document: A reasoning on/off decision travels as a raw provider request field, not through `ModelSettings.thinking` alone

CONCEPT:AU-ORCH.execution.delegation-reasoning-off

> `agent_utilities/agent/factory.py:234-256` (`_resolve_agent_extra_body`),
> `agent_utilities/core/model_factory.py:305-340`
> (`reasoning_wire_directives`).

## Decision — carry the reasoning on/off directive up as a raw `extra_body` field, because `pydantic_ai`'s unified `ModelSettings.thinking` silently no-ops for non-OpenAI-recognized model profiles

`reasoning_wire_directives` documents the regression this fixes precisely:
`pydantic_ai.models.Model.prepare_request` only forwards `ModelSettings.
thinking` into the actual outgoing request when the model's PROFILE is
recognized as reasoning-capable — and `openai_model_profile()` recognizes
ONLY OpenAI's own o-series/gpt-5(.1+) naming. A local/custom reasoning model
served through the generic `openai`-compatible provider (the concrete
example named in the code: `qwen/qwen3.6-27b` behind vLLM) gets
`supports_thinking=False` from that heuristic, so `thinking` — however it was
set, on the model OR the agent — never reaches the wire. The model's OWN
default (thinking ON, since it's a reasoning model) always wins instead. The
measured symptom: a call that explicitly requested "reasoning off" still took
~22s (full chain-of-thought) instead of the ~0.3s a directive that actually
reached the wire produces — **the disable directive was computed correctly
and then silently dropped before it reached the request**, not miscomputed.

The fix routes the decision through the model's raw `extra_body` (vLLM's own
scheduler-level knob), which `_resolve_agent_extra_body` explicitly carries
up from the model layer to the agent layer — needed because `pydantic-ai`
merges agent-over-model `ModelSettings` with a SHALLOW dict union, so an
agent-level `extra_body` REPLACES (not deep-merges) the model-level one
`create_model` already built. Without carrying it up, setting ANY
agent-level `extra_body` for an unrelated reason would silently discard the
model's own reasoning directive.

**The rejected alternative is `ModelSettings.thinking` as the sole
mechanism** — the natural, `pydantic-ai`-idiomatic way to express this, and
the thing that was tried first (it's what regressed). It loses specifically
for the non-OpenAI-profile case: the key itself survives the shallow union
fine, but the LIBRARY silently drops it before it reaches the wire based on
a profile check the local/custom model never passes. The raw `extra_body`
directive bypasses that profile gate entirely, since it's forwarded to the
provider verbatim rather than interpreted by `pydantic-ai`'s own
reasoning-capability logic.

## Risk Assessment

- **Blast Radius**: `agent_utilities/agent/factory.py`,
  `agent_utilities/core/model_factory.py`,
  `tests/unit/test_delegation_reasoning_off.py`.
- **Backward Compatible**: Yes — a model whose profile IS recognized by
  `pydantic-ai` (genuine OpenAI o-series/gpt-5) still gets `thinking` set
  normally; the raw `extra_body` directive is additive coverage for the
  models that were being silently dropped, not a replacement path for models
  that already worked.
- **Known weak point**: the fix is specific to the OpenAI-compatible
  `extra_body`/vLLM-scheduler wire shape — a future provider integration
  (a different API shape entirely) would need its OWN raw-directive
  translation; `reasoning_wire_directives`'s fix does not generalize
  automatically to a non-OpenAI-compatible transport.
