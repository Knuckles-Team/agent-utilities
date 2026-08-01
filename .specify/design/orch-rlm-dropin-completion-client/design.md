# Design Document: RLM gets a drop-in `.completion()` client instead of forcing every caller onto the structured `run_rlm()` signature

CONCEPT:AU-ORCH.execution.drop-rlm-completion-client

> `agent_utilities/rlm/client.py` (the client), `agent_utilities/rlm/prompts.py` (the
> family-aware system prompt it depends on), `agent_utilities/rlm/repl.py` (where that
> prompt is applied), `agent_utilities/rlm/config.py` (`prompt_family`),
> `agent_utilities/rlm/__init__.py` (export), plus two real call sites —
> `agent_utilities/harness/agentic_evolution_engine.py` and
> `agent_utilities/mcp/tools/analysis_tools.py` — that consume it.

## Decision — a paper-shaped `RLM(...).completion(prompt)` wrapper sits in front of `run_rlm(task, input_text=...)`

The RLM package's real programmatic interface is `run_rlm(task, input_text=..., config=...)`
(`agent_utilities/rlm/runner.py`) — a two-argument, task/input-split signature that matches
the module's own Predict-RLM signature model, not how a plain LLM client is normally called.
`agent_utilities/rlm/client.py:1-13` states the problem directly: this is "a thin, paper-shaped
client so RLM can replace a plain `llm.completion(prompt)` call **without learning** the
structured `run_rlm(task, input_text=...)` signature." The `RLM` class
(`client.py:43-79`) mirrors the reference RLM library's own ergonomics —
`RLM(backend="openai", backend_kwargs={"model_name": "..."})` then `.completion(prompt)` /
`.acompletion(prompt, context=...)` — and internally resolves those kwargs into
`RLMConfig.sub_llm_model_large` and delegates to `run_rlm`. `RLMResponse` (`client.py:29-40`)
gives back the plain `.response`/`.text`/`.ok`/`.usage`/`.error` shape any existing
"pass a prompt, get a string back" call site already expects.

**The rejected alternative** is making every caller learn and adopt the structured
`run_rlm(task, input_text=...)` call directly. That is the module's actual native API and
it works fine for code that is written RLM-aware from the start — but it loses for exactly
the two real production call sites that exist today, both of which are swap-ins for an
existing plain-completion contract, not new RLM-native code:

- `agent_utilities/harness/agentic_evolution_engine.py:497` — the Monte-Carlo graph-search
  code evolver's `coder_fn` hook needs *a function that turns a step plan into code*; the
  MCP `evolve_code` action supplies `RLM().completion(prompt)` as that real LLM-backed coder
  (`agent_utilities/mcp/tools/analysis_tools.py:1289-1304`), with a deterministic
  `f"{prior_code}\n# step for: {plan}"` fallback on any exception. Nothing here calls
  `run_rlm` directly or knows about `task`/`input_text`.
- `agent_utilities/mcp/tools/analysis_tools.py:1326-1329` — the night-shift vault swarm's
  `_llm_extract` cataloger needs *a function that turns source text into a list of atomic
  claims*; again `RLM().completion(prompt)` is dropped in, with a deterministic
  paragraph/sentence splitter as the offline fallback.

Both call sites are "give me a completion" adapters plugged into pre-existing
`coder_fn`/extractor function signatures that predate RLM. Requiring `run_rlm`'s two-argument
task/input-text split at both sites would mean reshaping the calling code's own contract
around RLM's internal API, instead of RLM presenting the ergonomics its callers already
expect. The accepted cost is a second, thinner API surface to keep in sync with `run_rlm`'s
real signature (`acompletion` at `client.py:64-79` is the sync/async seam: `completion` is a
plain wrapper — not shown in the excerpted lines above but referenced from the module
docstring's own usage example — that must stay callable outside an event loop while
`acompletion` is required inside one).

**A second, coupled decision in the same commit**: the RLM REPL's system prompt is
model-family-aware, not one fixed string. `agent_utilities/rlm/prompts.py:1-7` names the
failure mode directly — "Zhang et al. (2025) report that a single fixed RLM system prompt
fails to transfer across model families" — and `build_system_prompt(family, model_id)` keeps
one shared `_BASE` helper contract plus a per-family addendum (terser for Qwen, which can
exhaust output tokens; code-first for Anthropic, which tends to narrate). `RLMConfig.prompt_family`
(`agent_utilities/rlm/config.py:150-159`) defaults to `"auto"`, which infers the family from
the root model id; a pinned value overrides the inference. `repl.py:650-652` is where this is
actually applied: `build_system_prompt(self.config.prompt_family, model_id)` is called every
`run_full_rlm` turn rather than a module-level constant. **The rejected alternative here is
the paper's own documented failure mode** — one hardcoded prompt for every backend — which the
docstring explicitly calls out as broken, not merely suboptimal.

## Risk Assessment

- **Blast Radius**: `agent_utilities/rlm/client.py`, `agent_utilities/rlm/prompts.py`,
  `agent_utilities/rlm/repl.py`, `agent_utilities/rlm/config.py`, `agent_utilities/rlm/__init__.py`,
  plus the two consumer sites in `agentic_evolution_engine.py` and `analysis_tools.py`.
- **Backward Compatible**: Yes — `RLM.completion()` is additive; `run_rlm` is unchanged and
  still the module's structured entry point.
- **Known weak point**: `RLM.__init__` only threads `backend`/`backend_kwargs` into
  `sub_llm_model_large`; a caller relying on the drop-in client gets no control over
  `sub_llm_model_small` (the depth>0 model) without also constructing an explicit
  `RLMConfig`, so the "don't learn the RLM config" promise is partial for anything beyond a
  single root call.
- **Test coverage**: `tests/unit/rlm/test_orch_1_54_dropin_and_prompts.py:1` exercises this
  pairing (drop-in client + family-aware prompt) directly, confirming both halves were
  designed and shipped together, not accreted separately.
