# Design Document: The SWE agent grounds itself in the code KG BEFORE reading files, using Pydantic-AI's own tool-calling loop — not a bespoke CodeAct loop that reads whole files

CONCEPT:AU-ORCH.execution.swe-agent-system-prompt · CONCEPT:AU-ORCH.execution.swe-workspace-tools

> `agent_utilities/agent/swe_prompts.py` (the system prompt, the decision's
> clearest statement) and `agent_utilities/orchestration/swe_agent.py` (the
> loop it drives), wired through `agent_utilities/orchestration/engine.py`
> (`execute_swe`) and gated by `agent_utilities/tools/tool_registry.py`
> (`register_swe_tools`, the `SWE_TOOLS` flag). The action-tool half is
> `agent_utilities/tools/swe_workspace_tools.py` (pointer, below). Tested in
> `tests/orchestration/test_swe_agent_loop.py`.

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| `AU-KG.code.` code-ontology family (KG-2.65) | the code-intelligence graph tools (`find_definition`/`who_calls`/`impacted_tests`/`call_graph`/`dependencies`) this agent is built to prefer over raw reads | 0.45 | KG |
| `AU-ORCH.execution.computer-use-agent` | a sibling opt-in tool group registered right after this one in `tool_registry.py`, same registration pattern, unrelated capability (GUI control vs. code editing) | 0.20 | ORCH |

### Extension Analysis

- **Primary Extension Point**: `register_swe_tools` (`tool_registry.py:322`),
  shared between the `SWE_TOOLS`-gated general agent path and the lean
  `orchestration.swe_agent.build_swe_agent` path "so the SWE surface is
  defined once" (`tool_registry.py:326-327`).
- **Extension Strategy**: augment — new code-intelligence or workspace tools
  are added to the two constituent tool lists
  (`CODE_INTELLIGENCE_TOOLS`, `SWE_WORKSPACE_TOOLS`) without touching the
  registration seam.
- **New Concept Required?**: No.

## Decision — graph-first grounding via a fixed system prompt + Pydantic-AI's native tool loop, opt-in behind `SWE_TOOLS`

`agent_utilities/agent/swe_prompts.py:1-5`, `agent_utilities/orchestration/swe_agent.py:1-9`

`swe_prompts.py`'s module docstring states the goal directly: "The prompt's
job is to make the agent *graph-first*: reason over the code ontology
(KG-2.65) before reading files, so it scales to repos it has never read in
full — the behaviour that lets us surpass a context-stuffing CodeActAgent."
`SWE_SYSTEM_PROMPT` encodes this as an explicit numbered method the model
must follow: (1) GROUND FIRST — use the graph tools (`find_definition`,
`who_calls`, `impacted_tests`, `call_graph`, `dependencies`) to locate
symbols/callers/covering-tests *before* reading whole files, preferring them
"over blindly grepping or reading large files"; (2) read only the specific
line ranges needed; (3) make the smallest correct edit; (4) VERIFY by running
the impacted tests and iterating on failure, never stopping on an unverified
change.

`swe_agent.py`'s docstring names the mechanism: "The loop *is* Pydantic-AI's
own tool-calling loop: a focused `pydantic_ai.Agent` bound to the
code-intelligence (graph) tools (KG-2.65) and the SWE workspace (action)
tools" — not a custom ReAct/CodeAct control loop reimplemented on top of raw
LLM completions.

**The rejected alternative is named explicitly: "a context-stuffing
CodeActAgent."** That is the standard SWE-agent shape — dump repository
context (file trees, grepped snippets, or whole files) into the prompt so the
model has everything up front, then let it emit and execute code/shell
actions. It loses on the dimension the docstring calls out: it does not scale
to repositories the agent has never read in full, because "everything up
front" grows with repository size. The graph-first alternative instead
front-loads targeted graph lookups (definition location, call sites,
covering tests) that are cheap regardless of repository size, and only reads
the specific regions the graph pointed at.

**The surface is opt-in, not default-on.** `DEFAULT_SWE_TOOLS` gates on
`SWE_TOOLS` (default `False`, `tool_registry.py:80-81`): "the swe_engineer
agent enables it" — general-purpose agents do not carry the code-intelligence
+ workspace tool surface unless they are specifically the SWE role, keeping
it symmetric with the other optional tool groups (`DB_TOOLS`, `MEDIA_TOOLS`).
`execute_swe` (`orchestration/engine.py:224-243`) is the dispatch-mode entry
point (`mode="swe"`) that drives the loop inside a developer workspace
(OS-5.33), creating and tearing one down when the caller doesn't supply one
already-cloned (e.g. the SWE-bench harness, AHE-3.22).

`build_swe_agent` is deliberately model-injectable specifically so the full
loop can be driven deterministically in tests with a Pydantic-AI
`FunctionModel`/`TestModel` — no live LLM required (`swe_agent.py:11-12`).

### Pointer — `CONCEPT:AU-ORCH.execution.swe-workspace-tools`

`agent_utilities/tools/swe_workspace_tools.py:1-11`

The action-tool half of the same agent, factored into its own module and
concept because it is a narrower, separately-statable decision: "This is the
**sole** mutation and execution surface. General developer tools remain
read-only. It is distinct from `workspace_tools.py`, which manages SKILL.md
and project-memory metadata." Concretely, `run_command`/`read_file`/
`write_file`/`edit_file`/`run_tests` translate the model's tool calls into
typed runtime actions (`agent_utilities.runtime.events`) executed inside
`deps.workspace` (`DevWorkspace`, OS-5.33) — and because the workspace
mirrors every action back to the KG (KG-2.64) and gates mutations via
`ActionPolicy` (OS-5.24), the SWE agent inherits provenance and governance
"for free" rather than needing its own audit/gate logic. `register_swe_tools`
(`tool_registry.py:322-340`) notes the tools "no-op safely when no workspace
is attached" — a call outside a workspace context degrades to an explicit
"no workspace" message rather than failing unpredictably.

## Data Flow

1. **ORCH**: `execute_swe` (dispatch mode `"swe"`) → `run_swe_task` →
   `build_swe_agent`'s Pydantic-AI tool-calling loop.
2. **KG**: code-intelligence tools (`find_definition`, `who_calls`,
   `impacted_tests`, `call_graph`, `dependencies`) query the code ontology
   (KG-2.65); every workspace action mirrors back to the KG (KG-2.64).
3. **AHE**: the full trajectory is provenance the golden loop (AHE-3.23) can
   learn from; a pre-cloned workspace can come from the SWE-bench harness
   (AHE-3.22).
4. **ECO**: none directly.
5. **OS**: mutations are gated via `ActionPolicy` (OS-5.24) inside
   `DevWorkspace` (OS-5.33).

## Risk Assessment

- **Blast Radius**: `agent_utilities/agent/swe_prompts.py`,
  `agent_utilities/orchestration/swe_agent.py`,
  `agent_utilities/orchestration/engine.py`,
  `agent_utilities/tools/tool_registry.py`,
  `agent_utilities/tools/swe_workspace_tools.py`.
- **Backward Compatible**: Yes — opt-in via `SWE_TOOLS`/the `swe_engineer`
  role; general agents are unaffected.
- **Breaking Changes**: None.
- **Known weak point**: the graph-first discipline is prompt-enforced, not
  mechanically enforced — nothing stops the model from reading whole files
  first if it chooses to; the design relies on the system prompt's explicit
  ordering ("GROUND FIRST") plus the model's own tool-selection behavior.
