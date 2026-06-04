"""KG-driven orchestration SYNTHESIS engine (CONCEPT:KG-2.10 S4c).

Where ``orchestration.py`` (S4b) defines the artifact specs and their graph
serialisation, this layer *produces* those specs from the knowledge graph: given
a goal, it queries the KG for candidate tools/skills/prompts (capability search)
and uses an injected LLM to compose agents, decompose teams, and evolve prompts.
The output flows back through the same ``*_to_batch`` converters and
``registry.write_batch`` so synthesized orchestration persists through the one
``GraphBackend`` — exactly like every other source.

Follows the ``distill.py`` pattern: gather candidates → LLM compose → return
specs. ``capability_search`` and ``llm_fn`` are injected, so the whole engine is
testable with fakes and performs no network I/O at module scope.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from .orchestration import (
    AgentSpec,
    PromptSpec,
    TeamSpec,
    WorkflowSpec,
    agent_to_batch,
    prompt_to_batch,
    select_model,
    team_to_batch,
    workflow_to_batch,
)
from .registry import write_batch

logger = logging.getLogger(__name__)

# prompt -> raw JSON text
LLMFn = Callable[[str], str]
# (query, k) -> list of {id, type, name, ...} capability candidates from the KG
CapabilitySearchFn = Callable[[str, int], list[dict[str, Any]]]


def _loads_obj(raw: str) -> dict[str, Any]:
    """Lenient parse: extract the first ``{..}`` object from LLM output."""
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        obj = json.loads(raw[start:end])
        return obj if isinstance(obj, dict) else {}
    except (ValueError, json.JSONDecodeError):
        return {}


def _candidates_by_type(
    results: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group raw capability results by their ``type`` field."""
    out: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        t = str(r.get("type") or r.get("node_type") or "").strip()
        if t:
            out.setdefault(t, []).append(r)
    return out


def _ground(names: list[Any], candidate_names: set[str]) -> list[str]:
    """Prefer LLM-chosen names that match a KG candidate; keep novel ones too.

    Candidates first (grounded in the graph), then any extra names the LLM
    proposed, de-duplicated and order-preserving.
    """
    cleaned = [str(n).strip() for n in names if str(n).strip()]
    lower = {c.lower(): c for c in candidate_names}
    grounded: list[str] = []
    extra: list[str] = []
    seen: set[str] = set()
    for n in cleaned:
        if n.lower() in seen:
            continue
        seen.add(n.lower())
        (grounded if n.lower() in lower else extra).append(n)
    return grounded + extra


_AGENT_PROMPT = """You are designing a single specialized agent to achieve this goal:
{goal}

The knowledge graph offers these candidate capabilities (ground your design in
them where they fit):
Tools:
{tools}
Skills:
{skills}
Prompts:
{prompts}

Design one agent. Output ONLY a JSON object with keys:
"name" (short agent name), "system_prompt" (its instructions), "tools" (array of
tool names it should use, drawn from the candidates where possible), "skills"
(array of skill names), and "description" (one sentence). No other text."""


def synthesize_agent(
    goal: str,
    capability_search: CapabilitySearchFn,
    llm_fn: LLMFn,
    limit: int = 8,
    models: list[dict] | None = None,
    complexity: str = "normal",
) -> AgentSpec:
    """Compose an ``AgentSpec`` for ``goal`` from KG-retrieved capabilities.

    Fetches candidate tools/skills/prompts via ``capability_search`` and asks the
    injected LLM to assemble an agent, grounding tool/skill choices in the
    candidates where possible. When ``models`` (the KG-known model registry) is
    given, the right model is chosen for ``complexity`` ("light" routing vs
    "normal"/"super" heavy) and recorded on the agent. (CONCEPT:KG-2.10)
    """
    results = capability_search(goal, limit) or []
    by_type = _candidates_by_type(results)

    def _names(kind: str) -> set[str]:
        return {
            str(r.get("name") or "").strip()
            for r in by_type.get(kind, [])
            if r.get("name")
        }

    tool_names = _names("Tool")
    skill_names = _names("Skill")
    prompt_names = _names("Prompt")

    def _fmt(names: set[str]) -> str:
        return "\n".join(f"- {n}" for n in sorted(names)) or "- (none)"

    prompt = _AGENT_PROMPT.format(
        goal=goal,
        tools=_fmt(tool_names),
        skills=_fmt(skill_names),
        prompts=_fmt(prompt_names),
    )
    obj = _loads_obj(llm_fn(prompt))

    name = str(obj.get("name") or "").strip() or f"Agent for {goal}"[:80]
    return AgentSpec(
        name=name,
        goal=goal,
        system_prompt=str(obj.get("system_prompt") or "").strip(),
        tools=_ground(obj.get("tools") or [], tool_names),
        skills=_ground(obj.get("skills") or [], skill_names),
        model=select_model(models, complexity),
        description=str(obj.get("description") or "").strip(),
    )


_TEAM_PROMPT = """Decompose this goal into a small hierarchical team of agents:
{goal}

Pick at most {max_members} members. Output ONLY a JSON object with keys:
"team_name" (a short descriptive name for the team, e.g. "Codebase KG Squad"),
"lead" (name of the lead agent) and "members" (array of objects, each with
"name" and "subgoal" describing that member's responsibility). The lead may also
appear in members. No other text."""


def synthesize_team(
    goal: str,
    capability_search: CapabilitySearchFn,
    llm_fn: LLMFn,
    max_members: int = 5,
) -> tuple[TeamSpec, list[AgentSpec]]:
    """Decompose ``goal`` into roles, then synthesize an agent per role.

    One LLM call splits the goal into a lead + member sub-goals; each member is
    then built via :func:`synthesize_agent`. The returned ``TeamSpec`` defaults
    its hierarchy to every non-lead member reporting to the lead. (CONCEPT:KG-2.10)
    """
    obj = _loads_obj(llm_fn(_TEAM_PROMPT.format(goal=goal, max_members=max_members)))
    lead = str(obj.get("lead") or "").strip()

    raw_members = obj.get("members") or []
    member_specs: list[AgentSpec] = []
    member_names: list[str] = []
    seen: set[str] = set()
    for m in raw_members[:max_members]:
        if not isinstance(m, dict):
            continue
        mname = str(m.get("name") or "").strip()
        if not mname or mname.lower() in seen:
            continue
        seen.add(mname.lower())
        subgoal = str(m.get("subgoal") or "").strip() or goal
        spec = synthesize_agent(subgoal, capability_search, llm_fn)
        # Keep the LLM-assigned role name as the team-facing identity.
        spec = spec.model_copy(update={"name": mname})
        member_specs.append(spec)
        member_names.append(mname)

    if not lead and member_names:
        lead = member_names[0]

    team_name = str(obj.get("team_name") or "").strip()
    if not team_name:
        team_name = f"{lead} Team" if lead else "Synthesized Team"
    team = TeamSpec(
        name=team_name[:80],
        goal=goal,
        lead=lead,
        members=member_names,
        description=str(obj.get("description") or "").strip(),
    )
    return team, member_specs


_PROMPT_EVOLVE = """Draft (or evolve) a prompt named "{name}".
Problem to address: {problem}
{prior}
Output ONLY a JSON object with keys "content" (the prompt text) and "rationale"
(why this prompt addresses the problem). No other text."""


def evolve_prompts(needs: list[dict], llm_fn: LLMFn) -> list[PromptSpec]:
    """Draft or evolve prompts for identified needs.

    Each need is ``{"name", "problem", "prior_prompt_id"?}``. The LLM drafts new
    content; when a ``prior_prompt_id`` is given, the resulting ``PromptSpec``
    records it via ``evolved_from`` for lineage. (CONCEPT:KG-2.10)
    """
    specs: list[PromptSpec] = []
    for need in needs:
        if not isinstance(need, dict):
            continue
        name = str(need.get("name") or "").strip()
        if not name:
            continue
        problem = str(need.get("problem") or "").strip()
        prior = need.get("prior_prompt_id")
        prior_line = f"Evolve from prior prompt: {prior}" if prior else ""
        obj = _loads_obj(
            llm_fn(_PROMPT_EVOLVE.format(name=name, problem=problem, prior=prior_line))
        )
        specs.append(
            PromptSpec(
                name=name,
                content=str(obj.get("content") or "").strip(),
                evolved_from=str(prior).strip() if prior else None,
                rationale=str(obj.get("rationale") or "").strip(),
            )
        )
    return specs


_TO_BATCH: dict[type, Callable[[Any], Any]] = {
    AgentSpec: agent_to_batch,
    TeamSpec: team_to_batch,
    PromptSpec: prompt_to_batch,
    WorkflowSpec: workflow_to_batch,
}


def persist_synthesis(backend: Any, *specs: Any) -> tuple[int, int]:
    """Persist any mix of synthesized specs through the one ``GraphBackend``.

    Each spec is converted via its matching ``*_to_batch`` from
    ``orchestration.py`` and written via ``registry.write_batch``. Returns the
    total ``(nodes_written, edges_written)``. (CONCEPT:KG-2.10)
    """
    total_n = total_e = 0
    for spec in specs:
        to_batch = _TO_BATCH.get(type(spec))
        if to_batch is None:
            logger.debug("persist_synthesis: no converter for %s", type(spec))
            continue
        n, e = write_batch(backend, to_batch(spec))
        total_n += n
        total_e += e
    return total_n, total_e
