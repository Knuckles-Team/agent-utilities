"""
Investor Debate Team — CONCEPT:KG-2.6

Light wiring that slots the new investor personas, the forensic screener, and
the price-action pattern analyst into the EXISTING debate/consensus machinery:

* ``INVESTOR_PERSONAS`` maps each persona prompt (``agent_utilities/prompts/*``)
  onto a :class:`~agent_utilities.domains.finance.trading_swarm.SwarmRole` so the
  weighted ``SwarmConsensus`` and the Bull/Bear ``DebateEngine`` can consume them.
* :func:`build_financial_debate_team` produces the
  :class:`~agent_utilities.knowledge_graph.enrichment.orchestration.TeamSpec`
  (personas + risk officer reporting to a portfolio_manager).
* :func:`seed_financial_debate_team` persists that team (and its agents) into the
  KG through the same ``team_to_batch`` / ``agent_to_batch`` → ``write_batch``
  path every other orchestration source uses.

No new orchestration is introduced — this is content + seeding only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_utilities.domains.finance.trading_swarm import SwarmRole

logger = logging.getLogger(__name__)

TEAM_NAME = "financial_debate"
TEAM_CONFIG_TAG = "financial_debate"
PORTFOLIO_MANAGER = "portfolio_manager"
RISK_OFFICER = "risk_compliance_officer"

# Directory holding the persona prompt JSONs (agent_utilities/prompts).
_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

# Default voices for the Bull/Bear debate when a caller asks for personas but
# doesn't name them: Buffett (value bull) vs Burry (forensic bear).
DEFAULT_BULL_PERSONA = "buffett_investor"
DEFAULT_BEAR_PERSONA = "burry_investor"


@dataclass(frozen=True)
class PersonaRole:
    """Binding of an investor persona prompt to a swarm role + debate side."""

    prompt: str  # prompt file stem in agent_utilities/prompts/
    swarm_role: SwarmRole
    debate_side: str  # "bull" | "bear" | "neutral"
    philosophy: str


# Persona → SwarmRole mapping. The ``swarm_role`` mirrors the ``swarm_role`` key
# inside each persona JSON's ``metadata`` so the prompt and the swarm agree.
INVESTOR_PERSONAS: tuple[PersonaRole, ...] = (
    PersonaRole(
        prompt="buffett_investor",
        swarm_role=SwarmRole.FUNDAMENTAL_ANALYST,
        debate_side="bull",
        philosophy="value / durable moat / margin of safety",
    ),
    PersonaRole(
        prompt="burry_investor",
        swarm_role=SwarmRole.BEAR_RESEARCHER,
        debate_side="bear",
        philosophy="contrarian / forensic / what's quietly broken",
    ),
    PersonaRole(
        prompt="druckenmiller_investor",
        swarm_role=SwarmRole.DIRECTOR,
        debate_side="neutral",
        philosophy="macro / regime / position sizing",
    ),
    PersonaRole(
        prompt="damodaran_investor",
        swarm_role=SwarmRole.FUNDAMENTAL_ANALYST,
        debate_side="bull",
        philosophy="intrinsic valuation / DCF / narrative-and-numbers",
    ),
    PersonaRole(
        prompt="graham_investor",
        swarm_role=SwarmRole.FUNDAMENTAL_ANALYST,
        debate_side="bull",
        philosophy="quantitative deep value / margin of safety",
    ),
)

# The two engine-grounded specialists that feed the debate but are not personas.
SPECIALIST_ROLES: dict[str, SwarmRole] = {
    "forensic_screener": SwarmRole.BEAR_RESEARCHER,
    "pattern_analyst": SwarmRole.PATTERN_ANALYST,
}


def persona_for_role(role: SwarmRole) -> list[str]:
    """Return the persona prompt stems mapped to a given swarm role."""
    return [p.prompt for p in INVESTOR_PERSONAS if p.swarm_role == role]


def load_persona_prompt(stem: str) -> dict[str, Any]:
    """Load a persona prompt JSON body from ``agent_utilities/prompts``.

    Raises ``FileNotFoundError`` if the persona stem has no prompt file.
    """
    path = _PROMPTS_DIR / f"{stem}.json"
    if not path.exists():
        raise FileNotFoundError(f"No persona prompt for '{stem}' at {path}")
    return json.loads(path.read_text())


def persona_archetype(stem: str) -> str:
    """Return a persona's archetype label (e.g. ``BuffettInvestor``)."""
    try:
        return load_persona_prompt(stem).get("metadata", {}).get("archetype", stem)
    except (FileNotFoundError, json.JSONDecodeError):
        return stem


def persona_system_prompt(stem: str) -> str:
    """Compose a debate system prompt from a persona JSON body.

    Stitches ``identity.role`` + ``identity.goal`` + ``instructions.core_directive``
    into a system prompt so the Bull/Bear debate speaks in the investor's actual
    voice rather than a generic one. Returns ``""`` when the persona is missing,
    so callers can fall back to the generic prompt.
    """
    try:
        body = load_persona_prompt(stem)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.debug(
            "persona prompt '%s' unavailable; using generic debate voice", stem
        )
        return ""
    ident = body.get("identity", {})
    instr = body.get("instructions", {})
    role = ident.get("role", stem)
    goal = ident.get("goal", "")
    directive = instr.get("core_directive", "")
    archetype = body.get("metadata", {}).get("archetype", stem)
    parts = [
        f"You are arguing in a structured investment debate as the {role} "
        f"({archetype} archetype)."
    ]
    if goal:
        parts.append(goal)
    if directive:
        parts.append("Your discipline:\n" + directive)
    parts.append(
        "Make your case with specific, engine-grounded evidence and never invent "
        "numbers — cite the market/forensic/technical reports you are given."
    )
    return "\n\n".join(parts)


def build_financial_debate_team() -> Any:
    """Build the ``financial_debate`` :class:`TeamSpec`.

    The personas plus the forensic screener and pattern analyst report to the
    risk-compliance officer, who reports to the portfolio_manager (lead).
    """
    from agent_utilities.knowledge_graph.enrichment.orchestration import TeamSpec

    members = (
        [PORTFOLIO_MANAGER, RISK_OFFICER]
        + [p.prompt for p in INVESTOR_PERSONAS]
        + list(SPECIALIST_ROLES.keys())
    )

    # Personas + specialists report to the risk officer; risk officer reports to
    # the portfolio_manager. This mirrors the risk-first DebateEngine veto flow.
    reports_to: list[tuple[str, str]] = [(RISK_OFFICER, PORTFOLIO_MANAGER)]
    for p in INVESTOR_PERSONAS:
        reports_to.append((p.prompt, RISK_OFFICER))
    for spec in SPECIALIST_ROLES:
        reports_to.append((spec, RISK_OFFICER))

    return TeamSpec(
        name=TEAM_NAME,
        goal="Reach a risk-vetoed bull/bear consensus on an equity using "
        "investor personas grounded in engine forensic and price-action signals",
        lead=PORTFOLIO_MANAGER,
        members=members,
        reports_to=reports_to,
        description=(
            "Investor-persona debate team: Buffett/Damodaran/Graham (value, bull "
            "lean), Burry (forensic bear), Druckenmiller (macro/regime director), "
            "plus a forensic screener and price-action pattern analyst — all "
            "feeding the existing Bull/Bear DebateEngine and weighted "
            "SwarmConsensus with a risk-officer veto."
        ),
    )


def _persona_agent_specs() -> list[Any]:
    """Build :class:`AgentSpec` objects for each persona for KG seeding."""
    from agent_utilities.knowledge_graph.enrichment.orchestration import AgentSpec

    specs: list[Any] = []
    for p in INVESTOR_PERSONAS:
        specs.append(
            AgentSpec(
                name=p.prompt,
                goal="Argue the bull/bear case for an equity",
                prompt_id=f"prompt:{p.prompt}",
                description=f"{p.philosophy} (swarm role: {p.swarm_role})",
                tools=["graph-os", "data-science-mcp"],
            )
        )
    # Specialists as agents too, so the team's REPORTS_TO edges resolve.
    specs.append(
        AgentSpec(
            name="forensic_screener",
            goal="Screen earnings quality via engine forensic_report",
            description="Engine-grounded Beneish/Altman/Piotroski/Sloan screen",
            tools=["data-science-mcp"],
        )
    )
    specs.append(
        AgentSpec(
            name="pattern_analyst",
            goal="Classify price action into momentum/mean-reversion edge",
            description="Engine-grounded 4-pattern price-action classifier",
            tools=["data-science-mcp"],
        )
    )
    return specs


def seed_financial_debate_team(backend: Any) -> tuple[int, int]:
    """Persist the financial_debate team + its agents into the KG.

    Uses the shared ``agent_to_batch`` / ``team_to_batch`` → ``write_batch``
    path, so the team seeds through the one ``GraphBackend`` like every other
    orchestration source. Returns ``(nodes_written, edges_written)``.
    """
    from agent_utilities.knowledge_graph.enrichment.orchestration import (
        agent_to_batch,
        team_to_batch,
    )
    from agent_utilities.knowledge_graph.enrichment.registry import write_batch

    nodes = edges = 0
    for spec in _persona_agent_specs():
        n, e = write_batch(backend, agent_to_batch(spec))
        nodes += n
        edges += e

    team = build_financial_debate_team()
    n, e = write_batch(backend, team_to_batch(team))
    nodes += n
    edges += e
    logger.info("Seeded financial_debate team: %d nodes, %d edges", nodes, edges)
    return nodes, edges
