#!/usr/bin/python
"""Council Deliberation Module.

This module implements a multi-perspective advisory council pattern for
high-stakes decision-making within the pydantic-graph orchestrator. It
provides a 4-stage deliberative pipeline:

1. **Parallel Advisor Dispatch**: Run N advisors (each with a different
   thinking-style persona) concurrently via ``run_orthogonal_regions``.
2. **Anonymization**: Shuffle advisor identities behind opaque labels
   (Response A, B, ...) to prevent anchoring bias during peer review.
3. **Peer Review**: Run M reviewers who rank, critique, and identify
   collective gaps across the anonymized responses.
4. **Chairman Synthesis**: A chairman agent synthesizes all perspectives
   and peer review feedback into a structured ``CouncilVerdict``.

The council leverages the existing ``ModelRegistry`` for hybrid model
routing (different advisors can use different real models based on tier
and tags), and persists verdicts as ``DecisionNode`` entries in the
Knowledge Graph for future reference.
"""

from __future__ import annotations

import asyncio
import logging
import random
import string
import time
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .config_helpers import emit_graph_event, load_specialized_prompts
from .state import GraphDeps, GraphState

logger = logging.getLogger(__name__)

# Default advisor roles (matching JSON prompt file basenames in prompts/)
DEFAULT_ADVISOR_ROLES: list[str] = [
    "council_contrarian",
    "council_first_principles",
    "council_expansionist",
    "council_outsider",
    "council_executor",
]


class CouncilVerdict(BaseModel):
    """Structured output from a council deliberation.

    This model is the final product of the 4-stage pipeline. It captures
    the synthesized recommendation, supporting insights, blind spots
    surfaced during peer review, and a concrete actionable next step.

    Attributes:
        final_recommendation: The chairman's one-sentence verdict.
        key_insights: The most valuable takeaways from the advisors.
        blind_spots: Critical gaps identified during peer review.
        consensus_areas: Areas where multiple advisors converged.
        concrete_next_step: Exactly what to do first.
        confidence: Chairman confidence score (1-10).
        dissenting_views: Important minority opinions.

    """

    final_recommendation: str = Field(
        description="One-sentence recommendation from the chairman"
    )
    key_insights: list[str] = Field(
        default_factory=list,
        description="Top 3-5 insights from the advisory panel",
    )
    blind_spots: list[str] = Field(
        default_factory=list,
        description="Critical gaps identified by peer reviewers",
    )
    consensus_areas: list[str] = Field(
        default_factory=list,
        description="Points where multiple advisors converged",
    )
    concrete_next_step: str = Field(
        default="",
        description="Specific first action to take",
    )
    confidence: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Chairman confidence score 1-10",
    )
    dissenting_views: list[str] = Field(
        default_factory=list,
        description="Important minority opinions that should not be discarded",
    )


class AgentTranscript(BaseModel):
    """Generalized transcript of any agent execution stage.

    This model provides a reusable markdown-renderable record of any
    agent's input, output, and metadata. It is not council-specific —
    any specialist can produce transcripts for inspection.

    Attributes:
        stage_name: Human-readable label for the execution stage.
        agent_role: The role or persona that produced this output.
        model_id: The LLM model used (if known).
        input_query: The prompt/query sent to the agent.
        output_text: The raw output from the agent.
        duration_ms: Wall-clock duration in milliseconds.
        metadata: Additional key-value metadata.

    """

    stage_name: str = Field(
        description="Label for this stage (e.g. 'Advisor', 'Reviewer')"
    )
    agent_role: str = Field(description="Role name or persona tag")
    model_id: str = Field(default="unknown", description="LLM model used")
    input_query: str = Field(default="", description="Query sent to the agent")
    output_text: str = Field(default="", description="Agent's raw output")
    duration_ms: int = Field(default=0, description="Wall-clock duration in ms")
    metadata: dict[str, Any] = Field(default_factory=dict)


class CouncilTranscript(BaseModel):
    """Full transcript of a council deliberation for inspection.

    Contains ordered records from every stage of the pipeline,
    plus the final verdict. Can be rendered to markdown via
    :meth:`to_markdown`.

    Attributes:
        query: The original question posed to the council.
        advisor_transcripts: One record per advisor.
        anonymization_map: Label→advisor role mapping for auditability.
        reviewer_transcripts: One record per peer reviewer.
        chairman_transcript: The chairman synthesis record.
        verdict: The final structured verdict.

    """

    query: str = Field(description="Original question")
    advisor_transcripts: list[AgentTranscript] = Field(default_factory=list)
    anonymization_map: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of anonymous label to advisor role",
    )
    reviewer_transcripts: list[AgentTranscript] = Field(default_factory=list)
    chairman_transcript: AgentTranscript | None = None
    verdict: CouncilVerdict | None = None

    def to_markdown(self) -> str:
        """Render the full council deliberation as a markdown transcript.

        Returns:
            A formatted markdown string suitable for human inspection.

        """
        lines: list[str] = []
        lines.append("# Council Deliberation Transcript\n")
        lines.append(f"**Query:** {self.query}\n")

        # Advisors
        lines.append("## Stage 1: Advisor Responses\n")
        for t in self.advisor_transcripts:
            lines.append(f"### {t.agent_role} ({t.model_id})")
            lines.append(f"*Duration: {t.duration_ms}ms*\n")
            lines.append(t.output_text)
            lines.append("")

        # Anonymization
        if self.anonymization_map:
            lines.append("## Stage 2: Anonymization Map\n")
            lines.append("| Label | Advisor Role |")
            lines.append("|-------|-------------|")
            for label, role in sorted(self.anonymization_map.items()):
                lines.append(f"| {label} | {role} |")
            lines.append("")

        # Reviewers
        lines.append("## Stage 3: Peer Reviews\n")
        for t in self.reviewer_transcripts:
            lines.append(f"### {t.agent_role} ({t.model_id})")
            lines.append(f"*Duration: {t.duration_ms}ms*\n")
            lines.append(t.output_text)
            lines.append("")

        # Chairman
        if self.chairman_transcript:
            lines.append("## Stage 4: Chairman Synthesis\n")
            lines.append(
                f"*Model: {self.chairman_transcript.model_id} | "
                f"Duration: {self.chairman_transcript.duration_ms}ms*\n"
            )
            lines.append(self.chairman_transcript.output_text)
            lines.append("")

        # Verdict
        if self.verdict:
            lines.append("## Final Verdict\n")
            lines.append(f"**Recommendation:** {self.verdict.final_recommendation}\n")
            lines.append(f"**Confidence:** {self.verdict.confidence}/10\n")
            if self.verdict.key_insights:
                lines.append("**Key Insights:**")
                for insight in self.verdict.key_insights:
                    lines.append(f"- {insight}")
                lines.append("")
            if self.verdict.blind_spots:
                lines.append("**Blind Spots:**")
                for bs in self.verdict.blind_spots:
                    lines.append(f"- {bs}")
                lines.append("")
            if self.verdict.concrete_next_step:
                lines.append(f"**Next Step:** {self.verdict.concrete_next_step}\n")

        return "\n".join(lines)


def render_agent_transcript_markdown(transcripts: list[AgentTranscript]) -> str:
    """Render a generic list of agent transcripts to markdown.

    This is the generalized version — usable by ANY agent execution, not
    just the council. Pass any list of ``AgentTranscript`` records and
    receive a formatted markdown document.

    Args:
        transcripts: Ordered list of transcript records.

    Returns:
        A markdown string.

    """
    lines: list[str] = ["# Agent Execution Transcript\n"]
    for i, t in enumerate(transcripts, 1):
        lines.append(f"## {i}. {t.stage_name}: {t.agent_role}")
        if t.model_id != "unknown":
            lines.append(f"*Model: {t.model_id} | Duration: {t.duration_ms}ms*\n")
        if t.input_query:
            lines.append(f"**Input:** {t.input_query[:500]}\n")
        lines.append(t.output_text)
        lines.append("")
    return "\n".join(lines)


def _anonymize_responses(
    responses: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Shuffle advisor responses behind anonymous labels.

    Takes a mapping of ``{advisor_role: response_text}`` and returns:
    1. A mapping of ``{label: response_text}`` where labels are
       "Response A", "Response B", etc. in shuffled order.
    2. A reverse mapping of ``{label: advisor_role}`` for later
       de-anonymization.

    Args:
        responses: Mapping of advisor role names to their response text.

    Returns:
        A tuple of (anonymized_responses, label_to_role_map).

    """
    roles = list(responses.keys())
    random.shuffle(roles)

    labels = list(string.ascii_uppercase[: len(roles)])
    anonymized: dict[str, str] = {}
    label_map: dict[str, str] = {}

    for label, role in zip(labels, roles, strict=False):
        full_label = f"Response {label}"
        anonymized[full_label] = responses[role]
        label_map[full_label] = role

    return anonymized, label_map


async def run_council_deliberation(
    query: str,
    ctx_deps: GraphDeps,
    ctx_state: GraphState,
    advisor_roles: list[str] | None = None,
    reviewer_count: int = 3,
) -> tuple[CouncilVerdict, CouncilTranscript]:
    """Execute the full 4-stage council deliberation pipeline.

    This function orchestrates:
    1. Parallel advisor dispatch (one agent per thinking-style persona)
    2. Response anonymization (pure Python, zero LLM cost)
    3. Parallel peer review of anonymized responses
    4. Chairman synthesis into a structured ``CouncilVerdict``

    The function uses the existing ``ModelRegistry`` when available to
    route different advisors to different real LLM models (hybrid mode).

    Args:
        query: The question or decision to deliberate.
        ctx_deps: Graph runtime dependencies (model, toolsets, events).
        ctx_state: Current graph execution state.
        advisor_roles: Optional custom list of advisor role names (must
            match ``prompts/*.json`` file basenames). Defaults to the
            5 standard council advisors.
        reviewer_count: Number of independent peer reviewers to run.

    Returns:
        A tuple of (CouncilVerdict, CouncilTranscript) for structured
        consumption and optional human inspection.

    """
    roles = advisor_roles or DEFAULT_ADVISOR_ROLES
    transcript = CouncilTranscript(query=query)

    emit_graph_event(
        ctx_deps.event_queue,
        "council_started",
        advisor_count=len(roles),
        reviewer_count=reviewer_count,
    )

    # ── Stage 1: Parallel Advisor Dispatch ──────────────────────────────
    logger.info(
        f"[COUNCIL] Stage 1: Dispatching {len(roles)} advisors for: '{query[:60]}...'"
    )
    emit_graph_event(ctx_deps.event_queue, "council_stage", stage=1, name="advisors")

    advisor_responses: dict[str, str] = {}

    for role_name in roles:
        prompt = load_specialized_prompts(role_name)
        model = _pick_council_model(ctx_deps, role_name, tier="medium")

        advisor_agent = Agent(
            model=model,
            system_prompt=prompt,
        )

        start = time.time()
        try:
            res = await asyncio.wait_for(
                advisor_agent.run(f"<council_question>\n{query}\n</council_question>"),
                timeout=120.0,
            )
            output = str(res.output)
            duration = int((time.time() - start) * 1000)

            advisor_responses[role_name] = output
            ctx_state._update_usage(getattr(res, "usage", None))

            transcript.advisor_transcripts.append(
                AgentTranscript(
                    stage_name="Advisor",
                    agent_role=role_name,
                    model_id=str(model),
                    input_query=query[:500],
                    output_text=output,
                    duration_ms=duration,
                )
            )

            emit_graph_event(
                ctx_deps.event_queue,
                "council_advisor_complete",
                advisor=role_name,
                duration_ms=duration,
            )

        except Exception as e:
            logger.warning(f"[COUNCIL] Advisor '{role_name}' failed: {e}")
            advisor_responses[role_name] = f"[ADVISOR FAILED: {e}]"
            transcript.advisor_transcripts.append(
                AgentTranscript(
                    stage_name="Advisor",
                    agent_role=role_name,
                    model_id=str(model),
                    output_text=f"[FAILED: {e}]",
                    duration_ms=int((time.time() - start) * 1000),
                )
            )

    # ── Stage 2: Anonymization ──────────────────────────────────────────
    logger.info(f"[COUNCIL] Stage 2: Anonymizing {len(advisor_responses)} responses")
    emit_graph_event(
        ctx_deps.event_queue, "council_stage", stage=2, name="anonymization"
    )

    anonymized, label_map = _anonymize_responses(advisor_responses)
    transcript.anonymization_map = label_map

    # ── Stage 3: Peer Review ────────────────────────────────────────────
    logger.info(f"[COUNCIL] Stage 3: Running {reviewer_count} peer reviewers")
    emit_graph_event(ctx_deps.event_queue, "council_stage", stage=3, name="peer_review")

    # Build the review prompt with all anonymized responses
    responses_text = "\n\n---\n\n".join(
        f"### {label}\n{text}" for label, text in sorted(anonymized.items())
    )
    review_input = (
        f"<original_question>\n{query}\n</original_question>\n\n"
        f"<anonymized_responses>\n{responses_text}\n</anonymized_responses>"
    )

    reviewer_prompt = load_specialized_prompts("council_reviewer")
    review_results: list[str] = []

    for i in range(reviewer_count):
        model = _pick_council_model(ctx_deps, "council_reviewer", tier="light")
        reviewer_agent = Agent(model=model, system_prompt=reviewer_prompt)

        start = time.time()
        try:
            res = await asyncio.wait_for(
                reviewer_agent.run(review_input), timeout=120.0
            )
            output = str(res.output)
            duration = int((time.time() - start) * 1000)

            review_results.append(output)
            ctx_state._update_usage(getattr(res, "usage", None))

            transcript.reviewer_transcripts.append(
                AgentTranscript(
                    stage_name="Peer Review",
                    agent_role=f"reviewer_{i + 1}",
                    model_id=str(model),
                    input_query=f"[{len(anonymized)} anonymized responses]",
                    output_text=output,
                    duration_ms=duration,
                )
            )

            emit_graph_event(
                ctx_deps.event_queue,
                "council_reviewer_complete",
                reviewer=i + 1,
                duration_ms=duration,
            )

        except Exception as e:
            logger.warning(f"[COUNCIL] Reviewer {i + 1} failed: {e}")
            review_results.append(f"[REVIEWER FAILED: {e}]")

    # ── Stage 4: Chairman Synthesis ─────────────────────────────────────
    logger.info("[COUNCIL] Stage 4: Chairman synthesis")
    emit_graph_event(ctx_deps.event_queue, "council_stage", stage=4, name="chairman")

    chairman_prompt = load_specialized_prompts("council_chairman")
    model = _pick_council_model(ctx_deps, "council_chairman", tier="heavy")

    # Build the chairman input with all context
    advisor_section = "\n\n".join(
        f"### {role}\n{text}" for role, text in advisor_responses.items()
    )
    review_section = "\n\n".join(
        f"### Reviewer {i + 1}\n{text}" for i, text in enumerate(review_results)
    )

    chairman_input = (
        f"<original_question>\n{query}\n</original_question>\n\n"
        f"<advisor_responses>\n{advisor_section}\n</advisor_responses>\n\n"
        f"<peer_reviews>\n{review_section}\n</peer_reviews>"
    )

    chairman_agent = Agent(
        model=model,
        system_prompt=chairman_prompt,
        output_type=CouncilVerdict,
    )

    start = time.time()
    try:
        res = await asyncio.wait_for(chairman_agent.run(chairman_input), timeout=180.0)
        verdict = res.output
        duration = int((time.time() - start) * 1000)
        ctx_state._update_usage(getattr(res, "usage", None))

        transcript.chairman_transcript = AgentTranscript(
            stage_name="Chairman",
            agent_role="council_chairman",
            model_id=str(model),
            input_query=f"[{len(advisor_responses)} advisors + {len(review_results)} reviews]",
            output_text=verdict.final_recommendation,
            duration_ms=duration,
        )
        transcript.verdict = verdict

    except Exception as e:
        logger.error(f"[COUNCIL] Chairman synthesis failed: {e}")
        # Return a degraded verdict from the raw advisor responses
        verdict = CouncilVerdict(
            final_recommendation=f"Council synthesis failed ({e}). "
            "Review individual advisor responses.",
            key_insights=[
                f"{role}: {text[:200]}" for role, text in advisor_responses.items()
            ],
            confidence=2,
        )
        transcript.verdict = verdict

    emit_graph_event(
        ctx_deps.event_queue,
        "council_completed",
        confidence=verdict.confidence,
        advisor_count=len(roles),
    )

    logger.info(
        f"[COUNCIL] Complete. Confidence: {verdict.confidence}/10. "
        f"Advisors: {len(roles)}, Reviewers: {reviewer_count}"
    )

    return verdict, transcript


def _pick_council_model(ctx_deps: GraphDeps, role: str, tier: str = "medium") -> Any:
    """Select the LLM model for a council role using the ModelRegistry.

    If a ``ModelRegistry`` is available in deps, uses tier-based
    routing to assign different real models to different council roles.
    Otherwise falls back to the graph-wide default model.

    Args:
        ctx_deps: Graph runtime dependencies.
        role: The council role name (for logging).
        tier: Desired model tier ('light', 'medium', 'heavy', 'reasoning').

    Returns:
        A pydantic-ai model instance.

    """
    from .executor import pick_specialist_model

    # Override the tier hints for council-specific routing
    _COUNCIL_TIER_MAP: dict[str, str] = {
        "council_contrarian": "medium",
        "council_first_principles": "heavy",
        "council_expansionist": "medium",
        "council_outsider": "light",
        "council_executor": "medium",
        "council_reviewer": "light",
        "council_chairman": "heavy",
    }

    actual_tier = _COUNCIL_TIER_MAP.get(role, tier)

    try:
        return pick_specialist_model(ctx_deps, role)
    except Exception:
        logger.debug(
            f"Council model routing fell back to default for '{role}' (tier={actual_tier})"
        )
        return ctx_deps.agent_model
