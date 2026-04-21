#!/usr/bin/python
"""Self-Improvement Tools Module.

This module provides AI tools for autonomous self-improvement, allowing agents
to trigger background optimization cycles (Lightning style) and propose
new skills based on experience.
"""

import logging
from typing import Any

from pydantic_ai import RunContext

from ..knowledge_graph.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


async def run_self_improvement_cycle(ctx: RunContext[Any]) -> str:
    """Trigger an autonomous self-improvement cycle.

    This cycle analyzes recent failures, generates critiques, optimizes
    system prompts, and proposes new skills based on successful patterns.

    Args:
        ctx: The agent run context.

    Returns:
        A summary of the self-improvement actions taken.
    """
    engine = IntelligenceGraphEngine.get_active()
    if not engine:
        return "Error: Knowledge Graph engine is not active."

    try:
        # Run the cycle
        # Note: In a real system, this might be offloaded to a background task
        engine.run_self_improvement_cycle()
        return "Self-improvement cycle completed successfully. Prompts optimized and new skills proposed where applicable."
    except Exception as e:
        return f"Error during self-improvement cycle: {e}"


async def propose_skills_from_history(ctx: RunContext[Any]) -> str:
    """Analyze recent history and propose new skills for the agent library.

    Args:
        ctx: The agent run context.

    Returns:
        The ID of the newly proposed skill, or a message if none was found.
    """
    engine = IntelligenceGraphEngine.get_active()
    if not engine:
        return "Error: Knowledge Graph engine is not active."

    skill_id = engine.propose_new_skill_from_experience()
    if skill_id:
        return f"Proposed new skill node: {skill_id}. Review the 'ProposedSkill' nodes in the KG for details."
    else:
        return "No significant patterns found to warrant a new skill proposal at this time."


async def query_experiment_results(ctx: RunContext[Any], experiment_name: str) -> str:
    """Query the results of an A/B experiment for prompt or tool variants.

    Args:
        ctx: The agent run context.
        experiment_name: The name of the experiment to query.

    Returns:
        A summary of the experiment results and the current leader.
    """
    engine = IntelligenceGraphEngine.get_active()
    if not engine:
        return "Error: Knowledge Graph engine is not active."

    # Query logic for experiment performance
    query = """
    MATCH (exp:Experiment {name: $name})
    MATCH (exp)-[:HAS_VARIANT]->(v)
    OPTIONAL MATCH (v)-[:USED_BY]->(e:Episode)-[:PRODUCED_OUTCOME]->(o:OutcomeEvaluation)
    RETURN v.id as variant, avg(o.reward) as avg_reward, count(o) as sample_size
    ORDER BY avg_reward DESC
    """
    results = engine.query_cypher(query, {"name": experiment_name})

    if not results:
        return f"No results found for experiment '{experiment_name}'."

    summary = [f"Experiment: {experiment_name}"]
    for row in results:
        summary.append(
            f"- Variant {row['variant']}: Reward {row['avg_reward']:.2f} (n={row['sample_size']})"
        )

    return "\n".join(summary)
