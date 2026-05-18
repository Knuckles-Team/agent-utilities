import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _get_analysis_model_id() -> str | None:
    """Return the model ID for analysis tasks.

    Checks ANALYSIS_MODEL_ID → LITE_LLM_MODEL_ID → None (use default).
    This allows operators to point analysis at a non-thinking model
    (e.g. gemma-4) while keeping the main LLM as a thinking model.
    """
    return os.environ.get("ANALYSIS_MODEL_ID") or os.environ.get("LITE_LLM_MODEL_ID")


def _extract_result_text(result: Any) -> str:
    """Extract the actual text output from a pydantic-ai RunResult.

    Handles the qwen3/thinking-model edge case where the model puts
    all output into `reasoning_content` and leaves `content` empty.
    """
    # Try the standard pydantic-ai attributes first
    text = ""
    if hasattr(result, "output") and result.output:
        text = str(result.output)
    elif hasattr(result, "data") and result.data:
        text = str(result.data)

    # If still empty, dig into the raw messages for reasoning_content
    if not text and hasattr(result, "all_messages"):
        for msg in reversed(result.all_messages()):
            # Check for model response parts
            if hasattr(msg, "parts"):
                for part in msg.parts:
                    if hasattr(part, "content") and part.content:
                        return str(part.content)
            # Direct content check
            if hasattr(msg, "content") and msg.content:
                return str(msg.content)

    return text


async def _run_l2_synthesis(
    ctx: Any, engine: Any, query: str, enriched: list[dict]
) -> dict[str, Any]:
    """Layer 2: Freeform LLM synthesis — no strict JSON schema required.

    Uses natural language output from the LLM and returns the raw text
    as the synthesis result. This avoids Pydantic output_type failures
    with models that cannot reliably produce constrained JSON.
    """
    import asyncio

    # Build synthesis prompt from L1 results
    match_lines = []
    for r in enriched[:15]:
        score = r.get("score", 0)
        score = float(score) if score is not None else 0.0
        node = r.get("node", r)
        name = node.get("name", node.get("id", r.get("name", r.get("id", ""))))
        desc = node.get("description", "")[:200] if node.get("description") else ""
        match_lines.append(f"- **{name}** (score={score:.3f})")
        if desc:
            match_lines.append(f"  {desc}")
        for claim in r.get("innovation_claims", [])[:2]:
            match_lines.append(f"  > {claim[:200]}")

    synthesis_prompt = (
        f"## Cross-Reference Analysis: {query}\n\n"
        f"The following {len(enriched)} results were found via semantic "
        f"cross-reference against the Knowledge Graph:\n\n"
        + "\n".join(match_lines)
        + "\n\n---\n\n"
        "Analyze these matches and extract actionable feature recommendations "
        "for enhancing an agent orchestration framework. "
        "For each recommendation, describe: the feature name, what concepts it "
        "enhances, a brief implementation sketch, expected impact, and priority. "
        "Use clear markdown formatting with headers for each recommendation."
    )

    system_prompt = (
        "You are an expert software architect analyzing codebases and research "
        "cross-referenced against an agent framework's Knowledge Graph. "
        "Produce a clear, actionable analysis in markdown. Do NOT output JSON."
    )

    try:
        from pydantic_ai import Agent

        from agent_utilities.core.model_factory import create_model

        analysis_model = _get_analysis_model_id()
        agent = Agent(
            model=create_model(model_id=analysis_model),
            system_prompt=system_prompt,
        )
        result = await asyncio.to_thread(agent.run_sync, synthesis_prompt)
        synthesis_text = _extract_result_text(result)

        return {
            "layer": 2,
            "synthesis": synthesis_text,
            "items_analyzed": len(enriched),
        }
    except Exception as e:
        logger.warning("L2 synthesis failed: %s", e)
        # Fallback: return raw L1 match summaries as the synthesis
        return {
            "layer": 2,
            "synthesis": "\n".join(match_lines) if match_lines else "No matches found.",
            "items_analyzed": len(enriched),
            "note": f"LLM synthesis unavailable ({e}), returning L1 summaries.",
        }


async def _run_l3_extraction(
    ctx: Any, engine: Any, query: str, enriched: list[dict]
) -> dict[str, Any]:
    """Layer 3: Freeform deep extraction — no strict JSON schema required.

    Filters high-weight matches and asks the LLM for deep technical
    analysis in natural language markdown format.
    """
    import asyncio

    high_weight = []
    for r in enriched:
        score = r.get("score", 0)
        score = float(score) if score is not None else 0.0
        if score > 0.3:
            high_weight.append(r)

    if not high_weight:
        return {
            "layer": 3,
            "papers_analyzed": 0,
            "extraction": "No high-weight matches for deep extraction.",
        }

    # Build a SINGLE batched prompt for all high-weight matches
    match_sections = []
    for i, hw in enumerate(high_weight[:10], 1):
        node = hw.get("node", hw)
        name = node.get(
            "name", node.get("id", hw.get("name", hw.get("id", f"Match-{i}")))
        )
        score = hw.get("score", 0)
        score = float(score) if score is not None else 0.0
        desc = node.get("description", "")[:300] if node.get("description") else ""
        match_sections.append(
            f"### Match {i}: {name}\n"
            f"- Score: {score:.3f}\n"
            f"- Description: {desc or '(none)'}\n"
        )

    batched_prompt = (
        f"## Deep Technical Extraction for: {query}\n\n"
        f"Analyze the following {len(match_sections)} high-scoring matches and "
        f"extract deep technical knowledge for each:\n\n"
        + "\n".join(match_sections)
        + "\n---\n\n"
        "For EACH match, extract:\n"
        "1. Key algorithms/techniques\n"
        "2. Data structures and architectural patterns\n"
        "3. Integration blueprint for an agent orchestration framework\n"
        "4. Specific enhancement recommendations\n\n"
        "Use clear markdown formatting. Do NOT output JSON."
    )

    system_prompt = (
        "You are a deep technical analyst extracting knowledge from "
        "research papers and codebases for integration into an agent framework. "
        "Produce a detailed markdown analysis. Do NOT output JSON."
    )

    try:
        from pydantic_ai import Agent

        from agent_utilities.core.model_factory import create_model

        analysis_model = _get_analysis_model_id()
        agent = Agent(
            model=create_model(model_id=analysis_model),
            system_prompt=system_prompt,
        )
        result = await asyncio.to_thread(agent.run_sync, batched_prompt)
        extraction_text = _extract_result_text(result)

        return {
            "layer": 3,
            "papers_analyzed": len(high_weight),
            "extraction": extraction_text,
        }
    except Exception as e:
        logger.warning("L3 deep extraction failed: %s", e)
        # Fallback: return raw match summaries
        return {
            "layer": 3,
            "extraction": "\n".join(match_sections)
            if match_sections
            else "No matches.",
            "papers_analyzed": len(high_weight),
            "note": f"LLM extraction unavailable ({e}), returning match summaries.",
        }


def _run_owl_cycle(engine: Any) -> dict[str, Any]:
    """Trigger a lightweight OWL reasoning cycle on the engine's graph."""
    try:
        from agent_utilities.knowledge_graph.backends.owl import create_owl_backend
        from agent_utilities.knowledge_graph.core.owl_bridge import OWLBridge

        owl_backend = create_owl_backend()
        bridge = OWLBridge(
            graph=engine.graph,
            owl_backend=owl_backend,
            backend=engine.backend,
        )
        stats = bridge.run_cycle(lightweight=True)
        return {"status": "success", **stats}
    except Exception as e:
        logger.debug("OWL cycle skipped: %s", e)
        return {"status": "skipped", "reason": str(e)}


class GraphAnalyzer:
    """Performs cross-referencing and deep analysis on the Knowledge Graph."""

    def __init__(self, engine: Any):
        self.engine = engine

    async def synthesize(self, query: str, top_k: int) -> dict[str, Any]:
        results = self.engine.search_hybrid(query=query, top_k=top_k)
        if not results:
            return {"error": f"No results found for {query}"}
        return await _run_l2_synthesis(None, self.engine, query, results)

    async def deep_extract(self, query: str) -> dict[str, Any]:
        results = self.engine.search_hybrid(query=query, top_k=20)
        if not results:
            return {"error": f"No results found for {query}"}
        return await _run_l3_extraction(None, self.engine, query, results)

    async def background_research(self, query: str) -> dict[str, Any]:
        """Runs the complete L1 -> L2 -> L3 -> OWL pipeline."""
        results = self.engine.search_hybrid(query=query, top_k=15)
        if not results:
            return {"error": f"No results found for {query}"}

        l2 = await _run_l2_synthesis(None, self.engine, query, results)
        l3 = await _run_l3_extraction(None, self.engine, query, results)
        owl = _run_owl_cycle(self.engine)

        return {
            "status": "completed",
            "query": query,
            "l2_synthesis": l2,
            "l3_extraction": l3,
            "owl_reasoning": owl,
        }

    async def relevance_sweep(self, query: str) -> dict[str, Any]:
        """Scores all ingested items and extracts enhancement features for a target."""
        results = self.engine.search_hybrid(query=query, top_k=50)
        if not results:
            return {"error": f"No results found for {query}"}

        l2 = await _run_l2_synthesis(None, self.engine, query, results)
        return {
            "status": "sweep_completed",
            "query": query,
            "items_analyzed": len(results),
            "synthesis": l2.get("synthesis", ""),
            "note": l2.get("note", ""),
        }
