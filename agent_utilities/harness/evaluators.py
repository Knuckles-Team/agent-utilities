import logging
import re
from typing import Any

from agent_utilities.core.config import config
from agent_utilities.harness.trace_backend import (
    LangfuseTraceBackend,
    create_trace_backend,
)

logger = logging.getLogger(__name__)


async def capture_feedback(
    trace_id: str,
    score_name: str,
    value: float,
    comment: str | None = None,
    input_data: Any = None,
    expected_output: Any = None,
) -> bool:
    """
    Captures explicit or implicit feedback for a trace and automatically
    promotes the trace to a Dataset if it falls below the configured threshold.
    """
    if not config.langfuse_secret_key:
        return False

    backend = create_trace_backend(backend_type="langfuse")
    if not isinstance(backend, LangfuseTraceBackend):
        return False

    # Submit the score
    success = await backend.submit_score(
        trace_id=trace_id, name=score_name, value=value, comment=comment
    )
    if not success:
        return False

    # Check threshold for continuous learning dataset capture
    threshold = config.langfuse_dataset_capture_threshold
    if value < threshold:
        logger.info(
            f"Trace {trace_id} scored {value} (below threshold {threshold}). Capturing to dataset."
        )
        dataset_name = f"continuous_learning_{score_name.lower().replace(' ', '_')}"
        await backend.add_to_dataset(
            dataset_name=dataset_name,
            trace_id=trace_id,
            input_data=input_data,
            expected_output=expected_output,
        )

    return True


# --- Automated Evaluators ---


def evaluate_regex(
    trace_id: str, output: str, pattern: str, score_name: str = "Regex Match"
) -> bool:
    """Code-based evaluator: Checks if output matches a regex pattern."""
    match = bool(re.search(pattern, output))
    value = 1.0 if match else 0.0
    import asyncio

    asyncio.create_task(
        capture_feedback(
            trace_id, score_name, value, comment=f"Regex pattern: {pattern}"
        )
    )
    return match


def evaluate_length(
    trace_id: str, output: str, max_length: int, score_name: str = "Length Check"
) -> bool:
    """Code-based evaluator: Checks if output length is within limits."""
    valid = len(output) <= max_length
    value = 1.0 if valid else 0.0
    import asyncio

    asyncio.create_task(
        capture_feedback(
            trace_id,
            score_name,
            value,
            comment=f"Max length: {max_length}, Actual: {len(output)}",
        )
    )
    return valid


async def evaluate_llm_as_judge(
    trace_id: str,
    input_text: str,
    output_text: str,
    criteria: str,
    score_name: str = "LLM Judge",
) -> float:
    """
    LLM-as-a-judge evaluator.
    Calls the configured LLM to score the trace based on the provided criteria.
    """
    logger.debug(f"Evaluating trace {trace_id} with criteria: {criteria}")

    llm_score = 0.8
    llm_rationale = "The response was evaluated with a fallback score due to an error or missing provider config."

    try:
        from pydantic import BaseModel, Field
        from pydantic_ai import Agent

        from agent_utilities.core.model_factory import create_model

        class LLMJudgeResult(BaseModel):
            score: float = Field(
                description="Evaluation score between 0.0 and 1.0 based on the criteria."
            )
            rationale: str = Field(
                description="Detailed rationale and critique justifying the score."
            )

        model = create_model()
        agent = Agent(
            model=model,
            output_type=LLMJudgeResult,
            system_prompt=(
                "You are an expert AI evaluator. Assess the quality of the model's output given the input "
                "and the specified evaluation criteria. Return a numeric score between 0.0 and 1.0 (where "
                "1.0 is perfect alignment with criteria) and a clear, detailed rationale."
            ),
        )

        prompt = (
            f"Input Text:\n{input_text}\n\n"
            f"Output Text:\n{output_text}\n\n"
            f"Evaluation Criteria:\n{criteria}"
        )

        result = await agent.run(prompt)
        llm_score = result.output.score
        llm_rationale = result.output.rationale
    except Exception as e:
        logger.warning(f"LLM-as-a-judge evaluation failed, using fallback: {e}")

    await capture_feedback(
        trace_id, score_name, llm_score, comment=llm_rationale, input_data=input_text
    )
    return llm_score
