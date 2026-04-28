import logging

from ..capabilities.hooks import HookEvent, HookInput, HookResult
from .config import RLMConfig
from .repl import RLMEnvironment

logger = logging.getLogger(__name__)


async def rlm_large_output_hook(input: HookInput) -> HookResult | None:
    """
    Hook to intercept massive tool outputs and automatically wrap them in an RLM environment.
    This prevents the main context window from blowing up.
    """
    if input.event != HookEvent.POST_TOOL_USE:
        return None

    config = RLMConfig()
    if not config.enabled:
        return None

    result_str = str(input.result)

    # Check if the output exceeds the max_context_threshold
    if len(result_str) > config.max_context_threshold:
        logger.warning(
            f"Tool '{getattr(input.call, 'tool_name', 'unknown')}' output ({len(result_str)} chars) "
            f"exceeds threshold ({config.max_context_threshold}). Routing to RLM."
        )

        env = RLMEnvironment(
            context=result_str,
            config=config,
            graph_deps=getattr(input.ctx, "deps", None),
        )

        # Use RLM to analyze the output
        prompt = (
            f"The tool {getattr(input.call, 'tool_name', 'unknown')} returned a massive output. "
            f"Your job is to analyze this data in the `context` variable. "
            f"Extract the most relevant insights or summarize the key points needed to answer the user's original query. "
            f"Use FINAL_VAR('summary', <text_or_json>)."
        )

        try:
            rlm_result = await env.run_full_rlm(prompt)
            logger.info("RLM successfully summarized the massive tool output.")
            return HookResult(
                modify_result=f"[RLM Synthesized Summary of Massive Data]\n{rlm_result}"
            )
        except Exception as e:
            logger.error(f"RLM failed to process massive output: {e}")
            return HookResult(
                modify_result=f"[Error: Massive data returned and RLM failed: {e}]"
            )

    return None
