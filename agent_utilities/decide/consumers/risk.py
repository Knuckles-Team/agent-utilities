"""Advisory pre-tool risk scoring through EG ``Decide``.

The permission verdict stays deterministic: rules, the ontological guardrail
and the identity policy, merged most-restrictive-wins, decide every call. EG
is asked the same question (``allow`` / ``ask`` / ``deny``, security safety,
so never explored) and its answer is ATTACHED to the verdict as advisory
text -- shown to reviewers and logged, never obeyed. A score never grants or
removes authority: it is surfaced beside the deterministic decision, not a
substitute for it (AU-CONTROL-R005).
"""

from __future__ import annotations

from agent_utilities import decide
from agent_utilities.decide.options import Option, text_param

VERDICTS = ("allow", "ask", "deny")


def advisory_risk(tool_name: str, verdict: str, sensitive: bool) -> str | None:
    """EG's advisory verdict for one tool call, or ``None`` when it did not decide."""
    options = [
        Option(
            v,
            {
                "sensitive": 1.0 if sensitive else 0.0,
                "rule_verdict": 1.0 if v == verdict else 0.0,
            },
        )
        for v in VERDICTS
    ]
    choice = decide.choose(
        "au.tool.risk", options, lambda: verdict, params=[text_param("tool", tool_name)]
    )
    if not choice.decided:
        return None
    return f"{choice.option_id} (advisory; record {choice.record_id})"


__all__ = ["VERDICTS", "advisory_risk"]
