"""Security module for agent-utilities.

Provides tool-level authorization, prompt injection scanning, tool
repetition detection, and policy-based guardrails.

Modules:
    - ``tool_guard``: Tool-level sensitivity detection and approval gating
    - ``guardrails``: Policy engine with content filtering and cost budgets
    - ``prompt_scanner``: Pattern-based prompt injection detection (CONCEPT:OS-5.4)
    - ``repetition_guard``: Tool call loop detection (CONCEPT:OS-5.5)
    - ``permissions_kernel``: Role-based tool authorization (CONCEPT:OS-5.1)
"""

from agent_utilities.security.guardrails import (
    ContentFilterPolicy,
    CostBudgetPolicy,
    MaxTokensPolicy,
    OutputSchemaPolicy,
    PolicyEngine,
    PolicyResult,
    PolicyViolation,
)
from agent_utilities.security.prompt_scanner import (
    PromptInjectionPolicy,
    PromptInjectionScanner,
    RiskLevel,
    ScanResult,
    SecurityFindingNode,
)
from agent_utilities.security.repetition_guard import (
    RepetitionGuard,
    RepetitionPolicy,
    RepetitionResult,
    RepetitionVerdict,
)
from agent_utilities.security.tool_guard import (
    apply_tool_guard_approvals,
    build_sensitive_tool_names,
    flag_mcp_tool_definitions,
    is_safe_tool,
    is_sensitive_tool,
)

__all__ = [
    # guardrails
    "ContentFilterPolicy",
    "CostBudgetPolicy",
    "MaxTokensPolicy",
    "OutputSchemaPolicy",
    "PolicyEngine",
    "PolicyResult",
    "PolicyViolation",
    # prompt_scanner (CONCEPT:OS-5.4)
    "PromptInjectionPolicy",
    "PromptInjectionScanner",
    "RiskLevel",
    "ScanResult",
    "SecurityFindingNode",
    # repetition_guard (CONCEPT:OS-5.5)
    "RepetitionGuard",
    "RepetitionPolicy",
    "RepetitionResult",
    "RepetitionVerdict",
    # tool_guard
    "apply_tool_guard_approvals",
    "build_sensitive_tool_names",
    "flag_mcp_tool_definitions",
    "is_safe_tool",
    "is_sensitive_tool",
]
