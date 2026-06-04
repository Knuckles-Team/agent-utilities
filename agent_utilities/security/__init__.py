"""Security module for agent-utilities.

Provides tool-level authorization, prompt injection scanning, tool
repetition detection, and policy-based guardrails.

Modules:
    - ``tool_guard``: Tool-level sensitivity detection and approval gating
    - ``guardrails``: Policy engine with content filtering and cost budgets
    - ``threat_defense_engine``: Pattern-based prompt injection detection (CONCEPT:OS-5.1)
    - ``execution_stability_engine``: Tool call loop detection (CONCEPT:OS-5.1)
    - ``permissions_kernel``: Role-based tool authorization (CONCEPT:OS-5.1)
    - ``sandboxed_executor``: Process-isolated code execution sandbox (CONCEPT:OS-5.6)
"""

from agent_utilities.security.browser_auth import (
    BaseBrowserAuthManager,
    BaseLoopbackCallbackHandler,
    BaseLoopbackCallbackServer,
    generate_pkce,
)
from agent_utilities.security.execution_stability_engine import (
    RepetitionGuard,
    RepetitionPolicy,
    RepetitionResult,
    RepetitionVerdict,
)
from agent_utilities.security.guardrails import (
    ContentFilterPolicy,
    CostBudgetPolicy,
    MaxTokensPolicy,
    OutputSchemaPolicy,
    PolicyEngine,
    PolicyResult,
    PolicyViolation,
)
from agent_utilities.security.sandboxed_executor import (
    SandboxedExecutor,
    SandboxLimits,
    SandboxResult,
)
from agent_utilities.security.threat_defense_engine import (
    PromptInjectionPolicy,
    PromptInjectionScanner,
    RiskLevel,
    ScanResult,
    SecurityFindingNode,
)
from agent_utilities.security.tool_guard import (
    apply_tool_guard_approvals,
    build_sensitive_tool_names,
    flag_mcp_tool_definitions,
    is_safe_tool,
    is_sensitive_tool,
)

__all__ = [
    # browser_auth
    "BaseBrowserAuthManager",
    "BaseLoopbackCallbackHandler",
    "BaseLoopbackCallbackServer",
    "generate_pkce",
    # guardrails
    "ContentFilterPolicy",
    "CostBudgetPolicy",
    "MaxTokensPolicy",
    "OutputSchemaPolicy",
    "PolicyEngine",
    "PolicyResult",
    "PolicyViolation",
    # threat_defense_engine (CONCEPT:OS-5.1)
    "PromptInjectionPolicy",
    "PromptInjectionScanner",
    "RiskLevel",
    "ScanResult",
    "SecurityFindingNode",
    # execution_stability_engine (CONCEPT:OS-5.1)
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
    # sandboxed_executor (CONCEPT:OS-5.6)
    "SandboxedExecutor",
    "SandboxLimits",
    "SandboxResult",
]
