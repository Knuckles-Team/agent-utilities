"""Security module for agent-utilities.

Provides tool-level authorization, prompt injection scanning, tool
repetition detection, and policy-based guardrails.

Modules:
    - ``tool_guard``: Tool-level sensitivity detection and approval gating
    - ``guardrails``: Policy engine with content filtering and cost budgets
    - ``threat_defense_engine``: Pattern-based prompt injection detection (CONCEPT:AU-OS.config.secrets-authentication)
    - ``execution_stability_engine``: Tool call loop detection (CONCEPT:AU-OS.config.secrets-authentication)
    - ``permissions_kernel``: Role-based tool authorization (CONCEPT:AU-OS.config.secrets-authentication)
    - ``sandboxed_executor``: Process-isolated code execution sandbox (CONCEPT:AU-OS.observability.deterministic-replay)
"""

from agent_utilities.security.browser_auth import (
    BaseBrowserAuthManager,
    BaseLoopbackCallbackHandler,
    BaseLoopbackCallbackServer,
    generate_pkce,
)
from agent_utilities.security.credential_provider import (
    CredentialProvider,
    get_credential_provider,
)
from agent_utilities.security.execution_stability_engine import (
    RepetitionGuard,
    RepetitionPolicy,
    RepetitionResult,
    RepetitionVerdict,
    RetryConfig,
    RetryManager,
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
from agent_utilities.security.source_credentials import (
    ApiKeyCredential,
    AuthMaterial,
    BasicAuthCredential,
    CookieSessionCredential,
    NoCredential,
    OAuth2Credential,
    SourceCredential,
    build_credential,
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
    # credential_provider (CONCEPT:AU-OS.deployment.universal-outbound-credentialprovider) + source_credentials (CONCEPT:AU-OS.config.source-credential-registry)
    "CredentialProvider",
    "get_credential_provider",
    "SourceCredential",
    "NoCredential",
    "ApiKeyCredential",
    "CookieSessionCredential",
    "BasicAuthCredential",
    "OAuth2Credential",
    "AuthMaterial",
    "build_credential",
    # guardrails
    "ContentFilterPolicy",
    "CostBudgetPolicy",
    "MaxTokensPolicy",
    "OutputSchemaPolicy",
    "PolicyEngine",
    "PolicyResult",
    "PolicyViolation",
    # threat_defense_engine (CONCEPT:AU-OS.config.secrets-authentication)
    "PromptInjectionPolicy",
    "PromptInjectionScanner",
    "RiskLevel",
    "ScanResult",
    "SecurityFindingNode",
    # execution_stability_engine (CONCEPT:AU-OS.config.secrets-authentication)
    "RepetitionGuard",
    "RepetitionPolicy",
    "RepetitionResult",
    "RepetitionVerdict",
    # execution_stability_engine (CONCEPT:AU-ORCH.execution.execution-budget-caps) — agent-run retry loops
    # verified by shell SuccessChecks; for HTTP/in-process retry use
    # orchestration.resilience.ResiliencePolicy instead.
    "RetryConfig",
    "RetryManager",
    # tool_guard
    "apply_tool_guard_approvals",
    "build_sensitive_tool_names",
    "flag_mcp_tool_definitions",
    "is_safe_tool",
    "is_sensitive_tool",
    # sandboxed_executor (CONCEPT:AU-OS.observability.deterministic-replay)
    "SandboxedExecutor",
    "SandboxLimits",
    "SandboxResult",
]
