"""Security module for agent-utilities.

Provides tool-level authorization, prompt injection scanning, tool
repetition detection, and content guardrails.

Modules:
    - ``tool_guard``: Tool-level sensitivity detection and approval gating
    - ``guardrails``: PII sanitization primitives (see
      ``capabilities/content_guardrails.py`` for the live PII/forbidden-content/
      output-schema guardrails wired through ``pydantic-ai-harness``)
    - ``threat_defense_engine``: Pattern-based prompt injection detection (CONCEPT:AU-OS.config.secrets-authentication)
    - ``execution_stability_engine``: Tool call loop detection (CONCEPT:AU-OS.config.secrets-authentication)
    - ``permissions_kernel``: Role-based tool authorization (CONCEPT:AU-OS.config.secrets-authentication)
    - ``sandboxed_executor``: Process-isolated code execution sandbox (CONCEPT:AU-OS.observability.deterministic-replay)
"""

from importlib import import_module
from typing import Any

_LAZY_EXPORTS: dict[str, str] = {
    **dict.fromkeys(
        (
            "BaseBrowserAuthManager",
            "BaseLoopbackCallbackHandler",
            "BaseLoopbackCallbackServer",
            "generate_pkce",
        ),
        ".browser_auth",
    ),
    **dict.fromkeys(
        ("CredentialProvider", "get_credential_provider"),
        ".credential_provider",
    ),
    **dict.fromkeys(
        ("RepetitionGuard", "RepetitionResult", "RepetitionVerdict"),
        ".execution_stability_engine",
    ),
    "PiiSanitizer": ".guardrails",
    **dict.fromkeys(
        ("SandboxedExecutor", "SandboxLimits", "SandboxResult"),
        ".sandboxed_executor",
    ),
    **dict.fromkeys(
        (
            "ApiKeyCredential",
            "AuthMaterial",
            "BasicAuthCredential",
            "CookieSessionCredential",
            "NoCredential",
            "OAuth2Credential",
            "SourceCredential",
            "build_credential",
        ),
        ".source_credentials",
    ),
    **dict.fromkeys(
        ("PromptInjectionScanner", "RiskLevel", "ScanResult", "SecurityFindingNode"),
        ".threat_defense_engine",
    ),
    **dict.fromkeys(
        (
            "apply_tool_guard_approvals",
            "build_sensitive_tool_names",
            "flag_mcp_tool_definitions",
            "is_safe_tool",
            "is_sensitive_tool",
        ),
        ".tool_guard",
    ),
}


def __getattr__(name: str) -> Any:
    """Load an explicitly exported security surface only when requested."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name, package=__name__), name)


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
    "PiiSanitizer",
    # threat_defense_engine (CONCEPT:AU-OS.config.secrets-authentication)
    "PromptInjectionScanner",
    "RiskLevel",
    "ScanResult",
    "SecurityFindingNode",
    # execution_stability_engine (CONCEPT:AU-OS.config.secrets-authentication)
    "RepetitionGuard",
    "RepetitionResult",
    "RepetitionVerdict",
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
