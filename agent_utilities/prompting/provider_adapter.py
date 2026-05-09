#!/usr/bin/python
"""CONCEPT:ECO-4.5 — Provider Prompt Adaptation.

Implements an abstracted provider adapter framework for per-provider
prompt optimization, inspired by The Rosetta Prompt research (Source 3).

Architecture:
    Uses an **abstracted backend pattern** (matching vector-mcp):
    ``ProviderBackend`` is the abstract interface; concrete backends
    (``StaticRuleBackend``, ``KGRuleBackend``) implement rule storage
    and retrieval. This enables swapping rule sources without changing
    the adapter logic.

Key capabilities:
    - **Provider-aware prompt transformation**: Applies structural
      patterns optimized for each LLM provider (OpenAI, Anthropic,
      Google, and any future provider via backend registration)
    - **Abstracted rule storage**: Rules can live in a static dict,
      KG ``EngineeringRule`` nodes, or any custom backend
    - **Composable transformations**: Multiple rules can apply to a
      single prompt in priority order

Environment Variables:
    ``KG_PROVIDER_ADAPTER_BACKEND``: Backend to use (``static`` or ``kg``).
        Defaults to ``static``.

See docs/overview.md §CONCEPT:ECO-4.5.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

_DEFAULT_BACKEND = os.getenv("KG_PROVIDER_ADAPTER_BACKEND", "static")


# ── Rule Model ────────────────────────────────────────────────────────


class ProviderRule(BaseModel):
    """A single prompt transformation rule for a specific provider.

    Each rule defines a structural or stylistic optimization pattern
    for a target LLM provider, with optional ``applicable_when``
    conditions for contextual activation (mirrors CONCEPT:KG-2.2
    Engineering Rules Engine).
    """

    rule_id: str = Field(description="Unique identifier for this rule")
    provider: str = Field(
        description="Target provider (e.g., 'openai', 'anthropic', 'google')"
    )
    name: str = Field(description="Human-readable rule name")
    description: str = Field(default="", description="What this rule optimizes")
    priority: int = Field(
        default=50, ge=0, le=100, description="Execution priority (higher = first)"
    )
    transformation_type: str = Field(
        default="structural",
        description="Type: 'structural', 'formatting', 'injection', 'wrapper'",
    )

    # Transformation definition
    prefix: str = Field(default="", description="Text to prepend to the prompt")
    suffix: str = Field(default="", description="Text to append to the prompt")
    replacements: dict[str, str] = Field(
        default_factory=dict,
        description="Key-value pairs for text substitution",
    )
    wrapper_template: str = Field(
        default="",
        description="Template with {prompt} placeholder for wrapping",
    )

    # Contextual activation (CONCEPT:KG-2.2 pattern)
    applicable_when: dict[str, Any] = Field(
        default_factory=dict,
        description="Conditions for rule activation (e.g., {'task_type': 'code'})",
    )


# ── Backend Interface ──────────────────────────────────────────────────


class ProviderBackend(ABC):
    """Abstract backend for provider rule storage and retrieval.

    Follows the vector-mcp abstracted backend pattern: concrete
    implementations handle rule persistence, the adapter handles
    transformation logic.
    """

    @abstractmethod
    def get_rules(self, provider: str) -> list[ProviderRule]:
        """Return all rules for a given provider, ordered by priority."""
        ...

    @abstractmethod
    def add_rule(self, rule: ProviderRule) -> None:
        """Register a new rule."""
        ...

    @abstractmethod
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID. Returns True if found and removed."""
        ...

    @abstractmethod
    def list_providers(self) -> list[str]:
        """Return all providers with registered rules."""
        ...


# ── Static Backend ─────────────────────────────────────────────────────


# Built-in rules derived from Rosetta Prompt research + provider docs
_BUILTIN_RULES: list[ProviderRule] = [
    # --- OpenAI ---
    ProviderRule(
        rule_id="openai-markdown-headers",
        provider="openai",
        name="Markdown Header Optimization",
        description="OpenAI models respond best to markdown-structured prompts with clear H2/H3 sections",
        priority=80,
        transformation_type="structural",
        prefix="",
        suffix="",
    ),
    ProviderRule(
        rule_id="openai-system-role",
        provider="openai",
        name="System Role Reinforcement",
        description="OpenAI models benefit from explicit role definition at the start",
        priority=90,
        transformation_type="injection",
        prefix="You are a specialized AI assistant. Follow all instructions precisely.\n\n",
    ),
    # --- Anthropic ---
    ProviderRule(
        rule_id="anthropic-direct-instructions",
        provider="anthropic",
        name="Direct Instruction Style",
        description="Anthropic models prefer direct, imperative instructions over role-play framing",
        priority=85,
        transformation_type="structural",
        replacements={
            "You are a": "Your task is to act as a",
            "As an AI assistant,": "Complete the following task:",
        },
    ),
    ProviderRule(
        rule_id="anthropic-xml-tags",
        provider="anthropic",
        name="XML Tag Structure",
        description="Anthropic models respond well to XML-tagged sections for input/output separation",
        priority=70,
        transformation_type="wrapper",
        wrapper_template="<instructions>\n{prompt}\n</instructions>",
    ),
    # --- Google ---
    ProviderRule(
        rule_id="google-structured-examples",
        provider="google",
        name="Structured Example Patterns",
        description="Google models benefit from few-shot examples with clear input/output pairs",
        priority=75,
        transformation_type="structural",
        suffix="\n\nProvide your response in a clear, structured format.",
    ),
    ProviderRule(
        rule_id="google-grounding",
        provider="google",
        name="Grounding Context",
        description="Google models perform better when given explicit grounding context",
        priority=80,
        transformation_type="injection",
        prefix="Based on the provided context and your knowledge, ",
    ),
]


class StaticRuleBackend(ProviderBackend):
    """In-memory rule backend with built-in provider optimization rules.

    Pre-loaded with rules derived from Rosetta Prompt research for
    OpenAI, Anthropic, and Google providers. Additional rules can be
    registered at runtime.
    """

    def __init__(self) -> None:
        self._rules: dict[str, list[ProviderRule]] = {}
        for rule in _BUILTIN_RULES:
            self._rules.setdefault(rule.provider, []).append(rule)

    def get_rules(self, provider: str) -> list[ProviderRule]:
        """Return rules for provider, sorted by priority (highest first)."""
        rules = self._rules.get(provider.lower(), [])
        return sorted(rules, key=lambda r: r.priority, reverse=True)

    def add_rule(self, rule: ProviderRule) -> None:
        provider = rule.provider.lower()
        self._rules.setdefault(provider, []).append(rule)

    def remove_rule(self, rule_id: str) -> bool:
        for provider, rules in self._rules.items():
            for i, r in enumerate(rules):
                if r.rule_id == rule_id:
                    rules.pop(i)
                    if not rules:
                        del self._rules[provider]
                    return True
        return False

    def list_providers(self) -> list[str]:
        return list(self._rules.keys())


# ── KG Backend ─────────────────────────────────────────────────────────


class KGRuleBackend(ProviderBackend):
    """Knowledge Graph-backed rule backend.

    Stores rules as ``EngineeringRule`` nodes (CONCEPT:KG-2.2) with
    ``applicable_when`` conditions for contextual activation. Rules
    evolve with the harness — the KG becomes a living prompt
    optimization knowledge base.

    Args:
        engine: The IntelligenceGraphEngine to read/write rules from.
    """

    def __init__(self, engine: IntelligenceGraphEngine) -> None:
        self.engine = engine

    def get_rules(self, provider: str) -> list[ProviderRule]:
        rules: list[ProviderRule] = []
        provider_lower = provider.lower()

        for nid, data in self.engine.graph.nodes(data=True):
            if (
                data.get("type") == "provider_prompt_rule"
                and data.get("provider", "").lower() == provider_lower
            ):
                try:
                    rules.append(
                        ProviderRule(
                            rule_id=nid,
                            provider=provider_lower,
                            name=data.get("name", nid),
                            description=data.get("description", ""),
                            priority=data.get("priority", 50),
                            transformation_type=data.get(
                                "transformation_type", "structural"
                            ),
                            prefix=data.get("prefix", ""),
                            suffix=data.get("suffix", ""),
                            replacements=data.get("replacements", {}),
                            wrapper_template=data.get("wrapper_template", ""),
                            applicable_when=data.get("applicable_when", {}),
                        )
                    )
                except Exception as e:
                    logger.debug("Failed to load KG rule %s: %s", nid, e)

        return sorted(rules, key=lambda r: r.priority, reverse=True)

    def add_rule(self, rule: ProviderRule) -> None:
        self.engine.graph.add_node(
            rule.rule_id,
            type="provider_prompt_rule",
            provider=rule.provider.lower(),
            name=rule.name,
            description=rule.description,
            priority=rule.priority,
            transformation_type=rule.transformation_type,
            prefix=rule.prefix,
            suffix=rule.suffix,
            replacements=rule.replacements,
            wrapper_template=rule.wrapper_template,
            applicable_when=rule.applicable_when,
            importance_score=0.4,
        )

    def remove_rule(self, rule_id: str) -> bool:
        if rule_id in self.engine.graph:
            self.engine.graph.remove_node(rule_id)
            return True
        return False

    def list_providers(self) -> list[str]:
        providers: set[str] = set()
        for _, data in self.engine.graph.nodes(data=True):
            if data.get("type") == "provider_prompt_rule":
                providers.add(data.get("provider", "unknown"))
        return sorted(providers)


# ── Adapter ────────────────────────────────────────────────────────────


class ProviderPromptAdapter:
    """Transforms prompts based on target provider optimization rules.

    CONCEPT:ECO-4.5 — Provider Prompt Adaptation

    Uses an abstracted backend (static or KG) to retrieve and apply
    provider-specific prompt transformation rules. Transformations
    are applied in priority order and are composable.

    Args:
        backend: The rule storage backend. If None, auto-selects based
            on ``KG_PROVIDER_ADAPTER_BACKEND`` env var.
        engine: Optional KG engine for KG-backed rules.
    """

    def __init__(
        self,
        backend: ProviderBackend | None = None,
        engine: IntelligenceGraphEngine | None = None,
    ) -> None:
        if backend is not None:
            self._backend = backend
        elif _DEFAULT_BACKEND == "kg" and engine is not None:
            self._backend = KGRuleBackend(engine)
        else:
            self._backend = StaticRuleBackend()

    @property
    def backend(self) -> ProviderBackend:
        """The active rule backend."""
        return self._backend

    def adapt(
        self,
        prompt: str,
        provider: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Transform a prompt for the target provider.

        Applies all matching rules in priority order. Rules with
        ``applicable_when`` conditions are filtered by the provided
        context dict.

        Args:
            prompt: The original prompt text.
            provider: Target provider name (e.g., 'openai', 'anthropic').
            context: Optional context for conditional rule activation.

        Returns:
            The transformed prompt.
        """
        rules = self._backend.get_rules(provider.lower())
        if not rules:
            return prompt

        # Filter by applicable_when conditions
        active_rules = [r for r in rules if self._check_conditions(r, context or {})]

        if not active_rules:
            return prompt

        result = prompt
        applied: list[str] = []

        for rule in active_rules:
            result = self._apply_rule(result, rule)
            applied.append(rule.rule_id)

        if applied:
            logger.debug(
                "[CONCEPT:ECO-4.5] Applied %d rules for provider '%s': %s",
                len(applied),
                provider,
                applied,
            )

        return result

    def _check_conditions(self, rule: ProviderRule, context: dict[str, Any]) -> bool:
        """Check if a rule's applicable_when conditions are met."""
        if not rule.applicable_when:
            return True  # No conditions = always active

        for key, expected in rule.applicable_when.items():
            actual = context.get(key)
            if actual is None:
                return False
            if isinstance(expected, list):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False

        return True

    def _apply_rule(self, prompt: str, rule: ProviderRule) -> str:
        """Apply a single transformation rule to the prompt."""
        result = prompt

        # Apply prefix
        if rule.prefix and not result.startswith(rule.prefix):
            result = rule.prefix + result

        # Apply suffix
        if rule.suffix and not result.endswith(rule.suffix):
            result = result + rule.suffix

        # Apply replacements
        for old, new in rule.replacements.items():
            result = result.replace(old, new)

        # Apply wrapper template
        if rule.wrapper_template and "{prompt}" in rule.wrapper_template:
            result = rule.wrapper_template.replace("{prompt}", result)

        return result

    def get_supported_providers(self) -> list[str]:
        """Return all providers with registered rules."""
        return self._backend.list_providers()

    def register_rule(self, rule: ProviderRule) -> None:
        """Register a new provider optimization rule."""
        self._backend.add_rule(rule)
        logger.info(
            "[CONCEPT:ECO-4.5] Registered rule '%s' for provider '%s'",
            rule.rule_id,
            rule.provider,
        )

    def unregister_rule(self, rule_id: str) -> bool:
        """Remove a registered rule."""
        return self._backend.remove_rule(rule_id)
