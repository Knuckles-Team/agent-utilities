"""Security Policy Middleware (OS-5.4, OS-5.5, OS-5.11, OS-5.12 synthesis).

Combines Jailbreak Hardening, Prompt Injection, Doom-Loop tracking,
and Repetition Guards into a single cohesive security gateway.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SecurityViolation(Exception):
    """Raised when a security guard blocks an action."""

    pass


class SecurityPolicyMiddleware:
    """Central gateway for all security checks across I/O boundaries."""

    def __init__(self):
        self.max_tool_repetitions = 3
        self._tool_history: dict[str, int] = {}

    def intercept_input(self, prompt: str) -> str:
        """Scan incoming user prompts for injections and jailbreaks."""
        if self._detect_jailbreak(prompt):
            logger.error("Jailbreak pattern detected in input prompt.")
            raise SecurityViolation("Prompt injection/jailbreak detected.")
        return prompt

    def intercept_tool_call(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Scan outbound tool calls for repetitions and doom loops."""
        key = f"{tool_name}_{str(args)}"
        count = self._tool_history.get(key, 0)

        if count >= self.max_tool_repetitions:
            logger.error(
                f"Doom-loop detected: Tool {tool_name} repeated {count} times."
            )
            raise SecurityViolation(f"Tool repetition guard triggered for {tool_name}.")

        self._tool_history[key] = count + 1
        return True

    def _detect_jailbreak(self, prompt: str) -> bool:
        """Detect jailbreak patterns (DAN, AIM, UCAR) and topological risks."""
        # Simple static checks for the synthesized concept.
        # Advanced implementations will use subgraph comparisons.
        suspicious_keywords = ["ignore all previous", "DAN", "developer mode", "bypass"]
        return any(k.lower() in prompt.lower() for k in suspicious_keywords)
