#!/usr/bin/python
from __future__ import annotations

"""Dynamic two-layer PreToolUse gate for the Claude Code CLI.

CONCEPT:AU-OS.deployment.dynamic-two-fail-closed — Dynamic two-layer fail-closed PreToolUse ActionPolicy permission gate

This is the hook body Claude Code invokes (via ``agent-utilities harness-gate``)
before every gated tool call. It reads the PreToolUse event JSON from stdin and
returns a permission verdict JSON. It is the *dynamic* half of the fence — the
static :mod:`claude_fence` settings.json is the floor; this consults the live
:class:`ActionPolicy` (OS-5.24) at decision time so governance rules (a new
``forbidden`` kind in the YAML or a KG ``governance_rule`` node) take effect
without re-writing the fence.

Two layers, in order:

1. **Static (always-on, daemon-independent).** A secret-path / irreversible-command
   deny built from :mod:`claude_fence` and enforced through the ECO-4.13
   :class:`PermissionPolicyEngine`. Runs first and works even when the graph-os
   engine is unreachable — so secrets stay protected with the daemon down.
2. **Dynamic (governed).** The Bash verb / file op is mapped to an
   :class:`ActionRequest` and classified by ``ActionPolicy.classify`` (a
   side-effect-free tier read — no audit/approval spam per IDE call). The tier
   maps to allow / ask / deny.

**Fail-closed:** any exception, unparseable stdin, or import failure returns a
``deny`` — never a silent allow (mirrors ``ActionPolicy.decide``'s own contract).
"""

import json
import logging
import re
import sys
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["run", "decide_event", "map_event_to_request"]

# Tools this gate adjudicates. Anything else (read-only, MCP, etc.) is deferred
# to Claude Code's own permission system — we only fence mutations + shell.
_GATED_TOOLS = {"Bash", "Edit", "Write", "MultiEdit"}
_FILE_TOOLS = {"Edit", "Write", "MultiEdit"}

# Irreversible / exfiltration command signatures for the static layer. Kept as
# regexes (the static engine matches against the raw command string). This is the
# runtime twin of claude_fence._IRREVERSIBLE_DENY.
_DANGEROUS_CMD = re.compile(
    r"|".join(
        (
            r"\brm\s+-[a-z]*[rf]",  # rm -rf / -fr / -r ... -f
            r"\bgit\s+push\b[^\n]*(--force\b|-f\b|--force-with-lease\b)",
            r"\bgit\s+reset\s+--hard\b",
            r"\bgit\s+clean\s+-[a-z]*f",
            r"\bsudo\b",
            r"\bmkfs\b",
            r"\bdd\s+if=",
            r"\bcurl\b",
            r"\bwget\b",
            r":\(\)\s*\{\s*:\|:",  # fork bomb
            r"\bchmod\s+-R\b",
            r"\bchown\s+-R\b",
            r">\s*/dev/sd",  # raw block-device write
        )
    ),
    re.IGNORECASE,
)

# Bash leading-verb → ActionPolicy fleet kind, so the dynamic layer asks the same
# governance question the fleet does. Unmapped commands are sandboxed dev work
# (workspace.cmd → tier auto → allow).
_VERB_TO_KIND: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bdocker(-compose)?\s+(stack|service)\b"), "redeploy_stack"),
    (re.compile(r"\bdocker(-compose)?\b"), "deploy_service"),
    (re.compile(r"\bsystemctl\b"), "restart_service"),
    (re.compile(r"\bgit\s+merge\b"), "merge_promotion"),
    (re.compile(r"\bgit\s+push\b"), "merge_promotion"),
]


def _deny(reason: str) -> dict[str, Any]:
    return _verdict("deny", reason)


def _allow(reason: str) -> dict[str, Any]:
    return _verdict("allow", reason)


def _ask(reason: str) -> dict[str, Any]:
    return _verdict("ask", reason)


def _verdict(decision: str, reason: str) -> dict[str, Any]:
    """A Claude Code PreToolUse hook decision payload."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": reason,
        }
    }


def map_event_to_request(tool_name: str, tool_input: dict[str, Any]):
    """Map a Claude tool call to an :class:`ActionRequest`, or ``None`` if not gated."""
    from agent_utilities.orchestration.action_policy import ActionRequest

    if tool_name in _FILE_TOOLS:
        path = str(tool_input.get("file_path") or tool_input.get("path") or "")
        return ActionRequest(
            kind="workspace.write", target=path or "*", source="claude-code"
        )
    if tool_name == "Bash":
        command = str(tool_input.get("command") or "")
        for pattern, kind in _VERB_TO_KIND:
            if pattern.search(command):
                return ActionRequest(
                    kind=kind, target=_argv0(command), source="claude-code"
                )
        return ActionRequest(
            kind="workspace.cmd", target=_argv0(command), source="claude-code"
        )
    return None


def _argv0(command: str) -> str:
    parts = command.strip().split()
    return parts[0] if parts else "*"


def _static_check(tool_name: str, tool_input: dict[str, Any], cwd: str) -> None:
    """Raise ``PolicyViolation`` for a secret path or irreversible command.

    Reuses the ECO-4.13 :class:`PermissionPolicyEngine`, overlaying the
    fence-derived secret globs + the dangerous-command regex on top of any
    project ``.agents/permissions.json``. Engine-independent (no graph-os call).
    """
    from agent_utilities.claude_harness.claude_fence import derive_secret_globs
    from agent_utilities.ecosystem.permission_policy import (
        PermissionPolicyEngine,
        PermissionRule,
    )

    engine = PermissionPolicyEngine(workspace=cwd or ".")
    engine.policy.deny_paths = list(
        dict.fromkeys(engine.policy.deny_paths + derive_secret_globs())
    )
    engine.policy.rules.append(
        PermissionRule(
            tool="Bash",
            deny_args={"command": _DANGEROUS_CMD.pattern},
            description="irreversible / exfiltration command (OS-5.41 static floor)",
        )
    )
    engine.check_tool(tool_name, tool_input)


def decide_event(event: dict[str, Any], *, engine: Any = None) -> dict[str, Any]:
    """Adjudicate one PreToolUse event → a verdict payload. Fails closed."""
    from agent_utilities.ecosystem.permission_policy import PolicyViolation

    tool_name = str(event.get("tool_name") or "")
    tool_input = event.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        tool_input = {}
    cwd = str(event.get("cwd") or ".")

    if tool_name not in _GATED_TOOLS:
        return _allow(f"{tool_name or 'tool'} not gated — deferred to Claude")

    # Layer A — static floor (always available, daemon-independent).
    try:
        _static_check(tool_name, tool_input, cwd)
    except PolicyViolation as e:
        return _deny(f"blocked by static floor: {e}")
    except Exception as e:  # noqa: BLE001 — static layer fails CLOSED
        return _deny(f"static gate error (fail closed): {e}")

    # Layer B — governed verdict (side-effect-free tier read; consults KG rules
    # when an engine is supplied, file rules otherwise).
    from agent_utilities.orchestration.action_policy import (
        TIER_APPROVAL,
        TIER_AUTO,
        TIER_AUTO_NOTIFY,
        TIER_FORBIDDEN,
        ActionPolicy,
    )

    request = map_event_to_request(tool_name, tool_input)
    if request is None:
        return _allow("not a governed action")
    try:
        tier = ActionPolicy(engine=engine).classify(request)
    except Exception as e:  # noqa: BLE001 — dynamic layer fails CLOSED
        return _deny(f"governed gate error (fail closed): {e}")
    if tier == TIER_FORBIDDEN:
        return _deny(f"policy forbids {request.kind}")
    if tier == TIER_APPROVAL:
        return _ask(f"{request.kind} requires approval — halt and queue")
    if tier in (TIER_AUTO, TIER_AUTO_NOTIFY):
        return _allow(f"policy tier {tier} for {request.kind}")
    return _ask(f"unrecognized tier {tier!r} — pausing for human")


def run(stdin_text: str | None = None, *, engine: Any = None) -> dict[str, Any]:
    """Read a PreToolUse event from stdin (or ``stdin_text``) → verdict dict.

    The harness CLI prints the returned dict as JSON and exits 0. Any failure to
    read or parse the event denies (fail closed).
    """
    try:
        raw = stdin_text if stdin_text is not None else sys.stdin.read()
    except Exception as e:  # noqa: BLE001 — cannot even read → deny
        return _deny(f"could not read hook input (fail closed): {e}")
    if not raw.strip():
        return _deny("empty hook input (fail closed)")
    try:
        event = json.loads(raw)
        if not isinstance(event, dict):
            raise ValueError("event is not a JSON object")
    except Exception as e:  # noqa: BLE001 — unparseable → deny
        return _deny(f"unparseable hook input (fail closed): {e}")
    try:
        return decide_event(event, engine=engine)
    except Exception as e:  # noqa: BLE001 — any gate error → deny
        logger.warning("pretooluse_gate: decision error (fail closed): %s", e)
        return _deny(f"gate error (fail closed): {e}")
