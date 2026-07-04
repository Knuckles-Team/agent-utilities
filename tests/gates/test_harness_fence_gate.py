#!/usr/bin/python
"""Guardrail: the generated Claude Code fence never drifts from the safety floor.

CONCEPT:AU-OS.deployment.governance-derived-claude-code — Governance-derived Claude Code permission-fence generator

A drift guard (mirrors the DEFAULT_POLICY↔YAML parity gate): the fence built from
the SHIPPED ActionPolicy must always carry the irreversible + secret deny floor,
pin ``acceptEdits``, never emit ``bypassPermissions``, and reflect every
``forbidden`` policy rule (that maps to a CLI surface) in its deny list.
"""

from __future__ import annotations

import json

from agent_utilities.claude_harness import claude_fence as cf
from agent_utilities.claude_harness.claude_fence import _KIND_TO_PATTERNS
from agent_utilities.orchestration.action_policy import ActionPolicy, ActionRequest


def test_shipped_fence_has_irreversible_and_secret_floor():
    deny = set(cf.build_settings_dict()["permissions"]["deny"])
    floor = {
        "Bash(rm -rf:*)",
        "Bash(git push --force:*)",
        "Bash(git reset --hard:*)",
        "Bash(curl:*)",
        "Read(.env)",
        "Edit(.env)",
        "Write(secrets/**)",
    }
    missing = floor - deny
    assert not missing, f"fence lost safety-floor deny rules: {missing}"


def test_shipped_fence_pins_accept_edits_no_bypass():
    settings = cf.build_settings_dict()
    assert settings["permissions"]["defaultMode"] == "acceptEdits"
    assert "bypassPermissions" not in json.dumps(settings)


def test_forbidden_policy_rules_propagate_to_deny():
    """Every shipped forbidden rule with a CLI surface must be denied."""
    policy = ActionPolicy()  # shipped default
    deny = set(cf.derive_deny_allow_ask(policy).deny)
    for kind, patterns in _KIND_TO_PATTERNS.items():
        rule, _ = policy._match(ActionRequest(kind=kind, target="*"))
        if rule.tier == "forbidden":
            for pat in patterns:
                assert pat in deny, f"{kind} is forbidden but {pat} not denied"


def test_gate_hook_command_matches_installer():
    """The fence's gate hook and the hook installer must use ONE command string."""
    from agent_utilities.ecosystem.hook_installer import _CLAUDE_HOOKS

    pre = _CLAUDE_HOOKS["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert pre == cf.GATE_HOOK_COMMAND
