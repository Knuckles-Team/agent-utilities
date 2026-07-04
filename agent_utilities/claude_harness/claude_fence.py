#!/usr/bin/python
from __future__ import annotations

"""Governance-derived Claude Code permission fence generator.

CONCEPT:AU-OS.deployment.governance-derived-claude-code — Governance-derived Claude Code permission-fence generator

The unattended-Claude-Code pattern needs a ``settings.json`` permission fence
(``allow``/``ask``/``deny`` + ``defaultMode``) plus a ``.claudeignore`` so the
CLI can run while you sleep without a vague prompt force-pushing ``main`` or a
cleanup command nuking a directory. A hand-written file is the common approach;
this module **derives** the fence instead, from two sources of truth we already
own:

1. the irreversible-command + secret-file safety floor (hard-coded baseline), and
2. the live :class:`ActionPolicy` (``orchestration/action_policy``, OS-5.24) —
   every ``forbidden`` rule becomes a static ``deny`` and every
   ``approval_required`` rule becomes an ``ask`` (where the rule's fleet *kind*
   maps to a shell/MCP surface the CLI can emit).

Because the deny list is regenerated from the live policy on every run, adding a
``forbidden`` governance rule (in ``deploy/action-policy.default.yml`` or a KG
``governance_rule`` node) propagates into the IDE fence automatically — it
surpasses a static hand-edited file. The matching *dynamic* gate lives in
:mod:`pretooluse_gate` (OS-5.41); together they are the fence the article draws
once, made self-updating.

Precedence is Claude Code's own ``deny > allow > ask``; we additionally de-dup so
nothing in ``allow``/``ask`` shadows a ``deny``. ``defaultMode`` is hard-pinned to
``acceptEdits`` and ``bypassPermissions`` is never emitted.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from agent_utilities.deployment.config_generator import _SECRET_SUFFIXES

logger = logging.getLogger(__name__)

__all__ = [
    "GATE_HOOK_COMMAND",
    "FenceRules",
    "derive_secret_globs",
    "is_secret_path",
    "derive_deny_allow_ask",
    "build_settings_dict",
    "build_claudeignore_text",
    "write_fence",
]

# The single definition of the dynamic-gate hook command (imported by the
# installer paths so the string never drifts between the fence and the hook).
GATE_HOOK_COMMAND = "agent-utilities harness-gate"

# Tools whose calls the dynamic gate inspects (mutations + shell). Read is
# intentionally excluded so a normal session does not pay a hook round-trip on
# every file read — secret *reads* are already covered by the static deny list.
GATE_MATCHER = "Bash|Edit|Write|MultiEdit"

# Claude pins this; we never widen it and never emit bypassPermissions.
_DEFAULT_MODE = "acceptEdits"

# ── Safety floor — irreversible / exfiltration commands that must never run
# unattended. Claude Code Bash rules match a command prefix; ``:*`` allows args.
_IRREVERSIBLE_DENY: tuple[str, ...] = (
    "Bash(rm -rf:*)",
    "Bash(rm -fr:*)",
    "Bash(git push --force:*)",
    "Bash(git push -f:*)",
    "Bash(git push --force-with-lease:*)",
    "Bash(git reset --hard:*)",
    "Bash(git clean -fd:*)",
    "Bash(git clean -fdx:*)",
    "Bash(sudo:*)",
    "Bash(dd:*)",
    "Bash(mkfs:*)",
    "Bash(:(){ :|:& };:)",  # fork bomb
    "Bash(chmod -R:*)",
    "Bash(chown -R:*)",
    "Bash(curl:*)",  # raw outbound exfiltration path
    "Bash(wget:*)",
)

# Secret-bearing files — globs for ``.claudeignore`` AND Read/Edit/Write deny.
_SECRET_FILE_GLOBS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.keystore",
    "id_rsa",
    "id_rsa.*",
    "id_ed25519",
    "id_ed25519.*",
    "credentials.json",
    "*.credentials",
    "secrets/**",
    ".ssh/**",
)

# Routine, reversible work — runs without asking. Generic across stacks.
_ALLOW_BASELINE: tuple[str, ...] = (
    "Read(*)",
    "Edit(src/**)",
    "Edit(tests/**)",
    "Edit(test/**)",
    "Edit(docs/**)",
    "Bash(ls:*)",
    "Bash(cat:*)",
    "Bash(grep:*)",
    "Bash(rg:*)",
    "Bash(find:*)",
    "Bash(git status:*)",
    "Bash(git diff:*)",
    "Bash(git log:*)",
    "Bash(git add:*)",
    "Bash(git commit:*)",
    "Bash(git branch:*)",
    "Bash(git stash:*)",
    "Bash(npm run test:*)",
    "Bash(npm run lint:*)",
    "Bash(npm run build:*)",
    "Bash(pytest:*)",
    "Bash(uv run pytest:*)",
    "Bash(ruff:*)",
    "Bash(pre-commit run:*)",
    "Bash(cargo test:*)",
    "Bash(cargo build:*)",
    "Bash(make:*)",
)

# Medium-stakes work — pauses for a human (or, unattended, halts and is queued).
_ASK_BASELINE: tuple[str, ...] = (
    "Bash(git push:*)",
    "Bash(git merge:*)",
    "Bash(git rebase:*)",
    "Bash(npm install:*)",
    "Bash(pip install:*)",
    "Bash(uv add:*)",
    "Bash(docker:*)",
    "Bash(docker-compose:*)",
    "Bash(systemctl:*)",
    "Edit(package.json)",
    "Edit(pyproject.toml)",
)

# Map an ActionPolicy fleet *kind* to the Claude Code surfaces a CLI could use to
# perform it — so a policy verdict on the kind fences the matching tool calls.
# Only kinds with a concrete CLI/MCP surface appear; others govern at runtime
# through the dynamic gate (pretooluse_gate) only.
_KIND_TO_PATTERNS: dict[str, tuple[str, ...]] = {
    "deploy_service": ("Bash(docker:*)", "Bash(docker-compose:*)"),
    "redeploy_stack": ("Bash(docker stack:*)", "Bash(docker service update:*)"),
    "restart_service": ("Bash(systemctl:*)", "Bash(docker restart:*)"),
    "rollback_service": ("Bash(docker service rollback:*)",),
    "scale_service": ("Bash(docker service scale:*)",),
    "merge_promotion": ("Bash(git merge:*)", "Bash(git push:*)"),
}


class FenceRules:
    """The three permission buckets, de-duped with ``deny > allow > ask``."""

    def __init__(self, deny: list[str], ask: list[str], allow: list[str]) -> None:
        deny_set = list(dict.fromkeys(deny))
        # ask never shadows a deny; allow never shadows deny or ask.
        ask_set = [r for r in dict.fromkeys(ask) if r not in deny_set]
        allow_set = [
            r for r in dict.fromkeys(allow) if r not in deny_set and r not in ask_set
        ]
        self.deny = deny_set
        self.ask = ask_set
        self.allow = allow_set


def is_secret_path(path: str) -> bool:
    """True when ``path`` names a credential-bearing file.

    Reuses the deployment module's ``_SECRET_SUFFIXES`` (so a file whose stem
    ends in ``_TOKEN`` / ``_SECRET`` / ``PASSWORD`` … is caught) plus the
    standard secret-file globs. Shared by ``.claudeignore`` generation and the
    dynamic gate's static layer.
    """
    import fnmatch

    p = Path(path)
    name = p.name
    rel = str(path)
    for glob in _SECRET_FILE_GLOBS:
        if fnmatch.fnmatch(name, glob) or fnmatch.fnmatch(rel, glob):
            return True
    # Reuse the deployment secret suffixes against the full filename so a
    # credential-bearing extension (``db.token``) or stem (``service_secret``)
    # is caught even without a matching glob.
    up = name.upper()
    return any(up.endswith(suffix) for suffix in _SECRET_SUFFIXES)


def derive_secret_globs() -> list[str]:
    """Secret-file globs for ``.claudeignore`` and the Read/Edit/Write deny."""
    return list(_SECRET_FILE_GLOBS)


def _secret_tool_denies() -> list[str]:
    """Read/Edit/Write/MultiEdit deny entries for every secret glob."""
    out: list[str] = []
    for glob in _SECRET_FILE_GLOBS:
        for tool in ("Read", "Edit", "Write", "MultiEdit"):
            out.append(f"{tool}({glob})")
    return out


def derive_deny_allow_ask(policy: Any | None = None) -> FenceRules:
    """Build the three fence buckets from the safety floor + ``ActionPolicy``.

    ``policy`` is an :class:`ActionPolicy` (engine-optional). Its ``forbidden``
    rules become ``deny`` and its ``approval_required`` rules become ``ask`` for
    every fleet kind that maps to a CLI/MCP surface (:data:`_KIND_TO_PATTERNS`).
    Passing ``None`` constructs the default file-policy gate.
    """
    if policy is None:
        from agent_utilities.orchestration.action_policy import ActionPolicy

        policy = ActionPolicy()

    deny = list(_IRREVERSIBLE_DENY) + _secret_tool_denies()
    ask = list(_ASK_BASELINE)
    allow = list(_ALLOW_BASELINE)

    for kind, patterns in _KIND_TO_PATTERNS.items():
        tier = _kind_tier(policy, kind)
        if tier == "forbidden":
            deny.extend(patterns)
        elif tier in ("approval_required", "auto_notify"):
            ask.extend(patterns)
        # 'auto' leaves the baseline placement (kept conservative in ask) intact.

    return FenceRules(deny=deny, ask=ask, allow=allow)


def _kind_tier(policy: Any, kind: str) -> str:
    """Resolve the policy tier for a fleet ``kind`` (best-effort, never raises)."""
    try:
        from agent_utilities.orchestration.action_policy import ActionRequest

        rule, _defaults = policy._match(ActionRequest(kind=kind, target="*"))
        return str(rule.tier)
    except Exception as e:  # noqa: BLE001 — policy probe is best-effort
        logger.debug("claude_fence: tier probe for %s failed: %s", kind, e)
        return "approval_required"


def _gate_hooks() -> dict[str, Any]:
    """The PreToolUse dynamic-gate hook block (Claude Code settings shape)."""
    return {
        "PreToolUse": [
            {
                "matcher": GATE_MATCHER,
                "hooks": [{"type": "command", "command": GATE_HOOK_COMMAND}],
            }
        ]
    }


def build_settings_dict(policy: Any | None = None) -> dict[str, Any]:
    """Assemble the complete generated ``settings.json`` mapping (no merge).

    Invariants: ``defaultMode == 'acceptEdits'``; ``bypassPermissions`` never
    present; ``deny`` listed first; buckets de-duped so ``allow``/``ask`` never
    shadow a ``deny``.
    """
    rules = derive_deny_allow_ask(policy)
    permissions = {
        "defaultMode": _DEFAULT_MODE,
        "deny": rules.deny,
        "ask": rules.ask,
        "allow": rules.allow,
    }
    hooks = _gate_hooks()
    settings: dict[str, Any] = {
        "$schema": "https://json.schemastore.org/claude-code-settings.json",
        "_generated_by": "agent-utilities harness-fence (CONCEPT:AU-OS.deployment.governance-derived-claude-code)",
        "permissions": permissions,
        "hooks": hooks,
    }
    settings["_fingerprint"] = _fingerprint(permissions, hooks)
    return settings


def build_claudeignore_text() -> str:
    """Render the ``.claudeignore`` so secrets never even load into context."""
    header = (
        "# Generated by agent-utilities harness-fence (CONCEPT:AU-OS.deployment.governance-derived-claude-code).\n"
        "# Secrets listed here never enter Claude Code's context at all —\n"
        "# the deny list blocks edits; this blocks the read.\n"
    )
    return header + "\n".join(derive_secret_globs()) + "\n"


def _fingerprint(permissions: dict[str, Any], hooks: dict[str, Any]) -> str:
    """Stable hash of the generated content so an unchanged re-run is a no-op."""
    blob = json.dumps({"permissions": permissions, "hooks": hooks}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _merge_permission_lists(
    existing: dict[str, Any], generated: dict[str, Any]
) -> dict[str, Any]:
    """Union generated buckets into an operator's existing permissions.

    ``deny`` is monotonic (a generated/operator deny is never dropped — removing
    a governance rule fails *safe*, still denied). ``allow``/``ask`` union so
    operator hand-edits are preserved. Re-applied through :class:`FenceRules` so
    the ``deny > allow > ask`` de-dup invariant holds on the merged result.
    """
    merged = FenceRules(
        deny=list(existing.get("deny", [])) + list(generated.get("deny", [])),
        ask=list(existing.get("ask", [])) + list(generated.get("ask", [])),
        allow=list(existing.get("allow", [])) + list(generated.get("allow", [])),
    )
    return {
        "defaultMode": _DEFAULT_MODE,  # always pinned, never bypassPermissions
        "deny": merged.deny,
        "ask": merged.ask,
        "allow": merged.allow,
    }


def write_fence(
    target_dir: str | Path,
    policy: Any | None = None,
    *,
    merge: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Write/merge the fence into ``<target_dir>/settings.json`` + ``.claudeignore``.

    ``target_dir`` is the Claude config directory (e.g. ``~/.claude`` or
    ``<project>/.claude``); ``.claudeignore`` is written to its parent (the
    project / home root Claude reads it from). Idempotent: an unchanged policy
    re-merges to the same content and reports ``changed: False`` without
    rewriting. Operator hand-edits in ``allow``/``ask`` are preserved.
    """
    target = Path(target_dir).expanduser()
    settings_path = target / "settings.json"
    claudeignore_path = target.parent / ".claudeignore"

    generated = build_settings_dict(policy)
    gen_perms = generated["permissions"]

    existing: dict[str, Any] = {}
    if merge and settings_path.exists():
        try:
            existing = json.loads(settings_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {}
        except Exception as e:  # noqa: BLE001 — a corrupt file must not block the fence
            logger.warning(
                "claude_fence: unreadable %s (%s) — overwriting", settings_path, e
            )
            existing = {}

    out = dict(existing)
    raw_perms = existing.get("permissions")
    existing_perms: dict[str, Any] = raw_perms if isinstance(raw_perms, dict) else {}
    out["permissions"] = (
        _merge_permission_lists(existing_perms, gen_perms) if merge else gen_perms
    )
    # Merge hooks: preserve the operator's other hook events, refresh PreToolUse.
    raw_hooks = existing.get("hooks")
    existing_hooks: dict[str, Any] = raw_hooks if isinstance(raw_hooks, dict) else {}
    merged_hooks = dict(existing_hooks)
    merged_hooks["PreToolUse"] = generated["hooks"]["PreToolUse"]
    out["hooks"] = merged_hooks
    out["$schema"] = generated["$schema"]
    out["_generated_by"] = generated["_generated_by"]
    out["_fingerprint"] = _fingerprint(out["permissions"], out["hooks"])

    # Guard the cardinal invariant defensively (belt-and-suspenders for tests).
    if "bypassPermissions" in json.dumps(out):
        raise ValueError("refusing to emit bypassPermissions in a generated fence")

    ignore_text = build_claudeignore_text()
    unchanged = (
        settings_path.exists()
        and existing.get("_fingerprint") == out["_fingerprint"]
        and claudeignore_path.exists()
        and claudeignore_path.read_text(encoding="utf-8") == ignore_text
    )

    result: dict[str, Any] = {
        "target": str(target),
        "settings_path": str(settings_path),
        "claudeignore_path": str(claudeignore_path),
        "default_mode": _DEFAULT_MODE,
        "deny_count": len(out["permissions"]["deny"]),
        "ask_count": len(out["permissions"]["ask"]),
        "allow_count": len(out["permissions"]["allow"]),
        "fingerprint": out["_fingerprint"],
        "changed": not unchanged,
        "dry_run": dry_run,
    }
    if dry_run or unchanged:
        result["settings"] = out if dry_run else None
        return result

    target.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    claudeignore_path.parent.mkdir(parents=True, exist_ok=True)
    claudeignore_path.write_text(ignore_text, encoding="utf-8")
    return result
