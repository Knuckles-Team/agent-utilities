"""Parser modules + the full 36-agent registration (CONCEPT:ECO-4.38).

``load_all()`` registers every supported agent source. Coverage tiers:

* **bespoke** — Claude Code, Codex have dedicated, format-faithful parsers.
* **generic JSONL** — the large JSONL-family of agents share
  :mod:`generic_jsonl` (role+content+usage extraction); robust enough to
  ingest sessions and price them.
* **non-JSONL** (SQLite/protobuf/encrypted: cursor, zed, vscode-copilot,
  antigravity, warp, ...) are registered so they are *detected*, with a
  bespoke reader marked as a follow-up; until then their generic reader yields
  nothing rather than mis-parsing.

Mirrors agentsview ``internal/parser/types.go`` (dirs + env overrides).
"""

from __future__ import annotations

from ..registry import AgentSource, register_source
from . import claude, codex, generic_jsonl

# (agent_type, display_name, dirs, env_var, glob, parser, file_based)
# parser: "claude" | "codex" | "generic" | "nonjsonl"
_AGENTS: tuple[tuple, ...] = (
    ("claude", "Claude Code", ("~/.claude/projects",), "CLAUDE_PROJECTS_DIR",
     "**/*.jsonl", "claude", True),
    ("codex", "Codex", ("~/.codex/sessions",), "CODEX_SESSIONS_DIR",
     "**/*.jsonl", "codex", True),
    ("gemini", "Gemini CLI", ("~/.gemini",), "GEMINI_DIR", "**/*.jsonl",
     "generic", True),
    ("copilot", "Copilot CLI", ("~/.copilot",), "COPILOT_DIR", "**/*.jsonl",
     "generic", True),
    ("opencode", "OpenCode", ("~/.local/share/opencode",), "OPENCODE_DIR",
     "**/*.json", "generic", True),
    ("openhands", "OpenHands CLI", ("~/.openhands/conversations",),
     "OPENHANDS_CONVERSATIONS_DIR", "**/*.jsonl", "generic", True),
    ("amp", "Amp", ("~/.local/share/amp/threads",), "AMP_DIR", "**/*.jsonl",
     "generic", True),
    ("zencoder", "Zencoder", ("~/.zencoder/sessions",), "ZENCODER_DIR",
     "**/*.jsonl", "generic", True),
    ("iflow", "iFlow", ("~/.iflow/projects",), "IFLOW_DIR", "**/*.jsonl",
     "generic", True),
    ("pi", "Pi", ("~/.pi/agent/sessions",), "PI_DIR", "**/*.jsonl",
     "generic", True),
    ("qwen", "Qwen Code", ("~/.qwen/projects",), "QWEN_PROJECTS_DIR",
     "**/*.jsonl", "generic", True),
    ("commandcode", "Command Code", ("~/.commandcode/projects",),
     "COMMANDCODE_PROJECTS_DIR", "**/*.jsonl", "generic", True),
    ("openclaw", "OpenClaw", ("~/.openclaw/agents",), "OPENCLAW_DIR",
     "**/*.jsonl", "generic", True),
    ("qclaw", "QClaw", ("~/.qclaw/agents",), "QCLAW_DIR", "**/*.jsonl",
     "generic", True),
    ("kimi", "Kimi", ("~/.kimi/sessions",), "KIMI_DIR", "**/*.jsonl",
     "generic", True),
    ("kiro", "Kiro CLI", ("~/.kiro/sessions/cli", "~/.local/share/kiro-cli"),
     "KIRO_SESSIONS_DIR", "**/*.jsonl", "generic", True),
    ("cortex", "Cortex Code", ("~/.snowflake/cortex/conversations",),
     "CORTEX_DIR", "**/*.jsonl", "generic", True),
    ("hermes", "Hermes Agent", ("~/.hermes/sessions",), "HERMES_SESSIONS_DIR",
     "**/*.jsonl", "generic", True),
    ("workbuddy", "WorkBuddy", ("~/.workbuddy/projects",),
     "WORKBUDDY_PROJECTS_DIR", "**/*.jsonl", "generic", True),
    ("warp", "Warp", ("~/.warp",), "WARP_DIR", "**/*.jsonl", "generic", True),
    # claude.ai / chatgpt web exports (JSON arrays the user drops in)
    ("claude-ai", "Claude.ai", ("~/.agent-utilities/imports/claude-ai",),
     "CLAUDE_AI_DIR", "**/*.json", "generic", True),
    ("chatgpt", "ChatGPT", ("~/.agent-utilities/imports/chatgpt",),
     "CHATGPT_DIR", "**/*.json", "generic", True),
    # Non-JSONL (SQLite/protobuf/encrypted) — detected now, bespoke reader TODO.
    ("cursor", "Cursor", ("~/.cursor/projects",), "CURSOR_PROJECTS_DIR",
     "**/*.jsonl", "nonjsonl", True),
    ("vscode-copilot", "VSCode Copilot",
     ("~/.config/Code/User/workspaceStorage",), "VSCODE_COPILOT_DIR",
     "**/*.db", "nonjsonl", True),
    ("kiro-ide", "Kiro IDE", ("~/.config/Kiro/User",), "KIRO_IDE_DIR",
     "**/*.jsonl", "nonjsonl", True),
    ("positron", "Positron Assistant", ("~/.config/Positron/User",),
     "POSITRON_DIR", "**/*.db", "nonjsonl", True),
    ("zed", "Zed", ("~/.local/share/zed",), "ZED_DIR", "**/*.db", "nonjsonl",
     True),
    ("antigravity", "Antigravity", ("~/.gemini/antigravity",),
     "ANTIGRAVITY_DIR", "**/*.db", "nonjsonl", True),
    ("antigravity-cli", "Antigravity CLI", ("~/.gemini/antigravity-cli",),
     "ANTIGRAVITY_CLI_DIR", "**/*.jsonl", "nonjsonl", True),
    ("piebald", "Piebald", ("~/.local/share/piebald",), "PIEBALD_DIR",
     "**/*.db", "nonjsonl", True),
    ("forge", "Forge", ("~/.forge",), "FORGE_DIR", "**/*.jsonl", "nonjsonl",
     True),
    ("warp-sqlite", "Warp (SQLite)", ("~/.local/share/warp-terminal",),
     "WARP_SQLITE_DIR", "**/*.sqlite", "nonjsonl", True),
    ("kiro-cli-share", "Kiro CLI (shared)", ("~/.aws/kiro",),
     "KIRO_CLI_SHARE_DIR", "**/*.jsonl", "generic", True),
    ("goose", "Goose", ("~/.local/share/goose/sessions",), "GOOSE_DIR",
     "**/*.jsonl", "generic", True),
    ("aider", "Aider", ("~/.aider",), "AIDER_DIR", "**/*.jsonl", "generic",
     True),
    ("cline", "Cline", ("~/.config/cline/tasks",), "CLINE_DIR", "**/*.json",
     "generic", True),
)

_PARSERS = {
    "claude": claude.parse,
    "codex": codex.parse,
    "generic": generic_jsonl.parse,
    "nonjsonl": generic_jsonl.parse,  # placeholder until bespoke readers land
}

# agent_types whose bespoke (SQLite/protobuf) reader is still a follow-up.
NONJSONL_PENDING = frozenset(
    a[0] for a in _AGENTS if a[5] == "nonjsonl"
)

_loaded = False


def load_all() -> None:
    """Register all agent sources (idempotent)."""
    global _loaded
    if _loaded:
        return
    for atype, name, dirs, env, glob, parser, file_based in _AGENTS:
        register_source(
            AgentSource(
                agent_type=atype, display_name=name, default_dirs=dirs,
                env_var=env, file_glob=glob, parse_fn=_PARSERS[parser],
                file_based=file_based, id_prefix=atype,
            )
        )
    _loaded = True
