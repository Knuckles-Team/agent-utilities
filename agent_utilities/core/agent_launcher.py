#!/usr/bin/python
"""Agent Launcher Module.

Standardizes the launching of various agent CLIs (agent-terminal-ui, claude,
opencode, devin) with unified support for prompts and override/auto-approve modes.
"""

import logging
import shlex
import subprocess
import uuid
from typing import Any

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

# Standard mapping of agent types to their CLI flags
AGENT_CLI_CONFIG: dict[str, dict[str, str | None]] = {
    "agent-terminal-ui": {
        "command": "agent-terminal-ui",
        "prompt_flag": "--prompt",
        "override_flag": "--override",
    },
    "claude": {
        "command": "claude",
        "prompt_flag": "-p",
        "override_flag": None,  # Claude-code doesn't currently have a standard yolo flag
    },
    "claude-code": {
        "command": "claude",
        "prompt_flag": "-p",
        "override_flag": None,
    },
    "opencode": {
        "command": "opencode",
        "prompt_flag": "--prompt",
        "override_flag": "--yolo",
    },
    "devin": {
        "command": "devin",
        "prompt_flag": "--prompt",
        "override_flag": None,
    },
}


def launch_agent_in_terminal(
    prompt: str,
    agent_type: str | None = None,
    override: bool = False,
    session_name: str | None = None,
) -> dict[str, Any]:
    """Launch an agent in a visible terminal window using tmux.

    Standardizes the prompt and override flags across different agent CLIs.

    Args:
        prompt: The task or query to send to the agent.
        agent_type: The agent type (e.g., 'agent-terminal-ui', 'claude').
                    Defaults to DEFAULT_TERMINAL_AGENT from config.
        override: If True, use the agent's "yolo" or auto-approve mode.
        session_name: Optional custom tmux session/window name.

    Returns:
        Dictionary containing launch status and details.
    """
    from agent_utilities.core.config import DEFAULT_TERMINAL_AGENT

    target_agent = agent_type if agent_type else DEFAULT_TERMINAL_AGENT
    config = AGENT_CLI_CONFIG.get(
        target_agent,
        {
            "command": target_agent,
            "prompt_flag": "--prompt",
            "override_flag": None,
        },
    )

    base_cmd = config["command"]
    prompt_flag = config["prompt_flag"]
    override_flag = config["override_flag"]

    safe_prompt = shlex.quote(prompt)
    cmd = f"{base_cmd} {prompt_flag} {safe_prompt}"

    if override and override_flag:
        cmd += f" {override_flag}"

    final_session_name = session_name or f"agent_{uuid.uuid4().hex[:6]}"

    try:
        # Check if we are already inside a tmux session
        if setting("TMUX"):
            # Launch in a new window within current session
            launch_cmd = ["tmux", "new-window", "-n", final_session_name, cmd]
        else:
            # Launch in a new detached session
            launch_cmd = ["tmux", "new-session", "-d", "-s", final_session_name, cmd]

        subprocess.run(launch_cmd, check=True)

        return {
            "status": "launched",
            "agent_type": target_agent,
            "session_name": final_session_name,
            "command": cmd,
            "override_applied": bool(override and override_flag),
        }
    except Exception as e:
        logger.error(f"Failed to launch agent {target_agent} in tmux: {e}")
        raise RuntimeError(f"Terminal agent launch failed: {e}") from e
