#!/usr/bin/python
               
from __future__ import annotations

import os
import sys
import re
import shutil
import json
import logging
import asyncio
import yaml
import httpx
import argparse
import base64
import contextvars

                            
from typing import Any, Dict, List, Optional, Callable, TYPE_CHECKING
from datetime import datetime, timedelta

if TYPE_CHECKING:
    from fasta2a import Skill
    from fastapi import FastAPI
from pathlib import Path
from contextlib import asynccontextmanager
from importlib.resources import files, as_file

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response, StreamingResponse
from pydantic import ValidationError

from pydantic_ai import Agent, ModelSettings
from pydantic_ai.mcp import (
    load_mcp_servers,
    MCPServerStreamableHTTP,
    MCPServerSSE,
)

from universal_skills.skill_utilities import (
    resolve_mcp_reference,
    get_universal_skills_path,
)


from .config import *
from .workspace import *
from .base_utilities import (
    to_boolean,
    to_integer,
    to_float,
    to_list,
    to_dict,
    retrieve_package_name,
    GET_DEFAULT_SSL_VERIFY,
    load_env_vars,
)

                                                                   

from .models import PeriodicTask

                                 
tasks: List[PeriodicTask] = []
lock = asyncio.Lock()


from pydantic_ai.toolsets.fastmcp import FastMCPToolset

import logging
logger = logging.getLogger(__name__)

def is_sensitive_tool(name: str) -> bool:
    """Check if a tool name matches any sensitive pattern."""
    return any(
        re.match(pattern, name.lower()) for pattern in SENSITIVE_TOOL_PATTERNS
    )



def apply_tool_guard_approvals(agent: "Agent") -> None:
    """Apply requires_approval=True to all sensitive tools on an agent.

    Uses pydantic-ai's native Human-in-the-Loop mechanism.
    When TOOL_GUARD_MODE is 'native', tools matching SENSITIVE_TOOL_PATTERNS
    will require frontend approval before execution.
    The AG-UI / Vercel AI SDK frontend renders an ApprovalCard,
    and the user's response flows back via DeferredToolResults.

    Args:
        agent: The Pydantic AI Agent instance to modify.
    """
    if TOOL_GUARD_MODE == "off":
        logger.debug("Tool guard disabled (TOOL_GUARD_MODE=off)")
        return

    flagged = 0
                                                
    if hasattr(agent, "_function_toolset") and hasattr(
        agent._function_toolset, "tools"
    ):
        for tool_name, tool in agent._function_toolset.tools.items():
            if is_sensitive_tool(tool_name) and tool_name != "run_graph_flow" and not getattr(
                tool, "requires_approval", False
            ):
                tool.requires_approval = True
                flagged += 1

    if flagged:
        logger.info(
            f"Tool Guard (native): Flagged {flagged} sensitive tools with requires_approval=True"
        )
