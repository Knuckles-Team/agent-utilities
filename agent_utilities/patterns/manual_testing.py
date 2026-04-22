#!/usr/bin/python
"""Manual Testing Tools and Artifact Generation.

This module provides tools for python execution, curl exploration, and
Showboat-style artifact generation (ExecutionNotes).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .subagents import dispatch_subagent

logger = logging.getLogger(__name__)


@dataclass
class ExecutionNote:
    """A single note in an execution log."""
    timestamp: str
    type: str  # note, exec, image
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ExecutionNotes:
    """Showboat-style artifact generator for manual testing."""
    
    def __init__(self, goal: str):
        self.goal = goal
        self.notes: list[ExecutionNote] = []
        self.start_time = datetime.now()

    def note(self, message: str):
        """Add a text note to the log."""
        self.notes.append(ExecutionNote(
            timestamp=datetime.now().isoformat(),
            type="note",
            content=message
        ))
        logger.info(f"ExecutionNote [NOTE]: {message}")

    def exec(self, command: str, output: str, exit_code: int = 0):
        """Log a command execution."""
        self.notes.append(ExecutionNote(
            timestamp=datetime.now().isoformat(),
            type="exec",
            content=command,
            metadata={"output": output, "exit_code": exit_code}
        ))
        logger.info(f"ExecutionNote [EXEC]: {command} (exit: {exit_code})")

    def image(self, path: str, caption: str = ""):
        """Log an image artifact (e.g. screenshot)."""
        self.notes.append(ExecutionNote(
            timestamp=datetime.now().isoformat(),
            type="image",
            content=path,
            metadata={"caption": caption}
        ))
        logger.info(f"ExecutionNote [IMAGE]: {path}")

    def to_markdown(self) -> str:
        """Convert the notes to a markdown artifact."""
        md = [f"# Execution Log: {self.goal}\n"]
        md.append(f"**Started at**: {self.start_time.isoformat()}\n")
        
        for n in self.notes:
            ts = n.timestamp.split("T")[1].split(".")[0]
            if n.type == "note":
                md.append(f"### [{ts}] Note\n{n.content}\n")
            elif n.type == "exec":
                md.append(f"### [{ts}] Command\n`{n.content}`\n")
                md.append(f"```text\n{n.metadata.get('output', '')}\n```\n")
            elif n.type == "image":
                md.append(f"### [{ts}] Screenshot\n![{n.metadata.get('caption', '')}]({n.content})\n")
        
        return "\n".join(md)


async def run_manual_test_cycle(goal: str, deps: Any) -> str:
    """Orchestrate a manual testing cycle using subagents."""
    
    logger.info(f"Starting manual testing for goal: {goal}")
    notes = ExecutionNotes(goal)
    
    # In a real scenario, this would involve dispatching a subagent 
    # that uses tools and calls notes.note(), notes.exec(), etc.
    # For now, we provide the infrastructure.
    
    result = await dispatch_subagent(
        goal=f"Perform manual testing for: {goal}. Use python -c or curl to verify functionality. "
             f"Produce an ExecutionNotes artifact summarizing your steps.",
        deps=deps,
        name="Manual-Test-Agent",
        skill_types=["universal", "manual_testing"],
        system_prompt_suffix="You are an expert at manual verification and exploratory testing."
    )
    
    return result
