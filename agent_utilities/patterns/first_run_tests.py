#!/usr/bin/python
"""First Run Tests Orchestrator.

This module provides tools to run existing tests in a workspace and feed the
results back to the agent (Planner/Verifier).
"""

import asyncio
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Represents the result of a test run."""
    success: bool
    output: str
    exit_code: int
    command: str


async def run_first_tests(
    workspace_path: Path,
    test_command: str = "uv run pytest",
) -> TestResult:
    """Run tests in the workspace and capture results.

    Args:
        workspace_path: Path to the workspace/project root.
        test_command: The command to run tests.

    Returns:
        A TestResult object containing the outcome.
    """
    logger.info(f"Running first tests in {workspace_path} with command: {test_command}")
    
    # Ensure workspace exists
    if not workspace_path.exists():
        return TestResult(
            success=False,
            output=f"Error: Workspace path {workspace_path} does not exist.",
            exit_code=1,
            command=test_command
        )

    try:
        # Run the test command asynchronously
        process = await asyncio.create_subprocess_shell(
            test_command,
            cwd=str(workspace_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy()
        )
        
        stdout, stderr = await process.communicate()
        output = stdout.decode() + stderr.decode()
        exit_code = process.returncode if process.returncode is not None else 1
        
        success = exit_code == 0
        
        if success:
            logger.info("First tests passed successfully.")
        else:
            logger.warning(f"First tests failed with exit code {exit_code}.")
            
        return TestResult(
            success=success,
            output=output,
            exit_code=exit_code,
            command=test_command
        )
        
    except Exception as e:
        logger.error(f"Failed to execute tests: {e}")
        return TestResult(
            success=False,
            output=f"Execution Error: {str(e)}",
            exit_code=1,
            command=test_command
        )
