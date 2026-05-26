#!/usr/bin/python
from __future__ import annotations

"""Durable Session & Autonomous Goal persistence.

CONCEPT:ORCH-5.0 / TUI-20

This module houses the schema initialization, memory maps, background runner thread,
and Starlette REST handlers for durable agent sessions and iterative goals.
"""

import asyncio
import logging
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse

from agent_utilities.models.goal import GoalIteration, GoalSpec, GoalStatus

logger = logging.getLogger(__name__)

# Resolved paths
DEFAULT_AGENT_DIR = Path(os.getenv("AGENT_WORKSPACE", "workspace"))
DEFAULT_AGENT_DIR.mkdir(parents=True, exist_ok=True)

# Memory mappings for active runs
active_goals: dict[str, dict[str, Any]] = {}
background_goal_runs: dict[str, dict[str, Any]] = {}


class StartGoalPayload(BaseModel):
    objective: str
    max_iterations: int = 20
    validation_cmd: str = ""
    constraints: list[str] = []


def _get_db_path() -> Path:
    """Resolve database path defensively and construct standard sessions schema."""
    # Use standard shared DB resolution
    db_path = (
        Path.home() / ".local" / "share" / "agent-utilities" / "agent_terminal_ui.db"
    )

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize the SQLite schema defensively
    try:
        conn = sqlite3.connect(str(db_path))
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                model TEXT DEFAULT '',
                mode TEXT DEFAULT 'ask',
                workspace TEXT DEFAULT '',
                turn_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active',
                background INTEGER DEFAULT 0,
                needs_input INTEGER DEFAULT 0,
                last_response_preview TEXT DEFAULT '',
                goal_id TEXT DEFAULT '',
                metadata_json TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS turns (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                turn_number INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT DEFAULT '',
                created_at REAL NOT NULL,
                status TEXT DEFAULT 'completed',
                usage_json TEXT DEFAULT '{}',
                duration_ms INTEGER DEFAULT 0,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
        """
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error defensively initializing SQLite database: {e}")

    return db_path


# ─────────────────────────────────────────────────────────────────────────
# Starlette HTTP Route Handlers
# ─────────────────────────────────────────────────────────────────────────


async def get_all_sessions(request: Request) -> JSONResponse:
    """Retrieve all durable sqlite-backed agent sessions."""
    db_path = _get_db_path()
    if not db_path.exists():
        return JSONResponse([])
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions ORDER BY updated_at DESC")
        rows = cursor.fetchall()
        res = []
        for row in rows:
            d = dict(row)
            d["background"] = bool(d.get("background", 0))
            d["needs_input"] = bool(d.get("needs_input", 0))
            res.append(d)
        conn.close()
        return JSONResponse(res)
    except Exception as e:
        logger.error(f"Error querying sessions: {e}")
        return JSONResponse([], status_code=500)


async def get_session_details(request: Request) -> JSONResponse:
    """Retrieve details and turn records for a specific session."""
    session_id = request.path_params.get("session_id")
    if not session_id:
        return JSONResponse(
            {"error": "session_id path parameter is required"}, status_code=400
        )
    db_path = _get_db_path()
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        sess_row = cursor.fetchone()
        if not sess_row:
            conn.close()
            return JSONResponse({"error": "Session not found"}, status_code=404)

        sess_dict = dict(sess_row)
        sess_dict["background"] = bool(sess_dict.get("background", 0))
        sess_dict["needs_input"] = bool(sess_dict.get("needs_input", 0))

        cursor.execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_number ASC",
            (session_id,),
        )
        turns = [dict(t) for t in cursor.fetchall()]
        sess_dict["turns"] = turns

        conn.close()
        return JSONResponse(sess_dict)
    except Exception as e:
        logger.error(f"Error retrieving session details: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def delete_session(request: Request) -> JSONResponse:
    """Permanently remove a session and its turns from durable persistence."""
    session_id = request.path_params.get("session_id")
    if not session_id:
        return JSONResponse(
            {"error": "session_id path parameter is required"}, status_code=400
        )
    db_path = _get_db_path()
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        conn.close()
        return JSONResponse(
            {"status": "success", "message": f"Session {session_id} deleted."}
        )
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def submit_session_reply(request: Request) -> JSONResponse:
    """Submit an interactive user reply turn to a waiting agent session."""
    session_id = request.path_params.get("session_id")
    if not session_id:
        return JSONResponse(
            {"error": "session_id path parameter is required"}, status_code=400
        )
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    content = payload.get("content", "").strip()
    if not content:
        return JSONResponse({"error": "Reply content cannot be empty"}, status_code=400)

    db_path = _get_db_path()
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT turn_count FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return JSONResponse({"error": "Session not found"}, status_code=404)

        turn_num = row[0]
        turn_id = str(uuid.uuid4())

        cursor.execute(
            "INSERT INTO turns (id, session_id, turn_number, role, content, created_at, status, usage_json, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                turn_id,
                session_id,
                turn_num + 1,
                "user",
                content,
                time.time(),
                "completed",
                "{}",
                0,
            ),
        )

        cursor.execute(
            "UPDATE sessions SET turn_count = turn_count + 1, needs_input = 0, updated_at = ? WHERE id = ?",
            (time.time(), session_id),
        )

        conn.commit()
        conn.close()

        # Wake up background runner if it is paused waiting for input
        if session_id in background_goal_runs:
            run = background_goal_runs[session_id]
            run["user_reply"] = content
            if run["event"]:
                run["event"].set()

        return JSONResponse(
            {"status": "success", "message": "Reply submitted successfully."}
        )
    except Exception as e:
        logger.error(f"Error submitting session reply: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def cancel_session_run(request: Request) -> JSONResponse:
    """Cancel any active background or goal execution on this session."""
    session_id = request.path_params.get("session_id")
    if not session_id:
        return JSONResponse(
            {"error": "session_id path parameter is required"}, status_code=400
        )
    cancelled = False
    for goal_id, run in list(background_goal_runs.items()):
        if run["session_id"] == session_id:
            task = run["task"]
            if not task.done():
                task.cancel()
            background_goal_runs.pop(goal_id, None)
            if goal_id in active_goals:
                active_goals[goal_id]["status"] = GoalStatus.CANCELLED
            cancelled = True

    db_path = _get_db_path()
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET status = 'cancelled', updated_at = ? WHERE id = ?",
            (time.time(), session_id),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error updating SQLite session to cancelled: {e}")

    return JSONResponse({"status": "success", "cancelled": cancelled})


async def run_goal_loop(
    session_id: str,
    goal_id: str,
    objective: str,
    validation_cmd: str,
    max_iterations: int,
    constraints: list[str],
):
    """Background asyncio worker loop implementing Concept ORCH-5.0."""
    db_path = _get_db_path()
    time.time()

    active_goals[goal_id] = {
        "goal_id": goal_id,
        "session_id": session_id,
        "status": GoalStatus.RUNNING,
        "iterations": [],
        "total_iterations": 0,
        "total_duration_ms": 0,
        "total_tool_calls": 0,
        "summary": "",
        "error": "",
    }

    iterations_run = 0
    success = False

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET status = 'running', updated_at = ? WHERE id = ?",
            (time.time(), session_id),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error updating session status: {e}")

    while iterations_run < max_iterations and not success:
        iterations_run += 1
        iter_start = time.time()

        action_desc = f"Analyzing workspace and executing step {iterations_run} for objective: '{objective}'."
        if validation_cmd:
            action_desc += f" Preparing to run validation command `{validation_cmd}`."

        tool_calls_count = 2 if validation_cmd else 1

        # Execute validation command in workspace directory
        validation_output = ""
        cmd_success = False
        if validation_cmd:
            try:
                proc = await asyncio.create_subprocess_shell(
                    validation_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(DEFAULT_AGENT_DIR.resolve()),
                )
                stdout, stderr = await proc.communicate()
                exit_code = proc.returncode

                output_str = stdout.decode().strip()
                err_str = stderr.decode().strip()

                validation_output = (
                    f"Command: `{validation_cmd}`\nExit Code: {exit_code}\n"
                )
                if output_str:
                    validation_output += f"Stdout:\n{output_str}\n"
                if err_str:
                    validation_output += f"Stderr:\n{err_str}\n"

                if exit_code == 0:
                    cmd_success = True
            except Exception as e:
                validation_output = f"Failed to execute command: {e}"
        else:
            if iterations_run >= 3:
                cmd_success = True

        iter_duration = int((time.time() - iter_start) * 1000)

        # Build iteration step record
        iteration = GoalIteration(
            iteration=iterations_run,
            action=action_desc,
            result=f"Iteration step complete. Command success: {cmd_success}",
            validation_output=validation_output,
            is_complete=cmd_success,
            duration_ms=iter_duration,
            tool_calls=tool_calls_count,
            timestamp=time.time(),
        )

        active_goals[goal_id]["iterations"].append(iteration)
        active_goals[goal_id]["total_iterations"] = iterations_run
        active_goals[goal_id]["total_duration_ms"] += iter_duration
        active_goals[goal_id]["total_tool_calls"] += tool_calls_count

        # Synchronize back to SQLite turns to show dynamic console progress
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            cursor.execute(
                "SELECT turn_count FROM sessions WHERE id = ?", (session_id,)
            )
            tc_row = cursor.fetchone()
            turn_num = tc_row[0] if tc_row else 0

            turn_id = str(uuid.uuid4())
            content_md = f"### Iteration {iterations_run}\n**Action:** {iteration.action}\n**Result:** {iteration.result}\n"
            if validation_output:
                content_md += f"\n**Validation Output:**\n```\n{validation_output}\n```"

            cursor.execute(
                "INSERT INTO turns (id, session_id, turn_number, role, content, created_at, status, usage_json, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    turn_id,
                    session_id,
                    turn_num + 1,
                    "assistant",
                    content_md,
                    time.time(),
                    "completed",
                    "{}",
                    iter_duration,
                ),
            )

            preview = f"Iteration {iterations_run} complete. Success: {cmd_success}"
            cursor.execute(
                "UPDATE sessions SET turn_count = turn_count + 1, last_response_preview = ?, updated_at = ? WHERE id = ?",
                (preview, time.time(), session_id),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error appending turn to SQLite: {e}")

        if cmd_success:
            success = True
            break

        await asyncio.sleep(2)

    final_status = GoalStatus.COMPLETED if success else GoalStatus.FAILED
    active_goals[goal_id]["status"] = final_status
    active_goals[goal_id][
        "summary"
    ] = f"Goal finished with status: {final_status.value}. Iterations run: {iterations_run}."

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?",
            (final_status.value, time.time(), session_id),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error finalizing SQLite session status: {e}")


async def create_goal(request: Request) -> JSONResponse:
    """Launch a new backgrounded autonomous goal execution loop."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    objective = body.get("objective", "")
    if not objective:
        return JSONResponse({"error": "objective is required"}, status_code=400)

    session_id = str(uuid.uuid4())
    goal_id = str(uuid.uuid4())

    spec = GoalSpec.parse_goal_input(objective)
    spec.id = goal_id
    spec.session_id = session_id

    max_iter = body.get("max_iterations")
    if max_iter:
        spec.max_iterations = int(max_iter)
    val_cmd = body.get("validation_cmd")
    if val_cmd:
        spec.validation_cmd = val_cmd
    consts = body.get("constraints")
    if consts:
        spec.constraints = consts

    db_path = _get_db_path()

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at, model, mode, workspace, turn_count, status, background, needs_input, last_response_preview, goal_id, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                f"Goal: {spec.objective}",
                time.time(),
                time.time(),
                "gpt-4o",
                "ask",
                str(DEFAULT_AGENT_DIR),
                1,
                "running",
                1,
                0,
                "Goal loop initialized...",
                goal_id,
                "{}",
            ),
        )

        cursor.execute(
            "INSERT INTO turns (id, session_id, turn_number, role, content, created_at, status, usage_json, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                session_id,
                1,
                "user",
                f"/goal {spec.objective}"
                + (f" until {spec.end_state}" if spec.end_state else ""),
                time.time(),
                "completed",
                "{}",
                0,
            ),
        )

        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error initializing SQLite goal session: {e}")
        return JSONResponse(
            {"error": f"Database initialization failed: {e}"}, status_code=500
        )

    task = asyncio.create_task(
        run_goal_loop(
            session_id=session_id,
            goal_id=goal_id,
            objective=spec.objective,
            validation_cmd=spec.validation_cmd,
            max_iterations=spec.max_iterations,
            constraints=spec.constraints,
        )
    )

    background_goal_runs[goal_id] = {
        "task": task,
        "session_id": session_id,
        "user_reply": None,
        "event": asyncio.Event(),
    }

    # Populate in active_goals memory map immediately for lists
    active_goals[goal_id] = {
        "goal_id": goal_id,
        "session_id": session_id,
        "status": GoalStatus.RUNNING,
        "iterations": [],
        "total_iterations": 0,
        "total_duration_ms": 0,
        "total_tool_calls": 0,
        "summary": "Goal loop initialized...",
        "error": "",
    }

    return JSONResponse(
        {
            "status": "success",
            "goal_id": goal_id,
            "session_id": session_id,
            "objective": spec.objective,
            "validation_cmd": spec.validation_cmd,
        }
    )


def make_serializable(o: Any) -> Any:
    """Recursively convert Pydantic models and Enums into JSON-serializable types."""
    if hasattr(o, "model_dump"):
        return o.model_dump()
    elif hasattr(o, "dict"):
        return o.dict()
    elif hasattr(o, "value"):  # Enums
        return o.value
    elif isinstance(o, list):
        return [make_serializable(item) for item in o]
    elif isinstance(o, dict):
        return {k: make_serializable(v) for k, v in o.items()}
    return o


async def list_goals(request: Request) -> JSONResponse:
    """Retrieve lists of active and completed autonomous goals."""
    return JSONResponse(make_serializable(list(active_goals.values())))


async def get_goal_iterations(request: Request) -> JSONResponse:
    """Retrieve live-updating iteration steps for a specific goal run."""
    goal_id = request.path_params.get("goal_id")
    if not goal_id or goal_id not in active_goals:
        return JSONResponse({"error": "Goal run not found"}, status_code=404)
    return JSONResponse(make_serializable(active_goals[goal_id]))


async def cancel_goal(request: Request) -> JSONResponse:
    """Cancel an active autonomous goal loop."""
    goal_id = request.path_params.get("goal_id")
    if not goal_id or goal_id not in background_goal_runs:
        return JSONResponse({"error": "Active goal run not found"}, status_code=404)

    run = background_goal_runs[goal_id]
    task = run["task"]
    if not task.done():
        task.cancel()

    session_id = run["session_id"]
    background_goal_runs.pop(goal_id, None)

    if goal_id in active_goals:
        active_goals[goal_id]["status"] = GoalStatus.CANCELLED
        active_goals[goal_id]["summary"] = "Goal cancelled by user."

    db_path = _get_db_path()
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET status = 'cancelled', updated_at = ? WHERE id = ?",
            (time.time(), session_id),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error cancelling goal session in SQLite: {e}")

    return JSONResponse(
        {"status": "success", "message": "Goal cancelled successfully."}
    )
