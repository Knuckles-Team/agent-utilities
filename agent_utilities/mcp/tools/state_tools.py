"""Auto-extracted graph-os MCP tools: state_tools (register_state_tools).

Split out of kg_server._build_server to deepen the MCP surface into focused
modules without changing tool behavior or names.
"""

from __future__ import annotations

import json

from pydantic import Field
from starlette.responses import JSONResponse

from agent_utilities.mcp import kg_server


def register_state_tools(mcp):
    """Register the state_tools group on the given FastMCP server."""

    @mcp.tool(
        name="graph_sessions",
        description="Manage durable sessions (action in 'list', 'get', 'delete', 'reply', 'cancel').",
        tags=["graph-os", "sessions"],
    )
    async def graph_sessions(
        action: str = Field(
            description="Action: 'list', 'get', 'delete', 'reply', 'cancel'"
        ),
        session_id: str = Field(default="", description="Target session ID"),
        user_reply: str = Field(
            default="", description="Reply content for 'reply' action"
        ),
    ) -> str:
        """Manage durable sessions. Action: 'list', 'get', 'delete', 'reply', 'cancel'."""

        from agent_utilities.core.sessions import (
            cancel_session_run,
            delete_session,
            get_all_sessions,
            get_session_details,
            submit_session_reply,
        )

        try:
            req = kg_server._build_dummy_request(
                path_params={"session_id": session_id} if session_id else {},
                json_body={"content": user_reply} if user_reply else None,
            )
            if action == "list":
                resp = await get_all_sessions(req)
            elif action == "get":
                if not session_id:
                    return json.dumps({"error": "session_id is required"})
                resp = await get_session_details(req)
            elif action == "delete":
                if not session_id:
                    return json.dumps({"error": "session_id is required"})
                resp = await delete_session(req)
            elif action == "reply":
                if not session_id:
                    return json.dumps({"error": "session_id is required"})
                resp = await submit_session_reply(req)
            elif action == "cancel":
                if not session_id:
                    return json.dumps({"error": "session_id is required"})
                resp = await cancel_session_run(req)
            else:
                return json.dumps({"error": f"Unknown sessions action: {action}"})

            # Check if resp is JSONResponse
            if isinstance(resp, JSONResponse):
                # Return the decoded json string
                body_bytes = bytes(resp.body)
                return json.dumps(json.loads(body_bytes.decode("utf-8")))
            return str(resp)
        except Exception as e:
            return json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["graph_sessions"] = graph_sessions

    @mcp.tool(
        name="graph_goals",
        description="Orchestrate background/autonomous loops (action in 'create', 'list', 'iterations', 'cancel').",
        tags=["graph-os", "goals"],
    )
    async def graph_goals(
        action: str = Field(
            description="Action: 'create', 'list', 'iterations', 'cancel'"
        ),
        goal_id: str = Field(default="", description="Target goal ID"),
        goal: str = Field(
            default="", description="Goal description/instruction for 'create' action"
        ),
        max_iterations: int = Field(
            default=10, description="Max iterations for the autonomous loop"
        ),
    ) -> str:
        """Orchestrate background/autonomous loops. Action: 'create', 'list', 'iterations', 'cancel'."""

        from agent_utilities.core.sessions import (
            cancel_goal,
            create_goal,
            get_goal_iterations,
            list_goals,
        )

        try:
            req = kg_server._build_dummy_request(
                path_params={"goal_id": goal_id} if goal_id else {},
                json_body={"objective": goal, "max_iterations": max_iterations}
                if action == "create"
                else None,
            )
            if action == "list":
                resp = await list_goals(req)
            elif action == "create":
                if not goal:
                    return json.dumps({"error": "goal is required"})
                resp = await create_goal(req)
            elif action == "iterations":
                if not goal_id:
                    return json.dumps({"error": "goal_id is required"})
                req_iter = kg_server._build_dummy_request(
                    path_params={"goal_id": goal_id}
                )
                resp = await get_goal_iterations(req_iter)
            elif action == "cancel":
                if not goal_id:
                    return json.dumps({"error": "goal_id is required"})
                req_cancel = kg_server._build_dummy_request(
                    path_params={"goal_id": goal_id}
                )
                resp = await cancel_goal(req_cancel)
            else:
                return json.dumps({"error": f"Unknown goals action: {action}"})

            # Check if resp is JSONResponse
            if isinstance(resp, JSONResponse):
                # Return the decoded json string
                body_bytes = bytes(resp.body)
                return json.dumps(json.loads(body_bytes.decode("utf-8")))
            return str(resp)
        except Exception as e:
            return json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["graph_goals"] = graph_goals

    @mcp.tool(
        name="graph_loops",
        description=(
            "The single entrypoint for long-running objectives (CONCEPT:KG-2.78). A "
            "Loop is one objective of kind research|develop|skill; the LoopController "
            "advances every active Loop through ONE hot path. action in 'submit' "
            "(create a Loop: objective + kind [+ validation_cmd/end_state for develop, "
            "skill_ref for skill]), 'list' (active Loops), 'run' (advance all active "
            "Loops one cycle — research acquires/reasons, develop validates, skill "
            "executes), 'drive' (run ONE Loop by id to completion, durably — "
            "resume/checkpoint/corrigible, any kind), 'cancel' (terminate a Loop by id)."
        ),
        tags=["graph-os", "loops"],
    )
    async def graph_loops(
        action: str = Field(
            default="list",
            description="submit|list|run|drive|cancel|prioritize",
        ),
        objective: str = Field(default="", description="Objective text (submit)."),
        kind: str = Field(
            default="research", description="research|develop|skill (submit)."
        ),
        loop_id: str = Field(default="", description="Loop id (submit/cancel)."),
        validation_cmd: str = Field(
            default="",
            description="Shell command whose exit-0 completes a develop Loop.",
        ),
        end_state: str = Field(default="", description="Human end-state (develop)."),
        skill_ref: str = Field(
            default="", description="Skill / skill-workflow name or id (skill Loop)."
        ),
        max_topics: int = Field(default=5, description="Loops to advance per run."),
        limit: int = Field(default=10, description="Max rows (list)."),
        priority: str = Field(
            default="normal",
            description="Priority bucket 0-3 or critical|high|normal|background "
            "(submit/prioritize).",
        ),
    ) -> str:
        """Submit / list / run / drive / cancel / prioritize Loops — the one
        Loop-engine entrypoint."""
        import json as _json

        from agent_utilities.knowledge_graph.core.engine_tasks import (
            _coerce_prio_bucket,
        )
        from agent_utilities.knowledge_graph.research.loop_controller import (
            LoopController,
        )
        from agent_utilities.knowledge_graph.research.loops import (
            active_loops,
            mark_loop_status,
            prioritize_loop,
            submit_loop,
        )

        try:
            engine = kg_server._get_engine()
            if action == "submit":
                if not objective and not skill_ref:
                    return _json.dumps({"error": "submit needs objective or skill_ref"})
                loop = submit_loop(
                    engine,
                    objective,
                    kind=kind,  # type: ignore[arg-type]
                    validation_cmd=validation_cmd,
                    end_state=end_state,
                    skill_ref=skill_ref,
                    loop_id=loop_id,
                    prio_bucket=_coerce_prio_bucket(priority),
                )
                return _json.dumps({"action": "submit", "loop": loop}, default=str)
            if action == "list":
                return _json.dumps(
                    {"action": "list", "loops": active_loops(engine, limit)},
                    default=str,
                )
            if action == "run":
                rep = LoopController(engine).run_one_cycle(max_topics=max_topics)
                return _json.dumps(rep, indent=2, default=str)
            if action == "drive":
                # Drive ONE Loop to completion durably (resume/checkpoint/corrigible,
                # CONCEPT:OS-5.16) — works for any kind (research/develop/skill).
                if not loop_id:
                    return _json.dumps({"error": "drive needs a loop_id"})
                target = next(
                    (
                        loop_row
                        for loop_row in active_loops(engine, max(limit, 50))
                        if loop_row.get("id") == loop_id
                    ),
                    None,
                )
                if target is None:
                    return _json.dumps({"error": f"no active loop {loop_id!r}"})
                res = await LoopController(engine).run_loop(target, sleep_s=0)
                return _json.dumps({"action": "drive", "result": res}, default=str)
            if action == "cancel":
                if not loop_id:
                    return _json.dumps({"error": "cancel needs a loop_id"})
                ok = mark_loop_status(engine, loop_id, "cancelled", source="user")
                return _json.dumps({"action": "cancel", "id": loop_id, "ok": ok})
            if action == "prioritize":
                if not loop_id:
                    return _json.dumps({"error": "prioritize needs a loop_id"})
                bucket = _coerce_prio_bucket(priority)
                ok = prioritize_loop(engine, loop_id, bucket)
                return _json.dumps(
                    {
                        "action": "prioritize",
                        "id": loop_id,
                        "prio_bucket": bucket,
                        "ok": ok,
                    }
                )
            return _json.dumps({"error": f"unknown action {action!r}"})
        except Exception as e:
            return _json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["graph_loops"] = graph_loops

    @mcp.tool(
        name="research_artifact",
        description=(
            "Agent-Native Research Artifacts over the one ontology-driven KG "
            "(CONCEPT:KG-2.79/2.80). action in 'reason' (run OWL/RDF reasoning over "
            "the whole ecosystem and harvest extrapolated cross-domain relationships "
            "as research topics), 'compile' (paper -> ecosystem-grounded OWL-native "
            "4-layer ARA), 'review'/'seal' (L1/L2/L3 OWL/SHACL-grounded review + "
            "certificate), 'capture' (live research event w/ provenance), 'get', 'list'."
        ),
        tags=["graph-os", "research", "ontology"],
    )
    async def research_artifact(
        action: str = Field(
            default="reason",
            description="reason|compile|review|seal|capture|get|list",
        ),
        article_id: str = Field(
            default="", description="Paper/article id (compile/review/get)."
        ),
        query: str = Field(
            default="", description="Topic for 'reason' (reasoning is ecosystem-wide)."
        ),
        level: str = Field(default="L1", description="Seal level: L1|L2|L3 (review)."),
        text: str = Field(default="", description="Event text (capture)."),
        provenance: str = Field(
            default="ai_executed",
            description="capture provenance: user|ai_suggested|ai_executed|user_revised.",
        ),
        actor: str = Field(default="", description="Originating actor id (capture)."),
        event_type: str = Field(default="", description="Force event type (capture)."),
        target_codebase: str = Field(
            default="", description="Codebase to ground claims against (compile)."
        ),
        limit: int = Field(default=50, description="Max rows (list)."),
    ) -> str:
        """Run an ARA action over the one ontology-driven KG (single SoT)."""

        from agent_utilities.knowledge_graph.research.ara.service import ARAService

        try:
            service = ARAService(kg_server._get_engine())
            result = service.run(
                action,
                article_id=article_id,
                query=query,
                level=level,
                text=text,
                provenance=provenance,
                actor=actor,
                event_type=event_type,
                target_codebase=target_codebase or None,
                limit=limit,
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["research_artifact"] = research_artifact

    @mcp.tool(
        name="graph_hydrate",
        description="Hydrate the Knowledge Graph from configured external sources. ALIAS of `source_sync` (mode='full') kept for back-compat — `source_sync` is the canonical tool (it adds delta/reconcile modes and the same source='all' fleet sweep). source=<connector> hydrates one; source='all' sweeps every configured connector.",
        tags=["graph-os", "hydration"],
    )
    async def graph_hydrate(
        source: str = Field(
            default="all",
            description="The source connector to hydrate (any registered source), or 'all' to sweep every configured source.",
        ),
    ) -> str:
        """Hydrate the KG from external sources (thin alias of source_sync, mode=full)."""

        from agent_utilities.knowledge_graph.core.source_sync import sync_source

        try:
            engine = kg_server._get_engine()
            # Delegate to the one unified core so there is no divergent hydration
            # logic; 'all' fans out to the fleet sweep (CONCEPT:KG-2.9).
            res = sync_source(engine, source, mode="full")
            return json.dumps(res, default=str)
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    kg_server.REGISTERED_TOOLS["graph_hydrate"] = graph_hydrate

    # ══════════════════════════════════════════════════════════════════
    # Ontology System — Palantir Foundry parity (type/link/function layer)
    #   property types  (CONCEPT:KG-2.47)
    #   value types     (CONCEPT:KG-2.39)
    #   interfaces      (CONCEPT:KG-2.38)
    #   links           (CONCEPT:KG-2.26)
    #   functions       (CONCEPT:KG-2.41)
    #   derived props   (CONCEPT:KG-2.40)
    # All handlers are thin — they reach the live `KnowledgeGraph.ontology`
    # system (bound to the engine's backend) so Functions-on-Objects, derived
    # compute and interface targeting resolve against the real graph.
    # ══════════════════════════════════════════════════════════════════
