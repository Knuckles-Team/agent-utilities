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
            "The single entrypoint for long-running objectives (CONCEPT:AU-KG.research.these-properties-carry). A "
            "Loop is one objective of kind research|develop|skill; the LoopController "
            "advances every active Loop through ONE hot path. action in 'submit' "
            "(create a Loop: objective + kind [+ validation_cmd/end_state for develop, "
            "skill_ref for skill]), 'list' (active Loops), 'run' (advance all active "
            "Loops one cycle — research acquires/reasons, develop validates, skill "
            "executes), 'drive' (run ONE Loop by id to completion, durably — "
            "resume/checkpoint/corrigible, any kind), 'cancel' (terminate a Loop by id), "
            "'prioritize' (set claim bucket). TRANSPARENCY + STEERING (KG-2.290/292, "
            "OS-5.73): 'state' (LIVE EvolutionState — current stage + why, saturation "
            "gauge, open_gaps trend, velocity, spec backlog), 'specs' (distilled "
            "SpecProposal backlog; filter by status), 'review' (approve|edit|reject a "
            "distilled spec BEFORE it develops — spec_id + decision)."
        ),
        tags=["graph-os", "loops"],
    )
    async def graph_loops(
        action: str = Field(
            default="list",
            description="submit|list|run|drive|cancel|prioritize|state|specs|review",
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
        spec_id: str = Field(
            default="", description="SpecProposal id (review action)."
        ),
        decision: str = Field(
            default="",
            description="approve|edit|reject — spec-review decision (review action).",
        ),
        status: str = Field(
            default="",
            description="Filter SpecProposals by status (specs action): "
            "pending_review|approved|developing|published|reverted|rejected.",
        ),
    ) -> str:
        """Submit / list / run / drive / cancel / prioritize Loops + observe & steer
        the self-evolution flywheel (state / specs / review) — the one entrypoint."""
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
                # CONCEPT:AU-OS.state.unified-durable-state-externalization) — works for any kind (research/develop/skill).
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
            if action == "state":
                # LIVE EvolutionState (CONCEPT:AU-KG.research.evolutionstate-live-surface-per/2.291): current stage + why,
                # saturation gauge, open_gaps trend, velocity, distilled-spec backlog.
                from agent_utilities.knowledge_graph.research.evolution_state import (
                    read_evolution_state,
                )

                return _json.dumps(
                    {"action": "state", "evolution": read_evolution_state(engine)},
                    indent=2,
                    default=str,
                )
            if action == "specs":
                # The distilled-spec backlog (CONCEPT:AU-KG.research.close-distill-develop-seam).
                from agent_utilities.knowledge_graph.research.spec_proposals import (
                    list_specs,
                )

                return _json.dumps(
                    {
                        "action": "specs",
                        "specs": list_specs(
                            engine, status=(status or None), limit=limit
                        ),
                    },
                    default=str,
                )
            if action == "review":
                # Spec-level review/veto BEFORE develop (CONCEPT:AU-OS.config.autonomous-spec-develop-off).
                from agent_utilities.knowledge_graph.research.spec_proposals import (
                    review_spec,
                )

                sid = spec_id or loop_id
                if not sid or not decision:
                    return _json.dumps(
                        {
                            "error": "review needs spec_id and decision (approve|edit|reject)"
                        }
                    )
                return _json.dumps(
                    {
                        "action": "review",
                        "result": review_spec(engine, sid, decision, reviewer="user"),
                    },
                    default=str,
                )
            return _json.dumps({"error": f"unknown action {action!r}"})
        except Exception as e:
            return _json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["graph_loops"] = graph_loops

    @mcp.tool(
        name="graph_schedules",
        description=(
            "Inspect and control the unified scheduler (CONCEPT:AU-OS.state.unified-scheduling-one-intelligent). Every "
            "recurring job — the deploy/schedules.yml entries, the former "
            "fixed-interval maintenance ticks, the self-evolution loop, and the "
            "ScholarX RSS research feed — is a durable :Schedule node the one "
            "scheduler tick enqueues from. action in 'list' (registry + live "
            "run state), 'enable'/'disable' (toggle by name), 'prioritize' (set "
            "the claim bucket 0-3 of the enqueued job), 'set_interval' (retune "
            "cadence, seconds), 'run_now' (fire on the next tick)."
        ),
        tags=["graph-os", "scheduler"],
    )
    async def graph_schedules(
        action: str = Field(
            default="list",
            description="list|enable|disable|prioritize|set_interval|run_now",
        ),
        name: str = Field(default="", description="Schedule name (all but list)."),
        priority: str = Field(
            default="normal",
            description="Bucket 0-3 or critical|high|normal|background (prioritize).",
        ),
        interval_s: float = Field(
            default=0.0, description="New interval seconds (set_interval)."
        ),
    ) -> str:
        """List / enable / disable / prioritize / retune / run-now schedules."""
        import json as _json

        from agent_utilities.core import schedule_engine as _se

        try:
            engine = kg_server._get_engine()
            if action == "list":
                return _json.dumps(
                    {"action": "list", "schedules": _se.calendar(engine)},
                    default=str,
                )
            if not name:
                return _json.dumps({"error": f"{action} needs a schedule name"})
            if action == "enable":
                return _json.dumps(_se.set_enabled(engine, name, True))
            if action == "disable":
                return _json.dumps(_se.set_enabled(engine, name, False))
            if action == "prioritize":
                return _json.dumps(_se.set_priority(engine, name, priority))
            if action == "set_interval":
                return _json.dumps(_se.set_interval(engine, name, interval_s))
            if action == "run_now":
                return _json.dumps(_se.run_now(engine, name))
            return _json.dumps({"error": f"unknown action {action!r}"})
        except Exception as e:  # noqa: BLE001
            return _json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["graph_schedules"] = graph_schedules

    @mcp.tool(
        name="graph_sandbox",
        description=(
            "Inspect and control the native warm-fork sandbox runtime (CONCEPT:AU-ORCH.sandbox.graph-sandbox-surface). "
            "The RLM code-execution tier boots a runtime warm once and forks children from "
            "copy-on-write state (forkserver/os.fork, Wizer-warmed wasm, warm container pool, "
            "firecracker microVM) instead of cold-booting per snippet. action in 'status' "
            "(per-rung availability + pooled warm-parent count + per-rung reward EMA), 'reap' "
            "(close idle warm parents now + idle dev-workspaces), 'warm' (pre-pay a rung's "
            "start-up so the next fan-out forks cheaply — name it with rung). Code execution "
            "itself stays inside the governed RLM loop; this surface is lifecycle + visibility."
        ),
        tags=["graph-os", "sandbox", "warm-fork"],
    )
    async def graph_sandbox(
        action: str = Field(default="status", description="status|reap|warm"),
        rung: str = Field(
            default="", description="Rung to warm (warm): forkserver|container_fork|..."
        ),
    ) -> str:
        """Status / reap / warm the warm-fork sandbox rungs (CONCEPT:AU-ORCH.sandbox.graph-sandbox-surface, CONCEPT:AU-OS.host.so-they-are-idle)."""
        import json as _json

        try:
            if action == "status":
                from agent_utilities.deployment.doctor import _check_warm_fork
                from agent_utilities.rlm.sandboxes.reward import SandboxRewardTracker

                res = _check_warm_fork()
                data = res.get("data") or {}
                return _json.dumps(
                    {
                        "action": "status",
                        "status": res.get("status"),
                        "detail": res.get("detail"),
                        "rungs": data.get("rungs", {}),
                        "warm_rungs": data.get("warm_rungs", []),
                        "pool": data.get("pool", {}),
                        "rewards": SandboxRewardTracker.get().snapshot(),
                    },
                    default=str,
                )

            from agent_utilities.runtime.warm_registry import WarmParentRegistry

            if action == "reap":
                reaped = WarmParentRegistry.get().reap()
                workspaces: list[str] = []
                try:
                    from agent_utilities.runtime.docker_workspace import DockerWorkspace

                    workspaces = DockerWorkspace.reap_idle()
                except Exception:  # noqa: BLE001 - dev-workspace reap is best-effort
                    pass
                return _json.dumps(
                    {
                        "action": "reap",
                        "reaped_parents": reaped,
                        "reaped_workspaces": workspaces,
                        "pool": WarmParentRegistry.get().stats(),
                    }
                )

            if action == "warm":
                if not rung:
                    return _json.dumps({"error": "warm needs a rung name"})
                from agent_utilities.rlm.sandboxes.base import ForkableSandbox
                from agent_utilities.rlm.sandboxes.registry import default_sandboxes

                backend = next((b for b in default_sandboxes() if b.name == rung), None)
                if backend is None or not isinstance(backend, ForkableSandbox):
                    return _json.dumps(
                        {"error": f"{rung!r} is not a warm-fork rung on this host"}
                    )
                registry = WarmParentRegistry.get()
                spec = backend.warm_spec()
                already = registry.acquire(spec.key) is not None
                if not already:
                    parent = await backend.warm(spec)
                    registry.register(
                        spec.key, parent, close=parent.close, kind=backend.name
                    )
                return _json.dumps(
                    {
                        "action": "warm",
                        "rung": rung,
                        "already_warm": already,
                        "pool": registry.stats(),
                    }
                )

            return _json.dumps({"error": f"unknown action {action!r}"})
        except Exception as e:  # noqa: BLE001
            return _json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["graph_sandbox"] = graph_sandbox

    @mcp.tool(
        name="graph_feeds",
        description=(
            "Manage the unified RSS/Atom feed registry (CONCEPT:AU-KG.ingest.rss-feed-connector/2.122). "
            "Native RSS feeds, the FreshRSS aggregator, and ScholarX arXiv are "
            "first-class :FeedSource nodes ingested through ONE world-model gate "
            "(research items → prioritized fetch, news → relevance+novelty). action "
            "in 'list' (registered feeds), 'add' (register one OR many native "
            "RSS/Atom feeds — pass url=, or urls= a JSON array / comma-separated "
            "list for a bulk add), 'remove' (deregister one or many by url/urls), "
            "'sync' (run the feed sweep now — native RSS + ScholarX through the "
            "gate; feeds are fetched concurrently)."
        ),
        tags=["graph-os", "feeds"],
    )
    async def graph_feeds(
        action: str = Field(default="list", description="list|add|remove|sync"),
        url: str = Field(default="", description="Feed URL (single add/remove)."),
        urls: str = Field(
            default="",
            description=(
                "BULK add/remove: many feed URLs in ONE call — a JSON array "
                '(\'["https://a/feed","https://b/rss"]\') or a comma/newline-'
                "separated string. Combined with `url` and deduped."
            ),
        ),
        mode: str = Field(default="delta", description="delta|full (sync)."),
    ) -> str:
        """List / add / remove / sync unified RSS feed sources (add/remove are bulk-capable)."""
        import json as _json
        import re as _re

        from agent_utilities.automation.feed_sources import (
            list_feed_sources,
            remove_feed_source,
            upsert_feed_source,
        )

        def _url_list() -> list[str]:
            """Resolve url + urls into a deduped, ordered list (JSON array or delimited)."""
            out: list[str] = []
            raw = (urls or "").strip()
            if raw:
                parsed: object = None
                try:
                    parsed = _json.loads(raw)
                except Exception:  # noqa: BLE001 — not JSON → fall back to delimiters
                    parsed = None
                if isinstance(parsed, list):
                    out.extend(str(x).strip() for x in parsed)
                else:
                    out.extend(p.strip() for p in _re.split(r"[,\n]", raw))
            if url:
                out.append(url.strip())
            seen: set[str] = set()
            deduped: list[str] = []
            for u in out:
                if u and u not in seen:
                    seen.add(u)
                    deduped.append(u)
            return deduped

        try:
            engine = kg_server._get_engine()
            if action == "list":
                return _json.dumps(
                    {"action": "list", "feeds": list_feed_sources(engine)}, default=str
                )
            if action == "add":
                targets = _url_list()
                if not targets:
                    return _json.dumps(
                        {"error": "add needs a feed url (url=... or urls=[...])"}
                    )
                results: list[dict] = []
                for u in targets:
                    try:
                        nid = upsert_feed_source(
                            engine,
                            key=u,
                            source_system="rss",
                            feed_url=u,
                            kind="RssFeed",
                        )
                        results.append({"url": u, "id": nid})
                    except Exception as e:  # noqa: BLE001 — one bad feed never aborts the batch
                        results.append({"url": u, "error": str(e)})
                added = [r for r in results if "id" in r]
                return _json.dumps(
                    {
                        "action": "add",
                        "added": len(added),
                        "total": len(targets),
                        "results": results,
                    }
                )
            if action == "remove":
                targets = _url_list()
                if not targets:
                    return _json.dumps(
                        {"error": "remove needs a feed url (url=... or urls=[...])"}
                    )
                results = []
                for u in targets:
                    try:
                        ok = remove_feed_source(engine, key=u, source_system="rss")
                        results.append({"url": u, "ok": bool(ok)})
                    except Exception as e:  # noqa: BLE001
                        results.append({"url": u, "error": str(e)})
                return _json.dumps(
                    {
                        "action": "remove",
                        "removed": sum(1 for r in results if r.get("ok")),
                        "total": len(targets),
                        "results": results,
                    }
                )
            if action == "sync":
                # Run the sweep OFF the request path (CONCEPT:AU-KG.ingest.rss-feed-connector): enqueue a
                # feed_sweep task and return immediately, so a many-feed sweep (which
                # fetches + gates + enqueues per-article worldview/research tasks)
                # never rides — or times out — the 300s MCP call. ``url`` may name a
                # specific source (rss|freshrss|all); defaults to the native RSS sweep.
                submit = getattr(engine, "submit_task", None)
                feed_source = (url or "rss").strip().lower()
                if callable(submit):
                    job_id = submit(
                        target_path=f"feed_sweep:{feed_source}",
                        is_codebase=False,
                        provenance={"feed_sweep": feed_source},
                        task_type="feed_sweep",
                        priority=2,
                        skip_dedupe=True,
                        extra_meta={"feed_source": feed_source, "feed_mode": mode},
                    )
                    return _json.dumps(
                        {
                            "action": "sync",
                            "enqueued": True,
                            "job_id": job_id,
                            "source": feed_source,
                            "mode": mode,
                            "note": "sweep runs in the background (connectors lane); "
                            "watch the worldview/research lanes drain.",
                        }
                    )
                # Fallback: no queue (embedded engine) → run inline.
                from agent_utilities.knowledge_graph.core.source_sync import sync_source

                return _json.dumps(
                    sync_source(engine, feed_source, mode=mode), default=str
                )
            return _json.dumps({"error": f"unknown action {action!r}"})
        except Exception as e:  # noqa: BLE001
            return _json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["graph_feeds"] = graph_feeds

    @mcp.tool(
        name="research_artifact",
        description=(
            "Agent-Native Research Artifacts over the one ontology-driven KG "
            "(CONCEPT:AU-KG.research.best-effort-lightweight-never/2.80). action in 'reason' (run OWL/RDF reasoning over "
            "the whole ecosystem and harvest extrapolated cross-domain relationships "
            "as research topics), 'compile' (paper -> ecosystem-grounded OWL-native "
            "4-layer ARA), 'review'/'seal' (L1/L2/L3 OWL/SHACL-grounded review + "
            "certificate), 'capture' (live research event w/ provenance), 'get', 'list', "
            "'inquire' (native multi-perspective STORM inquiry: expert lenses -> "
            "contradiction/agreement/blind-spot map + self-critique, CONCEPT:AU-KG.research.perspectival-inquiry)."
        ),
        tags=["graph-os", "research", "ontology"],
    )
    async def research_artifact(
        action: str = Field(
            default="reason",
            description="reason|compile|review|seal|capture|get|list|inquire",
        ),
        topic: str = Field(default="", description="Topic to inquire into (inquire)."),
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
        materialize: bool = Field(
            default=True, description="Persist inquiry nodes (inquire)."
        ),
    ) -> str:
        """Run an ARA action over the one ontology-driven KG (single SoT)."""

        from agent_utilities.knowledge_graph.research.ara.service import ARAService

        try:
            service = ARAService(kg_server._get_engine())
            result = service.run(
                action,
                article_id=article_id,
                topic=topic or query,
                query=query,
                level=level,
                text=text,
                provenance=provenance,
                actor=actor,
                event_type=event_type,
                target_codebase=target_codebase or None,
                limit=limit,
                materialize=materialize,
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
            # logic; 'all' fans out to the fleet sweep (CONCEPT:AU-KG.ingest.enterprise-source-extractor).
            res = sync_source(engine, source, mode="full")
            return json.dumps(res, default=str)
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    kg_server.REGISTERED_TOOLS["graph_hydrate"] = graph_hydrate

    # ══════════════════════════════════════════════════════════════════
    # Ontology System — Palantir Foundry parity (type/link/function layer)
    #   property types  (CONCEPT:AU-KG.ontology.ontology-property-types)
    #   value types     (CONCEPT:AU-KG.ontology.value-type-shacl-load)
    #   interfaces      (CONCEPT:AU-KG.ontology.conformance-check)
    #   links           (CONCEPT:AU-KG.domains.trade-journal-bias-auditor)
    #   functions       (CONCEPT:AU-KG.ontology.default-runtime-bound-import)
    #   derived props   (CONCEPT:AU-KG.ontology.derived-property-registry)
    # All handlers are thin — they reach the live `KnowledgeGraph.ontology`
    # system (bound to the engine's backend) so Functions-on-Objects, derived
    # compute and interface targeting resolve against the real graph.
    # ══════════════════════════════════════════════════════════════════
