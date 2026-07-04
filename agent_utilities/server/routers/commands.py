import logging

from fastapi import APIRouter, Request

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/enhanced/commands", tags=["Command Center"])


@router.post(
    "/execute", summary="Execute a slash command centrally inside the backend gateway"
)
async def execute_slash_command(payload: dict, request: Request):
    """Execute a slash command centrally inside the backend."""
    command_str = payload.get("command", "").strip()

    if not command_str.startswith("/"):
        return {
            "response_markdown": "Error: Command must start with a slash `/`.",
            "client_actions": [],
        }

    parts = command_str[1:].split(maxsplit=1)
    cmd_name = parts[0].lower() if parts else ""
    args = parts[1] if len(parts) > 1 else ""

    # Standardize cmd_name aliases
    if cmd_name == "quit":
        cmd_name = "exit"

    client_actions = []

    if cmd_name == "help":
        response_md = (
            "### Available Commands:\n\n"
            "- `/help` - Show this help menu\n"
            "- `/clear` - Clear active chat session\n"
            "- `/model [model_id]` - View or change current LLM model\n"
            "- `/tools` - List all available MCP tools\n"
            "- `/skills` - List loaded custom skills\n"
            "- `/graph stats` - Display knowledge graph statistics\n"
            "- `/graph nodes [type]` - List graph nodes\n"
            "- `/graph search <query>` - Run semantic search on graph\n"
            "- `/graph impact <symbol>` - Run blast radius/impact analysis\n"
            "- `/kb list` - List connected knowledge bases\n"
            "- `/kb search <query>` - Query semantic knowledge base articles\n"
            "- `/kb ingest <url_or_path>` - Ingest folder/website to KB\n"
            "- `/sdd specs` - List active spec-driven specifications\n"
            "- `/sdd constitution` - Read spec governance rules\n"
            "- `/sdd sync` - Synchronize local files with KG specifications\n"
            "- `/cron calendar` - View scheduled background tasks\n"
            "- `/cron logs` - Check cron job execution logs\n"
            "- `/resources` - List spawned subagents and tasks\n"
            "- `/resources spawn <name>` - Deploy a new subagent\n"
        )
        return {"response_markdown": response_md, "client_actions": []}

    elif cmd_name == "clear":
        return {
            "response_markdown": "Chat session cleared.",
            "client_actions": [{"action": "clear_chat"}],
        }

    elif cmd_name == "model":
        registry = getattr(request.app.state, "model_registry", None)
        if not args:
            current_model = registry.get_default() if registry else None
            model_id = current_model.id if current_model else "unknown"
            response_md = f"Current active model: `{model_id}`.\n\nUse `/model <model_id>` to change it."
        else:
            client_actions.append({"action": "set_model", "value": args})
            response_md = f"Switched model to `{args}`."
        return {"response_markdown": response_md, "client_actions": client_actions}

    elif cmd_name == "tools":
        agent = getattr(request.app.state, "agent_instance", None)
        tools = []
        if agent and hasattr(agent, "_tools"):
            for t in agent._tools:
                tools.append(f"- `{t.name}`: {t.description}")
        mcp_toolsets = getattr(request.app.state, "mcp_toolsets", [])
        for toolset in mcp_toolsets:
            if hasattr(toolset, "tools"):
                for t in toolset.tools:
                    tools.append(f"- `[{toolset.name}] {t.name}`: {t.description}")
        if not tools:
            response_md = "No tools currently registered."
        else:
            response_md = "### Registered Tools:\n\n" + "\n".join(tools)
        return {"response_markdown": response_md, "client_actions": []}

    elif cmd_name == "skills":
        skills = []
        # Get from registered A2A skills or dynamic workspace skills
        agent_instance = getattr(request.app.state, "agent_instance", None)
        if agent_instance and hasattr(agent_instance, "skills"):
            for s in agent_instance.skills:
                skills.append(f"- **{s.name}** (`{s.id}`): {s.description}")
        if not skills:
            response_md = "No custom skills currently active."
        else:
            response_md = "### Active Custom Skills:\n\n" + "\n".join(skills)
        return {"response_markdown": response_md, "client_actions": []}

    elif cmd_name == "graph":
        sub_parts = args.split(maxsplit=1)
        sub = sub_parts[0].lower() if sub_parts else "stats"
        rest = sub_parts[1] if len(sub_parts) > 1 else ""

        # Route to the live engine (CONCEPT:EG-KG.storage.nonblocking-checkpoint). Never fabricate counts,
        # search hits, or impact percentages — if the engine is cold, say so.
        from agent_utilities.knowledge_graph.core.engine import (
            IntelligenceGraphEngine,
        )

        engine = IntelligenceGraphEngine.get_active()
        backend = getattr(engine, "backend", None) if engine else None

        if sub == "search":
            if not rest:
                response_md = "Usage: `/graph search <query>`"
            elif backend is None:
                response_md = "Graph backend not active — cannot run search."
            else:
                try:
                    rows = (
                        backend.execute(
                            "MATCH (n) WHERE toLower(n.name) CONTAINS toLower($q) "
                            "OR toLower(n.id) CONTAINS toLower($q) "
                            "RETURN n.id AS id, n.name AS name, labels(n)[0] AS type "
                            "LIMIT 10",
                            {"q": rest},
                        )
                        or []
                    )
                except Exception as e:  # noqa: BLE001
                    rows = []
                    logger.warning("Graph search failed: %s", e)
                if not rows:
                    response_md = f"No graph nodes matched `{rest}`."
                else:
                    lines = [
                        f"- **[{r.get('type', 'Node')}]** `{r.get('id')}`: "
                        f"{r.get('name') or r.get('id')}"
                        for r in rows
                    ]
                    response_md = (
                        f"### Graph Search Results for `{rest}`:\n\n" + "\n".join(lines)
                    )
        elif sub == "impact":
            if not rest:
                response_md = "Usage: `/graph impact <symbol>`"
            elif engine is None:
                response_md = "Graph backend not active — cannot run impact analysis."
            else:
                try:
                    radius = engine.get_blast_radius(rest, depth=2) or []
                except Exception as e:  # noqa: BLE001
                    radius = []
                    logger.warning("Blast radius query failed: %s", e)
                if not radius:
                    response_md = (
                        f"### Blast Radius Impact Analysis for `{rest}`\n\n"
                        f"No downstream dependencies found (or `{rest}` is not a known node)."
                    )
                else:
                    lines = [
                        f"- `{item.get('id')}` ({item.get('type', 'Node')}, "
                        f"depth {item.get('depth')})"
                        for item in radius
                    ]
                    response_md = (
                        f"### Blast Radius Impact Analysis for `{rest}`\n\n"
                        f"**{len(radius)}** downstream node(s) affected:\n\n"
                        + "\n".join(lines)
                    )
        else:
            # stats (default)
            if backend is None:
                response_md = (
                    "### Knowledge Graph Statistics\n\n"
                    "Graph backend not active — no live counts available."
                )
            else:
                try:
                    node_rows = backend.execute("MATCH (n) RETURN count(n) AS c") or []
                    edge_rows = (
                        backend.execute("MATCH ()-[r]->() RETURN count(r) AS c") or []
                    )
                    nodes = int(node_rows[0]["c"]) if node_rows else 0
                    edges = int(edge_rows[0]["c"]) if edge_rows else 0
                    response_md = (
                        "### Knowledge Graph Statistics\n\n"
                        f"- **Total Nodes**: {nodes}\n"
                        f"- **Total Relationships**: {edges}\n"
                        f"- **Backend**: {type(backend).__name__} (active)\n"
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("Graph stats query failed: %s", e)
                    response_md = f"Graph stats query failed: {e}"
        return {"response_markdown": response_md, "client_actions": []}

    elif cmd_name == "kb":
        sub_parts = args.split(maxsplit=1)
        sub = sub_parts[0].lower() if sub_parts else "list"
        rest = sub_parts[1] if len(sub_parts) > 1 else ""

        from agent_utilities.knowledge_graph.core.engine import (
            IntelligenceGraphEngine,
        )

        engine = IntelligenceGraphEngine.get_active()
        backend = getattr(engine, "backend", None) if engine else None

        if sub == "list":
            if backend is None:
                response_md = (
                    "### Connected Knowledge Bases:\n\n"
                    "Knowledge Graph backend not active — no knowledge bases available."
                )
            else:
                try:
                    rows = (
                        backend.execute(
                            "MATCH (kb:KnowledgeBase) RETURN kb.id AS id, "
                            "kb.name AS name, kb.description AS description"
                        )
                        or []
                    )
                except Exception as e:  # noqa: BLE001
                    rows = []
                    logger.warning("KB list query failed: %s", e)
                if not rows:
                    response_md = (
                        "### Connected Knowledge Bases:\n\n"
                        "No knowledge bases registered yet. "
                        "Use `/kb ingest <url_or_path>` to create one."
                    )
                else:
                    lines = [
                        f"- `{r.get('name') or r.get('id')}` "
                        f"({r.get('description', '') or 'no description'})"
                        for r in rows
                    ]
                    response_md = "### Connected Knowledge Bases:\n\n" + "\n".join(
                        lines
                    )
        elif sub == "search":
            if not rest:
                response_md = "Usage: `/kb search <query>`"
            elif backend is None:
                response_md = "Knowledge Graph backend not active — cannot search."
            else:
                try:
                    rows = (
                        backend.execute(
                            "MATCH (a:Article) "
                            "WHERE toLower(a.name) CONTAINS toLower($q) "
                            "OR toLower(a.content) CONTAINS toLower($q) "
                            "RETURN a.id AS id, a.name AS name, "
                            "a.description AS description LIMIT 5",
                            {"q": rest},
                        )
                        or []
                    )
                except Exception as e:  # noqa: BLE001
                    rows = []
                    logger.warning("KB search failed: %s", e)
                if not rows:
                    response_md = f"No KB articles matched `{rest}`."
                else:
                    lines = []
                    for i, r in enumerate(rows, 1):
                        excerpt = (r.get("description") or "")[:200]
                        lines.append(
                            f"{i}. **{r.get('name') or r.get('id')}**"
                            + (f"\n   > {excerpt}" if excerpt else "")
                        )
                    response_md = (
                        f"### KB Search Results for `{rest}`:\n\n" + "\n".join(lines)
                    )
        elif sub == "ingest":
            if not rest:
                response_md = "Usage: `/kb ingest <url_or_path>`"
            elif engine is None:
                response_md = (
                    "Knowledge Graph engine not active — cannot enqueue ingestion."
                )
            else:
                try:
                    job_id = engine.submit_task(
                        target_path=rest,
                        is_codebase=False,
                        provenance={"source": "slash_command:/kb ingest"},
                        task_type="document",
                    )
                    response_md = (
                        f"Enqueued KB ingestion task `{job_id}` for `{rest}`. "
                        "Track progress via the pipeline status."
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("KB ingest enqueue failed: %s", e)
                    response_md = f"Failed to enqueue ingestion for `{rest}`: {e}"
        else:
            response_md = f"Unknown `/kb` subcommand: `{sub}`"

        return {"response_markdown": response_md, "client_actions": []}

    elif cmd_name == "sdd":
        sub = args.strip().lower() or "specs"

        import os

        from agent_utilities.sdd import SDDManager

        workspace = setting("WORKSPACE_PATH") or os.getcwd()
        manager = SDDManager(workspace_path=workspace)

        if sub == "specs":
            try:
                specs = manager.list_specs()
            except Exception as e:  # noqa: BLE001
                specs = []
                logger.warning("SDD spec listing failed: %s", e)
            if not specs:
                response_md = (
                    "### Active Spec-Driven Specifications:\n\n"
                    f"No specs found under `{workspace}/.specify/specs/`."
                )
            else:
                lines = [
                    f"- **[{s.get('id')}]**: {s.get('title', s.get('id'))}"
                    for s in specs
                ]
                response_md = "### Active Spec-Driven Specifications:\n\n" + "\n".join(
                    lines
                )
        elif sub == "constitution":
            try:
                constitution = manager.get_constitution()
            except Exception as e:  # noqa: BLE001
                constitution = None
                logger.warning("SDD constitution load failed: %s", e)
            if not constitution:
                response_md = (
                    "### Spec-Driven Development Governance:\n\n"
                    f"No constitution found at `{workspace}/.specify/constitution.md`."
                )
            else:
                principles = constitution.get("core_principles") or []
                gates = constitution.get("quality_gates") or []
                sections = []
                vision = constitution.get("vision")
                mission = constitution.get("mission")
                if vision:
                    sections.append(f"**Vision**: {vision}")
                if mission:
                    sections.append(f"**Mission**: {mission}")
                if principles:
                    sections.append(
                        "**Core Principles**:\n"
                        + "\n".join(f"{i}. {p}" for i, p in enumerate(principles, 1))
                    )
                if gates:
                    sections.append(
                        "**Quality Gates**:\n" + "\n".join(f"- {g}" for g in gates)
                    )
                body = "\n\n".join(sections) if sections else "(constitution is empty)"
                response_md = "### Spec-Driven Development Governance:\n\n" + body
        elif sub == "sync":
            from agent_utilities.knowledge_graph.core.engine import (
                IntelligenceGraphEngine,
            )

            engine = IntelligenceGraphEngine.get_active()
            if engine is None or not getattr(engine, "backend", None):
                response_md = (
                    "Knowledge Graph backend not active — cannot sync specs to KG."
                )
            else:
                try:
                    from agent_utilities.models import Spec

                    specs = manager.list_specs()
                    for s in specs:
                        spec_model = manager.load(Spec, s.get("id"))
                        if spec_model is not None:
                            manager.record_sdd_outcome(spec_model, s.get("id"))
                    response_md = (
                        f"Synchronized {len(specs)} spec(s) from "
                        f"`{workspace}/.specify/specs/` into the Knowledge Graph "
                        "as `SDDArtifact` nodes."
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("SDD sync failed: %s", e)
                    response_md = f"SDD sync failed: {e}"
        else:
            response_md = f"Unknown `/sdd` subcommand: `{sub}`"

        return {"response_markdown": response_md, "client_actions": []}

    elif cmd_name == "cron":
        sub = args.strip().lower() or "calendar"
        # Real registry (CONCEPT:AU-OS.state.unified-scheduling-one-intelligent) — durable :Schedule nodes (seeded from
        # deploy/schedules.yml) + live last-run state, NOT placeholder text.
        try:
            from agent_utilities.core.schedule_engine import calendar
            from agent_utilities.knowledge_graph.core.engine import (
                IntelligenceGraphEngine,
            )

            entries = calendar(IntelligenceGraphEngine.get_active())
        except Exception as e:  # noqa: BLE001
            return {
                "response_markdown": f"Scheduler unavailable: {e}",
                "client_actions": [],
            }
        if sub in ("calendar", "logs"):
            if not entries:
                response_md = (
                    "No scheduled skills/workflows declared in `deploy/schedules.yml`."
                )
            elif sub == "calendar":
                lines = [
                    f"- `{e['name']}` (`{e['cron']}`, {e['kind']}:{e['ref']}): "
                    f"{e['description']} — last run: {e['last_run']}"
                    for e in entries
                ]
                response_md = "### Scheduled Skills / Workflows:\n\n" + "\n".join(lines)
            else:  # logs — last-run per declared schedule
                lines = [
                    f"- `{e['name']}` — last run: {e['last_run']}" for e in entries
                ]
                response_md = "### Schedule Last-Run:\n\n" + "\n".join(lines)
        else:
            response_md = f"Unknown `/cron` subcommand: `{sub}`"

        return {"response_markdown": response_md, "client_actions": []}

    elif cmd_name == "resources":
        sub_parts = args.split(maxsplit=1)
        sub = sub_parts[0].lower() if sub_parts else "list"
        rest = sub_parts[1] if len(sub_parts) > 1 else ""

        if sub in ("", "list"):
            lines = []
            # Registered specialist agents (same source as /agents).
            try:
                from agent_utilities.agent.discovery import (
                    discover_all_specialists,
                )

                for s in discover_all_specialists():
                    lines.append(
                        f"- **{s.name}** - Type: `{s.source or 'specialist'}`"
                        + (f" - Server: `{s.mcp_server}`" if s.mcp_server else "")
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning("Specialist discovery failed: %s", e)
            # Live dispatch workers (heartbeat-fresh).
            try:
                from agent_utilities.orchestration.agent_dispatch import (
                    list_dispatch_workers,
                )

                for w in list_dispatch_workers():
                    active = len(w.get("active_sessions", []))
                    lines.append(
                        f"- **worker `{w.get('worker_id')}`** on `{w.get('host')}` "
                        f"- active sessions: {active} - backend: "
                        f"`{w.get('queue_backend')}`"
                    )
            except Exception as e:  # noqa: BLE001
                logger.debug("Dispatch worker listing unavailable: %s", e)

            if not lines:
                response_md = (
                    "### Spawned Subagents and Background Tasks:\n\n"
                    "No registered specialists or live dispatch workers."
                )
            else:
                response_md = (
                    "### Registered Specialists and Live Workers:\n\n"
                    + "\n".join(lines)
                )
        elif sub == "spawn":
            if not rest:
                response_md = "Usage: `/resources spawn <name>`"
            else:
                from agent_utilities.orchestration.agent_dispatch import (
                    dispatch_queue_enabled,
                )

                if not dispatch_queue_enabled():
                    response_md = (
                        "Agent dispatch is in `inline` mode (no background queue). "
                        "Set `AGENT_DISPATCH_BACKEND=queue` to enqueue background "
                        f"subagent turns; `{rest}` was not spawned."
                    )
                else:
                    try:
                        import uuid as _uuid

                        from agent_utilities.orchestration.agent_dispatch import (
                            AgentTurnEnvelope,
                            enqueue_agent_turn,
                        )

                        envelope = AgentTurnEnvelope(
                            session_id=f"slash-spawn-{_uuid.uuid4().hex[:8]}",
                            agent_name=rest,
                        )
                        handle = enqueue_agent_turn(envelope)
                        response_md = (
                            f"Enqueued background agent turn `{handle['job_id']}` "
                            f"for **{rest}** (session `{handle['session_id']}`, "
                            f"status: {handle['status']})."
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Agent spawn enqueue failed: %s", e)
                        response_md = f"Failed to spawn **{rest}**: {e}"
        else:
            response_md = f"Unknown `/resources` subcommand: `{sub}`"

        return {"response_markdown": response_md, "client_actions": []}

    else:
        return {
            "response_markdown": f"Unknown slash command: `/{cmd_name}`. Type `/help` for a list of available commands.",
            "client_actions": [],
        }


@router.get(
    "/autocomplete",
    summary="Provide autocomplete dynamic options for client interfaces",
)
async def autocomplete_slash_command(query: str = ""):
    """Provide autocomplete dynamic options for client interfaces."""
    commands_list = [
        "/help",
        "/clear",
        "/model",
        "/tools",
        "/skills",
        "/graph stats",
        "/graph nodes",
        "/graph search",
        "/graph impact",
        "/kb list",
        "/kb search",
        "/kb ingest",
        "/sdd specs",
        "/sdd constitution",
        "/sdd sync",
        "/cron calendar",
        "/cron logs",
        "/resources list",
        "/resources spawn",
    ]
    if not query:
        return {"suggestions": commands_list}

    suggestions = [cmd for cmd in commands_list if cmd.startswith(query.lower())]
    return {"suggestions": suggestions}
