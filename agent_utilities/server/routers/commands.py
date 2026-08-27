import logging

from fastapi import APIRouter, Request

from agent_utilities.core.config import setting
from agent_utilities.security.error_surface import public_error_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/enhanced/commands", tags=["Command Center"])


async def _cmd_help(args: str, request: Request) -> dict:
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


async def _cmd_clear(args: str, request: Request) -> dict:
    return {
        "response_markdown": "Chat session cleared.",
        "client_actions": [{"action": "clear_chat"}],
    }


async def _cmd_model(args: str, request: Request) -> dict:
    registry = getattr(request.app.state, "model_registry", None)
    client_actions: list = []
    if not args:
        current_model = registry.get_default() if registry else None
        model_id = current_model.id if current_model else "unknown"
        response_md = (
            f"Current active model: `{model_id}`.\n\n"
            "Use `/model <model_id>` to change it."
        )
    else:
        client_actions.append({"action": "set_model", "value": args})
        response_md = f"Switched model to `{args}`."
    return {"response_markdown": response_md, "client_actions": client_actions}


async def _cmd_tools(args: str, request: Request) -> dict:
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


async def _cmd_skills(args: str, request: Request) -> dict:
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


def _graph_search(engine, backend, rest: str) -> str:
    if not rest:
        return "Usage: `/graph search <query>`"
    if backend is None:
        return "Graph backend not active — cannot run search."
    assert engine is not None  # backend is only set when engine is
    try:
        rows = (
            engine.query_cypher(
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
        logger.warning(
            "Graph search failed (exception_type=%s)", type(e).__name__
        )
    if not rows:
        return f"No graph nodes matched `{rest}`."
    lines = [
        f"- **[{r.get('type', 'Node')}]** `{r.get('id')}`: "
        f"{r.get('name') or r.get('id')}"
        for r in rows
    ]
    return f"### Graph Search Results for `{rest}`:\n\n" + "\n".join(lines)


def _graph_impact(engine, rest: str) -> str:
    if not rest:
        return "Usage: `/graph impact <symbol>`"
    if engine is None:
        return "Graph backend not active — cannot run impact analysis."
    try:
        radius = engine.get_blast_radius(rest, depth=2) or []
    except Exception as e:  # noqa: BLE001
        radius = []
        logger.warning(
            "Blast radius query failed (exception_type=%s)",
            type(e).__name__,
        )
    if not radius:
        return (
            f"### Blast Radius Impact Analysis for `{rest}`\n\n"
            f"No downstream dependencies found (or `{rest}` is not a known node)."
        )
    lines = [
        f"- `{item.get('id')}` ({item.get('type', 'Node')}, "
        f"depth {item.get('depth')})"
        for item in radius
    ]
    return (
        f"### Blast Radius Impact Analysis for `{rest}`\n\n"
        f"**{len(radius)}** downstream node(s) affected:\n\n" + "\n".join(lines)
    )


def _graph_stats(engine, backend) -> str:
    if backend is None:
        return (
            "### Knowledge Graph Statistics\n\n"
            "Graph backend not active — no live counts available."
        )
    assert engine is not None  # backend is only set when engine is
    try:
        node_rows = engine.query_cypher("MATCH (n) RETURN count(n) AS c") or []
        # See enhanced.py's get_graph_stats for why this binds a
        # source node (`a`) rather than the bare `()-[r]->()`
        # shape: an anonymous-only pattern has no variable
        # TenancyManager.scope_cypher_query can inject a tenant
        # predicate against, so it fails closed with
        # UnscopableQueryError instead of returning a count (D-W2T-1).
        edge_rows = (
            engine.query_cypher("MATCH (a)-[r]->() RETURN count(r) AS c") or []
        )
        nodes = int(node_rows[0]["c"]) if node_rows else 0
        edges = int(edge_rows[0]["c"]) if edge_rows else 0
        return (
            "### Knowledge Graph Statistics\n\n"
            f"- **Total Nodes**: {nodes}\n"
            f"- **Total Relationships**: {edges}\n"
            f"- **Backend**: {type(backend).__name__} (active)\n"
        )
    except Exception as exc:  # noqa: BLE001
        return public_error_text(exc, logger=logger)


async def _cmd_graph(args: str, request: Request) -> dict:
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
        response_md = _graph_search(engine, backend, rest)
    elif sub == "impact":
        response_md = _graph_impact(engine, rest)
    else:
        # stats (default)
        response_md = _graph_stats(engine, backend)
    return {"response_markdown": response_md, "client_actions": []}


def _kb_list(engine, backend) -> str:
    if backend is None:
        return (
            "### Connected Knowledge Bases:\n\n"
            "Knowledge Graph backend not active — no knowledge bases available."
        )
    assert engine is not None  # backend is only set when engine is
    try:
        rows = (
            engine.query_cypher(
                "MATCH (kb:KnowledgeBase) RETURN kb.id AS id, "
                "kb.name AS name, kb.description AS description"
            )
            or []
        )
    except Exception as e:  # noqa: BLE001
        rows = []
        logger.warning(
            "KB list query failed (exception_type=%s)", type(e).__name__
        )
    if not rows:
        return (
            "### Connected Knowledge Bases:\n\n"
            "No knowledge bases registered yet. "
            "Use `/kb ingest <url_or_path>` to create one."
        )
    lines = [
        f"- `{r.get('name') or r.get('id')}` "
        f"({r.get('description', '') or 'no description'})"
        for r in rows
    ]
    return "### Connected Knowledge Bases:\n\n" + "\n".join(lines)


def _kb_search(engine, backend, rest: str) -> str:
    if not rest:
        return "Usage: `/kb search <query>`"
    if backend is None:
        return "Knowledge Graph backend not active — cannot search."
    assert engine is not None  # backend is only set when engine is
    try:
        rows = (
            engine.query_cypher(
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
        logger.warning(
            "KB search failed (exception_type=%s)", type(e).__name__
        )
    if not rows:
        return f"No KB articles matched `{rest}`."
    lines = _kb_search_format_rows(rows)
    return f"### KB Search Results for `{rest}`:\n\n" + "\n".join(lines)


def _kb_search_format_rows(rows) -> list:
    lines = []
    for i, r in enumerate(rows, 1):
        excerpt = (r.get("description") or "")[:200]
        lines.append(
            f"{i}. **{r.get('name') or r.get('id')}**"
            + (f"\n   > {excerpt}" if excerpt else "")
        )
    return lines


def _kb_ingest(engine, rest: str) -> str:
    if not rest:
        return "Usage: `/kb ingest <url_or_path>`"
    if engine is None:
        return "Knowledge Graph engine not active — cannot enqueue ingestion."
    try:
        job_id = engine.submit_task(
            target_path=rest,
            is_codebase=False,
            provenance={"source": "slash_command:/kb ingest"},
            task_type="document",
        )
        return (
            f"Enqueued KB ingestion task `{job_id}` for `{rest}`. "
            "Track progress via the pipeline status."
        )
    except Exception as exc:  # noqa: BLE001
        return public_error_text(exc, logger=logger)


async def _cmd_kb(args: str, request: Request) -> dict:
    sub_parts = args.split(maxsplit=1)
    sub = sub_parts[0].lower() if sub_parts else "list"
    rest = sub_parts[1] if len(sub_parts) > 1 else ""

    from agent_utilities.knowledge_graph.core.engine import (
        IntelligenceGraphEngine,
    )

    engine = IntelligenceGraphEngine.get_active()
    backend = getattr(engine, "backend", None) if engine else None

    if sub == "list":
        response_md = _kb_list(engine, backend)
    elif sub == "search":
        response_md = _kb_search(engine, backend, rest)
    elif sub == "ingest":
        response_md = _kb_ingest(engine, rest)
    else:
        response_md = f"Unknown `/kb` subcommand: `{sub}`"

    return {"response_markdown": response_md, "client_actions": []}


def _sdd_specs(manager, workspace: str) -> str:
    try:
        specs = manager.list_specs()
    except Exception as e:  # noqa: BLE001
        specs = []
        logger.warning(
            "SDD spec listing failed (exception_type=%s)", type(e).__name__
        )
    if not specs:
        return (
            "### Active Spec-Driven Specifications:\n\n"
            f"No specs found under `{workspace}/.specify/specs/`."
        )
    lines = [
        f"- **[{s.get('id')}]**: {s.get('title', s.get('id'))}" for s in specs
    ]
    return "### Active Spec-Driven Specifications:\n\n" + "\n".join(lines)


def _sdd_constitution(manager, workspace: str) -> str:
    try:
        constitution = manager.get_constitution()
    except Exception as e:  # noqa: BLE001
        constitution = None
        logger.warning(
            "SDD constitution load failed (exception_type=%s)",
            type(e).__name__,
        )
    if not constitution:
        return (
            "### Spec-Driven Development Governance:\n\n"
            f"No constitution found at `{workspace}/.specify/constitution.md`."
        )
    sections = _sdd_constitution_sections(constitution)
    body = "\n\n".join(sections) if sections else "(constitution is empty)"
    return "### Spec-Driven Development Governance:\n\n" + body


def _sdd_constitution_sections(constitution: dict) -> list:
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
        sections.append("**Quality Gates**:\n" + "\n".join(f"- {g}" for g in gates))
    return sections


def _sdd_sync(manager, workspace: str) -> str:
    from agent_utilities.knowledge_graph.core.engine import (
        IntelligenceGraphEngine,
    )

    engine = IntelligenceGraphEngine.get_active()
    if engine is None or not getattr(engine, "backend", None):
        return "Knowledge Graph backend not active — cannot sync specs to KG."
    try:
        from agent_utilities.models import Spec

        specs = manager.list_specs()
        for s in specs:
            spec_model = manager.load(Spec, s.get("id"))
            if spec_model is not None:
                manager.record_sdd_outcome(spec_model, s.get("id"))
        return (
            f"Synchronized {len(specs)} spec(s) from "
            f"`{workspace}/.specify/specs/` into the Knowledge Graph "
            "as `SDDArtifact` nodes."
        )
    except Exception as exc:  # noqa: BLE001
        return public_error_text(exc, logger=logger)


async def _cmd_sdd(args: str, request: Request) -> dict:
    sub = args.strip().lower() or "specs"

    import os

    from agent_utilities.sdd import SDDManager

    workspace = setting("WORKSPACE_PATH") or os.getcwd()
    manager = SDDManager(workspace_path=workspace)

    if sub == "specs":
        response_md = _sdd_specs(manager, workspace)
    elif sub == "constitution":
        response_md = _sdd_constitution(manager, workspace)
    elif sub == "sync":
        response_md = _sdd_sync(manager, workspace)
    else:
        response_md = f"Unknown `/sdd` subcommand: `{sub}`"

    return {"response_markdown": response_md, "client_actions": []}


async def _cmd_cron(args: str, request: Request) -> dict:
    sub = args.strip().lower() or "calendar"
    # Real registry (CONCEPT:AU-OS.state.unified-scheduling-one-intelligent) — durable :Schedule nodes (seeded from
    # deploy/schedules.yml) + live last-run state, NOT placeholder text.
    try:
        from agent_utilities.core.schedule_engine import calendar
        from agent_utilities.knowledge_graph.core.engine import (
            IntelligenceGraphEngine,
        )

        entries = calendar(IntelligenceGraphEngine.get_active())
    except Exception as exc:  # noqa: BLE001
        return {
            "response_markdown": public_error_text(exc, logger=logger),
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
            lines = [f"- `{e['name']}` — last run: {e['last_run']}" for e in entries]
            response_md = "### Schedule Last-Run:\n\n" + "\n".join(lines)
    else:
        response_md = f"Unknown `/cron` subcommand: `{sub}`"

    return {"response_markdown": response_md, "client_actions": []}


def _resources_list() -> str:
    lines = []
    # Registered specialist agents (same source as /agents).
    try:
        from agent_utilities.agent.discovery import discover_all_specialists

        for s in discover_all_specialists():
            lines.append(
                f"- **{s.name}** - Type: `{s.source or 'specialist'}`"
                + (f" - Server: `{s.mcp_server}`" if s.mcp_server else "")
            )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Specialist discovery failed (exception_type=%s)",
            type(e).__name__,
        )
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
        logger.debug(
            "Dispatch worker listing unavailable (exception_type=%s)",
            type(e).__name__,
        )

    if not lines:
        return (
            "### Spawned Subagents and Background Tasks:\n\n"
            "No registered specialists or live dispatch workers."
        )
    return "### Registered Specialists and Live Workers:\n\n" + "\n".join(lines)


def _resources_spawn(rest: str) -> str:
    if not rest:
        return "Usage: `/resources spawn <name>`"
    try:
        import uuid as _uuid

        from agent_utilities.orchestration.agent_dispatch import (
            AgentTurnEnvelope,
            enqueue_agent_turn,
        )

        envelope = AgentTurnEnvelope(
            session_id=f"slash-spawn-{_uuid.uuid4().hex}",
            agent_name=rest,
        )
        handle = enqueue_agent_turn(envelope)
        return (
            f"Enqueued background agent turn `{handle['job_id']}` "
            f"for **{rest}** (session `{handle['session_id']}`, "
            f"status: {handle['status']})."
        )
    except Exception as exc:  # noqa: BLE001
        return public_error_text(exc, logger=logger)


async def _cmd_resources(args: str, request: Request) -> dict:
    sub_parts = args.split(maxsplit=1)
    sub = sub_parts[0].lower() if sub_parts else "list"
    rest = sub_parts[1] if len(sub_parts) > 1 else ""

    if sub in ("", "list"):
        response_md = _resources_list()
    elif sub == "spawn":
        response_md = _resources_spawn(rest)
    else:
        response_md = f"Unknown `/resources` subcommand: `{sub}`"

    return {"response_markdown": response_md, "client_actions": []}


_COMMAND_HANDLERS = {
    "help": _cmd_help,
    "clear": _cmd_clear,
    "model": _cmd_model,
    "tools": _cmd_tools,
    "skills": _cmd_skills,
    "graph": _cmd_graph,
    "kb": _cmd_kb,
    "sdd": _cmd_sdd,
    "cron": _cmd_cron,
    "resources": _cmd_resources,
}


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

    handler = _COMMAND_HANDLERS.get(cmd_name)
    if handler is not None:
        return await handler(args, request)

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
