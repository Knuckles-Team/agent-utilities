#!/usr/bin/python
"""Knowledge Graph Tools.

This module provides MCP-style tools for interacting with the unified
Knowledge Graph, allowing agents to persist and retrieve memories.
"""

import logging
import time
import uuid

from pydantic_ai import RunContext

from ..knowledge_graph.engine import RegistryGraphEngine
from ..models import (
    AgentDeps,
    ImplementationPlan,
    Spec,
    Tasks,
    TaskStatus,
)
from ..sdd import SDDManager

logger = logging.getLogger(__name__)


def get_knowledge_engine(ctx: RunContext[AgentDeps]) -> RegistryGraphEngine | None:
    """Helper to retrieve the knowledge engine from the context/deps."""
    return getattr(ctx.deps, "knowledge_engine", None)


async def search_knowledge_graph(ctx: RunContext[AgentDeps], query: str) -> str:
    """Search the Knowledge Graph for relevant agents, tools, code, or memories.

    Args:
        query: The keyword or concept to search for.
        ctx: The agent context containing the knowledge engine.
    """
    engine = get_knowledge_engine(ctx)
    if not engine:
        return "Knowledge Graph not available."

    results = engine.search_hybrid(query)

    output = []
    for res in results:
        ntype = res.get("type", "unknown")
        output.append(
            f"[{ntype.upper()}] [ID: {res.get('id')}] {res.get('name')}: {res.get('description', '')[:200]}"
        )

    if not output:
        return f"No results found in Knowledge Graph for '{query}'."

    return "\n---\n".join(output)


async def get_code_impact(ctx: RunContext[AgentDeps], symbol_or_file: str) -> str:
    """Analyze the potential impact of changing a specific code entity.

    Returns a list of dependent files and symbols that may be affected.
    """
    engine = get_knowledge_engine(ctx)
    if not engine:
        return "Knowledge Graph not available."

    impact = engine.query_impact(symbol_or_file)
    if not impact:
        return f"No impact found or entity '{symbol_or_file}' not recognized."

    output = [f"Impact Set for '{symbol_or_file}':"]
    for node in impact:
        output.append(
            f"- [{node.get('type')}] {node.get('id')} (File: {node.get('file_path', 'N/A')})"
        )

    return "\n".join(output)


async def add_knowledge_memory(
    ctx: RunContext[AgentDeps],
    content: str,
    name: str = "",
    category: str = "general",
    tags: list[str] | None = None,
) -> str:
    """Add a new long-term memory or observation to the Knowledge Graph (CREATE).

    Args:
        content: The memory content or observation.
        name: Optional title for the memory.
        category: Category of the memory (e.g., 'decision', 'preference', 'fact').
        tags: Optional list of descriptive tags.
        ctx: The agent context.
    """
    engine = get_knowledge_engine(ctx)
    if not engine:
        return "Knowledge Graph not available for persistence."

    mem_id = engine.add_memory(content, name=name, category=category, tags=tags)
    return f"Successfully added memory to Knowledge Graph with ID: {mem_id}"


async def get_knowledge_memory(ctx: RunContext[AgentDeps], memory_id: str) -> str:
    """Retrieve a specific memory from the Knowledge Graph by ID (READ).

    Args:
        memory_id: The unique identifier of the memory (e.g., 'mem:abc12345').
        ctx: The agent context.
    """
    engine = get_knowledge_engine(ctx)
    if not engine:
        return "Knowledge Graph not available."

    memory = engine.get_memory(memory_id)
    if not memory:
        return f"Memory with ID '{memory_id}' not found."

    return f"ID: {memory['id']}\nName: {memory['name']}\nTimestamp: {memory['timestamp']}\nCategory: {memory['category']}\nContent: {memory['description']}"


async def update_knowledge_memory(
    ctx: RunContext[AgentDeps],
    memory_id: str,
    content: str | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Update an existing memory in the Knowledge Graph (UPDATE).

    Args:
        memory_id: The ID of the memory to update.
        content: New content (if changing).
        category: New category (if changing).
        tags: New tags (if changing).
        ctx: The agent context.
    """
    engine = get_knowledge_engine(ctx)
    if not engine:
        return "Knowledge Graph not available."

    from typing import Any

    updates: dict[str, Any] = {}
    if content:
        updates["description"] = content
    if category:
        updates["category"] = category
    if tags:
        updates["tags"] = tags

    if not updates:
        return "No updates provided."

    engine.update_memory(memory_id, **updates)
    return f"Successfully updated memory '{memory_id}'."


async def delete_knowledge_memory(ctx: RunContext[AgentDeps], memory_id: str) -> str:
    """Permanently remove a memory from the Knowledge Graph (DELETE).

    Args:
        memory_id: The ID of the memory to remove.
        ctx: The agent context.
    """
    engine = get_knowledge_engine(ctx)
    if not engine:
        return "Knowledge Graph not available."

    engine.delete_memory(memory_id)
    return f"Successfully deleted memory '{memory_id}'."


async def link_knowledge_nodes(
    ctx: RunContext[AgentDeps],
    source_id: str,
    target_id: str,
    relationship: str = "related_to",
) -> str:
    """Establish a relationship between two nodes in the Knowledge Graph.

    Args:
        source_id: The ID of the source node.
        target_id: The ID of the target node.
        relationship: The type of relationship (e.g., 'depends_on', 'memory_of').
        ctx: The agent context.
    """
    engine = get_knowledge_engine(ctx)
    if not engine:
        return "Knowledge Graph not available."

    if source_id not in engine.graph or target_id not in engine.graph:
        return f"Error: One or both node IDs ({source_id}, {target_id}) not found in graph."

    engine.graph.add_edge(source_id, target_id, type=relationship)

    if engine.backend:
        engine.backend.execute(
            "MATCH (a {id: $source}), (b {id: $target}) MERGE (a)-[r:"
            + relationship.upper()
            + "]->(b)",
            {"source": source_id, "target": target_id},
        )

    return f"Successfully established {relationship} link between {source_id} and {target_id}."


async def sync_feature_to_memory(ctx: RunContext[AgentDeps], feature_id: str) -> str:
    """Synchronize the full SDD lifecycle of a feature into the Knowledge Graph memory.

    Captures the specification, technical plan, and final execution summary to ensure
    the agent can reference this work in the future.
    """
    engine = get_knowledge_engine(ctx)
    if not engine:
        return "Knowledge Graph not available for memory synchronization."

    workspace_path = getattr(ctx.deps, "workspace_path", None)
    if not workspace_path:
        return "Workspace path not available in context."

    manager = SDDManager(workspace_path)

    # Load artifacts
    spec = manager.load(Spec, feature_id=feature_id)
    plan = manager.load(ImplementationPlan, feature_id=feature_id)
    tasks = manager.load(Tasks, feature_id=feature_id)

    if not spec:
        return (
            f"Could not find Spec for feature '{feature_id}'. Synchronization aborted."
        )

    # Generate Summary
    summary = [f"### Feature: {spec.title} ({feature_id})"]
    summary.append(
        f"**Goal**: {spec.user_stories[0].description if spec.user_stories else 'N/A'}"
    )

    if plan:
        summary.append(f"**Technical Approach**: {plan.technical_context[:500]}...")

    if tasks:
        completed_tasks = [t for t in tasks.tasks if t.status == TaskStatus.COMPLETED]
        summary.append(
            f"**Execution Summary**: Completed {len(completed_tasks)}/{len(tasks.tasks)} tasks."
        )
        for t in completed_tasks:
            summary.append(f"- {t.title}: {t.result or 'No result recorded'}")

    content = "\n".join(summary)

    # Check if memory already exists to update or create
    mem_name = f"SDD Feature Memory: {feature_id}"
    existing_mem_id = None
    for node_id, data in engine.graph.nodes(data=True):
        if data.get("type") == "memory" and data.get("name") == mem_name:
            existing_mem_id = node_id
            break

    if existing_mem_id:
        engine.update_memory(existing_mem_id, description=content)
        return f"Successfully updated historical memory for feature '{feature_id}' in Knowledge Graph."
    else:
        mem_id = engine.add_memory(
            content=content,
            name=mem_name,
            category="sdd_feature",
            tags=["sdd", "feature_summary", feature_id],
        )
        return f"Successfully captured feature '{feature_id}' in Knowledge Graph memory (ID: {mem_id})."


async def log_heartbeat(
    ctx: RunContext[AgentDeps],
    agent_name: str,
    status: str,
    issues: list[str] | None = None,
    raw_data: str = "",
) -> str:
    """Log a heartbeat to the Knowledge Graph."""
    engine = get_knowledge_engine(ctx)
    if not engine:
        return "Knowledge Graph not available."

    hb_id = f"hb:{uuid.uuid4().hex[:8]}"
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    props = {
        "id": hb_id,
        "type": "heartbeat",
        "agent_name": agent_name,
        "timestamp": timestamp,
        "status": status,
        "issues": issues or [],
        "raw_data": raw_data,
    }

    if engine.backend:
        engine.backend.execute(
            "MERGE (n:Heartbeat {id: $id}) SET n += $props",
            {"id": hb_id, "props": props},
        )
        engine.backend.execute(
            "MERGE (a:Agent {id: $agent_id}) "
            "WITH a MATCH (h:Heartbeat {id: $hb_id}) "
            "MERGE (h)-[:HEARTBEAT_OF]->(a)",
            {"agent_id": f"agent:{agent_name}", "hb_id": hb_id},
        )
        return f"Heartbeat logged with ID: {hb_id}"
    return "Failed to log heartbeat."


async def create_client(
    ctx: RunContext[AgentDeps], name: str, description: str = ""
) -> str:
    """Create a Client node in the Knowledge Graph."""
    engine = get_knowledge_engine(ctx)
    if not engine:
        return "Knowledge Graph not available."

    client_id = f"client:{uuid.uuid4().hex[:8]}"
    props = {
        "id": client_id,
        "type": "client",
        "name": name,
        "description": description,
    }
    if engine.backend:
        engine.backend.execute(
            "MERGE (n:Client {id: $id}) SET n += $props",
            {"id": client_id, "props": props},
        )
        return f"Client created with ID: {client_id}"
    return "Failed to create client."


async def create_user(
    ctx: RunContext[AgentDeps],
    name: str,
    role: str = "user",
    client_id: str | None = None,
) -> str:
    """Create a User node in the Knowledge Graph and link to a Client."""
    engine = get_knowledge_engine(ctx)
    if not engine:
        return "Knowledge Graph not available."

    user_id = f"user:{uuid.uuid4().hex[:8]}"
    props = {"id": user_id, "type": "user", "name": name, "role": role}
    if engine.backend:
        engine.backend.execute(
            "MERGE (n:User {id: $id}) SET n += $props", {"id": user_id, "props": props}
        )
        if client_id:
            engine.backend.execute(
                "MATCH (u:User {id: $u_id}), (c:Client {id: $c_id}) MERGE (u)-[:BELONGS_TO]->(c)",
                {"u_id": user_id, "c_id": client_id},
            )
        return f"User created with ID: {user_id}"
    return "Failed to create user."


async def save_preference(
    ctx: RunContext[AgentDeps], user_id: str, category: str, value: str
) -> str:
    """Save a preference for a User in the Knowledge Graph."""
    engine = get_knowledge_engine(ctx)
    if not engine:
        return "Knowledge Graph not available."

    pref_id = f"pref:{uuid.uuid4().hex[:8]}"
    props = {"id": pref_id, "type": "preference", "category": category, "value": value}
    if engine.backend:
        engine.backend.execute(
            "MERGE (n:Preference {id: $id}) SET n += $props",
            {"id": pref_id, "props": props},
        )
        engine.backend.execute(
            "MATCH (u:User {id: $u_id}), (p:Preference {id: $p_id}) MERGE (u)-[:PREFERS]->(p)",
            {"u_id": user_id, "p_id": pref_id},
        )
        return f"Preference saved with ID: {pref_id}"
    return "Failed to save preference."


async def save_chat_message(
    ctx: RunContext[AgentDeps], thread_id: str, role: str, content: str
) -> str:
    """Save a chat message to the Knowledge Graph, with optional embedding."""
    engine = get_knowledge_engine(ctx)
    if not engine:
        return "Knowledge Graph not available."

    msg_id = f"msg:{uuid.uuid4().hex[:8]}"
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    props = {
        "id": msg_id,
        "type": "message",
        "role": role,
        "content": content,
        "timestamp": timestamp,
    }

    if engine.backend:
        engine.backend.execute(
            "MERGE (n:Message {id: $id}) SET n += $props",
            {"id": msg_id, "props": props},
        )
        engine.backend.execute(
            "MERGE (t:Thread {id: $t_id}) "
            "WITH t MATCH (m:Message {id: $m_id}) "
            "MERGE (m)-[:PART_OF]->(t)",
            {"t_id": thread_id, "m_id": msg_id},
        )
        # Note: Embedding logic (LM Studio) will be handled in a background task or specific pipeline phase.
        return f"Message saved with ID: {msg_id}"
    return "Failed to save message."


async def log_cron_execution(
    ctx: RunContext[AgentDeps], job_id: str, status: str, output: str
) -> str:
    """Log a cron job execution to the Knowledge Graph."""
    engine = get_knowledge_engine(ctx)
    if not engine:
        return "Knowledge Graph not available."

    log_id = f"log:{uuid.uuid4().hex[:8]}"
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    props = {
        "id": log_id,
        "type": "log",
        "timestamp": timestamp,
        "status": status,
        "output": output,
    }

    if engine.backend:
        engine.backend.execute(
            "MERGE (n:Log {id: $id}) SET n += $props", {"id": log_id, "props": props}
        )
        engine.backend.execute(
            "MERGE (j:Job {id: $j_id}) "
            "WITH j MATCH (l:Log {id: $l_id}) "
            "MERGE (l)-[:EXECUTED_BY]->(j)",
            {"j_id": job_id, "l_id": log_id},
        )
        return f"Cron execution logged with ID: {log_id}"
    return "Failed to log cron execution."


# Export all knowledge tools for easy registration
knowledge_tools = [
    search_knowledge_graph,
    get_code_impact,
    add_knowledge_memory,
    get_knowledge_memory,
    update_knowledge_memory,
    delete_knowledge_memory,
    link_knowledge_nodes,
    sync_feature_to_memory,
    log_heartbeat,
    create_client,
    create_user,
    save_preference,
    save_chat_message,
    log_cron_execution,
]


# ---------------------------------------------------------------------------
# Knowledge Base (KB) Tools — 8 tools for LLM-maintained personal wikis
# ---------------------------------------------------------------------------


def _get_kb_engine(ctx: RunContext[AgentDeps]):
    """Get or create a KBIngestionEngine from the agent context."""
    from ..knowledge_graph.kb.ingestion import KBIngestionEngine

    engine = get_knowledge_engine(ctx)
    graph = engine.graph if engine else None
    backend = engine.backend if engine else None

    import networkx as nx

    return KBIngestionEngine(
        graph=graph or nx.MultiDiGraph(),
        backend=backend,
    )


async def ingest_knowledge_base(
    ctx: RunContext[AgentDeps],
    source: str,
    kb_name: str,
    source_type: str = "auto",
    topic: str = "",
    force: bool = False,
) -> str:
    """Ingest documents into a named knowledge base.

    Supports directories, individual files, URLs, and skill-graph paths.
    Uses hash-based deduplication — unchanged files are skipped automatically.

    Args:
        source: File path, directory path, URL, or skill-graph path to ingest.
        kb_name: Name for the knowledge base (e.g., "pydantic-ai-docs", "my-research").
        source_type: One of "auto", "skill_graph", "directory", "url", "file".
            "auto" detects based on the source string.
        topic: Brief description of the KB topic (optional, inferred if not given).
        force: If True, re-ingest even unchanged files.

    Returns:
        Summary of ingestion results.
    """
    try:
        kb_engine = _get_kb_engine(ctx)
        from pathlib import Path

        # Auto-detect source type
        if source_type == "auto":
            if source.startswith("http://") or source.startswith("https://"):
                source_type = "url"
            elif Path(source).is_dir():
                # Check if it's a skill-graph (has SKILL.md)
                source_type = (
                    "skill_graph"
                    if (Path(source) / "SKILL.md").exists()
                    else "directory"
                )
            else:
                source_type = "file"

        meta = None
        if source_type == "skill_graph":
            meta = await kb_engine.ingest_skill_graph(source, force=force)
        elif source_type == "directory":
            meta = await kb_engine.ingest_directory(
                source, kb_name=kb_name, topic=topic or None, force=force
            )
        elif source_type == "url":
            meta = await kb_engine.ingest_url(
                source, kb_name=kb_name, topic=topic or None, force=force
            )
        elif source_type == "file":
            from pathlib import Path as _Path

            source = str(_Path(source).resolve())
            meta = await kb_engine.ingest_directory(
                str(_Path(source).parent),
                kb_name=kb_name,
                topic=topic or None,
                force=force,
            )

        if not meta:
            return f"❌ Ingestion failed for source: {source}"

        return (
            f"✅ Knowledge base '{meta.name}' ready.\n"
            f"   Topic: {meta.topic}\n"
            f"   Articles: {meta.article_count}\n"
            f"   Sources: {meta.source_count}\n"
            f"   Status: {meta.status}"
        )
    except Exception as e:
        logger.error(f"ingest_knowledge_base failed: {e}", exc_info=True)
        return f"❌ Ingestion error: {e}"


async def list_knowledge_bases(ctx: RunContext[AgentDeps]) -> str:
    """List all knowledge bases with their article counts and status.

    Returns a formatted table of all KBs in the graph so the agent
    can quickly discover what knowledge is available.

    Returns:
        Formatted list of knowledge bases with topic and article counts.
    """
    try:
        kb_engine = _get_kb_engine(ctx)
        kbs = kb_engine.list_knowledge_bases()
        if not kbs:
            return "No knowledge bases found. Use ingest_knowledge_base to add one."

        lines = ["📚 Knowledge Bases:\n"]
        for kb in kbs:
            status_icon = {
                "ready": "✅",
                "ingesting": "🔄",
                "error": "❌",
                "archived": "📦",
            }.get(kb["status"], "❓")
            lines.append(
                f"{status_icon} **{kb['name']}** (id: {kb['id']})\n"
                f"   Topic: {kb['topic']}\n"
                f"   Articles: {kb['article_count']} | Sources: {kb['source_count']} | Status: {kb['status']}\n"
            )
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"list_knowledge_bases failed: {e}")
        return f"❌ Error listing knowledge bases: {e}"


async def search_knowledge_base_tool(
    ctx: RunContext[AgentDeps],
    query: str,
    kb_id: str | None = None,
    top_k: int = 5,
) -> str:
    """Search knowledge bases for relevant articles and facts.

    Performs hybrid keyword search across all (or a specific) knowledge base.
    For precise semantic search, the embedding vector index is also queried
    when a backend is available.

    Args:
        query: What you want to know or find.
        kb_id: Optional KB to search (e.g., "kb:pydantic-ai-docs"). Searches all if not given.
        top_k: Maximum number of results to return.

    Returns:
        Relevant articles with excerpts.
    """
    try:
        kb_engine = _get_kb_engine(ctx)
        results = kb_engine.search_knowledge_base(query, kb_id=kb_id, top_k=top_k)
        if not results:
            scope = f" in '{kb_id}'" if kb_id else ""
            return f"No results found{scope} for: '{query}'"

        lines = [f"🔍 Search results for '{query}':\n"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. **{r['article_title']}** [{r['kb_name']}]\n"
                f"   {r['excerpt']}\n"
                f"   ID: {r['article_id']}\n"
            )
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"search_knowledge_base_tool failed: {e}")
        return f"❌ Search error: {e}"


async def get_kb_article(
    ctx: RunContext[AgentDeps],
    article_id: str,
) -> str:
    """Retrieve the full content of a specific KB article.

    Use the article ID from list_knowledge_bases or search_knowledge_base_tool.

    Args:
        article_id: The article node ID (e.g., "article:kb:pydantic-ai-docs:agents").

    Returns:
        Full article content in Markdown format.
    """
    try:
        engine = get_knowledge_engine(ctx)
        if not engine:
            return "Knowledge engine not available."

        graph = engine.graph
        if article_id not in graph.nodes:
            return f"Article not found: {article_id}"

        data = graph.nodes[article_id]
        title = data.get("name", article_id)
        summary = data.get("description", "")
        content = data.get("content", "")
        tags = data.get("tags", [])

        if not content:
            return (
                f"# {title}\n\n"
                f"*[Archived — summary only]*\n\n"
                f"{summary}\n\n"
                f"Tags: {', '.join(tags) if tags else 'none'}"
            )

        return f"# {title}\n\n{content}\n\nTags: {', '.join(tags) if tags else 'none'}"
    except Exception as e:
        logger.error(f"get_kb_article failed: {e}")
        return f"❌ Error retrieving article: {e}"


async def update_knowledge_base(
    ctx: RunContext[AgentDeps],
    kb_id: str,
) -> str:
    """Re-ingest changed source files for an existing knowledge base.

    Only files whose content hash has changed since last ingestion are
    re-processed, making this operation cheap for large corpora.

    Args:
        kb_id: The KB node ID (e.g., "kb:pydantic-ai-docs").

    Returns:
        Summary of what was updated.
    """
    try:
        kb_engine = _get_kb_engine(ctx)
        meta = await kb_engine.update_kb(kb_id)
        if not meta:
            return f"❌ KB not found: {kb_id}. Use list_knowledge_bases to see available KBs."
        return (
            f"✅ Updated '{meta.name}'\n"
            f"   Articles: {meta.article_count} | Status: {meta.status}"
        )
    except Exception as e:
        logger.error(f"update_knowledge_base failed: {e}")
        return f"❌ Update error: {e}"


async def run_kb_health_check(
    ctx: RunContext[AgentDeps],
    kb_id: str,
) -> str:
    """Run a health check on a knowledge base to find issues.

    Uses an LLM to audit the KB for:
    - Contradicting facts
    - Missing or sparse coverage
    - Orphaned articles (no backlinks)
    - Stale or outdated content
    - Suggestions for new articles to fill gaps

    Args:
        kb_id: The KB node ID to audit (e.g., "kb:pydantic-ai-docs").

    Returns:
        Health report with issues and suggestions.
    """
    try:
        kb_engine = _get_kb_engine(ctx)
        report = await kb_engine.run_health_check(kb_id)

        severity_icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        lines = [
            f"🏥 Health Check: {report.kb_name}\n",
            f"   Consistency Score: {report.consistency_score:.1%}\n",
            f"   {report.summary}\n",
        ]

        if report.issues:
            lines.append(f"\n**Issues ({len(report.issues)}):**")
            for issue in report.issues:
                icon = severity_icons.get(issue.severity, "⚪")
                lines.append(
                    f"{icon} [{issue.issue_type}] {issue.description}\n"
                    f"   → {issue.suggested_action}"
                )

        if report.suggested_articles:
            lines.append("\n**Suggested New Articles:**")
            for title in report.suggested_articles:
                lines.append(f"  - {title}")

        return "\n".join(lines)
    except Exception as e:
        logger.error(f"run_kb_health_check failed: {e}")
        return f"❌ Health check error: {e}"


async def archive_knowledge_base(
    ctx: RunContext[AgentDeps],
    kb_id: str,
    importance_threshold: float = 0.3,
) -> str:
    """Archive low-importance articles in a knowledge base (summary-only mode).

    Articles below the importance threshold have their full content removed
    from memory (keeping only the summary). This saves significant memory
    for large KBs while preserving discoverability.

    Args:
        kb_id: The KB to archive (e.g., "kb:pydantic-ai-docs").
        importance_threshold: Articles with importance < this value are compressed.
            Default 0.3 (30%). Use 0.0 to compress all, 1.0 to compress none.

    Returns:
        Archive result summary.
    """
    try:
        kb_engine = _get_kb_engine(ctx)
        result = await kb_engine.archive_kb(kb_id, threshold=importance_threshold)
        return (
            f"📦 Archived KB: {kb_id}\n"
            f"   Articles compressed: {result.articles_compressed}\n"
            f"   Nodes pruned: {result.nodes_pruned}\n"
            f"   Memory saved: {result.bytes_saved:,} bytes\n"
            f"   Timestamp: {result.archive_timestamp}"
        )
    except Exception as e:
        logger.error(f"archive_knowledge_base failed: {e}")
        return f"❌ Archive error: {e}"


async def export_knowledge_base(
    ctx: RunContext[AgentDeps],
    kb_id: str,
    output_dir: str,
) -> str:
    """Export a knowledge base as a directory of Markdown files.

    Creates Obsidian-compatible markdown files with YAML frontmatter,
    internal [[wiki-links]], and a table of contents index file.
    This allows viewing the KB in any Markdown editor (Obsidian, VS Code, etc.)

    Args:
        kb_id: The KB to export (e.g., "kb:pydantic-ai-docs").
        output_dir: Directory path where markdown files will be written.

    Returns:
        Export summary with file count and output path.
    """
    try:
        from pathlib import Path

        engine = get_knowledge_engine(ctx)
        if not engine:
            return "Knowledge engine not available."

        graph = engine.graph
        if kb_id not in graph.nodes:
            return f"KB not found: {kb_id}"

        kb_data = graph.nodes[kb_id]
        kb_name = kb_data.get("name", kb_id)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        from ...models.knowledge_graph import RegistryNodeType

        articles_exported = 0
        index_lines = [f"# {kb_name} Knowledge Base\n\n"]

        for n in graph.predecessors(kb_id):
            node_data = graph.nodes[n]
            if node_data.get("type") != RegistryNodeType.ARTICLE:
                continue

            title = node_data.get("name", n)
            summary = node_data.get("description", "")
            content = node_data.get("content", summary)
            tags = node_data.get("tags", [])

            # Build Obsidian-compatible frontmatter
            frontmatter = (
                f"---\n"
                f'title: "{title}"\n'
                f'kb: "{kb_id}"\n'
                f"tags: [{', '.join(tags)}]\n"
                f"importance: {node_data.get('importance_score', 0.5):.2f}\n"
                f"---\n\n"
            )

            # Sanitize filename
            safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title)[
                :80
            ]
            file_path = out_path / f"{safe_name}.md"
            file_path.write_text(frontmatter + content, encoding="utf-8")
            articles_exported += 1
            index_lines.append(f"- [[{safe_name}]]: {summary[:100]}")

        # Write index file
        index_path = out_path / "INDEX.md"
        index_path.write_text("\n".join(index_lines), encoding="utf-8")

        return (
            f"✅ Exported KB '{kb_name}' to: {output_dir}\n"
            f"   Articles: {articles_exported}\n"
            f"   Index: {index_path}"
        )
    except Exception as e:
        logger.error(f"export_knowledge_base failed: {e}")
        return f"❌ Export error: {e}"


KB_TOOLS = [
    ingest_knowledge_base,
    list_knowledge_bases,
    search_knowledge_base_tool,
    get_kb_article,
    update_knowledge_base,
    run_kb_health_check,
    archive_knowledge_base,
    export_knowledge_base,
]
