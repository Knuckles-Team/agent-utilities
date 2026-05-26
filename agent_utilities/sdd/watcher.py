#!/usr/bin/python
"""Plan and Tasks Watcher & Ingester.

CONCEPT:KG-2.6 — Implementation Plan & Tasks versioning and KG lineage.

This module watches for changes in implementation plans and task lists in the
active workspace or local IDE brain session directory, archives each unique state,
and ingests versioned nodes into the Knowledge Graph to maintain complete engineering lineage.
"""

import hashlib
import logging
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Cache of seen file hashes to avoid redundant writes/ingestions
# Key: absolute file path, Value: set of seen MD5 hashes
_SEEN_HASHES: dict[str, set[str]] = {}

# Cache of seen file modification times (st_mtime) to avoid redundant disk reads & MD5 hashing
# Key: absolute file path, Value: float modification timestamp
_SEEN_MTIMES: dict[str, float] = {}

# Global flag to temporarily pause the background watcher thread
_WATCHER_PAUSED = False


def _get_md5(content: str) -> str:
    """Calculate MD5 hash of string content."""
    return hashlib.md5(
        content.encode("utf-8", errors="ignore"), usedforsecurity=False
    ).hexdigest()


def _get_latest_version_from_history(
    history_dir: Path, prefix: str
) -> tuple[int, set[str]]:
    """Scan history directory for files with given prefix to find latest version and seen hashes.

    Files are expected to follow: {prefix}_v{version}_{timestamp}.md
    """
    version = 0
    seen_hashes: set[str] = set()
    if not history_dir.exists():
        return version, seen_hashes

    # Pattern to match plan_<feature_id>_v<version>_<timestamp>.md
    # or brain_plan_<session_id>_v<version>_<timestamp>.md
    pattern = re.compile(rf"^{prefix}_v(\d+)_\d+.*\.md$")

    for f in history_dir.iterdir():
        if f.is_file():
            match = pattern.match(f.name)
            if match:
                v = int(match.group(1))
                if v > version:
                    version = v
                # Read content to cache hash
                try:
                    content = f.read_text(encoding="utf-8", errors="ignore")
                    seen_hashes.add(_get_md5(content))
                except Exception as e:
                    logger.debug(f"Failed to read historical file {f}: {e}")

    return version, seen_hashes


def _parse_plan_metadata(content: str) -> dict[str, Any]:
    """Parse title and approach/context from implementation plan markdown."""
    title = "Implementation Plan"
    title_match = re.search(r"^#\s+(.*)", content)
    if title_match:
        title = title_match.group(1).strip()

    approach = ""
    approach_match = re.search(r"## Approach\n(.*?)(?=\n##|$)", content, re.DOTALL)
    if approach_match:
        approach = approach_match.group(1).strip()
    else:
        # Fallback to whole content
        approach = content

    return {
        "title": title,
        "approach": approach,
    }


def ingest_plan_version(
    engine: Any,
    feature_id: str,
    title: str,
    approach: str,
    version: int,
    session_id: str,
    content_hash: str,
    raw_content: str,
) -> str:
    """Ingest a versioned implementation plan into the Knowledge Graph via Cypher."""
    plan_id = f"plan:{feature_id}:v{version}"
    prev_plan_id = f"plan:{feature_id}:v{version - 1}" if version > 1 else None
    timestamp = datetime.now(UTC).isoformat()

    logger.info(f"Ingesting plan {plan_id} (v{version}) into KG")

    # 1. Merge ImplementationPlan node
    query_plan = (
        "MERGE (p:ImplementationPlan {id: $plan_id}) "
        "SET p.title = $title, "
        "    p.version = $version, "
        "    p.chatSessionId = $session_id, "
        "    p.planHash = $content_hash, "
        "    p.approach = $approach, "
        "    p.raw_content = $raw_content, "
        "    p.timestamp = $timestamp, "
        "    p.last_updated = $last_updated "
        "RETURN p.id"
    )
    engine.backend.execute(
        query_plan,
        {
            "plan_id": plan_id,
            "title": title,
            "version": version,
            "session_id": session_id,
            "content_hash": content_hash,
            "approach": approach[:2000]
            if approach
            else "",  # Limit size for graph props
            "raw_content": raw_content,
            "timestamp": timestamp,
            "last_updated": int(time.time() * 1000),
        },
    )

    # 2. Supersedes relationship
    if prev_plan_id:
        query_supersedes = (
            "MATCH (old:ImplementationPlan {id: $prev_plan_id}) "
            "MATCH (new:ImplementationPlan {id: $plan_id}) "
            "MERGE (new)-[:SUPERSEDES]->(old)"
        )
        try:
            engine.backend.execute(
                query_supersedes,
                {"prev_plan_id": prev_plan_id, "plan_id": plan_id},
            )
        except Exception as e:
            logger.warning(f"Could not link superseded plan: {e}")

    # 3. Link to SoftwareFeature (creating feature if missing)
    query_feature = (
        "MERGE (f:SoftwareFeature {id: $feature_id}) "
        "ON CREATE SET f.title = $feature_title "
        "WITH f "
        "MATCH (p:ImplementationPlan {id: $plan_id}) "
        "MERGE (p)-[:PLAN_FOR_FEATURE]->(f)"
    )
    engine.backend.execute(
        query_feature,
        {
            "feature_id": feature_id,
            "feature_title": f"Feature {feature_id}",
            "plan_id": plan_id,
        },
    )

    # 4. Link to active Project node
    query_project = (
        "MATCH (proj:Project) WHERE proj.name = 'current' "
        "MATCH (p:ImplementationPlan {id: $plan_id}) "
        "MERGE (proj)-[:HAS_ARTIFACT]->(p)"
    )
    try:
        engine.backend.execute(query_project, {"plan_id": plan_id})
    except Exception as e:
        logger.debug(f"Could not link plan to project: {e}")

    # 5. Link to Session node
    if session_id:
        query_session = (
            "MERGE (sess:Session {id: $session_id}) "
            "WITH sess "
            "MATCH (p:ImplementationPlan {id: $plan_id}) "
            "MERGE (sess)-[:HAS_ARTIFACT]->(p)"
        )
        try:
            engine.backend.execute(
                query_session, {"session_id": session_id, "plan_id": plan_id}
            )
        except Exception as e:
            logger.debug(f"Could not link plan to session: {e}")

    return plan_id


def ingest_tasks_version(
    engine: Any,
    feature_id: str,
    title: str,
    version: int,
    session_id: str,
    content_hash: str,
    raw_content: str,
) -> str:
    """Ingest a versioned tasks list into the Knowledge Graph via Cypher."""
    tasks_id = f"tasks:{feature_id}:v{version}"
    prev_tasks_id = f"tasks:{feature_id}:v{version - 1}" if version > 1 else None
    timestamp = datetime.now(UTC).isoformat()

    logger.info(f"Ingesting tasks {tasks_id} (v{version}) into KG")

    # 1. Merge Tasks node
    query_tasks = (
        "MERGE (t:Tasks {id: $tasks_id}) "
        "SET t.title = $title, "
        "    t.version = $version, "
        "    t.chatSessionId = $session_id, "
        "    t.tasksHash = $content_hash, "
        "    t.raw_content = $raw_content, "
        "    t.timestamp = $timestamp, "
        "    t.last_updated = $last_updated "
        "RETURN t.id"
    )
    engine.backend.execute(
        query_tasks,
        {
            "tasks_id": tasks_id,
            "title": title,
            "version": version,
            "session_id": session_id,
            "content_hash": content_hash,
            "raw_content": raw_content,
            "timestamp": timestamp,
            "last_updated": int(time.time() * 1000),
        },
    )

    # 2. Supersedes relationship
    if prev_tasks_id:
        query_supersedes = (
            "MATCH (old:Tasks {id: $prev_tasks_id}) "
            "MATCH (new:Tasks {id: $tasks_id}) "
            "MERGE (new)-[:SUPERSEDES]->(old)"
        )
        try:
            engine.backend.execute(
                query_supersedes,
                {"prev_tasks_id": prev_tasks_id, "tasks_id": tasks_id},
            )
        except Exception as e:
            logger.warning(f"Could not link superseded tasks: {e}")

    # 3. Link to SoftwareFeature
    query_feature = (
        "MERGE (f:SoftwareFeature {id: $feature_id}) "
        "WITH f "
        "MATCH (t:Tasks {id: $tasks_id}) "
        "MERGE (t)-[:TASKS_FOR_FEATURE]->(f)"
    )
    engine.backend.execute(
        query_feature,
        {"feature_id": feature_id, "tasks_id": tasks_id},
    )

    return tasks_id


def process_plan_file(engine: Any, file_path: Path, workspace_path: Path):
    """Processes a single implementation plan file, checking for changes and versioning."""
    if not file_path.exists():
        return

    file_key = str(file_path.resolve())
    try:
        mtime = file_path.stat().st_mtime
    except Exception as e:
        logger.debug(f"Failed to get stat for {file_path}: {e}")
        mtime = 0.0

    if _SEEN_MTIMES.get(file_key) == mtime:
        return

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return

    content_hash = _get_md5(content)

    # Initialize seen hashes cache for this file if not present
    if file_key not in _SEEN_HASHES:
        _SEEN_HASHES[file_key] = set()

    # Determine type and IDs
    is_brain = "brain" in file_path.parts
    session_id = "default_session"
    feature_id = "default_feature"

    if is_brain:
        # Path is typically /.../brain/<session_id>/implementation_plan.md
        # session_id is parent folder name
        session_id = file_path.parent.name
        feature_id = f"brain_{session_id}"
        prefix = f"brain_plan_{session_id}"
        history_dir = workspace_path / ".specify" / "history" / "brain"
    else:
        # Path is typically /.../.specify/specs/<feature_id>/plan.md
        feature_id = file_path.parent.name
        prefix = f"plan_{feature_id}"
        history_dir = workspace_path / ".specify" / "history" / "plans"

    # Scan history dir on first time to sync state
    if not _SEEN_HASHES[file_key]:
        latest_v, historical_hashes = _get_latest_version_from_history(
            history_dir, prefix
        )
        _SEEN_HASHES[file_key].update(historical_hashes)
        version = latest_v
    else:
        # Quick check: if we already saw this in memory, skip
        if content_hash in _SEEN_HASHES[file_key]:
            _SEEN_MTIMES[file_key] = mtime
            return
        # Find latest version from history dir directly to be safe
        latest_v, _ = _get_latest_version_from_history(history_dir, prefix)
        version = latest_v

    # Check if hash was already archived
    if content_hash in _SEEN_HASHES[file_key]:
        _SEEN_MTIMES[file_key] = mtime
        return

    # A new unique state is detected! Increment version and save
    version += 1
    timestamp_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    history_dir.mkdir(parents=True, exist_ok=True)
    history_file = history_dir / f"{prefix}_v{version}_{timestamp_str}.md"

    try:
        history_file.write_text(content, encoding="utf-8")
        logger.info(
            f"Archived plan version: {history_file.relative_to(workspace_path)}"
        )
    except Exception as e:
        logger.error(f"Failed to write history archive: {e}")
        return

    # Ingest into KG
    try:
        parsed = _parse_plan_metadata(content)
        ingest_plan_version(
            engine=engine,
            feature_id=feature_id,
            title=parsed["title"],
            approach=parsed["approach"],
            version=version,
            session_id=session_id,
            content_hash=content_hash,
            raw_content=content,
        )
    except Exception as e:
        logger.error(f"Failed to ingest plan to KG: {e}")

    # Cache the seen hash and modification time
    _SEEN_HASHES[file_key].add(content_hash)
    _SEEN_MTIMES[file_key] = mtime


def process_tasks_file(engine: Any, file_path: Path, workspace_path: Path):
    """Processes a single task list file, checking for changes and versioning."""
    if not file_path.exists():
        return

    file_key = str(file_path.resolve())
    try:
        mtime = file_path.stat().st_mtime
    except Exception as e:
        logger.debug(f"Failed to get stat for {file_path}: {e}")
        mtime = 0.0

    if _SEEN_MTIMES.get(file_key) == mtime:
        return

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return

    content_hash = _get_md5(content)

    if file_key not in _SEEN_HASHES:
        _SEEN_HASHES[file_key] = set()

    is_brain = "brain" in file_path.parts
    session_id = "default_session"
    feature_id = "default_feature"

    if is_brain:
        session_id = file_path.parent.name
        feature_id = f"brain_{session_id}"
        prefix = f"brain_tasks_{session_id}"
        history_dir = workspace_path / ".specify" / "history" / "brain"
    else:
        feature_id = file_path.parent.name
        prefix = f"tasks_{feature_id}"
        history_dir = workspace_path / ".specify" / "history" / "tasks"

    if not _SEEN_HASHES[file_key]:
        latest_v, historical_hashes = _get_latest_version_from_history(
            history_dir, prefix
        )
        _SEEN_HASHES[file_key].update(historical_hashes)
        version = latest_v
    else:
        if content_hash in _SEEN_HASHES[file_key]:
            _SEEN_MTIMES[file_key] = mtime
            return
        latest_v, _ = _get_latest_version_from_history(history_dir, prefix)
        version = latest_v

    if content_hash in _SEEN_HASHES[file_key]:
        _SEEN_MTIMES[file_key] = mtime
        return

    version += 1
    timestamp_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    history_dir.mkdir(parents=True, exist_ok=True)
    history_file = history_dir / f"{prefix}_v{version}_{timestamp_str}.md"

    try:
        history_file.write_text(content, encoding="utf-8")
        logger.info(
            f"Archived tasks version: {history_file.relative_to(workspace_path)}"
        )
    except Exception as e:
        logger.error(f"Failed to write history archive: {e}")
        return

    try:
        ingest_tasks_version(
            engine=engine,
            feature_id=feature_id,
            title=f"Tasks for {feature_id}",
            version=version,
            session_id=session_id,
            content_hash=content_hash,
            raw_content=content,
        )
    except Exception as e:
        logger.error(f"Failed to ingest tasks to KG: {e}")

    _SEEN_HASHES[file_key].add(content_hash)
    _SEEN_MTIMES[file_key] = mtime


def _safe_walk(root: Path, target_names: set[str], max_depth: int = 5):
    """Safely walk directories to find target files, skipping large/temp folders."""
    skip_dirs = {
        ".git",
        ".venv",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "build",
        "dist",
        "history",
        "cache",
        "temp",
        "tmp",
        ".mypy_cache",
    }
    if not root.exists():
        return

    def _walk(current: Path, depth: int):
        if depth > max_depth:
            return
        try:
            for item in current.iterdir():
                if item.is_dir():
                    if item.name in skip_dirs or (
                        item.name.startswith(".") and item.name != ".specify"
                    ):
                        continue
                    yield from _walk(item, depth + 1)
                elif item.is_file():
                    if item.name.lower() in target_names:
                        yield item
        except PermissionError:
            pass
        except Exception:
            pass

    yield from _walk(root, 1)


def get_all_skills_directories(workspace_path: Path) -> list[Path]:
    """Get all normalized skills directories across all platforms and the workspace."""
    raw_dirs = [
        "~/.gemini/antigravity/skills",
        "~/.claude/skills",
        "~/.config/claude/skills",
        "~/.devin/skills",
        "~/.codeium/windsurf/skills",
        "~/.windsurf/skills",
        "~/.config/agent-utilities/skills",
        os.environ.get("AGENT_SKILLS_DIR"),
    ]
    resolved = []
    for d in raw_dirs:
        if d:
            p = Path(os.path.expanduser(d)).resolve()
            if p.exists() and p.is_dir():
                resolved.append(p)

    # Also add the workspace's agent-packages skills folder
    wp_skills = workspace_path / "agent-packages" / "skills"
    if wp_skills.exists() and wp_skills.is_dir():
        resolved.append(wp_skills)
        try:
            for child in wp_skills.iterdir():
                if child.is_dir():
                    resolved.append(child)
        except Exception:
            pass

    return resolved


def get_scholarx_directories() -> list[Path]:
    """Get all normalized ScholarX and research download directories."""
    raw_dirs = [
        "~/.local/share/agent-utilities/research",
        "~/.local/share/scholarx/papers",
        os.environ.get("SCHOLARX_PAPERS_DIR"),
        os.environ.get("AGENT_RESEARCH_DIR"),
    ]
    resolved = []
    for d in raw_dirs:
        if d:
            p = Path(os.path.expanduser(d)).resolve()
            if p.exists() and p.is_dir():
                resolved.append(p)
    return resolved


def get_kg_ingest_paths(workspace_path: Path) -> list[Path]:
    """Get all core Knowledge Graph configuration and ontology files to watch."""
    raw_paths = [
        "~/.config/agent-utilities/inventory.yaml",
        "~/.config/agent-utilities/mcp_config.json",
        "~/.config/agent-utilities/config.json",
        "~/.local/share/agent-utilities/topology",
        str(workspace_path / "agent_utilities" / "knowledge_graph" / "ontology.ttl"),
        str(
            workspace_path
            / "agent_utilities"
            / "knowledge_graph"
            / "ontology_infrastructure.ttl"
        ),
        str(workspace_path / "agent_utilities" / "workflows" / "catalog.yaml"),
        os.environ.get("AGENT_INVENTORY_YAML"),
        os.environ.get("AGENT_MCP_CONFIG_JSON"),
    ]
    resolved = []
    for rp in raw_paths:
        if rp:
            p = Path(os.path.expanduser(rp)).resolve()
            if p.exists():
                resolved.append(p)
    return resolved


def process_skill_file(engine: Any, file_path: Path, workspace_path: Path):
    """Processes a single SKILL.md file, checking for changes and versioning."""
    if not file_path.exists():
        return

    file_key = str(file_path.resolve())
    try:
        mtime = file_path.stat().st_mtime
    except Exception as e:
        logger.debug(f"Failed to get stat for {file_path}: {e}")
        mtime = 0.0

    if _SEEN_MTIMES.get(file_key) == mtime:
        return

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return

    content_hash = _get_md5(content)

    if file_key not in _SEEN_HASHES:
        _SEEN_HASHES[file_key] = set()

    import yaml

    frontmatter: dict[str, Any] = {}
    skill_name = file_path.parent.name
    skill_desc = ""

    if content.startswith("---"):
        end_idx = content.find("---", 3)
        if end_idx != -1:
            try:
                frontmatter_str = content[3:end_idx].strip()
                frontmatter = yaml.safe_load(frontmatter_str) or {}
                skill_name = frontmatter.get("name", skill_name)
                skill_desc = frontmatter.get("description", "")
            except Exception as e:
                logger.debug(f"Failed to parse skill frontmatter for {file_path}: {e}")

    prefix = f"skill_{skill_name.lower().replace(' ', '_').replace('-', '_')}"
    history_dir = workspace_path / ".specify" / "history" / "skills"

    if not _SEEN_HASHES[file_key]:
        latest_v, historical_hashes = _get_latest_version_from_history(
            history_dir, prefix
        )
        _SEEN_HASHES[file_key].update(historical_hashes)
        version = latest_v
    else:
        if content_hash in _SEEN_HASHES[file_key]:
            _SEEN_MTIMES[file_key] = mtime
            return
        latest_v, _ = _get_latest_version_from_history(history_dir, prefix)
        version = latest_v

    if content_hash in _SEEN_HASHES[file_key]:
        _SEEN_MTIMES[file_key] = mtime
        return

    version += 1
    timestamp_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    history_dir.mkdir(parents=True, exist_ok=True)
    history_file = history_dir / f"{prefix}_v{version}_{timestamp_str}.md"

    try:
        history_file.write_text(content, encoding="utf-8")
        logger.info(
            f"Archived skill version: {history_file.relative_to(workspace_path) if workspace_path in history_file.parents else history_file}"
        )
    except Exception as e:
        logger.error(f"Failed to write skill history archive: {e}")
        return

    try:
        node_id = f"skill_{skill_name.lower().replace(' ', '_').replace('-', '_')}"
        props = {
            "name": skill_name,
            "description": skill_desc,
            "path": str(file_path.resolve()),
            "version": version,
            "last_updated": int(time.time() * 1000),
        }
        for k, v in frontmatter.items():
            if k not in ["name", "description"]:
                props[k] = str(v)

        engine.add_node(node_id=node_id, node_type="Skill", properties=props)

        query_project = (
            "MATCH (proj:Project) WHERE proj.name = 'current' "
            "MATCH (s:Skill {id: $node_id}) "
            "MERGE (proj)-[:HAS_ARTIFACT]->(s)"
        )
        try:
            engine.backend.execute(query_project, {"node_id": node_id})
        except Exception as e:
            logger.debug(f"Could not link skill to project: {e}")

    except Exception as e:
        logger.error(f"Failed to ingest skill to KG: {e}")

    _SEEN_HASHES[file_key].add(content_hash)
    _SEEN_MTIMES[file_key] = mtime


def process_scholarx_file(engine: Any, file_path: Path):
    """Processes a downloaded ScholarX research paper, triggering KG ingestion."""
    if not file_path.exists() or not file_path.is_file():
        return

    file_key = str(file_path.resolve())

    # G7: Fast-path mtime check — skip if file hasn't changed since last scan
    try:
        mtime = file_path.stat().st_mtime
    except Exception:
        mtime = 0.0
    if _SEEN_MTIMES.get(file_key) == mtime:
        return

    if file_key not in _SEEN_HASHES:
        _SEEN_HASHES[file_key] = set()

    mtime_str = str(mtime)
    if mtime_str in _SEEN_HASHES[file_key]:
        _SEEN_MTIMES[file_key] = mtime
        return

    try:
        logger.info(
            f"ScholarX download detected: {file_path}. Submitting ingestion task."
        )
        if hasattr(engine, "submit_task"):
            engine.submit_task(
                target_path=str(file_path.resolve()),
                is_codebase=False,
                task_type="document",
                provenance={"source": "watcher_scholarx"},
            )
        _SEEN_HASHES[file_key].add(mtime_str)
        _SEEN_MTIMES[file_key] = mtime
    except Exception as e:
        logger.error(f"Failed to submit ScholarX ingestion task: {e}")


def process_kg_ingest_location(engine: Any, file_path: Path):
    """Processes a Knowledge Graph ingestion location, re-triggering ingestion on changes."""
    if not file_path.exists() or not file_path.is_file():
        return

    file_key = str(file_path.resolve())
    if file_key not in _SEEN_HASHES:
        _SEEN_HASHES[file_key] = set()

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            content = str(file_path.stat().st_mtime)
        except Exception:
            return

    content_hash = _get_md5(content)
    if content_hash in _SEEN_HASHES[file_key]:
        return

    logger.info(
        f"Knowledge Graph Ingestion location modified: {file_path}. Re-ingesting."
    )

    if file_path.name == "mcp_config.json":
        try:
            from agent_utilities.mcp.kg_server import _ingest_capabilities

            _ingest_capabilities(engine)
        except Exception as e:
            logger.debug(f"Failed to re-ingest capabilities: {e}")
    else:
        try:
            if hasattr(engine, "submit_task"):
                engine.submit_task(
                    target_path=str(file_path.resolve()),
                    is_codebase=False,
                    task_type="document",
                    provenance={"source": "watcher_kg_ingest"},
                )
        except Exception as e:
            logger.error(f"Failed to submit KG re-ingestion task: {e}")

    _SEEN_HASHES[file_key].add(content_hash)


def run_watcher_scan(engine: Any, workspace_path: Path):
    """Executes a single synchronous directory scan for plans, tasks, skills, and downloads."""
    # 1. Scan active workspace specs
    specs_dir = workspace_path / ".specify" / "specs"
    if specs_dir.exists():
        for feature_dir in specs_dir.iterdir():
            if feature_dir.is_dir():
                plan_file = feature_dir / "plan.md"
                tasks_file = feature_dir / "tasks.md"
                if plan_file.exists():
                    process_plan_file(engine, plan_file, workspace_path)
                if tasks_file.exists():
                    process_tasks_file(engine, tasks_file, workspace_path)

    # 2. Scan Antigravity IDE brain directories
    brain_dir = Path(os.path.expanduser("~/.gemini/antigravity/brain"))
    if brain_dir.exists():
        for sess_dir in brain_dir.iterdir():
            if sess_dir.is_dir():
                plan_file = sess_dir / "implementation_plan.md"
                tasks_file = sess_dir / "task.md"
                if plan_file.exists():
                    process_plan_file(engine, plan_file, workspace_path)
                if tasks_file.exists():
                    process_tasks_file(engine, tasks_file, workspace_path)

    # 3. Recursive Specification Scan for nested sub-repositories
    try:
        target_plan_tasks = {
            "plan.md",
            "tasks.md",
            "task.md",
            "implementation_plan.md",
            "spec.md",
        }
        for f in _safe_walk(workspace_path, target_plan_tasks, max_depth=5):
            if f.name.lower() in {"plan.md", "implementation_plan.md", "spec.md"}:
                process_plan_file(engine, f, workspace_path)
            elif f.name.lower() in {"tasks.md", "task.md"}:
                process_tasks_file(engine, f, workspace_path)
    except Exception as e:
        logger.debug(f"Failed during nested specification scan: {e}")

    # 4. Multi-IDE / Platform Skills Scan
    try:
        skills_dirs = get_all_skills_directories(workspace_path)
        for s_dir in skills_dirs:
            for f in _safe_walk(s_dir, {"skill.md"}, max_depth=3):
                process_skill_file(engine, f, workspace_path)
            for f in _safe_walk(
                s_dir,
                {"plan.md", "tasks.md", "task.md", "implementation_plan.md"},
                max_depth=3,
            ):
                if f.name.lower() in {"plan.md", "implementation_plan.md"}:
                    process_plan_file(engine, f, workspace_path)
                elif f.name.lower() in {"tasks.md", "task.md"}:
                    process_tasks_file(engine, f, workspace_path)
    except Exception as e:
        logger.debug(f"Failed during skills scan: {e}")

    # 5. ScholarX Downloads Scan
    try:
        scholarx_dirs = get_scholarx_directories()
        target_exts = {".pdf", ".docx", ".doc", ".txt", ".md"}
        for s_dir in scholarx_dirs:
            try:
                for item in s_dir.iterdir():
                    if item.is_file() and item.suffix.lower() in target_exts:
                        if item.name.lower() != "skill.md":
                            process_scholarx_file(engine, item)
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"Failed during ScholarX scan: {e}")

    # 6. Core Knowledge Graph Ingest Locations Scan
    try:
        kg_paths = get_kg_ingest_paths(workspace_path)
        for p in kg_paths:
            if p.is_file():
                process_kg_ingest_location(engine, p)
            elif p.is_dir():
                try:
                    for child in p.iterdir():
                        if child.is_file():
                            process_kg_ingest_location(engine, child)
                except Exception:
                    pass
    except Exception as e:
        logger.debug(f"Failed during core KG ingestion location scan: {e}")


def run_plan_watcher_loop(engine: Any, workspace_path: Path, interval: float = 5.0):
    """Runs the plan watcher polling loop indefinitely (usually in a background thread)."""
    logger.info(
        f"Starting plan watcher loop for workspace {workspace_path} (interval: {interval}s)"
    )
    while True:
        try:
            if not _WATCHER_PAUSED:
                # Avoid lock contention/deadlocks with active database writers/ingestion processes
                db_path = getattr(getattr(engine, "backend", None), "db_path", None)
                if db_path and db_path != ":memory:":
                    from filelock import FileLock, Timeout

                    lock = FileLock(f"{db_path}.lock", timeout=0)
                    try:
                        with lock:
                            pass
                        run_watcher_scan(engine, workspace_path)
                    except Timeout:
                        logger.debug(
                            "Plan watcher: Skipping scan iteration because database lock is held (active ingestion)."
                        )
                else:
                    run_watcher_scan(engine, workspace_path)
            else:
                logger.debug("Plan watcher loop is paused.")
        except Exception as e:
            logger.error(f"Error in plan watcher loop: {e}")
        time.sleep(interval)


def seed_plans_from_prompts(
    engine: Any, workspace_path: Path, prompts_dir: Path
) -> int:
    """Seed the KG with verified plans from a prompts/plans directory."""
    if not prompts_dir.exists():
        logger.warning(f"Prompts/plans directory not found at {prompts_dir}")
        return 0

    count = 0
    history_dir = workspace_path / ".specify" / "history" / "plans"
    history_dir.mkdir(parents=True, exist_ok=True)

    for md_file in prompts_dir.glob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8", errors="ignore")
            # Generate feature_id from filename
            feature_id = md_file.stem.lower().replace(" ", "_").replace("-", "_")
            parsed = _parse_plan_metadata(content)
            content_hash = _get_md5(content)

            # Ingest as version 1
            ingest_plan_version(
                engine=engine,
                feature_id=feature_id,
                title=parsed["title"],
                approach=parsed["approach"],
                version=1,
                session_id="seeded_session",
                content_hash=content_hash,
                raw_content=content,
            )

            # Write versioned copy to history folder
            timestamp_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            history_file = history_dir / f"plan_{feature_id}_v1_{timestamp_str}.md"
            history_file.write_text(content, encoding="utf-8")

            count += 1
        except Exception as e:
            logger.error(f"Failed to seed plan from {md_file}: {e}")

    logger.info(f"Successfully seeded {count} verified plans into the KG and history.")
    return count


def get_workspace_path() -> Path:
    """Resolve the active workspace path using env vars or current folder hierarchy."""
    env_path = os.environ.get("WORKSPACE_PATH")
    if env_path:
        return Path(env_path).resolve()
    cwd = Path(os.getcwd()).resolve()
    for parent in [cwd] + list(cwd.parents):
        if (
            (parent / ".specify").exists()
            or (parent / ".git").exists()
            or (parent / "pyproject.toml").exists()
        ):
            return parent
    return cwd
