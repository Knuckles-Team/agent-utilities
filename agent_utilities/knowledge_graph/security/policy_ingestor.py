from __future__ import annotations

"""Constitution & Prompt Policy Ingestor — SDD governance into KG policies.

CONCEPT:AU-KG.ingest.engineering-rules — Engineering Rules Engine (Constitution Extension)

Parses project constitutions (``.specify/memory/constitution.md``) and
prompt JSON files to extract governance rules, normative statements,
quality gates, and embedded engineering guidance into ``PolicyNode``
entries in the Knowledge Graph.

This makes Spec-Driven Development (SDD) part of the rules reasoning
layer, alongside book-derived ``EngineeringRuleNode`` entries.

Three policy sources are unified:
    1. **Constitution**: Core principles, normative statements, quality gates,
       and governance rules from ``.specify/memory/constitution.md``
    2. **Prompt rules**: Engineering guidance extracted from the ``rules``
       key or ``core_directive`` of specialist prompt JSON files
    3. **Engineering rules**: Book-derived rules from ``rule_ingestor.py``

Usage::

    from agent_utilities.knowledge_graph.security.policy_ingestor import PolicyIngestor

    ingestor = PolicyIngestor(engine)
    stats = ingestor.ingest_constitution("/path/to/workspace")
    stats = ingestor.ingest_prompt_rules()
"""


import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.engine import IntelligenceGraphEngine

from ...models.knowledge_graph import (
    PolicyNode,
    RegistryEdgeType,
)
from ...security.persistence_privacy import (
    persistence_reference,
    sanitize_for_persistence,
)

logger = logging.getLogger(__name__)

# ── Policy source types ──────────────────────────────────────────────

POLICY_SOURCE_CONSTITUTION = "constitution"
POLICY_SOURCE_PROMPT = "prompt"
POLICY_SOURCE_ENGINEERING_BOOK = "engineering_book"

# ── Constitution section → policy category mapping ───────────────────

SECTION_CATEGORY_MAP: dict[str, str] = {
    "core principles": "principle",
    "guiding principles": "principle",
    "normative statements": "normative",
    "governance": "governance",
    "quality gates": "quality_gate",
    "testing": "quality_gate",
    "verification loop": "quality_gate",
    "prohibited uses": "constraint",
    "tech stack": "tech_stack",
}


def _extract_constitution_metadata(content: str) -> dict[str, str]:
    """Extract the title/vision/mission lines from constitution markdown."""
    project_name = ""
    vision = ""
    mission = ""

    title_match = re.search(
        r"^#\s+Project Constitution\s*-?\s*(.*)", content, re.MULTILINE
    )
    if title_match:
        project_name = title_match.group(1).strip()

    vision_match = re.search(r"\*\*Vision[^*]*\*\*:?\s*(.*?)(?:\n|$)", content)
    if not vision_match:
        # Try markdown format without bold
        vision_match = re.search(r"\*\*[^*]*\*\*\s+is\s+(.*?)(?:\.\s|\n)", content)
    if vision_match:
        vision = vision_match.group(1).strip()

    mission_match = re.search(r"\*\*Mission\*\*:?\s*(.*?)(?:\n|$)", content)
    if mission_match:
        mission = mission_match.group(1).strip()

    return {"project_name": project_name, "vision": vision, "mission": mission}


def _bullet_policy_entry(
    stripped: str, section: str, subsection: str
) -> dict[str, Any]:
    statement = stripped[2:].strip()
    # Clean up bold markers for readability
    clean_statement = re.sub(r"\*\*([^*]+)\*\*:?\s*", r"\1: ", statement)
    category = _determine_category(section, subsection)
    return {
        "category": category,
        "statement": clean_statement,
        "section": section,
        "subsection": subsection,
        "is_normative": _is_normative(statement),
    }


def _standalone_policy_entry(
    stripped: str, section: str, subsection: str
) -> dict[str, Any] | None:
    """A standalone (non-bullet) text line only counts as a policy statement
    inside a governance/quality-gate section, and only past a length floor —
    short standalone lines are prose noise, not rules."""
    category = _determine_category(section, subsection)
    if category not in ("governance", "quality_gate") or len(stripped) <= 20:
        return None
    return {
        "category": category,
        "statement": stripped,
        "section": section,
        "subsection": subsection,
        "is_normative": _is_normative(stripped),
    }


def _parse_constitution_policies(content: str) -> list[dict[str, Any]]:
    """Walk constitution markdown line by line, tracking the current H2/H3
    section, and collect every bullet or qualifying standalone line as a
    policy entry."""
    policies: list[dict[str, Any]] = []
    current_section = ""
    current_subsection = ""

    for line in content.split("\n"):
        h2_match = re.match(r"^##\s+(.+)$", line)
        if h2_match:
            current_section = h2_match.group(1).strip()
            current_subsection = ""
            continue

        h3_match = re.match(r"^###\s+(.+)$", line)
        if h3_match:
            current_subsection = h3_match.group(1).strip()
            continue

        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            policies.append(
                _bullet_policy_entry(stripped, current_section, current_subsection)
            )
            continue

        if stripped and not stripped.startswith("#") and current_section:
            entry = _standalone_policy_entry(
                stripped, current_section, current_subsection
            )
            if entry is not None:
                policies.append(entry)

    return policies


def parse_constitution_md(content: str) -> dict[str, Any]:
    """Parse a constitution.md file into structured policy categories.

    Extracts:
        - Vision & mission
        - Core principles (guiding + normative)
        - Governance rules
        - Quality gates (testing, verification, prohibited uses)
        - Tech stack constraints

    Args:
        content: Raw constitution markdown text.

    Returns:
        Dict with categorized policies and metadata.
    """
    metadata = _extract_constitution_metadata(content)
    return {
        "project_name": metadata["project_name"],
        "vision": metadata["vision"],
        "mission": metadata["mission"],
        "policies": _parse_constitution_policies(content),
    }


def _determine_category(section: str, subsection: str) -> str:
    """Map a section/subsection to a policy category."""
    # Check subsection first (more specific)
    for key, cat in SECTION_CATEGORY_MAP.items():
        if key in subsection.lower():
            return cat
    for key, cat in SECTION_CATEGORY_MAP.items():
        if key in section.lower():
            return cat
    return "general"


def _is_normative(statement: str) -> bool:
    """Detect if a policy statement is normative (MUST/SHALL/REQUIRED)."""
    normative_keywords = ["MUST", "SHALL", "REQUIRED", "MUST NOT", "SHALL NOT"]
    return any(kw in statement for kw in normative_keywords)


def _simple_prompt_rules(rules_data: list[Any], task: str) -> list[dict[str, str]]:
    """Format 1: a plain list of rule strings."""
    return [
        {
            "statement": rule,
            "category": "prompt_rule",
            "source": f"prompt:{task}:rules",
        }
        for rule in rules_data
        if isinstance(rule, str)
    ]


def _categorized_prompt_rules(
    rules_data: dict[str, Any], task: str
) -> list[dict[str, str]]:
    """Format 2: a dict mapping category names to lists of rule strings."""
    rules: list[dict[str, str]] = []
    for category, items in rules_data.items():
        if not isinstance(items, list):
            continue
        rules.extend(
            {
                "statement": item,
                "category": category,
                "source": f"prompt:{task}:rules.{category}",
            }
            for item in items
            if isinstance(item, str)
        )
    return rules


def _extract_prompt_rules(prompt_data: dict[str, Any]) -> list[dict[str, str]]:
    """Extract engineering rules from a prompt JSON file.

    Reads exclusively from the ``rules`` key in the prompt JSON.
    This is the single, canonical location for KG-ingestible policy
    statements within a prompt file. The agent still reads the full
    JSON for its system prompt, but only ``rules`` is parsed by the
    knowledge graph for policy ingestion.

    The ``rules`` key supports two formats:
        - **Simple**: A list of strings (each is a rule statement)
        - **Categorized**: A dict mapping category names to lists of
          rule strings (e.g., ``{"quality_gates": [...], "constraints": [...]}``

    Args:
        prompt_data: Parsed prompt JSON dict.

    Returns:
        List of dicts with ``statement``, ``category``, and ``source``.
    """
    task = prompt_data.get("task", "unknown")
    rules_data = prompt_data.get("rules")
    if rules_data is None:
        return []
    if isinstance(rules_data, list):
        return _simple_prompt_rules(rules_data, task)
    if isinstance(rules_data, dict):
        return _categorized_prompt_rules(rules_data, task)
    return []


def _find_constitution_path(root: Path) -> Path | None:
    """Search the standard constitution locations, in priority order."""
    candidates = [
        root / ".specify" / "memory" / "constitution.md",
        root / ".specify" / "constitution.md",
        root / "CONSTITUTION.md",
        root / "constitution.md",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_constitution_content(constitution_path: Path) -> str:
    """Fail-closed reads: reject a symlink source and an oversized payload
    before ever decoding/sanitizing it."""
    if constitution_path.is_symlink():
        raise ValueError("Constitution source must not be a symbolic link")
    payload = constitution_path.read_bytes()
    if len(payload) > 4 * 1024 * 1024:
        raise ValueError("Constitution source exceeds its ingestion bound")
    content, _privacy = sanitize_for_persistence(payload.decode("utf-8"))
    return str(content)


def _priority_for_policy(policy_data: dict[str, Any]) -> int:
    """Normative statements default to a higher priority; three categories
    override that default outright regardless of normativity."""
    priority = 80 if policy_data["is_normative"] else 50
    if policy_data["category"] == "constraint":
        return 90
    if policy_data["category"] == "quality_gate":
        return 70
    if policy_data["category"] == "tech_stack":
        return 60
    return priority


def _policy_condition(project_name: str, policy_data: dict[str, Any]) -> str:
    condition = f"When working on {project_name}"
    if policy_data["subsection"]:
        condition += f" ({policy_data['subsection']})"
    return condition


def _normalize_policy_metadata(data: dict[str, Any]) -> dict[str, Any]:
    """A policy node's ``metadata`` may be stored as a JSON-encoded string
    (backend round-trip) or a live dict; normalize to a dict either way,
    degrading to ``{}`` on malformed JSON."""
    meta = data.get("metadata", {})
    if not isinstance(meta, str):
        return meta
    try:
        return json.loads(meta)
    except (json.JSONDecodeError, TypeError):
        return {}


def _policy_applies_to(data: dict[str, Any]) -> list[str]:
    applies_to = data.get("applies_to", [])
    if isinstance(applies_to, str):
        return [applies_to]
    return applies_to


def _policy_matches_filters(
    data: dict[str, Any],
    meta: dict[str, Any],
    *,
    project_name: str | None,
    category: str | None,
    agent_role: str | None,
    include_normative_only: bool,
) -> bool:
    if project_name and project_name not in _policy_applies_to(data):
        return False
    if category and meta.get("category") != category:
        return False
    if agent_role and agent_role not in _policy_applies_to(data):
        return False
    if include_normative_only and not meta.get("is_normative", False):
        return False
    return True


def _policy_match_entry(
    node_id: str, data: dict[str, Any], meta: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": node_id,
        "statement": data.get("action", data.get("description", "")),
        "category": meta.get("category", "general"),
        "priority": int(data.get("priority", 50)),
        "source": meta.get("source", "unknown"),
        "is_normative": meta.get("is_normative", False),
        "applies_to": data.get("applies_to", []),
    }


def _load_prompt_json(json_file: Path) -> dict[str, Any] | None:
    """``None`` on a malformed file (invalid JSON) OR a non-dict payload —
    the caller cannot tell those apart, matching the original's silent
    ``continue`` for the second case (only the first is logged)."""
    try:
        prompt_data = json.loads(json_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Skipping invalid prompt policy (%s)", type(e).__name__)
        return None
    if not isinstance(prompt_data, dict):
        return None
    return prompt_data


@dataclass
class _PromptIngestContext:
    """Everything shared across every rule in one prompt file's ingestion,
    threaded through so per-rule helpers stay under the parameter cap."""

    task: str
    timestamp: str
    version: str
    json_file_name: str
    prompt_node_id: str


def _build_prompt_policy_node(
    rule_data: dict[str, str], ctx: _PromptIngestContext
) -> tuple[str, PolicyNode]:
    policy_id = f"policy:prompt:{ctx.task}:{uuid.uuid4().hex}"
    priority_map = {"prompt_rule": 60, "quality_gate": 70, "workflow": 50}
    node = PolicyNode(
        id=policy_id,
        name=rule_data["statement"][:80],
        description=rule_data["statement"],
        policy_id=f"prompt-{ctx.task}-{uuid.uuid4().hex}",
        condition=f"When acting as {ctx.task.replace('_', ' ')}",
        action=rule_data["statement"],
        priority=priority_map.get(rule_data["category"], 50),
        applies_to=[ctx.task],
        version=ctx.version,
        timestamp=ctx.timestamp,
        importance_score=0.5,
        is_permanent=True,
        metadata={
            "source": POLICY_SOURCE_PROMPT,
            "category": rule_data["category"],
            "prompt_source": rule_data.get("source", ""),
            "prompt_file": ctx.json_file_name,
        },
    )
    return policy_id, node


@dataclass
class _ConstitutionIngestContext:
    """Everything shared across every policy statement in one constitution
    ingestion round, threaded through so per-policy helpers stay under the
    parameter cap."""

    project_name: str
    project_ref: str
    project_node_id: str
    version: str
    timestamp: str
    constitution_ref: str


class PolicyIngestor:
    """Ingests constitutions and prompt rules into the KG as PolicyNodes.

    CONCEPT:AU-KG.ingest.engineering-rules — Engineering Rules Engine (Constitution Extension)

    Unifies three policy sources into the KG's governance layer:
      1. Constitution files (SDD governance)
      2. Prompt JSON files (embedded engineering guidance)
      3. Book-derived rules (via ``rule_ingestor.py``)

    Args:
        engine: The ``IntelligenceGraphEngine`` to ingest into.
    """

    def __init__(self, engine: IntelligenceGraphEngine) -> None:
        self.engine = engine

    def ingest_constitution(
        self,
        workspace_path: str | Path,
        version: str = "1.0.0",
    ) -> dict[str, Any]:
        """Ingest a project constitution into the KG as PolicyNodes.

        Searches for constitution files in:
            1. ``.specify/memory/constitution.md``
            2. ``.specify/constitution.md``
            3. ``CONSTITUTION.md`` (project root)
            4. ``constitution.md`` (project root)

        Args:
            workspace_path: Absolute path to the project workspace root.
            version: Semantic version for this ingestion round.

        Returns:
            Statistics dict with counts of ingested policies and edges.
        """
        root = Path(workspace_path)
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        constitution_path = _find_constitution_path(root)
        if not constitution_path:
            logger.info("No constitution found in the configured policy source")
            return {"policies_ingested": 0, "edges_created": 0}

        content = _load_constitution_content(constitution_path)
        parsed = parse_constitution_md(content)
        constitution_ref = persistence_reference(
            "constitution", constitution_path, namespace="policy-ingestion"
        )

        stats: dict[str, Any] = {
            "policies_ingested": 0,
            "edges_created": 0,
        }

        project_name = str(parsed["project_name"] or "project")
        project_ref = persistence_reference(
            "project", project_name, namespace="policy-ingestion"
        )

        # Create a project anchor node for linking
        project_node_id = f"project:{project_ref}"
        self.engine.graph.add_node(
            project_node_id,
            node_type="software_project",
            name=project_name,
            vision=parsed["vision"],
            mission=parsed["mission"],
            importance_score=1.0,
            is_permanent=True,
        )

        ctx = _ConstitutionIngestContext(
            project_name=project_name,
            project_ref=project_ref,
            project_node_id=project_node_id,
            version=version,
            timestamp=ts,
            constitution_ref=constitution_ref,
        )
        for policy_data in parsed["policies"]:
            self._ingest_one_policy(policy_data, ctx)
            stats["policies_ingested"] += 1
            stats["edges_created"] += 1

        logger.info(
            "Constitution ingestion complete: %d policies, %d edges",
            stats["policies_ingested"],
            stats["edges_created"],
        )
        # Surface the full constitution text (attached to the project anchor) so
        # the ingestion seam mines concepts + canonical facts from it — one
        # bounded payload per constitution, not per policy statement.
        stats["enrichable"] = [
            {
                "source_id": project_node_id,
                "text": content,
                "source_type": "policy",
                "title": project_name,
            }
        ]
        return stats

    def _build_policy_node(
        self, policy_data: dict[str, Any], ctx: _ConstitutionIngestContext
    ) -> tuple[str, PolicyNode]:
        policy_id = f"policy:{ctx.project_ref}:{uuid.uuid4().hex}"
        priority = _priority_for_policy(policy_data)
        statement = policy_data["statement"]
        condition = _policy_condition(ctx.project_name, policy_data)
        node = PolicyNode(
            id=policy_id,
            name=statement[:80],
            description=statement,
            policy_id=f"{ctx.project_ref}-{uuid.uuid4().hex}",
            condition=condition,
            action=statement,
            priority=priority,
            applies_to=[ctx.project_name],
            version=ctx.version,
            timestamp=ctx.timestamp,
            importance_score=0.8 if policy_data["is_normative"] else 0.6,
            is_permanent=True,
            metadata={
                "source": POLICY_SOURCE_CONSTITUTION,
                "category": policy_data["category"],
                "section": policy_data["section"],
                "subsection": policy_data["subsection"],
                "is_normative": policy_data["is_normative"],
                "constitution_ref": ctx.constitution_ref,
            },
        )
        return policy_id, node

    def _embed_policy_node(
        self, node: PolicyNode, project_name: str, statement: str
    ) -> None:
        if not self.engine.hybrid_retriever.embed_model:
            return
        try:
            node.embedding = (
                self.engine.hybrid_retriever.embed_model.get_text_embedding(
                    f"{project_name} policy: {statement}"
                )
            )
        except Exception as exc:
            logger.debug("Failed to embed policy (%s)", type(exc).__name__)

    def _persist_policy_node(self, node: PolicyNode, policy_id: str) -> None:
        # engine.graph.add_node() fail-closed rejects a literal 'type'
        # property (the pydantic model's own type field) — use the same
        # node/node_type translation _serialize_node already applies for
        # the self.engine.backend branch below, instead of the raw dump.
        self.engine.graph.add_node(node.id, **self.engine._serialize_node(node))
        if self.engine.backend:
            data = self.engine._serialize_node(node, label="Policy")
            self.engine._upsert_node("Policy", policy_id, data)

    def _ingest_one_policy(
        self, policy_data: dict[str, Any], ctx: _ConstitutionIngestContext
    ) -> None:
        policy_id, node = self._build_policy_node(policy_data, ctx)
        self._embed_policy_node(node, ctx.project_name, policy_data["statement"])
        self._persist_policy_node(node, policy_id)
        # Link: Policy → Project
        self.engine.link_nodes(
            policy_id,
            ctx.project_node_id,
            RegistryEdgeType.APPLIES_TO,
            {
                "source": POLICY_SOURCE_CONSTITUTION,
                "category": policy_data["category"],
            },
        )

    def ingest_prompt_rules(
        self,
        prompts_dir: str | Path | None = None,
        version: str = "1.0.0",
    ) -> dict[str, Any]:
        """Ingest engineering rules from prompt JSON files into the KG.

        Scans all ``.json`` files in the prompts directory and extracts
        rules from the ``rules`` key, ``quality_checklist``, and
        ``workflow`` sections.

        Args:
            prompts_dir: Path to the prompts directory. Defaults to the
                packaged ``agent_utilities/prompts/`` directory.
            version: Semantic version for this ingestion round.

        Returns:
            Statistics dict with counts.
        """
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if prompts_dir is None:
            from importlib.resources import files

            prompts_dir = files("agent_utilities") / "prompts"  # type: ignore

        prompts_path = Path(str(prompts_dir))

        if not prompts_path.is_dir():
            logger.warning("Configured prompts directory not found")
            return {"policies_ingested": 0, "edges_created": 0}

        stats: dict[str, int] = {
            "policies_ingested": 0,
            "prompts_scanned": 0,
            "edges_created": 0,
        }

        for json_file in sorted(prompts_path.glob("*.json")):
            self._ingest_prompt_file(json_file, version, ts, stats)

        logger.info(
            "Prompt rules ingestion: scanned %d prompts, ingested %d policies",
            stats["prompts_scanned"],
            stats["policies_ingested"],
        )
        return stats

    def _ingest_one_prompt_rule(
        self, rule_data: dict[str, str], ctx: _PromptIngestContext
    ) -> bool:
        """Returns True iff an APPLIES_TO edge was created (the prompt
        anchor node exists in the graph)."""
        policy_id, node = _build_prompt_policy_node(rule_data, ctx)
        # See ingest_constitution's identical comment above.
        self.engine.graph.add_node(node.id, **self.engine._serialize_node(node))
        if self.engine.backend:
            data = self.engine._serialize_node(node, label="Policy")
            self.engine._upsert_node("Policy", policy_id, data)

        if ctx.prompt_node_id not in self.engine.graph:
            return False
        self.engine.link_nodes(
            policy_id,
            ctx.prompt_node_id,
            RegistryEdgeType.APPLIES_TO,
            {
                "source": POLICY_SOURCE_PROMPT,
                "category": rule_data["category"],
            },
        )
        return True

    def _ingest_prompt_file(
        self, json_file: Path, version: str, ts: str, stats: dict[str, int]
    ) -> None:
        prompt_data = _load_prompt_json(json_file)
        if prompt_data is None:
            return

        stats["prompts_scanned"] += 1
        task = prompt_data.get("task", json_file.stem)

        rules = _extract_prompt_rules(prompt_data)
        if not rules:
            return

        ctx = _PromptIngestContext(
            task=task,
            timestamp=ts,
            version=version,
            json_file_name=json_file.name,
            prompt_node_id=f"prompt:{task}",
        )
        for rule_data in rules:
            if self._ingest_one_prompt_rule(rule_data, ctx):
                stats["edges_created"] += 1
            stats["policies_ingested"] += 1

    def ingest_all(
        self,
        workspace_path: str | Path,
        rules_books_path: str | None = None,
        version: str = "1.0.0",
    ) -> dict[str, Any]:
        """Ingest all policy sources into the KG in one call.

        Combines:
            1. Constitution policies (from workspace)
            2. Prompt rules (from agent_utilities/prompts/)
            3. Engineering rules (from agent-rules-books, if path given)

        Args:
            workspace_path: Project workspace root.
            rules_books_path: Optional path to agent-rules-books repo.
            version: Semantic version.

        Returns:
            Combined statistics dict.
        """
        combined: dict[str, Any] = {}

        # 1. Constitution
        const_stats = self.ingest_constitution(workspace_path, version=version)
        combined["constitution"] = const_stats

        # 2. Prompt rules
        prompt_stats = self.ingest_prompt_rules(version=version)
        combined["prompts"] = prompt_stats

        # 3. Engineering rules (optional)
        if rules_books_path:
            from .rule_ingestor import RuleIngestor

            eng_ingestor = RuleIngestor(self.engine)
            eng_stats = eng_ingestor.ingest_rules_books(
                rules_books_path, version=version
            )
            combined["engineering_rules"] = eng_stats

        # Log combined stats
        total_policies = const_stats.get("policies_ingested", 0) + prompt_stats.get(
            "policies_ingested", 0
        )
        total_rules = combined.get("engineering_rules", {}).get("rules_ingested", 0)
        logger.info(
            "Full policy ingestion: %d constitution policies, %d prompt policies, %d engineering rules",
            const_stats.get("policies_ingested", 0),
            prompt_stats.get("policies_ingested", 0),
            total_rules,
        )
        combined["total_policies"] = total_policies
        combined["total_engineering_rules"] = total_rules
        # Propagate enrichable payloads from sub-ingests so the seam enriches the
        # combined run too.
        combined["enrichable"] = [
            *const_stats.get("enrichable", []),
            *prompt_stats.get("enrichable", []),
            *combined.get("engineering_rules", {}).get("enrichable", []),
        ]

        return combined

    def query_policies_for_context(
        self,
        project_name: str | None = None,
        category: str | None = None,
        agent_role: str | None = None,
        include_normative_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Query policies relevant to the current working context.

        Args:
            project_name: Filter by project (constitution source).
            category: Filter by category (principle, normative, quality_gate, etc.).
            agent_role: Filter by agent role / prompt task.
            include_normative_only: Only return MUST/SHALL policies.

        Returns:
            List of policy dicts sorted by priority (highest first).
        """
        matches: list[dict[str, Any]] = []

        from ..core.bounded_read import iter_nodes_by_types

        # Bounded per-label fetch (CONCEPT:AU-KG.ingest.never-scan-whole-graph) — never a whole-graph node pull.
        for node_id, data in iter_nodes_by_types(self.engine.graph, "policy"):
            meta = _normalize_policy_metadata(data)
            if not _policy_matches_filters(
                data,
                meta,
                project_name=project_name,
                category=category,
                agent_role=agent_role,
                include_normative_only=include_normative_only,
            ):
                continue
            matches.append(_policy_match_entry(node_id, data, meta))

        # Sort by priority descending
        matches.sort(key=lambda x: x["priority"], reverse=True)
        return matches

    def render_policies_for_prompt(
        self,
        policies: list[dict[str, Any]],
        max_items: int = 20,
    ) -> str:
        """Render policies as markdown for prompt injection.

        Groups by source (constitution / prompt / engineering_book) and
        marks normative statements.

        Args:
            policies: Policies from ``query_policies_for_context()``.
            max_items: Maximum items to render.

        Returns:
            Formatted markdown string.
        """
        if not policies:
            return ""

        policies = policies[:max_items]
        lines = ["## Active Policies\n"]

        # Group by source
        by_source: dict[str, list[dict[str, Any]]] = {}
        for p in policies:
            src = p.get("source", "unknown")
            by_source.setdefault(src, []).append(p)

        source_titles = {
            POLICY_SOURCE_CONSTITUTION: "📜 Project Constitution",
            POLICY_SOURCE_PROMPT: "🤖 Agent Rules",
            POLICY_SOURCE_ENGINEERING_BOOK: "📚 Engineering Principles",
        }

        for source, source_policies in by_source.items():
            title = source_titles.get(source, f"📋 {source.title()}")
            lines.append(f"### {title}\n")

            for p in source_policies:
                prefix = "**[MUST]** " if p.get("is_normative") else ""
                lines.append(f"- {prefix}{p['statement']}")

            lines.append("")

        return "\n".join(lines)
