from __future__ import annotations

"""Constitution & Prompt Policy Ingestor — SDD governance into KG policies.

CONCEPT:KG-2.2 — Engineering Rules Engine (Constitution Extension)

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
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.engine import IntelligenceGraphEngine

from ...models.knowledge_graph import (
    PolicyNode,
    RegistryEdgeType,
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
    result: dict[str, Any] = {
        "project_name": "",
        "vision": "",
        "mission": "",
        "policies": [],  # List of (category, statement, section) tuples
    }

    # Extract project name from title
    title_match = re.search(
        r"^#\s+Project Constitution\s*-?\s*(.*)", content, re.MULTILINE
    )
    if title_match:
        result["project_name"] = title_match.group(1).strip()

    # Extract vision
    vision_match = re.search(r"\*\*Vision[^*]*\*\*:?\s*(.*?)(?:\n|$)", content)
    if not vision_match:
        # Try markdown format without bold
        vision_match = re.search(r"\*\*[^*]*\*\*\s+is\s+(.*?)(?:\.\s|\n)", content)
    if vision_match:
        result["vision"] = vision_match.group(1).strip()

    # Extract mission
    mission_match = re.search(r"\*\*Mission\*\*:?\s*(.*?)(?:\n|$)", content)
    if mission_match:
        result["mission"] = mission_match.group(1).strip()

    # Parse sections and extract policy statements
    current_section = ""
    current_subsection = ""

    for line in content.split("\n"):
        # H2 section headers
        h2_match = re.match(r"^##\s+(.+)$", line)
        if h2_match:
            current_section = h2_match.group(1).strip()
            current_subsection = ""
            continue

        # H3 subsection headers
        h3_match = re.match(r"^###\s+(.+)$", line)
        if h3_match:
            current_subsection = h3_match.group(1).strip()
            continue

        # Bullet points are policy statements
        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            statement = stripped[2:].strip()

            # Clean up bold markers for readability
            clean_statement = re.sub(r"\*\*([^*]+)\*\*:?\s*", r"\1: ", statement)

            # Determine category from section context
            category = _determine_category(current_section, current_subsection)

            result["policies"].append(
                {
                    "category": category,
                    "statement": clean_statement,
                    "section": current_section,
                    "subsection": current_subsection,
                    "is_normative": _is_normative(statement),
                }
            )

        # Standalone text lines in governance/quality sections (non-bullet)
        elif stripped and not stripped.startswith("#") and current_section:
            category = _determine_category(current_section, current_subsection)
            if category in ("governance", "quality_gate") and len(stripped) > 20:
                result["policies"].append(
                    {
                        "category": category,
                        "statement": stripped,
                        "section": current_section,
                        "subsection": current_subsection,
                        "is_normative": _is_normative(stripped),
                    }
                )

    return result


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
    rules: list[dict[str, str]] = []
    task = prompt_data.get("task", "unknown")

    rules_data = prompt_data.get("rules")
    if rules_data is None:
        return rules

    # Format 1: Simple list of strings
    if isinstance(rules_data, list):
        for rule in rules_data:
            if isinstance(rule, str):
                rules.append(
                    {
                        "statement": rule,
                        "category": "prompt_rule",
                        "source": f"prompt:{task}:rules",
                    }
                )

    # Format 2: Categorized dict — {"category_name": ["rule1", "rule2"]}
    elif isinstance(rules_data, dict):
        for category, items in rules_data.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, str):
                        rules.append(
                            {
                                "statement": item,
                                "category": category,
                                "source": f"prompt:{task}:rules.{category}",
                            }
                        )

    return rules


class PolicyIngestor:
    """Ingests constitutions and prompt rules into the KG as PolicyNodes.

    CONCEPT:KG-2.2 — Engineering Rules Engine (Constitution Extension)

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

        # Search for constitution in standard locations
        candidates = [
            root / ".specify" / "memory" / "constitution.md",
            root / ".specify" / "constitution.md",
            root / "CONSTITUTION.md",
            root / "constitution.md",
        ]

        constitution_path = None
        for candidate in candidates:
            if candidate.exists():
                constitution_path = candidate
                break

        if not constitution_path:
            logger.info("No constitution found in %s", root)
            return {"policies_ingested": 0, "edges_created": 0}

        content = constitution_path.read_text(encoding="utf-8")
        parsed = parse_constitution_md(content)

        stats: dict[str, int] = {
            "policies_ingested": 0,
            "edges_created": 0,
        }

        project_name = parsed["project_name"] or root.name

        # Create a project anchor node for linking
        project_node_id = f"project:{project_name}"
        self.engine.graph.add_node(
            project_node_id,
            type="software_project",
            name=project_name,
            vision=parsed["vision"],
            mission=parsed["mission"],
            importance_score=1.0,
            is_permanent=True,
        )

        # Ingest each policy statement
        for policy_data in parsed["policies"]:
            policy_id = f"policy:{project_name}:{uuid.uuid4().hex[:8]}"

            # Map normative statements to higher priority
            priority = 80 if policy_data["is_normative"] else 50
            if policy_data["category"] == "constraint":
                priority = 90
            elif policy_data["category"] == "quality_gate":
                priority = 70
            elif policy_data["category"] == "tech_stack":
                priority = 60

            # Determine action from statement
            statement = policy_data["statement"]
            condition = f"When working on {project_name}"
            if policy_data["subsection"]:
                condition += f" ({policy_data['subsection']})"

            node = PolicyNode(
                id=policy_id,
                name=statement[:80],
                description=statement,
                policy_id=f"{project_name}-{uuid.uuid4().hex[:6]}",
                condition=condition,
                action=statement,
                priority=priority,
                applies_to=[project_name],
                version=version,
                timestamp=ts,
                importance_score=0.8 if policy_data["is_normative"] else 0.6,
                is_permanent=True,
                metadata={
                    "source": POLICY_SOURCE_CONSTITUTION,
                    "category": policy_data["category"],
                    "section": policy_data["section"],
                    "subsection": policy_data["subsection"],
                    "is_normative": policy_data["is_normative"],
                    "constitution_path": str(constitution_path),
                },
            )

            # Generate embedding if available
            if self.engine.hybrid_retriever.embed_model:
                try:
                    node.embedding = (
                        self.engine.hybrid_retriever.embed_model.get_text_embedding(
                            f"{project_name} policy: {statement}"
                        )
                    )
                except Exception as e:
                    logger.debug("Failed to embed policy %s: %s", policy_id, e)

            self.engine.graph.add_node(node.id, **node.model_dump())
            if self.engine.backend:
                data = self.engine._serialize_node(node, label="Policy")
                self.engine._upsert_node("Policy", policy_id, data)

            # Link: Policy → Project
            self.engine.link_nodes(
                policy_id,
                project_node_id,
                RegistryEdgeType.APPLIES_TO,
                {
                    "source": POLICY_SOURCE_CONSTITUTION,
                    "category": policy_data["category"],
                },
            )
            stats["policies_ingested"] += 1
            stats["edges_created"] += 1

        logger.info(
            "Constitution ingestion complete for '%s': %d policies, %d edges",
            project_name,
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
            logger.warning("Prompts directory not found: %s", prompts_path)
            return {"policies_ingested": 0, "edges_created": 0}

        stats: dict[str, int] = {
            "policies_ingested": 0,
            "prompts_scanned": 0,
            "edges_created": 0,
        }

        for json_file in sorted(prompts_path.glob("*.json")):
            try:
                prompt_data = json.loads(json_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.debug("Skipping %s: %s", json_file.name, e)
                continue

            if not isinstance(prompt_data, dict):
                continue

            stats["prompts_scanned"] += 1
            task = prompt_data.get("task", json_file.stem)

            rules = _extract_prompt_rules(prompt_data)
            if not rules:
                continue

            # Create a prompt anchor node
            prompt_node_id = f"prompt:{task}"

            for rule_data in rules:
                policy_id = f"policy:prompt:{task}:{uuid.uuid4().hex[:8]}"

                priority_map = {
                    "prompt_rule": 60,
                    "quality_gate": 70,
                    "workflow": 50,
                }

                node = PolicyNode(
                    id=policy_id,
                    name=rule_data["statement"][:80],
                    description=rule_data["statement"],
                    policy_id=f"prompt-{task}-{uuid.uuid4().hex[:6]}",
                    condition=f"When acting as {task.replace('_', ' ')}",
                    action=rule_data["statement"],
                    priority=priority_map.get(rule_data["category"], 50),
                    applies_to=[task],
                    version=version,
                    timestamp=ts,
                    importance_score=0.5,
                    is_permanent=True,
                    metadata={
                        "source": POLICY_SOURCE_PROMPT,
                        "category": rule_data["category"],
                        "prompt_source": rule_data.get("source", ""),
                        "prompt_file": json_file.name,
                    },
                )

                self.engine.graph.add_node(node.id, **node.model_dump())
                if self.engine.backend:
                    data = self.engine._serialize_node(node, label="Policy")
                    self.engine._upsert_node("Policy", policy_id, data)

                # Link: Policy → Prompt
                if prompt_node_id in self.engine.graph:
                    self.engine.link_nodes(
                        policy_id,
                        prompt_node_id,
                        RegistryEdgeType.APPLIES_TO,
                        {
                            "source": POLICY_SOURCE_PROMPT,
                            "category": rule_data["category"],
                        },
                    )
                    stats["edges_created"] += 1

                stats["policies_ingested"] += 1

        logger.info(
            "Prompt rules ingestion: scanned %d prompts, ingested %d policies",
            stats["prompts_scanned"],
            stats["policies_ingested"],
        )
        return stats

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

        for node_id, data in self.engine.graph.nodes(data=True):
            if data.get("type") != "policy":
                continue

            meta = data.get("metadata", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}

            # Filter by project
            if project_name:
                applies_to = data.get("applies_to", [])
                if isinstance(applies_to, str):
                    applies_to = [applies_to]
                if project_name not in applies_to:
                    continue

            # Filter by category
            if category and meta.get("category") != category:
                continue

            # Filter by agent role
            if agent_role:
                applies_to = data.get("applies_to", [])
                if isinstance(applies_to, str):
                    applies_to = [applies_to]
                if agent_role not in applies_to:
                    continue

            # Filter normative only
            if include_normative_only and not meta.get("is_normative", False):
                continue

            matches.append(
                {
                    "id": node_id,
                    "statement": data.get("action", data.get("description", "")),
                    "category": meta.get("category", "general"),
                    "priority": int(data.get("priority", 50)),
                    "source": meta.get("source", "unknown"),
                    "is_normative": meta.get("is_normative", False),
                    "applies_to": data.get("applies_to", []),
                }
            )

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
