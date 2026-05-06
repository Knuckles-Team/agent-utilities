"""Engineering Rules Ingestor — Parses agent-rules-books into KG nodes.

CONCEPT:KG-2.2 — Engineering Rules Engine

Parses structured markdown rule files (mini/nano tiers) from the
agent-rules-books repository and creates versioned KG nodes for
context-sensitive retrieval, OWL reasoning, and AHE efficacy tracking.

Architecture:
    - Parses the standard mini.md section structure:
      1. Title  2. When to use  3. Primary bias to correct
      4. Decision rules  5. Trigger rules  6. Final checklist
    - Creates ``RuleBookNode`` for each book
    - Creates ``EngineeringRuleNode`` for each decision/trigger rule
    - Wires SKOS broader/narrower and PROV-O wasDerivedFrom relationships
    - Generates embeddings for semantic retrieval (when model available)

Usage::

    from agent_utilities.knowledge_graph.rule_ingestor import RuleIngestor

    ingestor = RuleIngestor(engine)
    stats = ingestor.ingest_rules_books()  # Uses bundled data by default
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .engine import IntelligenceGraphEngine

from ..models.knowledge_graph import (
    EngineeringRuleNode,
    RegistryEdgeType,
    RuleBookNode,
)

logger = logging.getLogger(__name__)


def get_bundled_rules_path() -> Path:
    """Resolve the path to the bundled engineering rules data.

    Returns the ``agent_utilities/policies/engineering_rules/`` directory
    shipped with the package. Falls back to importlib.resources for
    installed packages.

    Returns:
        Path to the bundled rules directory.
    """
    # Direct path (development / editable installs)
    pkg_dir = Path(__file__).parent.parent / "policies" / "engineering_rules"
    if pkg_dir.is_dir():
        return pkg_dir

    # importlib.resources fallback (installed packages)
    try:
        from importlib.resources import files

        data_dir = files("agent_utilities") / "policies" / "engineering_rules"
        return Path(str(data_dir))
    except Exception:
        return pkg_dir  # Return even if it doesn't exist yet


# ── Book Metadata Registry ───────────────────────────────────────────
# Maps book slug → (author, domain_tags) from README.md
BOOK_METADATA: dict[str, dict[str, Any]] = {
    "a-philosophy-of-software-design": {
        "author": "John Ousterhout",
        "domain_tags": ["refactoring", "api-design", "code-quality"],
    },
    "clean-architecture": {
        "author": "Robert C. Martin",
        "domain_tags": ["architecture", "boundaries", "code-quality"],
    },
    "clean-code": {
        "author": "Robert C. Martin",
        "domain_tags": ["code-quality", "readability", "testing"],
    },
    "code-complete": {
        "author": "Steve McConnell",
        "domain_tags": ["code-quality", "construction", "testing"],
    },
    "designing-data-intensive-applications": {
        "author": "Martin Kleppmann",
        "domain_tags": ["data-systems", "architecture", "production"],
    },
    "domain-driven-design": {
        "author": "Eric Evans",
        "domain_tags": ["domain-modeling", "architecture", "boundaries"],
    },
    "domain-driven-design-distilled": {
        "author": "Vaughn Vernon",
        "domain_tags": ["domain-modeling", "architecture"],
    },
    "implementing-domain-driven-design": {
        "author": "Vaughn Vernon",
        "domain_tags": ["domain-modeling", "architecture", "implementation"],
    },
    "patterns-of-enterprise-application-architecture": {
        "author": "Martin Fowler",
        "domain_tags": ["architecture", "patterns", "data-systems"],
    },
    "refactoring": {
        "author": "Martin Fowler",
        "domain_tags": ["refactoring", "code-quality"],
    },
    "release-it": {
        "author": "Michael T. Nygard",
        "domain_tags": ["production", "reliability", "architecture"],
    },
    "the-pragmatic-programmer": {
        "author": "Andrew Hunt, David Thomas",
        "domain_tags": ["code-quality", "engineering-style"],
    },
    "working-effectively-with-legacy-code": {
        "author": "Michael Feathers",
        "domain_tags": ["legacy", "refactoring", "testing"],
    },
}


@dataclass
class ParsedRuleSet:
    """Intermediate representation of a parsed mini/nano markdown file."""

    book_slug: str
    tier: str  # "mini" or "nano"
    title: str = ""
    when_to_use: str = ""
    primary_bias: str = ""
    decision_rules: list[str] = field(default_factory=list)
    trigger_rules: list[str] = field(default_factory=list)
    checklist_items: list[str] = field(default_factory=list)
    raw_content: str = ""
    # Frontmatter fields (from YAML header)
    frontmatter_name: str = ""
    frontmatter_description: str = ""
    frontmatter_author: str = ""
    frontmatter_version: str = ""


def _parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    """Extract YAML frontmatter from markdown content.

    Args:
        content: Raw markdown with optional ``---`` delimited frontmatter.

    Returns:
        Tuple of (frontmatter dict, remaining body content).
    """
    if not content.startswith("---"):
        return {}, content

    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        return {}, content

    fm_block = content[3 : 3 + end_match.start()]
    body = content[3 + end_match.end() :]

    # Simple YAML-like key: value parsing (no nested structures)
    fm: dict[str, str] = {}
    for line in fm_block.strip().split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            fm[key.strip()] = value.strip().strip('"').strip("'")

    return fm, body


def parse_mini_markdown(content: str, book_slug: str, tier: str) -> ParsedRuleSet:
    """Parse a mini/nano markdown file into structured sections.

    Supports optional YAML frontmatter with ``name``, ``description``,
    ``author``, ``version``, and ``tier`` fields for versioning.

    Expects the standard format from agent-rules-books PROCESS.md:
    1. title, 2. when to use, 3. primary bias to correct,
    4. decision rules, 5. trigger rules, 6. final checklist

    Args:
        content: Raw markdown text (may include frontmatter).
        book_slug: Kebab-case book directory name.
        tier: "mini" or "nano".

    Returns:
        A ``ParsedRuleSet`` with extracted sections and frontmatter.
    """
    # Extract frontmatter if present
    frontmatter, body = _parse_frontmatter(content)

    result = ParsedRuleSet(
        book_slug=book_slug,
        tier=frontmatter.get("tier", tier),
        raw_content=content,
        frontmatter_name=frontmatter.get("name", ""),
        frontmatter_description=frontmatter.get("description", ""),
        frontmatter_author=frontmatter.get("author", ""),
        frontmatter_version=frontmatter.get("version", ""),
    )

    # Extract title (first H1 from body, not frontmatter)
    title_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    if title_match:
        result.title = title_match.group(1).strip()

    # Extract sections by H2 headers (from body, not frontmatter)
    sections = _split_by_h2(body)

    for header, section_body in sections.items():
        header_lower = header.lower().strip()
        if "when to use" in header_lower:
            result.when_to_use = section_body.strip()
        elif "primary bias" in header_lower:
            result.primary_bias = section_body.strip()
        elif "decision rule" in header_lower:
            result.decision_rules = _extract_list_items(section_body)
        elif "trigger rule" in header_lower:
            result.trigger_rules = _extract_list_items(section_body)
        elif "checklist" in header_lower or "final" in header_lower:
            result.checklist_items = _extract_list_items(section_body)

    return result


def _split_by_h2(content: str) -> dict[str, str]:
    """Split markdown content into sections by H2 headers."""
    sections: dict[str, str] = {}
    current_header = ""
    current_body: list[str] = []

    for line in content.split("\n"):
        h2_match = re.match(r"^##\s+(.+)$", line)
        if h2_match:
            if current_header:
                sections[current_header] = "\n".join(current_body)
            current_header = h2_match.group(1)
            current_body = []
        else:
            current_body.append(line)

    if current_header:
        sections[current_header] = "\n".join(current_body)

    return sections


def _extract_list_items(text: str) -> list[str]:
    """Extract markdown list items (- bullet points) from text."""
    items: list[str] = []
    current_item: list[str] = []

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("- "):
            if current_item:
                items.append(" ".join(current_item))
            current_item = [stripped[2:].strip()]
        elif stripped and current_item:
            # Continuation line
            current_item.append(stripped)

    if current_item:
        items.append(" ".join(current_item))

    return items


class RuleIngestor:
    """Ingests agent-rules-books into the Knowledge Graph.

    CONCEPT:KG-2.2 — Engineering Rules Engine

    Parses structured markdown files and creates versioned KG nodes
    with embeddings, SKOS taxonomy, and PROV-O provenance.

    Args:
        engine: The ``IntelligenceGraphEngine`` to ingest into.
    """

    def __init__(self, engine: IntelligenceGraphEngine) -> None:
        self.engine = engine

    def ingest_rules_books(
        self,
        rules_books_path: str | None = None,
        tiers: list[str] | None = None,
        version: str = "1.0.0",
    ) -> dict[str, Any]:
        """Ingest all books from bundled data or an external repository.

        Args:
            rules_books_path: Optional external path to agent-rules-books root.
                If None, uses the bundled ``data/engineering_rules/`` directory.
            tiers: Which tiers to ingest. Defaults to ``["mini"]``.
            version: Semantic version for this ingestion round.

        Returns:
            Statistics dict with counts of ingested books, rules, and edges.
        """
        tiers = tiers or ["mini"]

        # Resolve data source
        if rules_books_path:
            root = Path(rules_books_path)
            use_bundled = False
        else:
            root = get_bundled_rules_path()
            use_bundled = True

        if not root.is_dir():
            raise FileNotFoundError(f"Rules data path not found: {root}")

        stats: dict[str, int] = {
            "books_ingested": 0,
            "rules_ingested": 0,
            "edges_created": 0,
        }
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        for book_slug, meta in BOOK_METADATA.items():
            # Resolve markdown file based on layout
            if use_bundled:
                # Bundled: flat layout — files are {slug}.{tier}.md in one dir
                book_found = False
                for tier in tiers:
                    md_file = root / f"{book_slug}.{tier}.md"
                    if md_file.exists():
                        book_found = True
                        break
                if not book_found:
                    logger.debug("No bundled data for %s", book_slug)
                    continue
            else:
                # External repo: per-book subdirectories
                book_dir = root / book_slug
                if not book_dir.is_dir():
                    logger.warning("Book directory not found: %s", book_dir)
                    continue

            # Create RuleBookNode
            book_node = self._ingest_book(book_slug, meta, ts, version)
            stats["books_ingested"] += 1

            # Parse and ingest each tier
            for tier in tiers:
                if use_bundled:
                    md_file = root / f"{book_slug}.{tier}.md"
                else:
                    md_file = root / book_slug / f"{book_slug}.{tier}.md"
                    if tier == "full":
                        md_file = root / book_slug / f"{book_slug}.md"

                if not md_file.exists():
                    logger.debug("Tier file not found: %s", md_file)
                    continue

                content = md_file.read_text(encoding="utf-8")
                parsed = parse_mini_markdown(content, book_slug, tier)

                # Use frontmatter version if present, else fall back to method param
                rule_version = parsed.frontmatter_version or version

                # Update book node with parsed metadata
                if parsed.when_to_use and not book_node.when_to_use:
                    book_node.when_to_use = parsed.when_to_use
                if parsed.primary_bias and not book_node.primary_bias:
                    book_node.primary_bias = parsed.primary_bias
                if parsed.frontmatter_author and not book_node.author:
                    book_node.author = parsed.frontmatter_author

                # Ingest individual rules
                rule_count = self._ingest_rules_from_parsed(
                    parsed, book_node.id, meta, ts, rule_version
                )
                stats["rules_ingested"] += rule_count
                stats["edges_created"] += rule_count  # Each rule → book edge

                # Update tier counts
                total_rules = (
                    len(parsed.decision_rules)
                    + len(parsed.trigger_rules)
                    + len(parsed.checklist_items)
                )
                if tier == "mini":
                    book_node.mini_rule_count = total_rules
                elif tier == "nano":
                    book_node.nano_rule_count = total_rules
                elif tier == "full":
                    book_node.full_rule_count = total_rules

            # Re-persist book node with updated metadata
            self.engine.graph.nodes[book_node.id].update(book_node.model_dump())
            if self.engine.backend:
                data = self.engine._serialize_node(book_node, label="RuleBook")
                self.engine._upsert_node("RuleBook", book_node.id, data)

        logger.info(
            "Engineering rules ingestion complete: %d books, %d rules, %d edges",
            stats["books_ingested"],
            stats["rules_ingested"],
            stats["edges_created"],
        )
        return stats

    def _ingest_book(
        self,
        book_slug: str,
        meta: dict[str, Any],
        ts: str,
        version: str,
    ) -> RuleBookNode:
        """Create or update a RuleBookNode for a book."""
        book_id = f"book:{book_slug}"
        node = RuleBookNode(
            id=book_id,
            name=book_slug.replace("-", " ").title(),
            book_id=book_slug,
            author=meta.get("author", ""),
            domain_tags=meta.get("domain_tags", []),
            description=f"Engineering rules from '{book_slug.replace('-', ' ').title()}'",
            timestamp=ts,
            importance_score=0.9,
            is_permanent=True,
            version=version,
        )

        # Generate embedding if available
        if self.engine.hybrid_retriever.embed_model:
            try:
                node.embedding = (
                    self.engine.hybrid_retriever.embed_model.get_text_embedding(
                        f"{node.name}: {node.description}"
                    )
                )
            except Exception as e:
                logger.debug("Failed to embed book %s: %s", book_id, e)

        self.engine.graph.add_node(node.id, **node.model_dump())
        if self.engine.backend:
            data = self.engine._serialize_node(node, label="RuleBook")
            self.engine._upsert_node("RuleBook", book_id, data)

        return node

    def _ingest_rules_from_parsed(
        self,
        parsed: ParsedRuleSet,
        book_id: str,
        meta: dict[str, Any],
        ts: str,
        version: str,
    ) -> int:
        """Create EngineeringRuleNode instances from a parsed rule set."""
        count = 0

        # Decision rules
        for rule_text in parsed.decision_rules:
            self._create_rule_node(
                rule_text=rule_text,
                book_id=book_id,
                parsed=parsed,
                meta=meta,
                rule_class="decision-changing",
                source_section="Decision rules",
                ts=ts,
                version=version,
            )
            count += 1

        # Trigger rules
        for rule_text in parsed.trigger_rules:
            self._create_rule_node(
                rule_text=rule_text,
                book_id=book_id,
                parsed=parsed,
                meta=meta,
                rule_class="trigger",
                source_section="Trigger rules",
                trigger_condition=rule_text,
                ts=ts,
                version=version,
            )
            count += 1

        # Checklist items
        for rule_text in parsed.checklist_items:
            self._create_rule_node(
                rule_text=rule_text,
                book_id=book_id,
                parsed=parsed,
                meta=meta,
                rule_class="checklist-only",
                source_section="Final checklist",
                ts=ts,
                version=version,
            )
            count += 1

        return count

    def _create_rule_node(
        self,
        rule_text: str,
        book_id: str,
        parsed: ParsedRuleSet,
        meta: dict[str, Any],
        rule_class: str,
        source_section: str,
        ts: str,
        version: str,
        trigger_condition: str = "",
    ) -> EngineeringRuleNode:
        """Create a single EngineeringRuleNode and wire KG relationships."""
        rule_id = f"rule:{parsed.book_slug}:{uuid.uuid4().hex[:8]}"

        # Determine strength from rule_class
        strength_map = {
            "book-thesis": 0.9,
            "decision-changing": 0.8,
            "micro-decision": 0.6,
            "trigger": 0.7,
            "conflict-resolver": 0.7,
            "checklist-only": 0.4,
            "framing": 0.3,
            "default": 0.5,
        }
        # Determine conflict weight from evidence depth
        # Books with more rules = more material backing = higher weight
        meta_domain_count = len(meta.get("domain_tags", []))
        base_weight = min(1.0, 0.3 + (meta_domain_count * 0.1))

        node = EngineeringRuleNode(
            id=rule_id,
            name=rule_text[:80],
            principle_id=f"{parsed.book_slug}-{uuid.uuid4().hex[:6]}",
            statement=rule_text,
            description=f"[{parsed.tier}] {rule_text}",
            tier=parsed.tier,  # type: ignore
            rule_class=rule_class,  # type: ignore
            bias_corrected=parsed.primary_bias,
            trigger_condition=trigger_condition,
            task_relevance_tags=meta.get("domain_tags", []),
            source_book_id=book_id,
            source_section=source_section,
            strength=strength_map.get(rule_class, 0.5),
            conflict_weight=base_weight,
            timestamp=ts,
            importance_score=0.7,
            is_permanent=True,
            version=version,
        )

        # Generate embedding
        if self.engine.hybrid_retriever.embed_model:
            try:
                embed_text = f"{parsed.title} — {rule_text}"
                node.embedding = (
                    self.engine.hybrid_retriever.embed_model.get_text_embedding(
                        embed_text
                    )
                )
            except Exception as e:
                logger.debug("Failed to embed rule %s: %s", rule_id, e)

        # Persist
        self.engine.graph.add_node(node.id, **node.model_dump())
        if self.engine.backend:
            data = self.engine._serialize_node(node, label="EngineeringRule")
            self.engine._upsert_node("EngineeringRule", rule_id, data)

        # Wire: Rule -[WAS_DERIVED_FROM]-> Book
        self.engine.link_nodes(
            rule_id,
            book_id,
            RegistryEdgeType.WAS_DERIVED_FROM,
            {"provenance": "agent-rules-books", "tier": parsed.tier},
        )

        return node

    def query_rules_for_task(
        self,
        task_tags: list[str],
        tier: str = "mini",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Query engineering rules relevant to a task's domain tags.

        Uses both graph traversal and semantic matching to find rules
        that are relevant to the given task context.

        Args:
            task_tags: Domain tags describing the current task
                (e.g., ["architecture", "refactoring"]).
            tier: Maximum tier to include ("nano", "mini", or "full").
            limit: Maximum number of rules to return.

        Returns:
            List of rule dicts sorted by relevance (conflict_weight × strength).
        """
        tier_order = {"nano": 0, "mini": 1, "full": 2}
        max_tier_val = tier_order.get(tier, 1)

        matches: list[dict[str, Any]] = []

        for node_id, data in self.engine.graph.nodes(data=True):
            if data.get("type") != "engineering_rule":
                continue

            rule_tier = data.get("tier", "mini")
            if tier_order.get(rule_tier, 1) > max_tier_val:
                continue

            # Check tag overlap
            rule_tags = data.get("task_relevance_tags", [])
            overlap = set(task_tags) & set(rule_tags)
            if not overlap:
                continue

            relevance = len(overlap) / max(len(task_tags), 1)
            strength = float(data.get("strength", 0.5))
            efficacy = float(data.get("efficacy_score", 0.5))
            weight = float(data.get("conflict_weight", 0.5))

            # Composite score
            score = (
                (relevance * 0.3) + (strength * 0.2) + (efficacy * 0.3) + (weight * 0.2)
            )

            matches.append(
                {
                    "id": node_id,
                    "statement": data.get("statement", ""),
                    "tier": rule_tier,
                    "rule_class": data.get("rule_class", ""),
                    "source_book_id": data.get("source_book_id", ""),
                    "source_section": data.get("source_section", ""),
                    "score": round(score, 4),
                    "tag_overlap": list(overlap),
                }
            )

        # Sort by score descending
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[:limit]

    def resolve_conflicts(
        self,
        rules: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Remove conflicting rules, keeping the higher-weighted one.

        Uses the ``CONFLICTS_WITH`` edges in the KG and deterministic
        ``conflict_weight`` comparison to resolve conflicts.

        Args:
            rules: List of rule dicts from ``query_rules_for_task()``.

        Returns:
            Filtered list with conflicts resolved.
        """
        excluded_ids: set[str] = set()

        for rule in rules:
            if rule["id"] in excluded_ids:
                continue

            # Check for conflict edges
            if rule["id"] in self.engine.graph:
                for neighbor in self.engine.graph.neighbors(rule["id"]):
                    edge_data = self.engine.graph.get_edge_data(rule["id"], neighbor)
                    if not edge_data:
                        continue
                    for _, edata in edge_data.items():
                        if edata.get("type") == RegistryEdgeType.CONFLICTS_WITH:
                            # Find the conflicting rule in our result set
                            conflicting = next(
                                (r for r in rules if r["id"] == neighbor),
                                None,
                            )
                            if conflicting and conflicting["id"] not in excluded_ids:
                                # Deterministic resolution: higher weight wins
                                rule_weight = float(
                                    self.engine.graph.nodes[rule["id"]].get(
                                        "conflict_weight", 0.5
                                    )
                                )
                                conflict_weight = float(
                                    self.engine.graph.nodes[neighbor].get(
                                        "conflict_weight", 0.5
                                    )
                                )
                                if rule_weight >= conflict_weight:
                                    excluded_ids.add(neighbor)
                                else:
                                    excluded_ids.add(rule["id"])

        return [r for r in rules if r["id"] not in excluded_ids]

    def render_rules_for_prompt(
        self,
        rules: list[dict[str, Any]],
        include_source: bool = True,
    ) -> str:
        """Render a list of rules into markdown text for prompt injection.

        Args:
            rules: Rule dicts from ``query_rules_for_task()``.
            include_source: Whether to include source book attribution.

        Returns:
            Formatted markdown string ready for prompt insertion.
        """
        if not rules:
            return ""

        lines = ["## Engineering Rules\n"]

        # Group by source book
        by_book: dict[str, list[dict[str, Any]]] = {}
        for rule in rules:
            book_id = rule.get("source_book_id", "unknown")
            by_book.setdefault(book_id, []).append(rule)

        for book_id, book_rules in by_book.items():
            if include_source:
                book_name = book_id.replace("book:", "").replace("-", " ").title()
                lines.append(f"### {book_name}\n")

            # Separate by section
            for rule in book_rules:
                prefix = ""
                if rule.get("rule_class") == "trigger":
                    prefix = "⚡ "
                elif rule.get("rule_class") == "checklist-only":
                    prefix = "☑ "
                lines.append(f"- {prefix}{rule['statement']}")

            lines.append("")

        return "\n".join(lines)
