#!/usr/bin/python
"""Dynamic Skill Evolution (CONCEPT:ECO-4.1 Enhancement).

Derived from: Skill Neologisms — Towards Skill-based Continual Learning
(arXiv:2605.04970v1, Score 11.9)

Key insight: Skill graphs should not be static — new skill representations
can be created on-the-fly from execution traces to avoid catastrophic
forgetting during continual learning.

Provides:
- SkillNeologismDetector — identifies when existing skills don't cover a new capability
- SkillFactory — creates new skill nodes from execution traces
- SkillMerger — detects and consolidates overlapping skills
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from datetime import UTC, datetime

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SkillNode(BaseModel):
    """A skill representation in the evolvable skill graph.

    CONCEPT:ECO-4.1 — Each skill has a name, description, trigger patterns,
    and provenance tracking to its creation source.

    Attributes:
        skill_id: Unique identifier for this skill.
        name: Human-readable skill name.
        description: What this skill does.
        trigger_patterns: Regex patterns that activate this skill.
        keywords: Keywords associated with this skill.
        provenance: How this skill was created (manual, trace, merge).
        source_trace_id: ID of the execution trace that created this skill.
        created_at: When the skill was created.
        activation_count: How many times this skill has been activated.
        confidence: Confidence that this is a valid, useful skill (0–1).
    """

    skill_id: str
    name: str
    description: str = ""
    trigger_patterns: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    provenance: str = "manual"  # manual, trace, merge, split
    source_trace_id: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    activation_count: int = 0
    confidence: float = 0.5

    def matches(self, text: str) -> bool:
        """Check if this skill matches the given text.

        Args:
            text: Input text to check against trigger patterns and keywords.

        Returns:
            True if any trigger pattern or keyword matches.
        """
        text_lower = text.lower()

        # Check trigger patterns
        for pattern in self.trigger_patterns:
            try:
                if re.search(pattern, text_lower):
                    return True
            except re.error:
                if pattern.lower() in text_lower:
                    return True

        # Check keywords
        return any(kw.lower() in text_lower for kw in self.keywords)


class SkillGap(BaseModel):
    """A detected gap in skill coverage.

    Attributes:
        task_text: The task that couldn't be matched.
        closest_skill: The nearest existing skill (if any).
        similarity_score: How similar the task is to the closest skill (0–1).
        gap_keywords: Keywords in the task not covered by any skill.
        suggested_name: Suggested name for a new skill.
    """

    task_text: str
    closest_skill: str = ""
    similarity_score: float = 0.0
    gap_keywords: list[str] = Field(default_factory=list)
    suggested_name: str = ""


class SkillMergeCandidate(BaseModel):
    """Two skills that may overlap and should be consolidated.

    Attributes:
        skill_a_id: First skill ID.
        skill_b_id: Second skill ID.
        overlap_score: How much these skills overlap (0–1).
        shared_keywords: Keywords present in both skills.
        recommendation: merge, keep_both, or split.
    """

    skill_a_id: str
    skill_b_id: str
    overlap_score: float = 0.0
    shared_keywords: list[str] = Field(default_factory=list)
    recommendation: str = "keep_both"


# ---------------------------------------------------------------------------
# Skill Neologism Detector
# ---------------------------------------------------------------------------


class SkillNeologismDetector:
    """Detects when existing skills don't cover a new capability.

    CONCEPT:ECO-4.1 — Identifies skill gaps by comparing task features
    against the existing skill graph.
    """

    def __init__(
        self,
        skills: list[SkillNode] | None = None,
        gap_threshold: float = 0.3,
    ) -> None:
        """Initialize the detector.

        Args:
            skills: Existing skill nodes to check against.
            gap_threshold: Similarity threshold below which a gap is
                detected (0–1). Default: 0.3.
        """
        self.skills = list(skills or [])
        self.gap_threshold = gap_threshold

    def detect_gap(self, task_text: str) -> SkillGap | None:
        """Check if the task reveals a skill gap.

        Args:
            task_text: The task description to evaluate.

        Returns:
            A SkillGap if no existing skill covers this task, else None.
        """
        if not self.skills:
            return SkillGap(
                task_text=task_text,
                gap_keywords=task_text.lower().split()[:10],
                suggested_name=self._suggest_name(task_text),
            )

        # Find the closest matching skill
        best_score = 0.0
        best_skill = ""
        task_words = set(task_text.lower().split())

        for skill in self.skills:
            if skill.matches(task_text):
                return None  # Covered by existing skill

            # Keyword similarity
            skill_words = set(kw.lower() for kw in skill.keywords) | set(
                skill.name.lower().split()
            )
            if not skill_words:
                continue

            overlap = len(task_words & skill_words)
            score = overlap / max(len(task_words | skill_words), 1)

            if score > best_score:
                best_score = score
                best_skill = skill.skill_id

        if best_score < self.gap_threshold:
            # Gap detected
            covered_words = set()
            for skill in self.skills:
                covered_words.update(kw.lower() for kw in skill.keywords)

            gap_words = [w for w in task_words if w not in covered_words and len(w) > 3]

            return SkillGap(
                task_text=task_text,
                closest_skill=best_skill,
                similarity_score=best_score,
                gap_keywords=gap_words[:15],
                suggested_name=self._suggest_name(task_text),
            )

        return None

    @staticmethod
    def _suggest_name(task_text: str) -> str:
        """Generate a suggested skill name from task text."""
        # Extract significant words (>3 chars, not stop words)
        stop_words = {
            "the",
            "and",
            "for",
            "with",
            "this",
            "that",
            "from",
            "have",
            "will",
        }
        words = [
            w
            for w in task_text.split()[:8]
            if len(w) > 3 and w.lower() not in stop_words
        ]
        return "-".join(words[:3]).lower().replace(".", "")


# ---------------------------------------------------------------------------
# Skill Factory
# ---------------------------------------------------------------------------


class SkillFactory:
    """Creates new skill nodes from execution traces.

    CONCEPT:ECO-4.1 — When a gap is detected, the factory creates a new
    skill node with auto-generated trigger patterns, keywords, and
    provenance tracking.
    """

    def __init__(self, prefix: str = "auto") -> None:
        """Initialize the factory.

        Args:
            prefix: Prefix for auto-generated skill IDs.
        """
        self.prefix = prefix
        self._created_count = 0

    def create_from_gap(
        self,
        gap: SkillGap,
        trace_id: str = "",
    ) -> SkillNode:
        """Create a new skill node from a detected gap.

        Args:
            gap: The skill gap to fill.
            trace_id: Optional trace ID for provenance.

        Returns:
            A new SkillNode with auto-generated properties.
        """
        self._created_count += 1
        skill_id = (
            f"{self.prefix}:{hashlib.sha256(gap.task_text.encode()).hexdigest()[:10]}"
        )

        # Generate trigger patterns from gap keywords
        patterns = []
        for kw in gap.gap_keywords[:5]:
            patterns.append(re.escape(kw))

        # Generate description
        description = (
            f"Auto-generated skill for: {gap.task_text[:100]}. "
            f"Created to fill gap (similarity to nearest skill: {gap.similarity_score:.2f})."
        )

        return SkillNode(
            skill_id=skill_id,
            name=gap.suggested_name or f"skill-{self._created_count}",
            description=description,
            trigger_patterns=patterns,
            keywords=gap.gap_keywords,
            provenance="trace",
            source_trace_id=trace_id,
            confidence=0.3,  # Low confidence until validated by use
        )

    def create_from_execution(
        self,
        task_text: str,
        result_summary: str,
        success: bool = True,
        trace_id: str = "",
    ) -> SkillNode:
        """Create a skill from a successful execution trace.

        Args:
            task_text: The original task description.
            result_summary: Summary of what the execution accomplished.
            success: Whether the execution was successful.
            trace_id: Optional trace ID for provenance.

        Returns:
            A new SkillNode.
        """
        self._created_count += 1
        skill_id = (
            f"{self.prefix}:{hashlib.sha256(task_text.encode()).hexdigest()[:10]}"
        )

        # Extract keywords from both task and result
        all_text = f"{task_text} {result_summary}".lower()
        words = re.findall(r"\b[a-z]{4,}\b", all_text)
        word_freq: dict[str, int] = defaultdict(int)
        for w in words:
            word_freq[w] += 1

        keywords = sorted(word_freq.keys(), key=lambda w: word_freq[w], reverse=True)[
            :10
        ]

        return SkillNode(
            skill_id=skill_id,
            name=SkillNeologismDetector._suggest_name(task_text),
            description=f"Learned from execution: {result_summary[:200]}",
            trigger_patterns=[re.escape(kw) for kw in keywords[:3]],
            keywords=keywords,
            provenance="trace",
            source_trace_id=trace_id,
            confidence=0.5 if success else 0.2,
        )


# ---------------------------------------------------------------------------
# Skill Merger
# ---------------------------------------------------------------------------


class SkillMerger:
    """Detects and consolidates overlapping skills.

    CONCEPT:ECO-4.1 — Prevents skill graph bloat by merging skills
    with high keyword overlap.
    """

    def __init__(self, merge_threshold: float = 0.7) -> None:
        """Initialize the merger.

        Args:
            merge_threshold: Overlap threshold above which skills should
                be merged (0–1). Default: 0.7.
        """
        self.merge_threshold = merge_threshold

    def find_merge_candidates(
        self,
        skills: list[SkillNode],
    ) -> list[SkillMergeCandidate]:
        """Find pairs of skills that may overlap.

        Args:
            skills: List of skill nodes to analyze.

        Returns:
            List of merge candidates with overlap scores.
        """
        candidates = []

        for i, skill_a in enumerate(skills):
            for skill_b in skills[i + 1 :]:
                overlap = self._compute_overlap(skill_a, skill_b)
                if overlap > 0.1:  # Report any non-trivial overlap
                    shared = list(
                        set(k.lower() for k in skill_a.keywords)
                        & set(k.lower() for k in skill_b.keywords)
                    )
                    recommendation = (
                        "merge" if overlap >= self.merge_threshold else "keep_both"
                    )
                    candidates.append(
                        SkillMergeCandidate(
                            skill_a_id=skill_a.skill_id,
                            skill_b_id=skill_b.skill_id,
                            overlap_score=overlap,
                            shared_keywords=shared,
                            recommendation=recommendation,
                        )
                    )

        return candidates

    def merge(self, skill_a: SkillNode, skill_b: SkillNode) -> SkillNode:
        """Merge two skills into one.

        The merged skill inherits keywords, patterns, and the higher
        confidence from both source skills.

        Args:
            skill_a: First skill.
            skill_b: Second skill.

        Returns:
            A new merged SkillNode.
        """
        merged_id = f"merged:{hashlib.sha256(f'{skill_a.skill_id}+{skill_b.skill_id}'.encode()).hexdigest()[:10]}"

        # Combine keywords (deduplicated)
        all_keywords = list(dict.fromkeys(skill_a.keywords + skill_b.keywords))

        # Combine patterns (deduplicated)
        all_patterns = list(
            dict.fromkeys(skill_a.trigger_patterns + skill_b.trigger_patterns)
        )

        return SkillNode(
            skill_id=merged_id,
            name=f"{skill_a.name}+{skill_b.name}",
            description=(
                f"Merged from [{skill_a.skill_id}] and [{skill_b.skill_id}]. "
                f"{skill_a.description} | {skill_b.description}"
            ),
            trigger_patterns=all_patterns,
            keywords=all_keywords,
            provenance="merge",
            activation_count=skill_a.activation_count + skill_b.activation_count,
            confidence=max(skill_a.confidence, skill_b.confidence),
        )

    @staticmethod
    def _compute_overlap(skill_a: SkillNode, skill_b: SkillNode) -> float:
        """Compute Jaccard similarity between two skills' keywords."""
        set_a = set(k.lower() for k in skill_a.keywords)
        set_b = set(k.lower() for k in skill_b.keywords)
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)
