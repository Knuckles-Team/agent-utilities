"""
Year-over-Year Filing Diff — CONCEPT:AU-KG.research.research-pipeline-runner

Given this-year and last-year Risk Factors / MD&A text from two annual filings,
produce a structured "what is NEW or REMOVED, ignore boilerplate" diff and an
agent runner that turns the diff into the signal a Bear/Burry persona reads.

The actual filings are fetched elsewhere (an EDGAR MCP); this helper only
accepts text inputs. A deterministic paragraph-level pre-diff runs locally (no
engine, no LLM) so callers always get a structured ``FilingDiff`` even offline;
the LLM runner refines it into NEW/REMOVED/MODIFIED material-change findings and
gracefully falls back to the deterministic diff when no model is configured.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Boilerplate sentence/paragraph signatures that recur unchanged in 10-K risk
# sections and should not be reported as a material change.
_BOILERPLATE_PATTERNS = (
    r"the following risk factors",
    r"you should carefully consider",
    r"these risks are not the only",
    r"additional risks and uncertainties not (currently|presently) known",
    r"could materially (and adversely )?affect",
    r"forward[- ]looking statements",
    r"see .*part [iv]+",
)
_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE)


def _paragraphs(text: str) -> list[str]:
    """Split filing text into normalized, non-empty paragraphs."""
    if not text:
        return []
    chunks = re.split(r"\n\s*\n|\r\n\s*\r\n", text)
    out: list[str] = []
    for c in chunks:
        norm = re.sub(r"\s+", " ", c).strip()
        if norm:
            out.append(norm)
    return out


def _is_boilerplate(paragraph: str) -> bool:
    return bool(_BOILERPLATE_RE.search(paragraph))


@dataclass
class FilingDiff:
    """Deterministic paragraph-level diff of two filing sections.

    Boilerplate paragraphs are filtered out of ``added`` / ``removed`` so the
    output isolates genuinely new or dropped disclosure language.
    """

    section: str
    added: list[str] = field(default_factory=list)  # paragraphs NEW this year
    removed: list[str] = field(default_factory=list)  # paragraphs REMOVED this year
    boilerplate_skipped: int = 0

    @property
    def has_material_change(self) -> bool:
        return bool(self.added or self.removed)

    def to_prompt(self) -> str:
        """Render the diff as a structured prompt block for an analyst agent."""
        new_block = (
            "\n".join(f"  [NEW] {p}" for p in self.added) if self.added else "  (none)"
        )
        removed_block = (
            "\n".join(f"  [REMOVED] {p}" for p in self.removed)
            if self.removed
            else "  (none)"
        )
        return (
            f"Year-over-year diff of the '{self.section}' section "
            f"(boilerplate ignored; {self.boilerplate_skipped} boilerplate "
            "paragraphs skipped).\n\n"
            f"NEW language THIS year:\n{new_block}\n\n"
            f"REMOVED language vs LAST year:\n{removed_block}\n\n"
            "Identify which of these are MATERIAL changes (new risk exposure, "
            "going-concern, customer/supplier concentration, covenant/leverage, "
            "litigation, hedging, or a quietly dropped risk) vs. cosmetic edits."
        )


def diff_filing_sections(
    section: str,
    this_year_text: str,
    last_year_text: str,
    similarity_threshold: float = 0.6,
) -> FilingDiff:
    """Compute a boilerplate-filtered paragraph diff between two filing sections.

    A paragraph counts as unchanged when its closest match in the other year is
    at least ``similarity_threshold`` similar (``difflib`` ratio). Everything
    else is reported as NEW (present this year) or REMOVED (present last year).
    """
    this_paras = _paragraphs(this_year_text)
    last_paras = _paragraphs(last_year_text)

    skipped = 0
    added: list[str] = []
    removed: list[str] = []

    def _best_ratio(p: str, pool: list[str]) -> float:
        best = 0.0
        for q in pool:
            r = difflib.SequenceMatcher(None, p, q).ratio()
            if r > best:
                best = r
                if best >= similarity_threshold:
                    break
        return best

    for p in this_paras:
        if _is_boilerplate(p):
            skipped += 1
            continue
        if _best_ratio(p, last_paras) < similarity_threshold:
            added.append(p)

    for p in last_paras:
        if _is_boilerplate(p):
            skipped += 1
            continue
        if _best_ratio(p, this_paras) < similarity_threshold:
            removed.append(p)

    return FilingDiff(
        section=section,
        added=added,
        removed=removed,
        boilerplate_skipped=skipped,
    )


class FilingDiffFinding(BaseModel):
    """A single material year-over-year disclosure change."""

    change_type: str = Field(..., description="NEW | REMOVED | MODIFIED")
    summary: str = Field(..., description="Plain-language description of the change")
    severity: str = Field("info", description="info | watch | red_flag")
    excerpt: str = Field(
        "", description="Quoted filing language supporting the finding"
    )


class FilingDiffResult(BaseModel):
    """Structured output of the YoY filing-diff agent run."""

    section: str
    findings: list[FilingDiffFinding] = Field(default_factory=list)
    overall_assessment: str = ""
    has_material_change: bool = False


class FilingDiffAgent:
    """Runs the YoY filing diff and refines it into material-change findings.

    The deterministic ``diff_filing_sections`` always runs first; the LLM only
    classifies the resulting NEW/REMOVED paragraphs. With no model configured it
    falls back to reporting the deterministic diff directly so it works offline.
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def run(
        self,
        section: str,
        this_year_text: str,
        last_year_text: str,
    ) -> FilingDiffResult:
        diff = diff_filing_sections(section, this_year_text, last_year_text)

        try:
            from pydantic_ai import Agent

            from agent_utilities.core.model_factory import create_model

            model = self.llm or create_model()
            agent = Agent(
                model=model,
                output_type=FilingDiffResult,
                system_prompt=(
                    "You are a forensic filing analyst. You are given a "
                    "year-over-year diff of a 10-K Risk Factors / MD&A section "
                    "with boilerplate already removed. Classify each NEW or "
                    "REMOVED block as a material change or cosmetic edit, assign "
                    "a severity (info/watch/red_flag), and give an overall "
                    "assessment. Quote the filing language; do NOT invent text."
                ),
            )
            result = agent.run_sync(diff.to_prompt())
            out = result.output
            out.section = section
            out.has_material_change = diff.has_material_change or bool(out.findings)
            return out
        except Exception as exc:  # noqa: BLE001 — degrade to deterministic diff
            logger.warning(
                "Filing-diff LLM refinement unavailable, using deterministic diff: %s",
                exc,
            )
            findings = [
                FilingDiffFinding(
                    change_type="NEW", summary=p[:200], severity="watch", excerpt=p
                )
                for p in diff.added
            ] + [
                FilingDiffFinding(
                    change_type="REMOVED", summary=p[:200], severity="watch", excerpt=p
                )
                for p in diff.removed
            ]
            return FilingDiffResult(
                section=section,
                findings=findings,
                overall_assessment=(
                    f"{len(diff.added)} new and {len(diff.removed)} removed "
                    "non-boilerplate disclosure blocks detected (deterministic diff)."
                ),
                has_material_change=diff.has_material_change,
            )
