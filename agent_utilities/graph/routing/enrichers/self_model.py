"""Self-Model routing enrichers (R4, R5 — CONCEPT:AU-KG.memory.tiered-memory-caching).

Pure formatters extracted from the router monolith so the discovery-context
injection is defined once and independently testable:

* R4 — domain-proficiency injection (from the Self-Model success rates).
* R5 — ACO pheromone-trail specialist affinities.

Both return a context string to append to the router's discovery context (empty
string when there is nothing to inject), so they never alter control flow.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def format_proficiency_context(domain_success_rates: Mapping[str, float] | None) -> str:
    """R4: top-5 domain proficiencies as a discovery-context block."""
    if not domain_success_rates:
        return ""
    lines = [
        f"- {domain}: {rate:.0%} success rate"
        for domain, rate in sorted(
            domain_success_rates.items(), key=lambda x: x[1], reverse=True
        )[:5]
    ]
    return (
        "\n### YOUR DOMAIN PROFICIENCY (from Self-Model)\n"
        + "\n".join(lines)
        + "\nPrefer routing to domains where you have proven competence.\n\n"
    )


def format_pheromone_affinities(
    pheromone_trails: Mapping[str, Mapping[str, float]] | None,
) -> str:
    """R5: top-7 specialist→domain ACO pheromone affinities as a context block."""
    if not pheromone_trails:
        return ""
    lines: list[str] = []
    for specialist_id, domains in sorted(
        pheromone_trails.items(),
        key=lambda x: max(x[1].values()) if x[1] else 0,
        reverse=True,
    )[:7]:
        if not domains:
            continue
        top_domain = max(domains, key=domains.get)  # type: ignore[arg-type]
        lines.append(
            f"- {specialist_id} → {top_domain} (affinity: {domains[top_domain]:.0%})"
        )
    if not lines:
        return ""
    return (
        "### SPECIALIST AFFINITIES (ACO Pheromone Trails)\n"
        + "\n".join(lines)
        + "\nThese specialists have proven track records for these domains.\n\n"
    )


def self_model_context(current: Any) -> str:
    """Combined R4+R5 context for a Self-Model snapshot (``current``).

    Tolerates a snapshot missing either attribute; returns the concatenated
    proficiency + affinity blocks (possibly empty).
    """
    if current is None:
        return ""
    block = format_proficiency_context(getattr(current, "domain_success_rates", None))
    # R5 was only injected when proficiency was present in the monolith; preserve
    # that ordering but make each block independently safe.
    if block:
        block += format_pheromone_affinities(getattr(current, "pheromone_trails", None))
    return block
