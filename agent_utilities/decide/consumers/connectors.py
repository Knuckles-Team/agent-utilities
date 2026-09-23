"""EH-041 / EH-042 / EH-043: connector-side decisions through EG ``Decide``.

All three are EVALUATE-ONLY (DECIDE-LAYER-DESIGN §4.5): high-frequency,
low-stakes, so their records are sampled into the decision log rather than
committed per call, and D18 write-back AUTHORIZATION stays deterministic --
a decision only ever proposes.

* EH-041 inbound event triage: which registered playbook handles a fleet
  event; the exact-key -> source -> default lookup is the fallback.
* EH-042 connector-internal tool choice: which of a connector's tools serves
  one request; the connector's own pick is the fallback.
* EH-043 write-back proposals: which candidate mutation (or none) to PROPOSE;
  the proposal still goes through the governed action executor
  (authorization, approval, audit) before anything is written.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agent_utilities import decide
from agent_utilities.decide.options import Option, text_param

NO_WRITE = "no_write"


def _heuristic_options(
    ids: Sequence[str], pick: str, **numbers: Mapping[str, float]
) -> list[Option]:
    return [
        Option(
            i,
            {
                "heuristic": 1.0 if i == pick else 0.0,
                **{k: v[i] for k, v in numbers.items()},
            },
        )
        for i in ids
    ]


def triage_playbook(source: str, severity: str, registered: Mapping[str, Any]) -> str:
    """The playbook key for one event (EH-041); the specificity lookup is the fallback."""
    keys = [k for k in (f"{source}:{severity}", source, "default") if k in registered]
    if not keys:
        return "default"
    rank = {k: float(2 - i) for i, k in enumerate(keys)}
    choice = decide.choose(
        "au.connector.triage",
        _heuristic_options(keys, keys[0], severity_rank=rank),
        lambda: keys[0],
        params=[text_param("source", source), text_param("severity", severity)],
    )
    return str(choice.option_id or keys[0])


def connector_tool(connector: str, tools: Sequence[str], picked: str) -> str:
    """Which of ``tools`` serves one request (EH-042); ``picked`` is the fallback."""
    if picked not in tools:
        return picked
    choice = decide.choose(
        "au.connector.tool",
        _heuristic_options(sorted(set(tools)), picked),
        lambda: picked,
        params=[text_param("connector", connector)],
    )
    return str(choice.option_id or picked)


def propose_writeback(
    proposals: Mapping[str, Mapping[str, Any]], default: str = NO_WRITE
) -> tuple[str, Mapping[str, Any] | None]:
    """The write-back to PROPOSE (EH-043), never executed here.

    ``proposals`` maps a proposal id to its fleet ``execute_action`` parameters;
    ``no_write`` is always an option. Returns ``(id, params)`` -- ``params`` is
    ``None`` for ``no_write`` -- to hand to the governed action executor.
    """
    ids = sorted({*proposals, NO_WRITE})
    choice = decide.choose(
        "au.connector.writeback",
        _heuristic_options(ids, default),
        lambda: default,
    )
    chosen = str(choice.option_id or default)
    return chosen, proposals.get(chosen)


__all__ = ["NO_WRITE", "connector_tool", "propose_writeback", "triage_playbook"]
