"""EH-041 / EH-042 / EH-043: connector-side decisions through EG ``Decide``.

EH-042 (connector-internal tool choice) and EH-043 (write-back proposals)
are owned by the connector SDK -- :mod:`agent_connector_sdk.decide.consumers`
-- where the connector runtime calls them; AU re-exports them rather than
keeping a second copy. :func:`agent_utilities.decide.install_runner` installs
AU's runner into the SDK too, so both sides ask EG the same question through
the same runner.

EH-041 (inbound fleet event triage) is AU's own: it dispatches AU's fleet
events, so it lives here. Evaluate-only (DECIDE-LAYER-DESIGN §4.5): records
are sampled; the specificity lookup is the deterministic fallback.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_connector_sdk.decide.consumers import (
    NO_WRITE,
    connector_tool,
    propose_writeback,
)

from agent_utilities import decide
from agent_utilities.decide.options import Option, text_param


def triage_playbook(source: str, severity: str, registered: Mapping[str, Any]) -> str:
    """The playbook key for one event (EH-041); the specificity lookup is the fallback."""
    keys = [k for k in (f"{source}:{severity}", source, "default") if k in registered]
    if not keys:
        return "default"
    options = [
        Option(key, {"heuristic": float(rank == 0), "severity_rank": float(2 - rank)})
        for rank, key in enumerate(keys)
    ]
    choice = decide.choose(
        "au.connector.triage",
        options,
        lambda: keys[0],
        params=[text_param("source", source), text_param("severity", severity)],
    )
    return str(choice.option_id or keys[0])


__all__ = ["NO_WRITE", "connector_tool", "propose_writeback", "triage_playbook"]
