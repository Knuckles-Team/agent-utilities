"""Retired-``tiered``-backend migration.

The engine-authority consolidation removed L0/L1/L2/L3 tiering. What this test
module originally verified -- that a deployment whose GRAPH_BACKEND env/config
still says "tiered" keeps booting by mapping forward to the self-contained
engine authority (``epistemic_graph``) -- has since been superseded by a
stricter migration: ``GRAPH_BACKEND`` is now itself one of AgentConfig's
``_RETIRED_CONFIGURATION_KEYS`` (see agent_utilities/core/config.py), so
merely *having it set* now fails config load outright with a ValueError
("retired durable configuration key(s) are not accepted") -- there is no
forward-compat remap left to reach at all. The explicit
``backend_type="tiered"`` argument was retired the same way: it is simply not
one of ``create_backend()``'s recognized types anymore (it logs "Unknown
graph backend type" and returns None, like any other unrecognized string).
"""

from __future__ import annotations

import pytest

from agent_utilities.core.config import AgentConfig
from agent_utilities.knowledge_graph.backends import create_backend


def test_tiered_maps_to_epistemic_graph(monkeypatch):
    # Renamed in spirit only (kept the original test id/function name so this
    # slice's node id list still resolves): GRAPH_BACKEND no longer maps
    # forward to anything -- it is itself a rejected retired configuration
    # key, so merely setting it now fails config load outright.
    monkeypatch.setenv("GRAPH_BACKEND", "tiered")
    with pytest.raises(ValueError, match="GRAPH_BACKEND"):
        AgentConfig()


def test_explicit_tiered_arg_maps_too():
    # Same here: "maps too" described the old forward-compat remap, which no
    # longer exists -- backend_type="tiered" is just an unrecognized type now.
    backend = create_backend(backend_type="tiered")
    assert backend is None
