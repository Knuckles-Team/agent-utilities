#!/usr/bin/python
"""Compaction support for agent context management.

Holds the single tool-pair-safety invariant (``_shared.py``) that every history
compaction path (the live Memento sawtooth in ``capabilities/memento.py``) routes its
eviction decisions through, so a compacted message list never orphans a tool-call /
tool-return pair.
"""

from ._shared import enforce_tool_pair_safety, iter_tool_pairs

__all__ = ["enforce_tool_pair_safety", "iter_tool_pairs"]
