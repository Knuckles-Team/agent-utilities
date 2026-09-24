"""ST-1: the swarm-topology vocabulary is an EG core schema source.

The vocabulary (topology classes, slot roles, stop rules, task-shape
admissibility) and its template-projection shapes are authored by AU but OWNED
by EG as the immutable core source :data:`SOURCE_ID` (RF-ADR-009 clean cut: AU
ships, loads and parses no ``.ttl``). Every graph's composed schema carries it,
so EG entails admissibility with no attach step; AU refers to it by id and to
its terms by :data:`SWARM_NS` only. A tenant extends the vocabulary under its
own GraphSchema key.
"""

from __future__ import annotations

#: The EG core schema source the vocabulary and shapes live in.
SOURCE_ID = "core:swarm-topology@1"
#: The vocabulary namespace.
SWARM_NS = "http://knuckles.team/kg/swarm#"

__all__ = ["SOURCE_ID", "SWARM_NS"]
