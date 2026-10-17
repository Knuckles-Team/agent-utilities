"""Computer-use tier port for the developer workspace.

The harness only dispatches computer-use actions to a driver it is given. A
real driver (a desktop sandbox reached by the host) is supplied by the host;
without one the null driver answers every action with a refusal observation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .events import ComputerUseAction, Observation, ScreenObservation


@runtime_checkable
class ComputerUseDriver(Protocol):
    async def run(self, action: ComputerUseAction) -> Observation: ...


class NullComputerUseDriver:
    """The always-available floor: advertises the tier but performs no actuation."""

    name = "null"

    async def run(self, action: ComputerUseAction) -> Observation:
        return ScreenObservation(
            error="computer-use tier not provisioned "
            "(attach a ComputerUseDriver from the host to enable it)",
        )
