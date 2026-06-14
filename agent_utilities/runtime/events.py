"""CONCEPT:ORCH-1.46 — Action/Observation event protocol for the developer-workspace runtime.

The SWE loop (ORCH-1.47) speaks in *typed actions* that each yield a *typed observation*. This
mirrors OpenHands' action/observation split, but every event here is a first-class,
KG-mintable record (KG-2.64): it carries ``run_id``/``step``/``ts``/``actor`` so the whole
trajectory is replayable and attributable.

Two design rules keep this honest:

* **Frozen, discriminated unions.** Every action/observation is an immutable pydantic model
  keyed by ``kind``. ``ACTION_ADAPTER``/``OBSERVATION_ADAPTER`` parse a wire dict back to the
  exact subtype — so the protocol round-trips across the bridge (``bridge.py``) without
  bespoke parsing.
* **Mutation is explicit.** :func:`mutating_action_name` maps an action to the
  ``ActionPolicy`` (OS-5.24) action name it must be gated by — or ``None`` for read-only
  actions. The runtime never executes a mutating action without a policy decision.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #


class _Event(BaseModel):
    """Common envelope for every action/observation.

    ``run_id``/``step``/``ts``/``actor`` are stamped by :class:`~.workspace.DevWorkspace` when
    the event is admitted to the loop, so callers may construct bare actions (``run_id=""``)
    and let the workspace own correlation.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str = ""
    step: int = 0
    ts: float = 0.0
    actor: str | None = None


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #


class CmdRunAction(_Event):
    """Run a shell command in the workspace (stateful cwd persists across commands)."""

    kind: Literal["cmd_run"] = "cmd_run"
    command: str
    cwd: str | None = None
    timeout: float = 120.0


class FileReadAction(_Event):
    kind: Literal["file_read"] = "file_read"
    path: str
    start: int | None = None  # 1-based inclusive line range; None = whole file
    end: int | None = None


class FileWriteAction(_Event):
    kind: Literal["file_write"] = "file_write"
    path: str
    content: str


class FileEditAction(_Event):
    """Exact-string replace (``old`` -> ``new``) within a file, mirroring developer_tools."""

    kind: Literal["file_edit"] = "file_edit"
    path: str
    old: str
    new: str
    replace_all: bool = False


class TestRunAction(_Event):
    __test__ = False  # not a pytest test class despite the name
    kind: Literal["test_run"] = "test_run"
    selector: str | None = None  # e.g. "tests/test_x.py::test_y"; None = whole suite
    framework: str = "pytest"
    cwd: str | None = None
    timeout: float = 600.0


class PortExposeAction(_Event):
    kind: Literal["port_expose"] = "port_expose"
    port: int


class BrowseAction(_Event):
    """Optional browser tier (CONCEPT:ECO-4.44) — off the core edit→test critical path."""

    kind: Literal["browse"] = "browse"
    url: str = ""
    interaction: str = ""  # optional DSL/JSON describing a click/type sequence


class AgentFinishAction(_Event):
    kind: Literal["agent_finish"] = "agent_finish"
    summary: str = ""


Action = Annotated[
    CmdRunAction
    | FileReadAction
    | FileWriteAction
    | FileEditAction
    | TestRunAction
    | PortExposeAction
    | BrowseAction
    | AgentFinishAction,
    Field(discriminator="kind"),
]


# --------------------------------------------------------------------------- #
# Observations
# --------------------------------------------------------------------------- #


class CmdOutputObservation(_Event):
    kind: Literal["cmd_output"] = "cmd_output"
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    cwd: str = ""  # the cwd AFTER the command (captured so `cd` persists)


class FileContentObservation(_Event):
    kind: Literal["file_content"] = "file_content"
    path: str
    content: str


class FileWriteObservation(_Event):
    kind: Literal["file_write_ok"] = "file_write_ok"
    path: str
    bytes_written: int


class FileEditObservation(_Event):
    kind: Literal["file_edit_ok"] = "file_edit_ok"
    path: str
    diff: str = ""
    applied: bool = True
    replacements: int = 0


class TestResultObservation(_Event):
    __test__ = False  # not a pytest test class despite the name
    kind: Literal["test_result"] = "test_result"
    passed: int = 0
    failed: int = 0
    errors: int = 0
    exit_code: int = 0
    report: str = ""  # short human summary
    raw: str = ""  # truncated raw output


class PortObservation(_Event):
    kind: Literal["port"] = "port"
    port: int
    url: str


class ErrorObservation(_Event):
    """A typed failure (policy denial, unknown action, backend error) — the agent can read it."""

    kind: Literal["error"] = "error"
    message: str
    action_kind: str = ""


class BrowserObservation(_Event):
    kind: Literal["browser"] = "browser"
    url: str = ""
    status: int = 0
    title: str = ""
    text: str = ""
    screenshot_ref: str = ""
    error: str = ""


class NullObservation(_Event):
    kind: Literal["null"] = "null"


Observation = Annotated[
    CmdOutputObservation
    | FileContentObservation
    | FileWriteObservation
    | FileEditObservation
    | TestResultObservation
    | PortObservation
    | BrowserObservation
    | ErrorObservation
    | NullObservation,
    Field(discriminator="kind"),
]


# TypeAdapters for round-tripping a wire dict back to the exact subtype.
from pydantic import TypeAdapter  # noqa: E402

ACTION_ADAPTER: TypeAdapter = TypeAdapter(Action)
OBSERVATION_ADAPTER: TypeAdapter = TypeAdapter(Observation)


# --------------------------------------------------------------------------- #
# Policy mapping
# --------------------------------------------------------------------------- #

# Which ActionPolicy (OS-5.24) action name each action kind must be gated by.
# Read-only actions map to ``None`` and bypass the mutating-action gate.
_MUTATING_KIND_TO_POLICY: dict[str, str] = {
    "cmd_run": "workspace.cmd",
    "test_run": "workspace.cmd",
    "file_write": "workspace.write",
    "file_edit": "workspace.edit",
    "browse": "workspace.browse",
}

# The full set of mutating policy action names this runtime registers (used by callers that
# pre-register them with the ActionPolicy fail-closed gate).
WORKSPACE_MUTATING_ACTIONS: frozenset[str] = frozenset(
    _MUTATING_KIND_TO_POLICY.values()
)


def mutating_action_name(action: BaseModel) -> str | None:
    """Return the ActionPolicy action name to gate ``action`` by, or ``None`` if read-only."""
    return _MUTATING_KIND_TO_POLICY.get(getattr(action, "kind", ""))
