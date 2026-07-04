"""CONCEPT:AU-KG.memory.live-refreshable-artifact-models — Live Refreshable Artifact models + bounded-JSON + safe interpolation.

Assimilated from open-design's Live Artifact spec (specs/2026-04-29-live-artifacts): an output is a
``template`` + bounded ``data`` + ``provenance`` triad. Interpolation is **injection-safe** —
``{{data.path}}`` dotted lookups and a single ``{{#each path}}…{{/each}}`` repeat directive only; no
raw HTML, no expressions; every interpolated value is HTML-escaped.

Superiority delta: the data is re-derived from the **KG** (see :mod:`.refresh`), and ``provenance``
records the producing model/CLI (ORCH-1.33/1.34) and supporting KG evidence node ids (KG-2.18).
"""

from __future__ import annotations

import html
import re
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

# ── Bounded-JSON limits (mirror open-design's caps) ──────────────────
MAX_DEPTH = 8
MAX_KEYS = 100
MAX_ITEMS = 500
MAX_STRING = 16 * 1024
MAX_TOTAL = 256 * 1024


class BoundedJSONError(ValueError):
    """Raised when artifact data violates the bounded-JSON contract."""


def validate_bounded_json(data: Any, *, _depth: int = 0) -> None:
    """Validate that ``data`` respects the depth/keys/items/string/total caps. Raises on violation."""
    if _depth == 0:
        import json

        try:
            total = len(json.dumps(data))
        except (TypeError, ValueError) as exc:
            raise BoundedJSONError(f"data is not JSON-serializable: {exc}") from exc
        if total > MAX_TOTAL:
            raise BoundedJSONError(f"data exceeds {MAX_TOTAL} bytes total ({total})")
    if _depth > MAX_DEPTH:
        raise BoundedJSONError(f"data nesting exceeds depth {MAX_DEPTH}")
    if isinstance(data, dict):
        if len(data) > MAX_KEYS:
            raise BoundedJSONError(f"object exceeds {MAX_KEYS} keys ({len(data)})")
        for v in data.values():
            validate_bounded_json(v, _depth=_depth + 1)
    elif isinstance(data, list):
        if len(data) > MAX_ITEMS:
            raise BoundedJSONError(f"array exceeds {MAX_ITEMS} items ({len(data)})")
        for v in data:
            validate_bounded_json(v, _depth=_depth + 1)
    elif isinstance(data, str):
        if len(data) > MAX_STRING:
            raise BoundedJSONError(f"string exceeds {MAX_STRING} chars ({len(data)})")


# ── Safe interpolation ───────────────────────────────────────────────
_EACH_RE = re.compile(r"\{\{#each\s+([\w.]+)\s*\}\}(.*?)\{\{/each\}\}", re.DOTALL)
_VAR_RE = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")


def _lookup(data: Any, path: str) -> Any:
    cur = data
    for part in path.split("."):
        if part == "data":  # allow optional leading "data." namespace
            continue
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit():
            idx = int(part)
            cur = cur[idx] if 0 <= idx < len(cur) else None
        else:
            return None
    return cur


def render_template(template: str, data: dict[str, Any]) -> str:
    """Render ``template`` against ``data`` with injection-safe interpolation.

    Supports ``{{#each path}}…{{/each}}`` (path must resolve to a list; inside, ``{{item.x}}`` and
    ``{{.}}`` refer to the current element) and ``{{path}}`` scalar lookups. All substituted values
    are HTML-escaped. Unknown paths render as empty string.
    """

    def render_each(m: re.Match) -> str:
        seq = _lookup(data, m.group(1))
        body = m.group(2)
        if not isinstance(seq, list):
            return ""
        out = []
        for elem in seq:
            scope = {"item": elem, "data": data}

            def sub_var(vm: re.Match, elem: object = elem, scope: dict = scope) -> str:
                key = vm.group(1)
                if key in {"item", "."}:
                    val = elem
                elif key.startswith("item."):
                    val = _lookup({"item": elem}, key)
                else:
                    val = _lookup(scope, key)
                return html.escape("" if val is None else str(val))

            out.append(_VAR_RE.sub(sub_var, body))
        return "".join(out)

    rendered = _EACH_RE.sub(render_each, template)

    def sub_scalar(m: re.Match) -> str:
        val = _lookup(data, m.group(1))
        return html.escape("" if val is None else str(val))

    return _VAR_RE.sub(sub_scalar, rendered)


# ── Models ───────────────────────────────────────────────────────────
class Provenance(BaseModel):
    """Who/what produced an artifact generation (KG-2.18 evidence-weighted)."""

    model: str = ""  # producing model / CLI adapter id (ORCH-1.33/1.34)
    source_query: str = ""  # the KG query/source bound for refresh
    evidence_node_ids: list[str] = Field(default_factory=list)  # supporting KG nodes
    generated_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )


class LiveArtifact(BaseModel):
    """A refreshable artifact: template + data + provenance, bound to a KG source.

    CONCEPT:AU-KG.memory.live-refreshable-artifact-models — Live Refreshable Artifact.
    """

    artifact_id: str = Field(default_factory=lambda: f"artifact:{uuid.uuid4().hex[:8]}")
    schema_version: str = "live_artifact_v1"
    name: str = ""
    template: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    source_query: str = ""  # bound query re-run on refresh
    source_node_ids: list[str] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)
    last_rendered: str = ""  # last successful render (preserved on failed refresh)
    refresh_count: int = 0

    def validate_data(self) -> None:
        """Raise :class:`BoundedJSONError` if the current data violates the bounded-JSON contract."""
        validate_bounded_json(self.data)

    def render(self) -> str:
        """Render the artifact's template against its current data (injection-safe)."""
        return render_template(self.template, self.data)
