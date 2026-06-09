from __future__ import annotations

"""Pure In-Memory Graph Backend (CONCEPT:OS-5.0).

Zero-dependency, zero-disk backend using GraphComputeEngine (Rust/epistemic-graph).
Ideal for testing, edge devices, ephemeral containers,
and as the default lightweight backend.

Implements the full GraphBackend ABC with optional
JSON serialization for persistence.
"""


import json
import logging
from typing import Any

from .base import GraphBackend

logger = logging.getLogger(__name__)


class EpistemicGraphBackend(GraphBackend):
    """Pure in-memory graph backend using GraphComputeEngine (Rust-native).

    This is the lightest-weight backend: zero disk, zero external
    dependencies beyond the compiled graph engine. All data lives in
    process memory and is lost on shutdown unless explicitly saved
    via ``save_to_json()``.

    Use cases:
        - Unit testing (fast, deterministic)
        - Edge compute (minimal footprint)
        - Ephemeral containers (no persistent storage needed)
        - Development/prototyping
    """

    def __init__(self) -> None:
        from ..core.graph_compute import GraphComputeEngine

        self._graph = GraphComputeEngine(backend_type="rust")
        self._embeddings: dict[str, list[float]] = {}
        self._node_counter = 0
        logger.info(
            "EpistemicGraphBackend initialized (GraphComputeEngine, pure in-memory)"
        )

    @property
    def graph(self) -> Any:
        """Direct access to the underlying GraphComputeEngine."""
        return self._graph

    # --- GraphBackend ABC Implementation ---

    def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query against the in-memory graph.

        This backend has no Cypher engine, so it interprets the *operational
        subset* the orchestration engine relies on directly over the
        ``GraphComputeEngine`` node store. Supported for single-node
        ``MATCH`` patterns (no relationships):

          - label filter ``(v:Label ...)`` (matched against ``node_type``,
            ``label``, or ``labels``)
          - inline ``{prop: $param | 'literal'}`` and ``WHERE`` equality,
            ``IN [...]``, ``CONTAINS``, ``IS [NOT] NULL`` filters
          - ``SET v.prop = $param | 'literal'`` property mutation (upsert
            merge — critical for Task status transitions)
          - ``DETACH DELETE v``
          - ``RETURN`` projection: bare ``v`` (full node), ``v.prop AS alias``,
            and ``count(v) AS alias``; honours ``LIMIT``

        Anything outside this subset (relationship traversal, MERGE/CREATE,
        unrecognised shapes) falls back to the legacy id/label/all behaviour.
        """
        params = params or {}
        q = (query or "").strip()
        qu = q.upper()

        # Node upsert: ``MERGE (n:Label {id: $id}) SET n.k = $props_k, ...`` is the
        # persistence path used by the graph-writer daemon and sync phase. Without
        # this, ingested nodes never land in the in-memory store. (CONCEPT:KG-2.0)
        if qu.startswith("MERGE") and "->" not in q and "<-" not in q:
            handled, result = self._exec_merge_node(q, params)
            if handled:
                return result

        # Relationship upsert: ``MATCH (a),(b) WHERE a.id=$x AND b.id=$y
        # MERGE (a)-[:REL]->(b)``. Resolve both endpoints by id (O(1)) and add
        # the edge directly — otherwise this falls to the full-scan legacy reader
        # AND never creates the L1 edge. (CONCEPT:KG-2.8 ingestion throughput)
        if "->" in q and "MERGE" in qu and qu.startswith("MATCH"):
            handled, result = self._exec_rel_merge(q, params)
            if handled:
                return result

        # Single-hop relationship traversal read:
        # ``MATCH (a {id:$x})-[:REL]->(b:Label) RETURN b.prop...``. Without this,
        # such reads fall to the legacy full-scan and return every node, not the
        # edge targets (e.g. workflow-step load returned the anchor node too).
        if (
            qu.startswith("MATCH")
            and "->" in q
            and "MERGE" not in qu
            and "CREATE" not in qu
            and "DELETE" not in qu
        ):
            handled, result = self._exec_rel_match(q, params)
            if handled:
                return result

        # Single-node MATCH patterns are interpreted directly; relationship
        # traversals and other write-DDL fall through to the legacy reader.
        if (
            qu.startswith("MATCH")
            and "->" not in q
            and "<-" not in q
            and "MERGE" not in qu
            and "CREATE" not in qu
        ):
            handled, result = self._exec_node_match(q, params)
            if handled:
                return result

        return self._legacy_execute(params)

    def _exec_rel_match(
        self, q: str, params: dict[str, Any]
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Interpret a single-hop ``MATCH (a {id:$x})-[:REL]->(b:Label) RETURN b...``.

        Resolves the anchor ``a`` by id, walks ``REL`` successors, filters the
        targets ``b`` by label, and projects on ``b``. Returns ``(False, [])``
        for any shape outside this subset so the caller can fall back.
        """
        import re

        # Anchor id accepts either a ``$param`` placeholder or an inline quoted
        # literal (``{id:'foo'}``) — interactive callers commonly write the literal.
        m = re.search(
            r"MATCH\s*\(\s*(\w+)\s*(?::\w+)?\s*\{\s*id\s*:\s*(\$\w+|'[^']*'|\"[^\"]*\")\s*\}\s*\)"
            r"\s*-\s*\[\s*\w*\s*:?\s*(\w+)?[^\]]*\]\s*->\s*"
            r"\(\s*(\w+)\s*(?::(\w+))?\s*\)",
            q,
            re.I,
        )
        if not m:
            return False, []
        _src_var, id_token, rel, tgt_var, tgt_label = m.groups()

        # Only the simple "anchor by id, project the target" shape is supported.
        if re.search(r"\bSET\b|\bDETACH\b", q, re.I):
            return False, []

        anchor_id = self._coerce_literal(id_token, params)
        if anchor_id is None or not self._graph.has_node(anchor_id):
            return True, []

        rel_upper = rel.upper() if rel else None
        matched: list[tuple[str, dict[str, Any]]] = []
        for tgt in self._graph.get_successors(anchor_id):
            edge_props = self._graph._get_edge_properties(anchor_id, tgt) or {}
            if rel_upper:
                edge_rel = str(
                    edge_props.get("rel_type") or edge_props.get("type") or ""
                ).upper()
                if edge_rel != rel_upper:
                    continue
            data = self._graph._get_node_properties(tgt) or {}
            if tgt_label and not self._label_match(data, tgt_label):
                continue
            matched.append((tgt, data))

        # Honour ``ORDER BY <tgt>.<prop> [DESC]`` over the projected targets,
        # since successor order is not guaranteed to be meaningful.
        ob = re.search(
            rf"\bORDER\s+BY\s+{tgt_var}\.(\w+)\s*(DESC|ASC)?",
            q,
            re.I,
        )
        if ob:
            order_prop = ob.group(1)
            reverse = bool(ob.group(2)) and ob.group(2).upper() == "DESC"

            def _sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, Any]:
                val = item[1].get(order_prop)
                # None sorts last; keep numeric/string ordering stable otherwise.
                return (1, "") if val is None else (0, val)

            matched.sort(key=_sort_key, reverse=reverse)

        return True, self._project(q, tgt_var, matched, params)

    def _exec_rel_merge(
        self, q: str, params: dict[str, Any]
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Interpret ``MATCH (a),(b) ... MERGE (a)-[:REL]->(b)`` as an O(1) edge add."""
        import re

        mm = re.search(
            r"MERGE\s*\(\s*(\w+)\s*\)\s*-\s*\[\s*:?\s*(\w+)?[^\]]*\]\s*->\s*\(\s*(\w+)\s*\)",
            q,
            re.I,
        )
        if not mm:
            return False, []
        src_var, rel, tgt_var = mm.group(1), (mm.group(2) or "RELATED"), mm.group(3)

        # Resolve each var's id from WHERE (``v.id = $param``) or inline
        # (``(v:Label {id: $param})``).
        idmap: dict[str, Any] = {}
        for mv in re.finditer(r"(\w+)\.id\s*=\s*\$(\w+)", q, re.I):
            idmap[mv.group(1)] = params.get(mv.group(2))
        for mv in re.finditer(
            r"\(\s*(\w+)\s*(?::\w+)?\s*\{\s*id\s*:\s*\$(\w+)\s*\}", q, re.I
        ):
            idmap[mv.group(1)] = params.get(mv.group(2))

        src_id, tgt_id = idmap.get(src_var), idmap.get(tgt_var)
        if not src_id or not tgt_id:
            return False, []  # unrecognised shape → defer to legacy
        try:
            self._graph.add_edge(src_id, tgt_id, {"rel_type": rel})
        except Exception:  # noqa: BLE001
            pass
        return True, []

    def _exec_merge_node(
        self, q: str, params: dict[str, Any]
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Interpret ``MERGE (n:Label {id: $id}) [SET ...]`` as an upsert."""
        import re

        m = re.search(
            r"MERGE\s*\(\s*(\w+)\s*:(\w+)\s*\{\s*id\s*:\s*\$(\w+)\s*\}\s*\)", q, re.I
        )
        if not m:
            return False, []
        var, label, id_param = m.group(1), m.group(2), m.group(3)
        nid = params.get(id_param)
        if nid is None:
            return True, []

        existing = (
            self._graph._get_node_properties(nid) if self._graph.has_node(nid) else {}
        )
        merged = dict(existing or {})
        merged["node_type"] = label

        ms = re.search(r"\bSET\b(.+?)$", q, re.I | re.S)
        if ms:
            for frag in self._split_top_level(ms.group(1)):
                if "=" not in frag:
                    continue
                lhs, rhs = frag.split("=", 1)
                prop = lhs.strip()
                prop = prop[len(var) + 1 :] if prop.startswith(var + ".") else prop
                merged[prop] = self._coerce_literal(rhs, params)

        self._graph.add_node(nid, merged)
        return True, []

    # --- Operational Cypher subset interpreter ---------------------------

    def _legacy_execute(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Original best-effort reader: id lookup / label-param / return-all."""
        if "id" in params:
            node_id = params["id"]
            if self._graph.has_node(node_id):
                data = self._graph._get_node_properties(node_id)
                data["id"] = node_id
                return [data]
            return []

        if "label" in params:
            label = params["label"]
            results = []
            for nid in self._graph._get_all_nodes():
                data = self._graph._get_node_properties(nid)
                if data.get("label") == label:
                    entry = dict(data)
                    entry["id"] = nid
                    results.append(entry)
            return results

        results = []
        for nid in self._graph._get_all_nodes():
            data = self._graph._get_node_properties(nid)
            entry = dict(data)
            entry["id"] = nid
            results.append(entry)
        return results

    @staticmethod
    def _label_match(data: dict[str, Any], label: str) -> bool:
        # ``type`` is the system's canonical label key (the Rust node store
        # normalises node_type→type on read-back; see graph_compute
        # ``props.get("type", props.get("node_type", ...))``). Check all
        # conventions so a label filter matches regardless of writer path.
        # Compare case-insensitively: schema labels are PascalCase (``Prompt``)
        # while the node-type enum values are lowercase (``prompt``).
        target = label.lower()
        if target in (
            str(data.get("type", "")).lower(),
            str(data.get("node_type", "")).lower(),
            str(data.get("label", "")).lower(),
        ):
            return True
        labels = data.get("labels")
        return isinstance(labels, list | tuple) and any(
            str(lbl).lower() == target for lbl in labels
        )

    @staticmethod
    def _coerce_literal(raw: str, params: dict[str, Any]) -> Any:
        """Resolve a Cypher value token to a Python value."""
        raw = raw.strip()
        if raw.startswith("$"):
            return params.get(raw[1:])
        if (raw.startswith("'") and raw.endswith("'")) or (
            raw.startswith('"') and raw.endswith('"')
        ):
            return raw[1:-1]
        if raw.lower() in ("true", "false"):
            return raw.lower() == "true"
        if raw.lower() in ("null", "current_timestamp()"):
            import datetime

            if raw.lower() == "null":
                return None
            return datetime.datetime.now(datetime.UTC).isoformat()
        try:
            return int(raw)
        except ValueError:
            try:
                return float(raw)
            except ValueError:
                return raw

    def _node_value(self, nid: str, data: dict[str, Any], prop: str) -> Any:
        return nid if prop == "id" else data.get(prop)

    def _exec_node_match(
        self, q: str, params: dict[str, Any]
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Interpret a single-node MATCH. Returns (handled, rows)."""
        import re

        m = re.search(r"MATCH\s*\(\s*(\w+)\s*(?::(\w+))?\s*(\{[^}]*\})?\s*\)", q, re.I)
        if not m:
            return False, []
        var, label, inline = m.group(1), m.group(2), m.group(3)
        qu = q.upper()

        inline_conds: list[tuple[str, str, Any]] = []
        if inline:
            for pair in inline.strip("{}").split(","):
                if ":" not in pair:
                    continue
                k, v = pair.split(":", 1)
                inline_conds.append((k.strip(), "=", self._coerce_literal(v, params)))

        # WHERE is parsed into DNF — a list of AND-groups that are OR-combined (a row
        # matches if ANY group matches). An inline ``{prop:..}`` pattern ANDs into
        # every group. A genuinely unsupported shape returns None and is logged
        # loudly rather than silently returning [] — the old behaviour masqueraded
        # "I can't parse this" as "no rows", a debugging footgun. (CONCEPT:KG-2.0)
        where_groups: list[list[tuple[str, str, Any]]] = [inline_conds]
        mw = re.search(
            r"\bWHERE\b(.+?)(?:\bSET\b|\bRETURN\b|\bDETACH\b|\bDELETE\b|$)",
            q,
            re.I | re.S,
        )
        if mw:
            groups = self._parse_where_or(mw.group(1), var, params)
            if groups is None:
                logger.warning(
                    "epistemic_graph backend: unsupported WHERE shape, deferring to "
                    "legacy reader (may under-match): %s",
                    mw.group(1).strip()[:200],
                )
                return False, []  # unsupported WHERE → defer to legacy
            where_groups = [inline_conds + g for g in groups]

        # Gather matching nodes. Fast path: a single AND-group with an ``id = <value>``
        # equality resolves one node directly (O(1)) instead of scanning the whole
        # graph (O(N)). Ingestion issues many id-keyed MATCH/SET upserts, so without
        # this every write is O(N) → ingestion is O(N²) and degrades as the graph
        # grows. A disjunction (OR) can't use the fast path → full scan.
        # (CONCEPT:KG-2.8 ingestion throughput)
        matched: list[tuple[str, dict[str, Any]]] = []
        id_val = None
        if len(where_groups) == 1:
            id_val = next(
                (
                    v
                    for (p, op, v) in where_groups[0]
                    if p == "id" and op == "=" and v is not None
                ),
                None,
            )

        # Fast path: a bare aggregate count (``MATCH (n) RETURN count(n)``) with
        # no label/WHERE/SET/DELETE needs only the node *count* — never per-node
        # properties. The general scan below issues one ``_get_node_properties``
        # UDS round-trip per node, so an unfiltered count is O(N) round-trips and
        # gets slower as the graph grows (a full ``count(*)`` was taking minutes
        # on an accumulated graph). Resolve it from the id list directly.
        # (CONCEPT:KG-2.8 ingestion throughput)
        cnt_m = re.search(r"\bRETURN\b\s+count\s*\(\s*\*?\s*\w*\s*\)", q, re.I)
        if (
            id_val is None
            and not any(where_groups)
            and not label
            and cnt_m
            and "SET" not in qu
            and "DELETE" not in qu
        ):
            cnt = len(self._graph._get_all_nodes())
            alias_m = re.search(r"count\s*\([^)]*\)\s+as\s+(\w+)", q, re.I)
            alias = alias_m.group(1) if alias_m else "count"
            return True, [{alias: cnt}]

        if id_val is not None:
            if self._graph.has_node(id_val):
                data = self._graph._get_node_properties(id_val) or {}
                if (not label or self._label_match(data, label)) and self._eval_groups(
                    id_val, data, where_groups
                ):
                    matched.append((id_val, data))
        else:
            # Full-graph scan: fetch all nodes WITH their properties in a single
            # round-trip. Issuing one ``_get_node_properties`` call per node is an
            # N+1 that cost ~45s on a 40K-node graph (and held the GIL, starving
            # foreground ingestion). (CONCEPT:KG-2.8 ingestion throughput)
            for nid, data in self._graph._get_all_nodes_with_properties():
                data = data or {}
                if label and not self._label_match(data, label):
                    continue
                if not self._eval_groups(nid, data, where_groups):
                    continue
                matched.append((nid, data))

        if "DETACH DELETE" in qu or re.search(rf"\bDELETE\s+{var}\b", q, re.I):
            for nid, _ in matched:
                self._graph.remove_node(nid)
                self._embeddings.pop(nid, None)
            return True, []

        ms = re.search(r"\bSET\b(.+?)(?:\bRETURN\b|$)", q, re.I | re.S)
        if ms:
            assigns: dict[str, Any] = {}
            for frag in self._split_top_level(ms.group(1)):
                if "=" not in frag:
                    continue
                lhs, rhs = frag.split("=", 1)
                prop = lhs.strip()
                prop = prop[len(var) + 1 :] if prop.startswith(var + ".") else prop
                assigns[prop] = self._coerce_literal(rhs, params)
            for nid, data in matched:
                merged = dict(data)
                merged.update(assigns)
                self._graph.add_node(nid, merged)
                data.update(assigns)

        return True, self._project(q, var, matched, params)

    def _parse_where(
        self, text: str, var: str, params: dict[str, Any]
    ) -> list[tuple[str, str, Any]] | None:
        """Parse a conjunctive WHERE clause. None ⇒ unsupported shape."""
        import re

        conds: list[tuple[str, str, Any]] = []
        for clause in re.split(r"\bAND\b", text, flags=re.I):
            clause = clause.strip()
            if not clause:
                continue
            m_in = re.match(rf"{var}\.(\w+)\s+IN\s+\[(.*?)\]", clause, re.I | re.S)
            m_null = re.match(rf"{var}\.(\w+)\s+IS\s+(NOT\s+)?NULL", clause, re.I)
            m_has = re.match(rf"{var}\.(\w+)\s+CONTAINS\s+(.+)", clause, re.I)
            m_eq = re.match(rf"{var}\.(\w+)\s*=\s*(.+)", clause, re.I)
            if m_in:
                vals = [
                    self._coerce_literal(t, params)
                    for t in m_in.group(2).split(",")
                    if t.strip()
                ]
                conds.append((m_in.group(1), "IN", vals))
            elif m_null:
                conds.append(
                    (m_null.group(1), "NOTNULL" if m_null.group(2) else "ISNULL", None)
                )
            elif m_has:
                conds.append(
                    (
                        m_has.group(1),
                        "CONTAINS",
                        self._coerce_literal(m_has.group(2), params),
                    )
                )
            elif m_eq:
                conds.append(
                    (m_eq.group(1), "=", self._coerce_literal(m_eq.group(2), params))
                )
            else:
                return None
        return conds

    def _parse_where_or(
        self, text: str, var: str, params: dict[str, Any]
    ) -> list[list[tuple[str, str, Any]]] | None:
        """Parse a WHERE clause into DNF — OR of AND-groups.

        Splits on top-level ``OR`` and parses each disjunct with the conjunctive
        ``_parse_where``. Returns a list of AND-groups (a row matches if ANY group
        matches), or ``None`` if any disjunct is an unsupported shape. With no
        top-level ``OR`` this returns a single group, identical to the prior
        AND-only behaviour.
        """
        groups: list[list[tuple[str, str, Any]]] = []
        for part in self._split_or(text):
            parsed = self._parse_where(part, var, params)
            if parsed is None:
                return None
            groups.append(parsed)
        return groups or None

    @staticmethod
    def _split_or(text: str) -> list[str]:
        """Split on top-level ``OR`` (case-insensitive), respecting (), [], {}, quotes."""
        import re

        out: list[str] = []
        buf: list[str] = []
        depth = 0
        quote: str | None = None
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if quote:
                buf.append(ch)
                if ch == quote:
                    quote = None
                i += 1
            elif ch in "'\"":
                quote = ch
                buf.append(ch)
                i += 1
            elif ch in "([{":
                depth += 1
                buf.append(ch)
                i += 1
            elif ch in ")]}":
                depth -= 1
                buf.append(ch)
                i += 1
            elif depth == 0 and (m := re.match(r"\s+OR\s+", text[i:], re.I)):
                out.append("".join(buf))
                buf = []
                i += m.end()
            else:
                buf.append(ch)
                i += 1
        if buf:
            out.append("".join(buf))
        return [s for s in out if s.strip()]

    def _eval_groups(
        self,
        nid: str,
        data: dict[str, Any],
        groups: list[list[tuple[str, str, Any]]],
    ) -> bool:
        """DNF evaluation: the row matches if ANY AND-group matches."""
        return any(self._eval_conds(nid, data, g) for g in groups)

    def _eval_conds(
        self, nid: str, data: dict[str, Any], conds: list[tuple[str, str, Any]]
    ) -> bool:
        for prop, op, val in conds:
            actual = self._node_value(nid, data, prop)
            if op == "=":
                if actual != val:
                    return False
            elif op == "IN":
                if actual not in val:
                    return False
            elif op == "CONTAINS":
                if actual is None or val is None or str(val) not in str(actual):
                    return False
            elif op == "ISNULL":
                if actual is not None:
                    return False
            elif op == "NOTNULL":
                if actual is None:
                    return False
        return True

    @staticmethod
    def _split_top_level(text: str) -> list[str]:
        """Split on commas not nested inside brackets/quotes."""
        out, buf, depth, quote = [], [], 0, None
        for ch in text:
            if quote:
                buf.append(ch)
                if ch == quote:
                    quote = None
            elif ch in "'\"":
                quote = ch
                buf.append(ch)
            elif ch in "[{(":
                depth += 1
                buf.append(ch)
            elif ch in "]})":
                depth -= 1
                buf.append(ch)
            elif ch == "," and depth == 0:
                out.append("".join(buf))
                buf = []
            else:
                buf.append(ch)
        if buf:
            out.append("".join(buf))
        return [s for s in out if s.strip()]

    def _project(
        self,
        q: str,
        var: str,
        matched: list[tuple[str, dict[str, Any]]],
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        import re

        m_ret = re.search(r"\bRETURN\b(.+?)$", q, re.I | re.S)
        if not m_ret:
            return []
        ret = m_ret.group(1)

        limit = None
        m_lim = re.search(r"\bLIMIT\s+(\$?\w+)", ret, re.I)
        if m_lim:
            tok = m_lim.group(1)
            limit = params.get(tok[1:]) if tok.startswith("$") else int(tok)
            ret = ret[: m_lim.start()]
        ret = re.sub(r"\bORDER\s+BY\b.*$", "", ret, flags=re.I | re.S)

        items = self._split_top_level(ret)

        # Aggregate: count(var)
        for it in items:
            mc = re.match(r"\s*count\(\s*\*?\s*\w*\s*\)\s*(?:as\s+(\w+))?", it, re.I)
            if mc:
                alias = mc.group(1) or "count"
                return [{alias: len(matched)}]

        rows: list[dict[str, Any]] = []
        for nid, data in matched:
            row: dict[str, Any] = {}
            for it in items:
                it = it.strip()
                m_alias = re.match(rf"{var}\.(\w+)\s+as\s+(\w+)", it, re.I) or re.match(
                    rf"{var}\.(\w+)", it, re.I
                )
                if m_alias and "." in it:
                    prop = m_alias.group(1)
                    value = self._node_value(nid, data, prop)
                    has_explicit_alias = m_alias.lastindex and m_alias.lastindex >= 2
                    if has_explicit_alias:
                        row[m_alias.group(2)] = value
                    else:
                        # No explicit ``AS`` alias: expose both the standard
                        # Cypher column name (``var.prop``) and the bare prop
                        # name so callers using either convention resolve it.
                        row[f"{var}.{prop}"] = value
                        row.setdefault(prop, value)
                elif it == var or re.match(rf"{var}\s+as\s+\w+", it, re.I):
                    # Bare ``RETURN v`` → a single column named ``v`` holding the
                    # full node dict (Cypher semantics; callers read ``res["v"]``).
                    full = dict(data)
                    full["id"] = nid
                    alias_m = re.match(rf"{var}\s+as\s+(\w+)", it, re.I)
                    row[alias_m.group(1) if alias_m else var] = full
                else:
                    row[it] = None
            rows.append(row)

        if limit is not None:
            rows = rows[:limit]
        return rows

    def execute_batch(
        self, query: str, batch: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Execute batch operations."""
        results = []
        for params in batch:
            results.extend(self.execute(query, params))
        return results

    def create_schema(self) -> None:
        """Initialize schema metadata representation."""
        # Simple schema metadata tracking for in-memory backend
        self._schema_created = True

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Store an embedding vector for a node."""
        self._embeddings[node_id] = embedding

    def semantic_search(
        self, query_embedding: list[float], n_results: int = 5
    ) -> list[dict[str, Any]]:
        """Cosine similarity search over stored embeddings."""
        if not self._embeddings:
            return []

        import numpy as np

        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        scores: list[tuple[str, float]] = []
        for node_id, emb in self._embeddings.items():
            emb_vec = np.array(emb)
            emb_norm = np.linalg.norm(emb_vec)
            if emb_norm == 0:
                continue
            similarity = float(np.dot(query_vec, emb_vec) / (query_norm * emb_norm))
            scores.append((node_id, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for node_id, score in scores[:n_results]:
            if self._graph.has_node(node_id):
                data = self._graph._get_node_properties(node_id)
                data["id"] = node_id
                data["_similarity"] = score
                results.append(data)

        return results

    def prune(self, criteria: dict[str, Any]) -> None:
        """Prune nodes matching criteria."""
        to_remove = []
        for nid in self._graph._get_all_nodes():
            data = self._graph._get_node_properties(nid)
            match = all(data.get(k) == v for k, v in criteria.items())
            if match:
                to_remove.append(nid)

        for nid in to_remove:
            self._graph.remove_node(nid)
            self._embeddings.pop(nid, None)

        logger.info("Pruned %d nodes", len(to_remove))

    def close(self) -> None:
        """Reset the in-memory graph."""
        from ..core.graph_compute import GraphComputeEngine

        self._graph = GraphComputeEngine(backend_type="rust")
        self._embeddings.clear()

    # --- Extended API ---

    def health_check(self) -> bool:
        """Always healthy for in-memory backend."""
        return True

    def get_stats(self) -> dict[str, Any]:
        """Return graph statistics."""
        return {
            "backend": "memory",
            "nodes": self._graph.node_count(),
            "edges": self._graph.edge_count(),
            "embeddings": len(self._embeddings),
        }

    # --- Node/Edge Operations ---

    def add_node(self, node_id: str, label: str = "", **properties: Any) -> None:
        """Add a node to the graph."""
        props = {"label": label, **properties}
        self._graph.add_node(node_id, props)
        self._node_counter += 1

    def add_edge(
        self,
        source: str,
        target: str,
        rel_type: str = "",
        **properties: Any,
    ) -> None:
        """Add an edge between two nodes."""
        props = {"rel_type": rel_type, **properties}
        self._graph.add_edge(source, target, props)

    def get_node_properties(self, node_id: str) -> dict[str, Any] | None:
        """Return a node's current properties, or ``None`` if it doesn't exist.

        Cheap in-memory read used by the Company Brain write-path guard for
        field-level survivorship (CONCEPT:KG-2.6). Returns ``{}`` for a node that
        exists with no properties; ``None`` lets the guard fall back to
        node-level arbitration.
        """
        try:
            if not self._graph.has_node(node_id):
                return None
            props = self._graph._get_node_properties(node_id)
            return dict(props) if isinstance(props, dict) else {}
        except Exception:  # pragma: no cover - read best-effort
            return None

    # --- Persistence ---

    def save_to_json(self, path: str) -> None:
        """Serialize the graph to a JSON file."""
        graph_json = self._graph.to_json()
        data = json.loads(graph_json)
        data["_embeddings"] = {k: v for k, v in self._embeddings.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Graph saved to %s", path)

    def load_from_json(self, path: str) -> None:
        """Deserialize the graph from a JSON file."""
        with open(path) as f:
            data = json.load(f)

        embeddings = data.pop("_embeddings", {})
        self._graph.from_json(json.dumps(data))
        self._embeddings = embeddings
        logger.info(
            "Graph loaded from %s (%d nodes, %d edges)",
            path,
            self._graph.node_count(),
            self._graph.edge_count(),
        )
