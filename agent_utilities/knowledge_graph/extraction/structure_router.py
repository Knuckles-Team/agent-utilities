"""Structure router — prose vs. structured text classification (CONCEPT:KG-2.74).

The universal ingestion funnel (reader → **structure-router** → {open | schema}
extraction → ontology grounding → background closure) needs to know, *before* it
spends an LLM call, whether a blob of text is free-flowing prose (best mined by
the open ``fact_extractor`` triple LLM), a structured record (best mapped by
schema/field rules into ``key/value`` pairs), or a mix of both (a form header
over a prose body).

This module answers that with **cheap, dependency-free heuristics** — delimiter
density, ``key: value`` line shape, table/CSV columns, JSON structure, and known
form/record markers (invoices, bills, Jira/ticket fields) — so the router itself
adds no model cost and no optional dependency. It is the gate that lets
``IngestionEngine._enrich_text`` pick the *right* extractor instead of always
firing the prose-tuned fact LLM at a CSV.

Public surface:

* :func:`classify_text` — ``text -> "prose" | "structured" | "mixed"``.
* :func:`route_for_extraction` — ``text -> {mode, prose_text, records}`` where
  ``records`` is a ``list[dict]`` of parsed key/value records for schema mapping
  and ``prose_text`` is the free text handed to open extraction. ``mixed`` splits
  the leading record header from the trailing prose body.

Provenance: the classification approach (delimiter/key-value/table cues) is a
distillation of the funnel design in this repo's ingestion docs; the parsing is
all stdlib (``csv``, ``json``, ``re``). Best-effort and never-raise so it is safe
to call inline on any ingest path — a parse failure degrades to ``prose``.
"""

from __future__ import annotations

import csv
import io
import json
import re
from typing import Any

# --------------------------------------------------------------------------- #
# Tunables — one correct value each (config discipline: module constants, not
# env flags). These are classification thresholds, not deployment-varying knobs.
# --------------------------------------------------------------------------- #

# A line is "key: value" / "key = value" shaped if it has a short-ish key on the
# left of the first delimiter and some value on the right.
_KV_LINE = re.compile(
    r"""^\s*
    (?P<key>[A-Za-z0-9][\w .,/&()#'-]{0,59}?)   # a label, not a full sentence
    \s*[:=]\s+
    (?P<val>\S.*?)\s*$
    """,
    re.VERBOSE,
)

# Form/record marker words: their presence as a field label is a strong "this is
# a record" signal (invoices, bills, tickets, shipping, HR forms, …). Matched
# case-insensitively against the *key* side of a ``key: value`` line.
_RECORD_MARKERS = frozenset(
    {
        "invoice",
        "invoice number",
        "invoice no",
        "invoice date",
        "bill to",
        "bill to:",
        "ship to",
        "sold to",
        "po number",
        "purchase order",
        "order number",
        "order id",
        "account number",
        "account no",
        "amount due",
        "balance due",
        "total due",
        "subtotal",
        "tax",
        "due date",
        "issue date",
        "statement date",
        "customer id",
        "customer name",
        "vendor",
        "supplier",
        # Ticketing / issue-tracker field labels (Jira/ServiceNow/GitHub).
        "issue key",
        "issue type",
        "ticket",
        "ticket id",
        "ticket number",
        "priority",
        "assignee",
        "reporter",
        "status",
        "resolution",
        "sprint",
        "epic",
        "story points",
        "labels",
        "component",
        "severity",
        # Contact / shipping / HR record fields.
        "first name",
        "last name",
        "full name",
        "email",
        "phone",
        "address",
        "date of birth",
        "ssn",
        "employee id",
        "department",
        "tracking number",
    }
)

# Delimiters whose high per-line density signals tabular/structured content.
_COLUMN_DELIMS = ("\t", "|", ",", ";")

# How many leading lines a "mixed" header may span. A form/record header sits at
# the top; once prose begins it tends to continue, so we only scan the head.
_MIXED_HEADER_SCAN = 60

# Minimum populated rows for a CSV/TSV body to count as a real table (one header
# + at least two data rows — a lone "a, b" line is prose with commas).
_MIN_TABLE_ROWS = 3

# A record-ish run this long (in key:value lines) flips a doc to structured even
# without explicit markers — e.g. a config dump or a properties file.
_KV_RUN_STRUCTURED = 5


# --------------------------------------------------------------------------- #
# Cheap cue detectors
# --------------------------------------------------------------------------- #


def _looks_like_json(text: str) -> bool:
    """True if the whole text parses as a JSON object/array (a structured record)."""
    s = text.strip()
    if not s or s[0] not in "{[":
        return False
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return False
    return isinstance(obj, (dict, list))


def _kv_match(line: str) -> tuple[str, str] | None:
    """Return ``(key, value)`` if ``line`` is a ``key: value`` / ``key = value``
    record line, else ``None``. Rejects lines whose "key" reads like a sentence
    (too many words / ends with sentence punctuation before the delimiter)."""
    m = _KV_LINE.match(line)
    if not m:
        return None
    key = m.group("key").strip()
    val = m.group("val").strip()
    if not key or not val:
        return None
    # A real field label is short and not a full clause. A colon inside prose
    # ("As Plato said: ...") has a long, many-word left side — reject those.
    if len(key.split()) > 6:
        return None
    # A URL/time ("http://x", "12:30") trips the colon but isn't a field line.
    if re.match(r"^[a-z][a-z0-9+.-]*$", key) and val.startswith("//"):
        return None
    # A field *value* is short and atomic; a prose value is a full sentence.
    # Reject "There is one rule: never stop learning. Everything else follows" —
    # a multi-sentence, many-word right side is prose, not a record value.
    if _value_is_sentence(val):
        return None
    return key, val


def _value_is_sentence(val: str) -> bool:
    """True if a ``key: value`` right side reads as prose, not an atomic field.

    Heuristic: a long value (>=8 words) that also carries internal sentence
    punctuation (a period/!/? followed by a space, i.e. more than one clause) is
    prose. Short values, even with a trailing period, stay atomic fields.
    """
    words = val.split()
    if len(words) < 8:
        return False
    return bool(re.search(r"[.!?]\s+\S", val))


def _is_record_marker(key: str) -> bool:
    """True if a field key is a known invoice/bill/ticket/form record marker."""
    k = key.strip().lower().rstrip(":")
    if k in _RECORD_MARKERS:
        return True
    # Allow a marker to appear as a prefix token-run ("Invoice Number #" etc.).
    return any(k.startswith(mk) for mk in _RECORD_MARKERS if len(mk) >= 5)


def _column_table(lines: list[str]) -> tuple[str, list[list[str]]] | None:
    """Detect a delimited table among ``lines``.

    Returns ``(delimiter, rows)`` where the chosen delimiter yields a consistent
    multi-column shape over enough rows, else ``None``. Markdown table pipes and
    CSV/TSV are all handled by counting per-line delimiter occurrences and taking
    the delimiter whose column count is stable across the most lines.
    """
    nonempty = [ln for ln in lines if ln.strip()]
    if len(nonempty) < _MIN_TABLE_ROWS:
        return None
    best: tuple[str, list[list[str]]] | None = None
    best_score = 0
    for delim in _COLUMN_DELIMS:
        counts = [ln.count(delim) for ln in nonempty]
        # Need the delimiter to appear on most lines, more than once per line.
        rows_with = [c for c in counts if c >= 1]
        if len(rows_with) < max(_MIN_TABLE_ROWS, int(0.7 * len(nonempty))):
            continue
        # Column count should be roughly constant (a real table), not ragged.
        col_mode = max(set(counts), key=counts.count)
        if col_mode < 1:
            continue
        consistent = sum(1 for c in counts if abs(c - col_mode) <= 1)
        score = consistent * (col_mode + 1)
        if score > best_score:
            parsed = _parse_delimited(nonempty, delim)
            if parsed and len(parsed) >= _MIN_TABLE_ROWS:
                best = (delim, parsed)
                best_score = score
    return best


def _parse_delimited(lines: list[str], delim: str) -> list[list[str]]:
    """Parse delimited lines into rows. ``|`` uses a markdown-aware split
    (strips leading/trailing pipes + separator rows); ``,``/``;``/``\\t`` go
    through ``csv`` so quoting/escaping is handled correctly."""
    if delim == "|":
        rows: list[list[str]] = []
        for ln in lines:
            stripped = ln.strip().strip("|")
            cells = [c.strip() for c in stripped.split("|")]
            # Skip markdown separator rows (---|:--:|---).
            if cells and all(set(c) <= set("-: ") and c for c in cells):
                continue
            rows.append(cells)
        return rows
    try:
        reader = csv.reader(io.StringIO("\n".join(lines)), delimiter=delim)
        return [list(r) for r in reader if any(c.strip() for c in r)]
    except csv.Error:
        return []


def _table_to_records(rows: list[list[str]]) -> list[dict[str, Any]]:
    """Turn header+data rows into ``list[dict]`` keyed by the header cells."""
    if len(rows) < 2:
        return []
    header = [h.strip() or f"col_{i}" for i, h in enumerate(rows[0])]
    records: list[dict[str, Any]] = []
    for row in rows[1:]:
        if not any(c.strip() for c in row):
            continue
        rec: dict[str, Any] = {}
        for i, cell in enumerate(row):
            key = header[i] if i < len(header) else f"col_{i}"
            rec[key] = cell.strip()
        records.append(rec)
    return records


def _json_to_records(text: str) -> list[dict[str, Any]]:
    """Flatten parsed JSON into record dicts (single object → one record; array
    of objects → many; scalars/array-of-scalars → a single ``{"value": ...}``)."""
    try:
        obj = json.loads(text.strip())
    except (ValueError, TypeError):
        return []
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list):
        out = [it for it in obj if isinstance(it, dict)]
        if out:
            return out
        return [{"value": it} for it in obj]
    return [{"value": obj}]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def classify_text(text: str, doc_type: str = "") -> str:
    """Classify ``text`` as ``"prose"``, ``"structured"``, or ``"mixed"``.

    Cheap heuristics only (no LLM, no optional deps):

    * **JSON** (whole text parses as object/array) → ``structured``.
    * **A delimited table** (consistent CSV/TSV/markdown columns over enough
      rows) → ``structured`` (or ``mixed`` if prose precedes/follows it).
    * **Dense ``key: value`` record lines** — many of them, or any known
      invoice/bill/ticket/form marker key → ``structured``; a record *header*
      over a prose body → ``mixed``.
    * Otherwise → ``prose``.

    ``doc_type`` is an optional hint from :func:`detect_doc_type`
    (``email``/``paper``/…): a ``paper``/``book`` biases toward ``prose``, but
    the text cues win when they are unambiguous. (Email headers are detected by
    the same leading-header + prose-body split that produces ``mixed``.)
    """
    if not text or not text.strip():
        return "prose"

    if _looks_like_json(text):
        return "structured"

    lines = text.splitlines()
    nonempty = [ln for ln in lines if ln.strip()]
    if not nonempty:
        return "prose"

    # --- key/value record signal -----------------------------------------
    kv_total = 0
    marker_hit = False
    header_kv = 0  # kv lines within the leading header window
    for idx, ln in enumerate(nonempty):
        kv = _kv_match(ln)
        if kv is None:
            continue
        kv_total += 1
        if idx < _MIXED_HEADER_SCAN:
            header_kv += 1
        if _is_record_marker(kv[0]):
            marker_hit = True

    kv_ratio = kv_total / len(nonempty)

    # --- table signal -----------------------------------------------------
    table = _column_table(nonempty)

    # A leading contiguous KV header (form/email header block) + a real prose
    # body after it ⇒ mixed (split + open-extract the body). This is the key
    # discriminator between a pure record and a form-with-notes / email.
    has_header = header_kv >= 2 and _leading_kv_header(nonempty) >= 2
    has_body = _has_prose_body(nonempty)

    dt = (doc_type or "").strip().lower()
    if dt in ("paper", "book") and not marker_hit and kv_ratio < 0.3 and table is None:
        return "prose"

    if has_header and has_body:
        return "mixed"

    # Strong structured signals (no prose body to split off).
    if marker_hit and kv_total >= 2:
        return "structured"
    if kv_ratio >= 0.6 or (kv_total >= _KV_RUN_STRUCTURED and kv_ratio >= 0.4):
        return "structured"

    if table is not None:
        _delim, rows = table
        if len(rows) >= 0.7 * len(nonempty):
            return "structured"
        return "mixed"

    # A small record header over a body that didn't trip the strict prose test.
    if has_header and header_kv >= 3 and kv_ratio < 0.5:
        return "mixed"

    return "prose"


def _leading_kv_header(nonempty: list[str]) -> int:
    """Count the contiguous run of ``key: value`` lines at the very top.

    A form/email header is a *block* at the start; this rejects a doc whose
    key:value lines are scattered through prose (those aren't a header to split).
    """
    run = 0
    for ln in nonempty:
        if _kv_match(ln) is not None:
            run += 1
        else:
            break
    return run


def _has_prose_body(nonempty: list[str]) -> bool:
    """True if there is a meaningful prose body beyond the record lines.

    A prose body is non-key/value text with real sentence content — measured by
    word mass and sentence-shaped lines, so a single long paragraph (one line of
    many words) counts just as much as several shorter sentence lines.
    """
    prose_lines = [ln for ln in nonempty if _kv_match(ln) is None]
    if not prose_lines:
        return False
    prose_words = sum(len(ln.split()) for ln in prose_lines)
    # A sentence-shaped line: enough words and at least one sentence terminator.
    sentence_like = sum(
        1 for ln in prose_lines if len(ln.split()) >= 8 and re.search(r"[.!?]", ln)
    )
    return prose_words >= 20 and sentence_like >= 1


def route_for_extraction(text: str, doc_type: str = "") -> dict[str, Any]:
    """Route ``text`` to the right extractor, returning a structured plan.

    Returns a dict::

        {
          "mode": "prose" | "structured" | "mixed",
          "prose_text": str,              # free text for OPEN extraction
          "records": list[dict],          # key/value records for SCHEMA mapping
        }

    Contract by mode:

    * ``prose`` — ``prose_text`` is the whole input, ``records`` empty. The
      caller runs the open ``fact_extractor`` triple LLM.
    * ``structured`` — ``records`` holds the parsed key/value record(s) (JSON
      object(s), CSV/TSV/markdown table rows, or ``key: value`` lines),
      ``prose_text`` empty. The caller maps fields to ontology properties — no
      prose LLM call needed.
    * ``mixed`` — the leading record header is parsed into ``records`` and the
      remaining prose body is returned as ``prose_text``; the caller does BOTH
      (schema-map the header, open-extract the body).

    Never raises: any parse failure degrades to ``prose`` with the original text.
    """
    plan: dict[str, Any] = {"mode": "prose", "prose_text": text or "", "records": []}
    if not text or not text.strip():
        return plan

    try:
        mode = classify_text(text, doc_type)
    except Exception:  # noqa: BLE001 — classification never breaks ingest
        return plan

    if mode == "prose":
        return plan

    if mode == "structured":
        plan["mode"] = "structured"
        plan["prose_text"] = ""
        plan["records"] = _parse_structured(text)
        # If we somehow parsed nothing, fall back to prose so no text is lost.
        if not plan["records"]:
            plan["mode"] = "prose"
            plan["prose_text"] = text
        return plan

    # mixed: split a leading record header from the trailing prose body.
    header_records, body = _split_header_body(text, doc_type)
    if not header_records:
        # No clean header → treat as prose so the body is still mined.
        plan["mode"] = "prose"
        plan["prose_text"] = text
        return plan
    plan["mode"] = "mixed"
    plan["records"] = header_records
    plan["prose_text"] = body
    return plan


def _parse_structured(text: str) -> list[dict[str, Any]]:
    """Parse a fully-structured blob into records (JSON → table → key:value)."""
    if _looks_like_json(text):
        return _json_to_records(text)

    nonempty = [ln for ln in text.splitlines() if ln.strip()]
    table = _column_table(nonempty)
    if table is not None:
        _delim, rows = table
        recs = _table_to_records(rows)
        if recs:
            return recs

    # key:value lines → one flat record (last value wins on duplicate keys, but
    # repeated keys are collected into a list so nothing is dropped).
    record: dict[str, Any] = {}
    for ln in nonempty:
        kv = _kv_match(ln)
        if kv is None:
            continue
        key, val = kv
        key = key.strip().rstrip(":")
        if key in record:
            existing = record[key]
            if isinstance(existing, list):
                existing.append(val)
            else:
                record[key] = [existing, val]
        else:
            record[key] = val
    return [record] if record else []


def _split_header_body(
    text: str, doc_type: str = ""
) -> tuple[list[dict[str, Any]], str]:
    """Split ``text`` into (leading key/value header records, prose body).

    Walks the leading lines collecting contiguous ``key: value`` record lines
    (tolerating blank separators) until the prose body begins, then returns the
    header as one record dict and everything after as the prose body.
    """
    lines = text.splitlines()
    header: dict[str, Any] = {}
    body_start = 0
    seen_kv = False
    blanks_after_kv = 0
    for i, ln in enumerate(lines[:_MIXED_HEADER_SCAN]):
        if not ln.strip():
            # A blank line right after the header block ends it.
            if seen_kv:
                blanks_after_kv += 1
                if blanks_after_kv >= 1:
                    body_start = i + 1
                    break
            continue
        kv = _kv_match(ln)
        if kv is not None:
            key, val = kv
            key = key.strip().rstrip(":")
            header[key] = val
            seen_kv = True
            body_start = i + 1
            blanks_after_kv = 0
        else:
            # First non-kv, non-blank line after the header → prose body begins.
            if seen_kv:
                body_start = i
                break
            # No header yet and this is prose → no header at all.
            return [], text
    body = "\n".join(lines[body_start:]).strip()
    records = [header] if header else []
    return records, body
