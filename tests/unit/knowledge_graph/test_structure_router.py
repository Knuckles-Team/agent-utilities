"""Tests for the structure router (CONCEPT:AU-KG.backend.mirror-health-repair).

Covers ``classify_text`` (prose/structured/mixed) and ``route_for_extraction``
(records vs. prose split) across JSON, CSV/TSV, markdown tables, key:value
config dumps, invoice/bill/ticket forms, emails, and pure prose — all on the
stdlib-only fast path with no LLM and no optional deps.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.extraction.structure_router import (
    classify_text,
    route_for_extraction,
)

# --------------------------------------------------------------------------- #
# Fixtures of representative inputs
# --------------------------------------------------------------------------- #

PROSE = (
    "The Jina AI team released jina-embeddings-v3 in 2024. The model was trained "
    "on a large multilingual corpus and outperforms earlier baselines on the MTEB "
    "benchmark. As Plato said: the unexamined life is not worth living, and the "
    "authors echo that sentiment in their conclusion about evaluation rigor."
)

JSON_OBJ = '{"invoice_number": "INV-1001", "amount_due": "420.00", "currency": "USD"}'

JSON_ARRAY = '[{"name": "alice", "role": "admin"}, {"name": "bob", "role": "viewer"}]'

CSV_TABLE = (
    "name,role,team\nalice,admin,platform\nbob,viewer,growth\ncarol,editor,docs\n"
)

TSV_TABLE = "id\tstatus\tpriority\n1\topen\thigh\n2\tclosed\tlow\n3\topen\tmed\n"

MARKDOWN_TABLE = (
    "| Symbol | Price | Change |\n"
    "|--------|-------|--------|\n"
    "| AAPL   | 187.4 | +1.2   |\n"
    "| MSFT   | 412.1 | -0.4   |\n"
    "| NVDA   | 905.0 | +3.7   |\n"
)

KV_CONFIG = (
    "host: db.internal\n"
    "port: 5432\n"
    "user: app\n"
    "pool_size: 20\n"
    "ssl_mode: require\n"
    "timeout: 30\n"
)

INVOICE_FORM = (
    "Invoice Number: INV-2048\n"
    "Invoice Date: 2026-06-01\n"
    "Bill To: Acme Corp\n"
    "Amount Due: 1,250.00\n"
    "Due Date: 2026-07-01\n"
)

JIRA_TICKET = (
    "Issue Key: PROJ-512\n"
    "Issue Type: Bug\n"
    "Priority: High\n"
    "Assignee: jdoe\n"
    "Status: In Progress\n"
)

INVOICE_WITH_BODY = (
    "Invoice Number: INV-2048\n"
    "Bill To: Acme Corp\n"
    "Amount Due: 1,250.00\n"
    "\n"
    "Thank you for your continued business this quarter. The attached charges "
    "cover the migration engagement completed in May, including the data backfill "
    "and the cutover support window that ran over the holiday weekend. Please remit "
    "payment within thirty days to avoid a late fee on the outstanding balance.\n"
)

EMAIL = (
    "From: alice@example.com\n"
    "To: bob@example.com\n"
    "Subject: Q2 planning\n"
    "Date: 2026-06-01\n"
    "\n"
    "Hi Bob, I wanted to follow up on the roadmap discussion from last week. Could "
    "you put together the capacity estimates for the platform team before Friday so "
    "we can finalize the staffing plan and share it with leadership?\n"
)


# --------------------------------------------------------------------------- #
# classify_text
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text,expected",
    [
        (PROSE, "prose"),
        (JSON_OBJ, "structured"),
        (JSON_ARRAY, "structured"),
        (CSV_TABLE, "structured"),
        (TSV_TABLE, "structured"),
        (MARKDOWN_TABLE, "structured"),
        (KV_CONFIG, "structured"),
        (INVOICE_FORM, "structured"),
        (JIRA_TICKET, "structured"),
        (INVOICE_WITH_BODY, "mixed"),
        ("", "prose"),
        ("   \n\n  ", "prose"),
    ],
)
def test_classify_text(text: str, expected: str) -> None:
    assert classify_text(text) == expected


def test_email_doc_type_is_mixed() -> None:
    # An email body with a From/To/Subject header → header records + prose body.
    assert classify_text(EMAIL, doc_type="email") == "mixed"


def test_paper_hint_biases_prose() -> None:
    # A colon in prose ("As Plato said:") must not flip a paper to structured.
    assert classify_text(PROSE, doc_type="paper") == "prose"


def test_prose_with_one_colon_stays_prose() -> None:
    text = "There is one rule: never stop learning. Everything else follows from it."
    assert classify_text(text) == "prose"


# --------------------------------------------------------------------------- #
# route_for_extraction — structured
# --------------------------------------------------------------------------- #


def test_route_json_object_to_records() -> None:
    plan = route_for_extraction(JSON_OBJ)
    assert plan["mode"] == "structured"
    assert plan["prose_text"] == ""
    assert plan["records"] == [
        {"invoice_number": "INV-1001", "amount_due": "420.00", "currency": "USD"}
    ]


def test_route_json_array_to_records() -> None:
    plan = route_for_extraction(JSON_ARRAY)
    assert plan["mode"] == "structured"
    assert plan["records"] == [
        {"name": "alice", "role": "admin"},
        {"name": "bob", "role": "viewer"},
    ]


def test_route_csv_to_records() -> None:
    plan = route_for_extraction(CSV_TABLE)
    assert plan["mode"] == "structured"
    assert len(plan["records"]) == 3
    assert plan["records"][0] == {"name": "alice", "role": "admin", "team": "platform"}
    assert plan["records"][-1]["name"] == "carol"


def test_route_tsv_to_records() -> None:
    plan = route_for_extraction(TSV_TABLE)
    assert plan["mode"] == "structured"
    assert len(plan["records"]) == 3
    assert plan["records"][0] == {"id": "1", "status": "open", "priority": "high"}


def test_route_markdown_table_to_records() -> None:
    plan = route_for_extraction(MARKDOWN_TABLE)
    assert plan["mode"] == "structured"
    assert len(plan["records"]) == 3
    assert plan["records"][0]["Symbol"] == "AAPL"
    # The markdown separator row must be dropped, not parsed as data.
    assert all("---" not in "".join(r.values()) for r in plan["records"])


def test_route_kv_config_to_single_record() -> None:
    plan = route_for_extraction(KV_CONFIG)
    assert plan["mode"] == "structured"
    assert plan["records"] == [
        {
            "host": "db.internal",
            "port": "5432",
            "user": "app",
            "pool_size": "20",
            "ssl_mode": "require",
            "timeout": "30",
        }
    ]


def test_route_invoice_form_to_record() -> None:
    plan = route_for_extraction(INVOICE_FORM)
    assert plan["mode"] == "structured"
    assert len(plan["records"]) == 1
    rec = plan["records"][0]
    assert rec["Invoice Number"] == "INV-2048"
    assert rec["Amount Due"] == "1,250.00"


def test_route_jira_ticket_to_record() -> None:
    plan = route_for_extraction(JIRA_TICKET)
    assert plan["mode"] == "structured"
    rec = plan["records"][0]
    assert rec["Issue Key"] == "PROJ-512"
    assert rec["Status"] == "In Progress"


# --------------------------------------------------------------------------- #
# route_for_extraction — prose & mixed
# --------------------------------------------------------------------------- #


def test_route_prose_returns_whole_text() -> None:
    plan = route_for_extraction(PROSE)
    assert plan["mode"] == "prose"
    assert plan["prose_text"] == PROSE
    assert plan["records"] == []


def test_route_mixed_splits_header_and_body() -> None:
    plan = route_for_extraction(INVOICE_WITH_BODY)
    assert plan["mode"] == "mixed"
    # Header parsed into records...
    assert plan["records"]
    assert plan["records"][0]["Invoice Number"] == "INV-2048"
    # ...and the prose body is handed to open extraction, header stripped out.
    assert "Thank you for your continued business" in plan["prose_text"]
    assert "Invoice Number" not in plan["prose_text"]


def test_route_email_mixed_split() -> None:
    plan = route_for_extraction(EMAIL, doc_type="email")
    assert plan["mode"] == "mixed"
    assert plan["records"][0]["From"] == "alice@example.com"
    assert plan["records"][0]["Subject"] == "Q2 planning"
    assert "follow up on the roadmap" in plan["prose_text"]
    assert "From:" not in plan["prose_text"]


def test_empty_text_is_prose_noop() -> None:
    plan = route_for_extraction("")
    assert plan == {"mode": "prose", "prose_text": "", "records": []}


def test_never_loses_text_on_structured_parse_miss() -> None:
    # A blob must always surface either prose_text or records, never both empty.
    plan = route_for_extraction(PROSE)
    assert plan["prose_text"] or plan["records"]


def test_route_is_best_effort_on_garbage() -> None:
    # Binary-ish / control-char soup must not raise; degrades safely.
    junk = "\x00\x01 broken : not really \x07 a field \x00"
    plan = route_for_extraction(junk)
    assert plan["mode"] in {"prose", "structured", "mixed"}
    assert isinstance(plan["records"], list)
