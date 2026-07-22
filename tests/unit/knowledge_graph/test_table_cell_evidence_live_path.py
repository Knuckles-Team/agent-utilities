"""Live-path proof for the ``TableCellRange`` evidence-locus producer
(CONCEPT:AU-KG.identity.evidence-spine-convergence, Evidence seam completion).

``docs/architecture/evidence_spine_convergence.md`` named ``readers_office.py``'s
``read_xlsx`` as a candidate producer: it already iterates each worksheet's rows
in order (an implicit row index) but flattens them straight to tab-joined text,
discarding the extent. This test proves the gap is closed: the EXISTING xlsx
reader entry point (``read_xlsx``, reached from every ``.xlsx`` ingested through
``read_media``/``read_any``) now also writes ONE ``TableCellRange`` evidence
locus per worksheet covering its real used range, as a side effect.

Two layers: :func:`test_persist_table_cell_evidence_writes_the_full_used_range`
exercises the wiring function directly with synthetic rows (dependency-free, so
it runs everywhere); :func:`test_read_xlsx_writes_table_cell_evidence` goes
through the REAL ``read_xlsx`` + a real ``openpyxl`` workbook end-to-end
(skipped when ``openpyxl`` isn't installed, matching this test module's own
existing ``test_read_xlsx_extracts_sheet_rows`` convention).
"""

from __future__ import annotations

import importlib
import sys

import pytest

from agent_utilities.knowledge_graph.extraction import readers_office as ro
from agent_utilities.knowledge_graph.memory import native_ingest


class _FakeStore:
    def __init__(self) -> None:
        self.cell_calls: list[tuple[bytes, dict]] = []

    def store_table_cell_evidence(self, data: bytes, **kwargs):
        self.cell_calls.append((data, kwargs))
        return object()


def test_persist_table_cell_evidence_writes_the_full_used_range(monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr(native_ingest, "media_store", lambda: store)

    rows = [["Q", "Amount"], ["Q1", 100], ["Q2", 150]]
    ro._persist_table_cell_evidence("/data/book.xlsx", "Revenue", rows)

    assert len(store.cell_calls) == 1
    data, kw = store.cell_calls[0]
    assert data == ro._format_rows(rows).encode("utf-8")
    assert kw["table_id"] == "/data/book.xlsx#Revenue"
    assert kw["row_start"] == 0
    assert kw["row_end"] == 2  # 3 rows, 0-indexed inclusive
    assert kw["col_start"] == 0
    assert kw["col_end"] == 1  # 2 columns, 0-indexed inclusive
    assert kw["source"] == "xlsx"


def test_persist_table_cell_evidence_skips_empty_sheet(monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr(native_ingest, "media_store", lambda: store)
    ro._persist_table_cell_evidence("/data/book.xlsx", "Empty", [])
    assert store.cell_calls == []


@pytest.mark.skipif(
    importlib.util.find_spec("openpyxl") is None,
    reason="openpyxl not installed",
)
def test_read_xlsx_writes_table_cell_evidence(
    tmp_path, monkeypatch
):  # pragma: no cover - dep-gated
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Revenue"
    ws.append(["Q", "Amount"])
    ws.append(["Q1", 100])
    ws.append(["Q2", 150])
    p = tmp_path / "book.xlsx"
    wb.save(str(p))

    store = _FakeStore()
    monkeypatch.setattr(native_ingest, "media_store", lambda: store)

    # The EXISTING xlsx reader entry point — no new call site.
    text = ro.read_xlsx(str(p))

    # Text extraction is unchanged.
    assert "# Revenue" in text

    # AND a TableCellRange evidence locus was written for the sheet's full extent.
    assert len(store.cell_calls) == 1
    _data, kw = store.cell_calls[0]
    assert kw["table_id"] == f"{p!s}#Revenue"
    assert kw["row_start"] == 0 and kw["row_end"] == 2
    assert kw["col_start"] == 0 and kw["col_end"] == 1


def test_read_xlsx_skips_evidence_when_dep_missing(tmp_path, monkeypatch, caplog):
    """The existing no-openpyxl no-op degradation is unaffected by the wiring."""
    monkeypatch.setitem(sys.modules, "openpyxl", None)
    store = _FakeStore()
    monkeypatch.setattr(native_ingest, "media_store", lambda: store)
    p = tmp_path / "book.xlsx"
    p.write_bytes(b"PK\x03\x04 not really")
    out = ro.read_xlsx(str(p))
    assert out == ""
    assert store.cell_calls == []
