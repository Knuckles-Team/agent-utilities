"""Governed PDF extraction is killable and bounded outside GraphOS."""

from __future__ import annotations

from pathlib import Path

from agent_utilities.knowledge_graph.core import engine_tasks
from agent_utilities.knowledge_graph.extraction.pdf import (
    MAX_PDF_WALL_SECONDS,
    read_pdf_pages,
    read_pdf_text,
)

# ``_extract_pdf_raw`` spawns a real child process (``multiprocessing.get_context
# ("spawn")`` — a fresh interpreter, not a fork) and ``parent.poll(timeout_seconds)``
# is a genuine wall-clock wait for it. The 3 tests below assert a SUCCESSFUL
# extraction, so a short budget here doesn't buy speed (poll returns the instant
# the trivial fake worker replies) — it only adds risk: under the parallel suite's
# `-n auto` CPU contention a slow-but-working spawn can blow a tight deadline and
# return "" instead of the expected text (this was observed intermittently with
# `timeout_seconds=5`). Use the same generous ceiling production defaults to
# (``MAX_PDF_WALL_SECONDS``) so only a genuinely hung worker times out.
_SUCCESS_TIMEOUT_SECONDS = MAX_PDF_WALL_SECONDS


def _fake_pypdf(
    root: Path,
    monkeypatch,
    implementation: str,
) -> None:
    package = root / "pypdf"
    package.mkdir()
    (package / "__init__.py").write_text(implementation, encoding="utf-8")
    monkeypatch.syspath_prepend(str(root))


def test_engine_reader_always_selects_bounded_pypdf() -> None:
    reader = engine_tasks._pdf_file_extractor()[".pdf"]
    assert type(reader).__name__ == "_BoundedPypdfReader"


def test_worker_extracts_and_caps_text(tmp_path, monkeypatch) -> None:
    _fake_pypdf(
        tmp_path,
        monkeypatch,
        """
class Page:
    def __init__(self, text): self.text = text
    def extract_text(self): return self.text
class PdfReader:
    is_encrypted = False
    def __init__(self, *args, **kwargs): self.pages = [Page('first'), Page('second')]
""",
    )
    pdf = tmp_path / "document.pdf"
    pdf.write_bytes(b"%PDF fixture")
    assert (
        read_pdf_text(pdf, max_chars=8, timeout_seconds=_SUCCESS_TIMEOUT_SECONDS)
        == "first\nse"
    )


def test_read_pdf_pages_preserves_page_boundaries(tmp_path, monkeypatch) -> None:
    """D-ES-3: page boundaries must survive extraction for fragment_pdf.

    ``read_pdf_text`` joins pages on ``"\\n"`` — indistinguishable from a
    ``"\\n"`` that already occurs inside a page's own extracted text.
    ``read_pdf_pages`` must return the pages as a list instead, and
    ``read_pdf_text``'s own output must be unaffected by the refactor.
    """
    _fake_pypdf(
        tmp_path,
        monkeypatch,
        """
class Page:
    def __init__(self, text): self.text = text
    def extract_text(self): return self.text
class PdfReader:
    is_encrypted = False
    def __init__(self, *args, **kwargs):
        self.pages = [Page('line one\\nline two'), Page('second page')]
""",
    )
    pdf = tmp_path / "document.pdf"
    pdf.write_bytes(b"%PDF fixture")

    assert read_pdf_pages(pdf, timeout_seconds=_SUCCESS_TIMEOUT_SECONDS) == [
        "line one\nline two",
        "second page",
    ]
    # read_pdf_text's own contract (join on "\n") is unchanged by the refactor.
    assert (
        read_pdf_text(pdf, timeout_seconds=_SUCCESS_TIMEOUT_SECONDS)
        == "line one\nline two\nsecond page"
    )


def test_read_pdf_pages_fails_closed_like_read_pdf_text(tmp_path, monkeypatch) -> None:
    _fake_pypdf(
        tmp_path,
        monkeypatch,
        """
class PdfReader:
    def __init__(self, *args, **kwargs): raise RuntimeError('parser detail')
""",
    )
    pdf = tmp_path / "document.pdf"
    pdf.write_bytes(b"%PDF fixture")
    assert read_pdf_pages(pdf, timeout_seconds=5) == []


def test_hung_worker_is_killed_at_wall_deadline(tmp_path, monkeypatch) -> None:
    _fake_pypdf(
        tmp_path,
        monkeypatch,
        """
import time
class PdfReader:
    def __init__(self, *args, **kwargs): time.sleep(60)
""",
    )
    pdf = tmp_path / "document.pdf"
    pdf.write_bytes(b"%PDF fixture")
    assert read_pdf_text(pdf, timeout_seconds=0.5) == ""


def test_worker_parser_failure_is_detail_free(tmp_path, monkeypatch) -> None:
    _fake_pypdf(
        tmp_path,
        monkeypatch,
        """
class PdfReader:
    def __init__(self, *args, **kwargs): raise RuntimeError('parser detail')
""",
    )
    pdf = tmp_path / "document.pdf"
    pdf.write_bytes(b"%PDF fixture")
    assert read_pdf_text(pdf, timeout_seconds=5) == ""


def test_worker_rejects_oversized_page_count(tmp_path, monkeypatch) -> None:
    _fake_pypdf(
        tmp_path,
        monkeypatch,
        """
class Page:
    def extract_text(self): return 'text'
class PdfReader:
    is_encrypted = False
    def __init__(self, *args, **kwargs): self.pages = [Page(), Page()]
""",
    )
    pdf = tmp_path / "document.pdf"
    pdf.write_bytes(b"%PDF fixture")
    assert read_pdf_text(pdf, max_pages=1, timeout_seconds=5) == ""


def test_parent_rejects_oversized_file_before_worker(tmp_path) -> None:
    pdf = tmp_path / "document.pdf"
    pdf.write_bytes(b"%PDF fixture")
    assert read_pdf_text(pdf, max_bytes=4) == ""


def test_worker_rejects_oversized_ipc_output(tmp_path, monkeypatch) -> None:
    _fake_pypdf(
        tmp_path,
        monkeypatch,
        """
class Page:
    def extract_text(self): return 'abcdefghij'
class PdfReader:
    is_encrypted = False
    def __init__(self, *args, **kwargs): self.pages = [Page()]
""",
    )
    pdf = tmp_path / "document.pdf"
    pdf.write_bytes(b"%PDF fixture")
    assert (
        read_pdf_text(
            pdf,
            max_chars=10,
            max_output_bytes=4,
            timeout_seconds=5,
        )
        == ""
    )


def test_invalid_or_excessive_limit_request_fails_closed(tmp_path) -> None:
    pdf = tmp_path / "document.pdf"
    pdf.write_bytes(b"%PDF fixture")
    assert read_pdf_text(pdf, timeout_seconds=float("inf")) == ""
    assert read_pdf_text(pdf, max_pages=5_001) == ""
