"""Governed PDF extraction is killable and bounded outside GraphOS."""

from __future__ import annotations

from pathlib import Path

from agent_utilities.knowledge_graph.core import engine_tasks
from agent_utilities.knowledge_graph.extraction.pdf import read_pdf_text


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
    assert read_pdf_text(pdf, max_chars=8, timeout_seconds=5) == "first\nse"


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
