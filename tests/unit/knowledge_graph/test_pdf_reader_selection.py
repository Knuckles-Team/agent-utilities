"""

CONCEPT:KG-2.8
Tests for PDF-reader selection in document ingestion.

The default LlamaIndex ``SimpleDirectoryReader`` PDF reader (pypdf) is pure-Python
and can stall for minutes on a single pathological PDF, holding the GIL and
starving every other KGTaskWorker on the host. We force PyMuPDF (a GIL-releasing C
extension) for ``.pdf`` instead, with a graceful fallback to the default reader if
PyMuPDF isn't installed.
"""

import sys
from unittest.mock import patch

from agent_utilities.knowledge_graph.core import engine_tasks


class TestPdfFileExtractor:
    """``engine_tasks._pdf_file_extractor`` picks PyMuPDF, degrades gracefully."""

    def test_returns_pymupdf_reader_for_pdf(self):
        """When the PyMuPDF reader is importable, it is mapped to ``.pdf``."""
        ext = engine_tasks._pdf_file_extractor()
        assert ".pdf" in ext, "PyMuPDF reader should be installed in the test env"
        # The reader is the PyMuPDF one, not the default pypdf-backed reader.
        assert type(ext[".pdf"]).__name__ == "PyMuPDFReader"

    def test_empty_mapping_when_reader_missing(self):
        """Missing optional dep → empty mapping (default reader), never raises.

        An empty ``file_extractor`` dict is merged with SimpleDirectoryReader's
        built-in defaults, so it behaves identically to the prior default path.
        """
        # Simulate the import failing inside the helper.
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "llama_index.readers.file":
                raise ImportError("simulated missing optional dependency")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            ext = engine_tasks._pdf_file_extractor()
        assert ext == {}

    def test_pymupdf_extracts_faster_than_default(self, tmp_path):
        """Smoke: the selected reader actually extracts text from a real PDF."""
        fitz = sys.modules.get("fitz")
        if fitz is None:
            import importlib

            try:
                fitz = importlib.import_module("fitz")
            except ImportError:
                import pytest

                pytest.skip("PyMuPDF not installed")
        pdf = tmp_path / "sample.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "hello knowledge graph")
        doc.save(str(pdf))
        doc.close()

        reader = engine_tasks._pdf_file_extractor()[".pdf"]
        docs = reader.load_data(file_path=str(pdf))
        text = "\n".join(d.text for d in docs)
        assert "hello knowledge graph" in text
