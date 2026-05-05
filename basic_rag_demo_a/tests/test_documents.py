from __future__ import annotations

from pathlib import Path

import pytest

from rag_demo.documents import chunk_document, load_document
from rag_demo.sample_data import ensure_sample_docs


def test_load_markdown_and_chunk(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    ensure_sample_docs(docs)

    document = load_document(docs / "meal_notes.md")
    chunks = chunk_document(document, chunk_chars=800, overlap_chars=100)

    assert "Breakfast" in document.text
    assert chunks
    assert chunks[0].title


def test_load_html(tmp_path: Path) -> None:
    pytest.importorskip("bs4")
    docs = tmp_path / "docs"
    ensure_sample_docs(docs)

    document = load_document(docs / "hydration_guide.html")

    assert "72 ounces" in document.text


def test_load_pdf(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    docs = tmp_path / "docs"
    ensure_sample_docs(docs)

    document = load_document(docs / "pantry_guide.pdf")

    assert "Pantry Guide" in document.text


def test_load_docx(tmp_path: Path) -> None:
    pytest.importorskip("docx")
    docs = tmp_path / "docs"
    ensure_sample_docs(docs)

    document = load_document(docs / "fiber_notes.docx")

    assert "Fiber Notes" in document.text
