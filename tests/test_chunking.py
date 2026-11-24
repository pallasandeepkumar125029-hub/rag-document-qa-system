"""
Simple unit tests for document loading and chunking.
"""

from src.loader import load_all_pdfs
from src.chunker import chunk_documents


def test_pdf_loader():
    """Verify that the PDF loads and metadata contains page numbers."""
    pages = load_all_pdfs()
    assert len(pages) > 0, "No pages were loaded from data/raw/"
    assert pages[0].metadata["page"] == 1, "First page should be 1-indexed"
    assert "source" in pages[0].metadata, "Metadata missing 'source'"


def test_chunking():
    """Verify that pages are broken down into overlapping chunks."""
    pages = load_all_pdfs()[:3]  # Test with first 3 pages
    chunks = chunk_documents(pages, chunk_size=600, chunk_overlap=100)
    
    assert len(chunks) > 0, "Chunker returned 0 chunks"
    assert len(chunks[0].page_content) <= 650, "Chunk exceeded size threshold"
    assert "source" in chunks[0].metadata, "Chunk metadata missing 'source'"
    assert "page" in chunks[0].metadata, "Chunk metadata missing 'page'"
