"""
Document Ingestion Module
-------------------------
Loads all PDF documents from the raw data directory using LangChain's PyPDFLoader.
Extracts text page-by-page while preserving structured metadata (source filename and page number).
Adheres strictly to project settings defined in src/config.py.
Windows-safe: normalizes typographic ligatures (e.g. 'fi', 'fl') and handles console encodings.
"""

import sys
import unicodedata
import warnings
from pathlib import Path
from typing import List, Optional

# Ensure standard output can print Unicode characters on Windows CP1252 terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Suppress harmless third-party deprecation notices
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from src.config import settings


def normalize_pdf_text(text: str) -> str:
    """
    Normalizes extracted PDF text:
    - Decomposes typographic ligatures (e.g., 'ﬁ' -> 'fi', 'ﬂ' -> 'fl')
    - Strips leading and trailing whitespace
    """
    if not text:
        return ""
    # NFKD decomposes combined ligature glyphs into standard ASCII equivalents
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.strip()


def load_all_pdfs(raw_dir: Optional[Path] = None) -> List[Document]:
    """
    Scans the specified directory for all PDF files, loads each using LangChain's
    PyPDFLoader, normalizes metadata, and returns a consolidated list of page Documents.

    Args:
        raw_dir (Optional[Path]): Directory containing source PDFs. Defaults to settings.RAW_DATA_DIR.

    Returns:
        List[Document]: List of LangChain Document objects (one per page).

    Raises:
        FileNotFoundError: If the raw data directory does not exist or contains no PDF files.
    """
    directory = raw_dir or settings.RAW_DATA_DIR

    # 1. Verify directory existence
    if not directory.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: '{directory.resolve()}'. "
            "Please ensure the directory exists and contains PDF files."
        )

    # 2. Discover all PDF files in the directory
    pdf_files = sorted(list(directory.glob("*.pdf")))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{directory.resolve()}'. "
            "Please place your PDF document(s) in this folder before ingestion."
        )

    all_documents: List[Document] = []
    loaded_pdf_count = 0

    print(f"Discovered {len(pdf_files)} PDF file(s) in '{directory}'. Beginning ingestion...\n")

    # 3. Iterate through each PDF and load pages
    for pdf_path in pdf_files:
        try:
            print(f"-> Ingesting: {pdf_path.name}")
            loader = PyPDFLoader(str(pdf_path.resolve()))
            pages = loader.load()

            # Normalize and enrich metadata for each page
            for page_doc in pages:
                raw_page_num = page_doc.metadata.get("page", 0)
                # Ensure 1-indexed page number for intuitive citation in human interfaces
                page_doc.metadata["page"] = raw_page_num + 1
                page_doc.metadata["source"] = pdf_path.name
                page_doc.metadata["file_path"] = str(pdf_path.resolve())

                # Clean and decompose ligatures in page content
                page_doc.page_content = normalize_pdf_text(page_doc.page_content)

            all_documents.extend(pages)
            loaded_pdf_count += 1
            print(f"   [OK] Successfully loaded {len(pages)} pages from {pdf_path.name}")

        except Exception as e:
            # Robust error handling: Log error and continue with other files
            print(f"   [ERROR] Error loading '{pdf_path.name}': {e}")

    return all_documents


class PDFDocumentLoader:
    """
    Object-oriented wrapper around load_all_pdfs for pipeline compatibility.
    """

    def __init__(self, file_path: Optional[Path] = None):
        self.file_path = Path(file_path) if file_path else None

    def load(self) -> List[Document]:
        if self.file_path:
            loader = PyPDFLoader(str(self.file_path.resolve()))
            pages = loader.load()
            for doc in pages:
                raw_page = doc.metadata.get("page", 0)
                doc.metadata["page"] = raw_page + 1
                doc.metadata["source"] = self.file_path.name
                doc.page_content = normalize_pdf_text(doc.page_content)
            return pages
        return load_all_pdfs()


def print_ingestion_summary(documents: List[Document], sample_page_count: int = 3) -> None:
    """
    Prints a structured summary of the ingestion results including:
    - Total unique PDFs loaded
    - Total pages loaded
    - Sample text preview from the first N pages

    Args:
        documents (List[Document]): The loaded document pages.
        sample_page_count (int): Number of initial pages to preview (default: 3).
    """
    unique_sources = {doc.metadata.get("source", "unknown") for doc in documents}

    print("\n" + "=" * 70)
    print("                    DOCUMENT INGESTION SUMMARY                    ")
    print("=" * 70)
    print(f"Total PDFs Loaded   : {len(unique_sources)}")
    print(f"Source Files        : {', '.join(sorted(unique_sources))}")
    print(f"Total Pages Loaded  : {len(documents)}")
    print("=" * 70)

    # Display sample preview from first N pages
    print(f"\n--- SAMPLE PREVIEW (First {min(sample_page_count, len(documents))} Pages) ---\n")
    for i in range(min(sample_page_count, len(documents))):
        doc = documents[i]
        source = doc.metadata.get("source", "Unknown")
        page_num = doc.metadata.get("page", i + 1)
        preview_text = doc.page_content[:300].replace("\n", " ")

        print(f"[Page {page_num} | Source: {source}]")
        print(f"Excerpt: \"{preview_text}...\"")
        print("-" * 70)


if __name__ == "__main__":
    # Standalone execution for verification and lab demonstrations
    try:
        loaded_pages = load_all_pdfs()
        print_ingestion_summary(loaded_pages, sample_page_count=3)
    except Exception as err:
        print(f"\n[FATAL ERROR] Ingestion failed: {err}")
