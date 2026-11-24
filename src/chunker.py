"""
Document Chunking Module
------------------------
Splits document pages into semantically cohesive text passages using LangChain's
RecursiveCharacterTextSplitter. Preserves page-level metadata and assigns unique
chunk identifiers for verifiable source attribution.
"""

import sys
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings

# Ensure standard output can print Unicode characters safely on Windows CP1252 consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class DocumentChunker:
    """
    Splits LangChain Document objects into smaller, overlapping chunks
    using recursive character boundaries (paragraphs -> lines -> spaces -> chars).
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        """
        Initializes the chunker with configured size and overlap boundaries.

        Args:
            chunk_size (Optional[int]): Target character size per chunk. Defaults to settings.CHUNK_SIZE (600).
            chunk_overlap (Optional[int]): Number of overlapping characters. Defaults to settings.CHUNK_OVERLAP (100).
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        # Defensive validation: Overlap must be strictly smaller than chunk size
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than "
                f"chunk_size ({self.chunk_size})."
            )

        # Initialize LangChain's RecursiveCharacterTextSplitter
        # Separator hierarchy prioritizes semantic coherence:
        # 1. "\n\n" -> Paragraphs
        # 2. "\n"   -> Lines / Clauses
        # 3. " "    -> Words
        # 4. ""     -> Characters (fallback)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of page-level Documents into chunk-level Documents.
        Preserves original metadata (source, page) and enriches with chunk tracking IDs.

        Args:
            documents (List[Document]): Raw documents loaded from loader.py.

        Returns:
            List[Document]: Segmented chunk documents ready for vector embedding.

        Raises:
            ValueError: If input documents list is empty.
        """
        if not documents:
            raise ValueError("Cannot chunk an empty list of documents. Ensure loader.py found valid PDFs.")

        # Execute recursive character splitting
        raw_chunks = self.splitter.split_documents(documents)

        enriched_chunks: List[Document] = []
        page_chunk_counters = {}

        for global_idx, chunk in enumerate(raw_chunks):
            # Clean up any surrounding whitespace
            cleaned_text = chunk.page_content.strip()
            if not cleaned_text:
                continue

            # Extract existing metadata
            source_file = chunk.metadata.get("source", "document.pdf")
            page_num = chunk.metadata.get("page", 1)

            # Track per-page chunk indexing
            counter_key = (source_file, page_num)
            page_chunk_counters[counter_key] = page_chunk_counters.get(counter_key, 0) + 1
            local_page_chunk_idx = page_chunk_counters[counter_key]

            # Assign structured, deterministic chunk metadata
            chunk.metadata["chunk_id"] = f"{source_file}:p{page_num}:c{local_page_chunk_idx}"
            chunk.metadata["chunk_index"] = global_idx
            chunk.metadata["chunk_char_count"] = len(cleaned_text)
            chunk.page_content = cleaned_text

            enriched_chunks.append(chunk)

        return enriched_chunks


def chunk_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """
    Functional convenience wrapper for DocumentChunker.split_documents.
    """
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.split_documents(documents)


def print_chunking_summary(chunks: List[Document], sample_count: int = 3) -> None:
    """
    Prints a formatted summary of chunking statistics and previews initial chunks.

    Args:
        chunks (List[Document]): Processed chunk documents.
        sample_count (int): Number of initial chunks to preview (default: 3).
    """
    if not chunks:
        print("[WARNING] No chunks available to summarize.")
        return

    lengths = [len(c.page_content) for c in chunks]
    total_chunks = len(chunks)
    avg_length = sum(lengths) // total_chunks
    min_length = min(lengths)
    max_length = max(lengths)

    print("\n" + "=" * 70)
    print("                     DOCUMENT CHUNKING SUMMARY                     ")
    print("=" * 70)
    print(f"Total Chunks Created: {total_chunks:,}")
    print(f"Average Chunk Length: {avg_length:,} characters (~{avg_length // 4} tokens)")
    print(f"Min / Max Length    : {min_length} / {max_length} characters")
    print(f"Target Configuration: chunk_size={settings.CHUNK_SIZE}, chunk_overlap={settings.CHUNK_OVERLAP}")
    print("=" * 70)

    # Display preview of first N chunks
    print(f"\n--- SAMPLE CHUNK PREVIEW (First {min(sample_count, total_chunks)} Chunks) ---\n")
    for i in range(min(sample_count, total_chunks)):
        chunk = chunks[i]
        meta = chunk.metadata
        print(f"[Chunk #{i + 1}] ID: {meta.get('chunk_id')}")
        print(f"Source: {meta.get('source')} | Page: {meta.get('page')} | Length: {meta.get('chunk_char_count')} chars")
        print(f"Content:\n{chunk.page_content}")
        print("-" * 70)


if __name__ == "__main__":
    # Integration test with src.loader
    from src.loader import load_all_pdfs

    try:
        print("Loading source documents from raw data directory...")
        raw_pages = load_all_pdfs()

        print(f"Chunking {len(raw_pages)} pages with chunk_size={settings.CHUNK_SIZE}, chunk_overlap={settings.CHUNK_OVERLAP}...")
        processed_chunks = chunk_documents(raw_pages)

        # Print statistics and first 3 chunks as required
        print_chunking_summary(processed_chunks, sample_count=3)

    except Exception as err:
        print(f"\n[FATAL ERROR] Chunking failed: {err}")
