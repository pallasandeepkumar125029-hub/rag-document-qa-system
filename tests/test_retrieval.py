"""
Simple unit tests for ChromaDB vector store and semantic search.
"""

from src.vectorstore import get_vector_db, search_similar_chunks


def test_vector_store_persistence():
    """Verify that ChromaDB has indexed chunks saved to disk."""
    collection = get_vector_db()
    assert collection.count() > 0, "ChromaDB collection is empty"


def test_semantic_retrieval():
    """Verify that searching for a concept returns relevant chunks with scores."""
    hits = search_similar_chunks("What is confabulation in Generative AI?", top_k=2)
    assert len(hits) == 2, f"Expected 2 hits, got {len(hits)}"
    assert "content" in hits[0], "Result missing 'content'"
    assert "page" in hits[0], "Result missing 'page'"
    assert "similarity_score" in hits[0], "Result missing 'similarity_score'"
