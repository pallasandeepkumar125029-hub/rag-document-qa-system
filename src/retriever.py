"""
retriever.py
------------
Handles semantic retrieval from the ChromaDB vector database.
Takes a user question, searches for the closest vector chunks,
and returns the top-k most relevant text passages with page numbers.
"""

import sys
from typing import List, Dict, Any
import chromadb
from src.config import settings
from src.embeddings import get_embedding_function

# Ensure safe printing on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Name of the collection stored in ChromaDB
COLLECTION_NAME = "nist_ai_chunks"


def clean_display_text(text: str) -> str:
    """Replaces smart/curly quotes and dashes with standard ASCII for clean terminal display."""
    return (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u2014", " - ")
        .replace("\u2013", "-")
    )


def retrieve_relevant_chunks(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Connects to the persistent ChromaDB database and retrieves the top-k
    most semantically relevant chunks for a given user query.

    Args:
        query (str): The search query or question.
        top_k (int): Number of closest chunks to retrieve (default is 3).

    Returns:
        List[Dict[str, Any]]: List of dictionary objects containing chunk text,
                              source filename, page number, chunk ID, and similarity score.
    """
    # 1. Connect to our persistent ChromaDB on disk
    client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_DIR))
    embedding_fn = get_embedding_function()

    # 2. Load the existing collection with the embedding function
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    # 3. Query the collection (ChromaDB converts query text into an embedding vector automatically)
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # 4. Format results into clean, easy-to-use Python dictionaries
    formatted_chunks = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc_text, meta, dist in zip(documents, metadatas, distances):
        # In cosine distance: 0.0 means identical, 2.0 means opposite
        # Cosine similarity = 1 - cosine_distance
        similarity_score = round(1.0 - dist, 4)

        formatted_chunks.append({
            "content": clean_display_text(doc_text),
            "source": meta.get("source", "Unknown"),
            "page": meta.get("page", 1),
            "chunk_id": meta.get("chunk_id", "unknown_id"),
            "similarity_score": similarity_score
        })

    return formatted_chunks


def display_retrieval_results(query: str, chunks: List[Dict[str, Any]]) -> None:
    """
    Prints retrieved chunks in a clean, human-readable format.
    """
    print("\n" + "=" * 70)
    print("                    SEMANTIC RETRIEVAL RESULTS                    ")
    print("=" * 70)
    print(f"User Query          : \"{query}\"")
    print(f"Total Chunks Found  : {len(chunks)}")
    print("=" * 70)

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n[Result #{i}]")
        print(f"  Similarity Score : {chunk['similarity_score']} (higher = more relevant)")
        print(f"  Source File      : {chunk['source']}")
        print(f"  Page Number      : Page {chunk['page']}")
        print(f"  Chunk ID         : {chunk['chunk_id']}")
        print(f"  Content:")
        print(f"  \"{chunk['content']}\"")
        print("-" * 70)


if __name__ == "__main__":
    # Test query as requested: "What is confabulation?"
    test_query = "What is confabulation?"
    
    print(f"Running retrieval test for query: '{test_query}'...")
    retrieved_chunks = retrieve_relevant_chunks(test_query, top_k=3)
    display_retrieval_results(test_query, retrieved_chunks)
