"""
vectorstore.py
--------------
Handles storing text chunks in ChromaDB and running similarity searches.
Saves vectors permanently to disk in data/chroma_db so we only embed once.
"""

from typing import List, Dict, Any
import chromadb
from src.config import settings
from src.loader import load_all_pdfs
from src.chunker import chunk_documents
from src.embeddings import get_embedding_function

# Name of our collection in ChromaDB
COLLECTION_NAME = "nist_ai_chunks"


def get_vector_db(force_reload: bool = False):
    """
    Connects to ChromaDB and populates it with document chunks if not already stored.
    
    Args:
        force_reload (bool): If True, deletes existing collection and re-indexes.
    
    Returns:
        collection: The active ChromaDB collection ready for queries.
    """
    # 1. Connect to ChromaDB in persistent mode (saves to data/chroma_db on disk)
    client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_DIR))
    embedding_fn = get_embedding_function()

    # 2. If force_reload is requested, clear the old collection
    if force_reload:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Cleared existing collection: '{COLLECTION_NAME}'")
        except Exception:
            pass

    # 3. Create or get existing collection
    # We use 'cosine' distance: distance = 1 - cosine_similarity
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    # 4. Check if collection is already indexed to save time
    existing_count = collection.count()
    if existing_count > 0 and not force_reload:
        print(f"Vector DB already has {existing_count} chunks indexed. Ready to query!")
        return collection

    # 5. Load the PDF pages and break them into chunks
    print("\n--- Step 1: Loading PDF Documents ---")
    pages = load_all_pdfs()
    
    print("\n--- Step 2: Chunking Pages ---")
    chunks = chunk_documents(pages)

    # 6. Prepare data for ChromaDB
    # ChromaDB expects three simple parallel lists: IDs, texts, and metadata dicts
    ids = []
    texts = []
    metadatas = []

    for idx, chunk in enumerate(chunks):
        ids.append(f"chunk_{idx}")
        texts.append(chunk.page_content)
        metadatas.append({
            "source": chunk.metadata.get("source", "NIST.AI.600-1.pdf"),
            "page": chunk.metadata.get("page", 1),
            "chunk_id": chunk.metadata.get("chunk_id", f"chunk_{idx}"),
        })

    # 7. Store into ChromaDB in batches of 50 to keep memory low
    print(f"\n--- Step 3: Generating Embeddings & Storing in ChromaDB ---")
    batch_size = 50
    total_chunks = len(chunks)

    for i in range(0, total_chunks, batch_size):
        end_idx = min(i + batch_size, total_chunks)
        collection.add(
            ids=ids[i:end_idx],
            documents=texts[i:end_idx],
            metadatas=metadatas[i:end_idx],
        )
        print(f"Saved chunks {i + 1} to {end_idx} of {total_chunks}...")

    print(f"\n[SUCCESS] Stored all {collection.count()} chunks in ChromaDB!")
    return collection


def search_similar_chunks(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Takes a user question and finds the top-k most relevant chunks in ChromaDB.
    
    Args:
        query (str): The search query or question.
        top_k (int): Number of chunks to retrieve (default is 3).
        
    Returns:
        List of dicts containing the content, source, page, and similarity score.
    """
    client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_DIR))
    embedding_fn = get_embedding_function()
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

    # Query ChromaDB
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    formatted_results = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc_text, meta, dist in zip(docs, metas, distances):
        # Cosine distance to similarity score: similarity = 1 - distance
        similarity_score = round(1.0 - dist, 4)
        formatted_results.append({
            "content": doc_text,
            "source": meta.get("source", "Unknown"),
            "page": meta.get("page", 1),
            "chunk_id": meta.get("chunk_id", ""),
            "similarity_score": similarity_score
        })

    return formatted_results


if __name__ == "__main__":
    print("=" * 65)
    print("             PHASE 4: VECTOR STORE VERIFICATION             ")
    print("=" * 65)

    # 1. Initialize or load the database
    collection = get_vector_db(force_reload=True)

    # 2. Print required verification metrics
    total_indexed = collection.count()
    print("\n" + "=" * 65)
    print("                    INDEXING METRICS                     ")
    print("=" * 65)
    print(f"Total Chunks Indexed  : {total_indexed}")
    print(f"Number of Vectors     : {total_indexed} (stored permanently in data/chroma_db)")
    
    # 3. Print sample metadata entry
    sample_entry = collection.get(ids=["chunk_0"], include=["metadatas", "documents"])
    print(f"\nSample Metadata Entry for 'chunk_0':")
    print(sample_entry["metadatas"][0])
    print(f"\nSample Chunk Content:")
    print(f"\"{sample_entry['documents'][0][:200]}...\"")
    print("=" * 65)

    # 4. Quick semantic search test to prove retrieval works
    test_query = "What is confabulation in Generative AI?"
    print(f"\nTesting Semantic Search for: '{test_query}'")
    hits = search_similar_chunks(test_query, top_k=2)

    for i, hit in enumerate(hits, 1):
        print(f"\n[Result #{i}] Score: {hit['similarity_score']} | Source: {hit['source']} (Page {hit['page']})")
        print(f"Excerpt: \"{hit['content'][:250]}...\"")
    print("=" * 65)
