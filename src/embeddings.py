"""
embeddings.py
-------------
Helper module to load the HuggingFace embedding model for ChromaDB.
Model: all-MiniLM-L6-v2 (Sentence Transformers)
Dimension: 384 numbers per vector
Runs 100% locally on CPU without needing any API keys.
"""

from chromadb.utils import embedding_functions


def get_embedding_function():
    """
    Returns the all-MiniLM-L6-v2 embedding function for ChromaDB.
    ChromaDB downloads and runs this model locally using ONNX.
    """
    return embedding_functions.DefaultEmbeddingFunction()


if __name__ == "__main__":
    # Quick test to see if embeddings work
    fn = get_embedding_function()
    test_text = "What is Generative AI risk management?"
    vector = fn([test_text])[0]
    
    print("Embedding model loaded successfully!")
    print(f"Sample text     : '{test_text}'")
    print(f"Vector length   : {len(vector)} dimensions")
    print(f"First 5 numbers : {vector[:5]}")
