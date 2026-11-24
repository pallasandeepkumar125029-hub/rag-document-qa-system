"""
Interactive Command Line Interface (CLI)
----------------------------------------
Provides a quick, terminal-based REPL for asking questions,
inspecting retrieved passages, and comparing With RAG vs Without RAG.
"""

import sys
from src.pipeline import rag_pipeline
from src.config import settings


def print_banner():
    print("=" * 70)
    print("           ENTERPRISE RAG SYSTEM - INTERACTIVE CLI           ")
    print(f" Document: NIST AI 600-1 Generative AI Profile (64 Pages)")
    print(f" Vector DB: ChromaDB | Embedding: {settings.EMBEDDING_PROVIDER.upper()}")
    print("=" * 70)


def main():
    print_banner()

    # Step 1: Ensure documents are indexed
    print("\n[1/2] Checking Vector Database...")
    chunk_count = rag_pipeline.initialize_knowledge_base()
    print(f"[✓] Knowledge base ready ({chunk_count} chunks indexed).\n")

    print("[2/2] Ready for questions!")
    print("Commands:")
    print("  - Type your question and press Enter")
    print("  - Type 'compare <question>' to see With RAG vs Without RAG")
    print("  - Type 'reindex' to force re-indexing of documents")
    print("  - Type 'exit' or 'quit' to close\n")

    while True:
        try:
            user_input = input("Question > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting RAG CLI. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting RAG CLI. Goodbye!")
            break

        if user_input.lower() == "reindex":
            rag_pipeline.initialize_knowledge_base(force_reindex=True)
            continue

        if user_input.lower().startswith("compare "):
            query = user_input[8:].strip()
            print(f"\n--- Running A/B Comparison for: '{query}' ---")
            comparison = rag_pipeline.compare(query)

            print("\n[WITHOUT RAG (Vanilla LLM)]")
            print(comparison["without_rag"]["answer"])

            print("\n" + "-" * 50)
            print("[WITH RAG (Grounded Knowledge Base)]")
            print(comparison["with_rag"]["answer"])

            print("\n[CITATIONS]")
            for src in comparison["with_rag"]["sources"]:
                print(f"  • Snippet {src['snippet_number']}: {src['source']} (Page {src['page']}) [Score: {src['similarity_score']:.4f}]")
            print("=" * 70 + "\n")
            continue

        # Standard RAG Query
        print("\nRetrieving relevant chunks and generating grounded answer...")
        result = rag_pipeline.query(user_input)

        print("\n" + "=" * 50)
        print("ANSWER:")
        print("=" * 50)
        print(result["answer"])

        print("\n" + "-" * 50)
        print("VERIFIED SOURCES & CITATIONS:")
        print("-" * 50)
        if result["sources"]:
            for src in result["sources"]:
                print(f"[{src['snippet_number']}] {src['source']} | Page {src['page']} | Relevance: {src['similarity_score']:.4f}")
                print(f"    Preview: \"{src['preview']}...\"")
        else:
            print("No matching document chunks found above the relevance threshold.")

        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

