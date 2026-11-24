"""
RAG Pipeline Orchestrator
-------------------------
Integrates loader, chunker, vector store, retriever, and generator
into a unified, production-grade interface.
"""

from typing import Dict, Any, List, Optional
from src.loader import PDFDocumentLoader
from src.chunker import TextChunker
from src.vectorstore import vector_store, VectorStoreManager
from src.retriever import ContextRetriever
from src.generator import generator, GroundedGenerator
from src.config import settings


class RAGPipeline:
    """End-to-end RAG orchestrator for document querying and evaluation."""

    def __init__(
        self,
        store: Optional[VectorStoreManager] = None,
        retriever_instance: Optional[ContextRetriever] = None,
        generator_instance: Optional[GroundedGenerator] = None,
    ):
        self.store = store or vector_store
        self.retriever = retriever_instance or ContextRetriever(store=self.store)
        self.generator = generator_instance or generator

    def initialize_knowledge_base(self, force_reindex: bool = False) -> int:
        """
        Loads and indexes documents into the persistent vector store.
        If the store already contains chunks and force_reindex is False, skips re-indexing.
        """
        current_count = self.store.count()
        if current_count > 0 and not force_reindex:
            print(f"Knowledge base already initialized with {current_count} chunks. Skipping re-indexing.")
            return current_count

        if force_reindex and current_count > 0:
            print("Force re-index requested. Resetting existing collection...")
            self.store.reset()

        print("Starting document ingestion and chunking pipeline...")
        loader = PDFDocumentLoader()
        raw_pages = loader.load()
        print(f"Loaded {len(raw_pages)} pages from '{loader.file_path.name}'.")

        chunker = TextChunker()
        chunks = chunker.split_documents(raw_pages)
        print(f"Generated {len(chunks)} text chunks (Size: {settings.CHUNK_SIZE}, Overlap: {settings.CHUNK_OVERLAP}).")

        indexed_count = self.store.add_documents(chunks)
        return indexed_count

    def query(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Executes a complete Grounded RAG query cycle:
        1. Retrieve top-k relevant chunks from ChromaDB
        2. Format context with source citations
        3. Pass query + context to LLM
        4. Return structured answer with source attribution
        """
        chunks = self.retriever.retrieve(question, top_k=top_k)
        context, sources = self.retriever.format_context_for_prompt(chunks)
        answer = self.generator.generate_grounded_answer(query=question, context=context)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": chunks,
            "retrieval_count": len(chunks),
        }

    def compare(self, question: str) -> Dict[str, Any]:
        """
        A/B Comparison: Executes query with RAG vs without RAG.
        Used to demonstrate grounded factuality and hallucination prevention.
        """
        # 1. Grounded RAG Generation
        rag_result = self.query(question)

        # 2. Vanilla LLM Generation (Pre-training weights only)
        vanilla_answer = self.generator.generate_vanilla_answer(question)

        return {
            "question": question,
            "with_rag": {
                "answer": rag_result["answer"],
                "sources": rag_result["sources"],
            },
            "without_rag": {
                "answer": vanilla_answer,
            },
        }


# Global pipeline singleton instance
rag_pipeline = RAGPipeline()


if __name__ == "__main__":
    print("=" * 60)
    print("RAG PIPELINE STATUS")
    print("=" * 60)
    print(f"Indexed Chunks: {rag_pipeline.store.count()}")
    print("=" * 60)

