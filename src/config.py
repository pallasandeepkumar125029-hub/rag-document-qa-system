"""
Configuration Module for RAG System.
------------------------------------
Centralizes all hyperparameters, directory paths, and model selections.
Uses python-dotenv to load overrides from a local .env file.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Automatically locate the project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables from .env file in the project root
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Immutable project settings and hyperparameters."""

    # --------------------------------------------------------------------------
    # Filesystem Paths (OS-Independent using pathlib)
    # --------------------------------------------------------------------------
    ROOT_DIR: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
    CHROMA_DB_DIR: Path = PROJECT_ROOT / "data" / "chroma_db"

    # --------------------------------------------------------------------------
    # Document Ingestion & Chunking Parameters
    # --------------------------------------------------------------------------
    # 600 characters is approximately 120-150 tokens.
    # Overlap of 100 characters ensures semantic continuity between chunks.
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "600"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # --------------------------------------------------------------------------
    # Embedding Configuration
    # --------------------------------------------------------------------------
    # Options: "huggingface" (Free, local) | "openai" (API key required)
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()

    # Default local embedding model: 384-dimensional dense vectors, fast & accurate
    HUGGINGFACE_EMBEDDING_MODEL: str = os.getenv(
        "HUGGINGFACE_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    # Cloud alternative (OpenAI)
    OPENAI_EMBEDDING_MODEL: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-small",
    )

    # --------------------------------------------------------------------------
    # Vector Database (ChromaDB) Configuration
    # --------------------------------------------------------------------------
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_documents")

    # --------------------------------------------------------------------------
    # Retrieval Hyperparameters
    # --------------------------------------------------------------------------
    # Number of top relevant document chunks to inject into the prompt
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "3"))

    # --------------------------------------------------------------------------
    # LLM (Generator) Configuration
    # --------------------------------------------------------------------------
    # Options: "ollama" (Local LLaMA 3.2 / Mistral) | "openai" (GPT-4o-mini)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").lower()

    # Ollama local settings
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # OpenAI cloud settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_LLM_MODEL: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

    # Temperature = 0.0 enforces deterministic, factual, grounded answers
    TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    def ensure_directories_exist(self) -> None:
        """Create necessary directories if they do not already exist."""
        self.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)


# Global settings singleton instance for easy import across modules
settings = Settings()

# Automatically verify/create required directories on import
settings.ensure_directories_exist()


if __name__ == "__main__":
    # Self-test block: Run python -m src.config to verify paths
    print("=" * 60)
    print("RAG SYSTEM CONFIGURATION VERIFICATION")
    print("=" * 60)
    print(f"Project Root       : {settings.ROOT_DIR}")
    print(f"Raw Data Directory : {settings.RAW_DATA_DIR}")
    print(f"ChromaDB Directory : {settings.CHROMA_DB_DIR}")
    print(f"Chunk Size / Overlap: {settings.CHUNK_SIZE} / {settings.CHUNK_OVERLAP}")
    print(f"Embedding Provider : {settings.EMBEDDING_PROVIDER} ({settings.HUGGINGFACE_EMBEDDING_MODEL})")
    print(f"LLM Provider       : {settings.LLM_PROVIDER}")
    print(f"Retrieval Top-K    : {settings.RETRIEVAL_TOP_K}")
    print(f"Directories Created: Successfully verified.")
    print("=" * 60)

