"""
Configuration Module for Banking Document Intelligence Assistant.

Centralizes all configurable parameters (models, chunking sizes, directory paths)
so they can be tuned or inspected in one place without changing core logic.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# ==============================================================================
# File System Paths
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
EVAL_DIR = BASE_DIR / "eval"

# Ensure runtime directories exist
DATA_DIR.mkdir(exist_ok=True)
SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(exist_ok=True)

# ==============================================================================
# Model Configurations
# ==============================================================================
# Primary LLM used for grounded answer generation
# Default is set to GPT-5 mini as requested; can be overridden via environment variable
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5-mini")

# Embedding model used to convert text chunks into vector representations
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# LLM Generation Parameters
# Note: GPT-5 mini uses max_completion_tokens and default temperature (1.0)
LLM_MAX_COMPLETION_TOKENS = int(os.getenv("LLM_MAX_COMPLETION_TOKENS", 1000))

# Explicit standard response when context is missing or irrelevant
INSUFFICIENT_EVIDENCE_PHRASE = (
    "I cannot find sufficient evidence in the uploaded documents to answer this question."
)

# ==============================================================================
# Chunking & Retrieval Parameters (Baseline parameters to evaluate)
# ==============================================================================
# Baseline token-based chunk size (~500 tokens covers typical banking clauses/paragraphs)
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", 500))

# Baseline token overlap (~100 tokens prevents cutting legal definitions across chunks)
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", 100))

# Number of top relevant chunks to retrieve from Chroma for each question
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 4))

# Chroma Vector Database Collection Name
CHROMA_COLLECTION_NAME = "banking_documents"

# ==============================================================================
# PDF Ingestion & Layout Configuration
# ==============================================================================
# When True, uses PyMuPDF block coordinate extraction and removes top/bottom header/footer noise
PDF_FILTER_HEADERS_FOOTERS = os.getenv("PDF_FILTER_HEADERS_FOOTERS", "true").lower() == "true"

# Top header margin threshold in points (blocks whose bottom boundary y1 <= margin are classified as headers)
PDF_HEADER_MARGIN_PT = float(os.getenv("PDF_HEADER_MARGIN_PT", 50.0))

# Bottom footer margin threshold in points (blocks whose top boundary y0 >= page_height - margin are classified as footers)
PDF_FOOTER_MARGIN_PT = float(os.getenv("PDF_FOOTER_MARGIN_PT", 50.0))

# ==============================================================================
# Adjacent-Page Context Augmentation Parameters
# ==============================================================================
# When True, supplements primary retrieved chunks with immediate neighboring pages (N-1, N+1) from the same document
ENABLE_ADJACENT_CONTEXT = os.getenv("ENABLE_ADJACENT_CONTEXT", "false").lower() == "true"

# Maximum number of adjacent supporting chunks to attach per query
MAX_ADJACENT_CHUNKS = int(os.getenv("MAX_ADJACENT_CHUNKS", 2))

# ==============================================================================
# API Key Helper
# ==============================================================================
def get_openai_api_key() -> str:
    """
    Retrieve and validate the OpenAI API key.
    Raises ValueError if the key is missing.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.strip() == "your_openai_api_key_here":
        raise ValueError(
            "OPENAI_API_KEY not found or invalid. Please set your OpenAI API key in a .env file."
        )
    return api_key.strip()
