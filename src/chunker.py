"""
Chunking Module for Banking Document Intelligence Assistant.

Splits extracted PDF page text into token-aware, overlapping chunks using tiktoken.
Preserves page-level metadata (source, page number, unique chunk ID, token count).
"""

import re
import logging
from typing import List, Dict, Any
import tiktoken
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the BPE tokenizer for OpenAI embeddings / models
try:
    # cl100k_base is the standard tokenizer for text-embedding-3-small and modern OpenAI models
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception as err:
    logger.warning(f"Could not load 'cl100k_base', falling back to gpt-4o encoder: {err}")
    _ENCODER = tiktoken.encoding_for_model("gpt-4o")


def count_tokens(text: str) -> int:
    """
    Count the exact number of tokens in a given string.
    """
    if not text:
        return 0
    return len(_ENCODER.encode(text))


def clean_text(text: str) -> str:
    """
    Perform lightweight text normalization on raw PDF text:
    - Replaces null bytes and non-printable control characters.
    - Normalizes multiple spaces and tabs to single spaces.
    - Collapses 3+ consecutive newlines into 2 newlines (preserves paragraph breaks).
    - Trims leading and trailing whitespace.
    """
    if not text:
        return ""
    # Remove null bytes and control chars (except standard newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Normalize horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse multiple blank lines to double newlines (paragraphs)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def _sanitize_slug(name: str) -> str:
    """
    Convert a filename into a clean, alphanumeric identifier for chunk IDs.
    e.g., 'Credit_Agreement (2024).pdf' -> 'Credit_Agreement_2024'
    """
    base_name = name.rsplit(".", 1)[0]
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", base_name)
    return slug.strip("_")


def chunk_pages(
    pages_data: List[Dict[str, Any]],
    chunk_size: int = config.CHUNK_SIZE_TOKENS,
    chunk_overlap: int = config.CHUNK_OVERLAP_TOKENS,
) -> List[Dict[str, Any]]:
    """
    Chunk extracted PDF pages into token-bounded segments with overlap.

    Page boundaries are preserved: chunking is performed page-by-page so every chunk
    cleanly maps to an exact 1-indexed page number for unambiguous citations.

    Args:
        pages_data: List of page dictionaries from load_pdf_pages()
                    [{ "source": str, "page": int, "text": str, ... }, ...]
        chunk_size: Maximum tokens per chunk (default from config.py)
        chunk_overlap: Number of tokens overlapping between adjacent chunks on the same page

    Returns:
        A list of chunk dictionaries with the following structure:
        [
            {
                "source": "loan_terms.pdf",
                "page": 1,
                "chunk_id": "loan_terms_p1_c0",
                "chunk_text": "Clause 1.2 ...",
                "token_count": 412
            },
            ...
        ]
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    step_size = chunk_size - chunk_overlap
    all_chunks = []

    for page_dict in pages_data:
        source_name = page_dict.get("source", "document.pdf")
        page_num = page_dict.get("page", 1)
        raw_text = page_dict.get("text", "")

        cleaned = clean_text(raw_text)
        if not cleaned:
            # Skip completely empty pages
            continue

        slug = _sanitize_slug(source_name)
        token_ids = _ENCODER.encode(cleaned)
        total_tokens = len(token_ids)

        # Case 1: Page fits entirely within a single chunk
        if total_tokens <= chunk_size:
            chunk_id = f"{slug}_p{page_num}_c0"
            all_chunks.append({
                "source": source_name,
                "page": page_num,
                "chunk_id": chunk_id,
                "chunk_text": cleaned,
                "token_count": total_tokens
            })
            continue

        # Case 2: Page exceeds chunk size -> split into overlapping token slices
        chunk_idx = 0
        for start_idx in range(0, total_tokens, step_size):
            end_idx = min(start_idx + chunk_size, total_tokens)
            slice_tokens = token_ids[start_idx:end_idx]

            chunk_text = _ENCODER.decode(slice_tokens).strip()
            if not chunk_text:
                continue

            chunk_id = f"{slug}_p{page_num}_c{chunk_idx}"
            all_chunks.append({
                "source": source_name,
                "page": page_num,
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "token_count": len(slice_tokens)
            })
            chunk_idx += 1

            # Stop if we reached the end of the page's token stream
            if end_idx >= total_tokens:
                break

    logger.info(
        f"Chunking complete: Created {len(all_chunks)} chunks from {len(pages_data)} pages "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})."
    )
    return all_chunks
