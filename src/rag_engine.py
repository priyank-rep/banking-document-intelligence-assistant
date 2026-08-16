"""
RAG Engine Module for Banking Document Intelligence Assistant.

Orchestrates the grounded answer generation pipeline:
1. Receives user query.
2. Retrieves top-K context chunks from ChromaDB.
3. Builds a structured, guardrailed prompt.
4. Synthesizes a factual answer with source citations via OpenAI LLM (GPT-5 mini).
5. Distinguishes cleanly between:
   - Genuine insufficient document evidence (grounding limitation)
   - Database/retrieval failures
   - OpenAI API / network / configuration errors
"""

import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI, OpenAIError

import config
from src.vector_store import (
    query_vector_store,
    get_openai_client,
    get_or_create_collection
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# Prompt Construction & Guardrails
# ==============================================================================

SYSTEM_PROMPT = f"""You are an expert Banking Document Intelligence Assistant.
Your primary task is to answer user questions about banking documents with complete fidelity to the provided context.

STRICT GROUNDING & COMPLIANCE RULES:
1. Answer ONLY using information explicitly stated in the provided Context Chunks.
2. Do NOT use outside knowledge, prior assumptions, or extrapolate beyond the text.
3. Do NOT guess or invent numbers, percentages, fees, deadlines, names, or legal terms.
4. When answering, cite your sources inline using the format [Document: filename, Page: X] or clause references where applicable.
5. If the provided context does NOT contain enough information to answer the question with certainty, you MUST respond EXACTLY with:
   "{config.INSUFFICIENT_EVIDENCE_PHRASE}"
6. Do NOT provide speculative answers, partial guesses, or conversational apologies when returning the insufficient evidence phrase.
"""


def format_context_chunks(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a clean, structured context string for the LLM prompt.
    """
    if not chunks:
        return "No context available."

    formatted_blocks = []
    for idx, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page", 1)
        chunk_id = chunk.get("chunk_id", f"chunk_{idx}")
        similarity = chunk.get("similarity_score", 0.0)
        text = chunk.get("chunk_text", "").strip()

        block = (
            f"--- CONTEXT CHUNK {idx} ---\n"
            f"Source Document: {source} (Page {page})\n"
            f"Chunk ID: {chunk_id} | Similarity: {similarity:.4f}\n"
            f"Content:\n{text}"
        )
        formatted_blocks.append(block)

    return "\n\n".join(formatted_blocks)


def build_user_prompt(query: str, formatted_context: str) -> str:
    """
    Build the structured user prompt containing context documents and the user's question.
    """
    return (
        f"CONTEXT DOCUMENTS:\n"
        f"{formatted_context}\n\n"
        f"USER QUESTION:\n"
        f"{query.strip()}\n\n"
        f"Please provide a grounded answer based strictly on the context above. "
        f"Include inline citations [Document: filename, Page: X]. "
        f"If the context is insufficient, reply with the exact required phrase."
    )


# ==============================================================================
# Grounded Answer Generation Pipeline
# ==============================================================================

def generate_grounded_answer(
    query: str,
    top_k: int = config.RETRIEVAL_TOP_K,
    collection: Optional[Any] = None,
    openai_client: Optional[OpenAI] = None,
    model: str = config.LLM_MODEL
) -> Dict[str, Any]:
    """
    Execute the end-to-end RAG pipeline for a user question.

    Distinguishes strictly between:
    1. Grounding fallback (insufficient evidence in context)
    2. Retrieval failure (database error)
    3. OpenAI API errors (e.g. invalid params, quota, connection)
    4. Programming / input validation errors

    Returns:
        Structured dictionary for UI presentation:
        {
            "query": str,
            "answer": str,
            "sources": list,
            "retrieved_chunks": list,
            "insufficient_evidence": bool,
            "error": Optional[str],
            "error_type": Optional[str],
            "model_used": str,
            "usage": dict
        }
    """
    # 0. Input validation
    if not query or not query.strip():
        return {
            "query": query,
            "answer": "Please enter a valid question.",
            "sources": [],
            "retrieved_chunks": [],
            "insufficient_evidence": False,
            "error": "Empty question submitted.",
            "error_type": "validation_error",
            "model_used": model,
            "usage": {}
        }

    # 1. Retrieve top-K relevant chunks from ChromaDB
    try:
        retrieved_chunks = query_vector_store(
            query_text=query,
            top_k=top_k,
            collection=collection,
            openai_client=openai_client
        )
    except Exception as e:
        logger.error(f"Retrieval error querying vector store: {str(e)}")
        return {
            "query": query,
            "answer": f"Database retrieval error: Unable to search indexed documents.",
            "sources": [],
            "retrieved_chunks": [],
            "insufficient_evidence": False,
            "error": f"Vector search failure: {str(e)}",
            "error_type": "retrieval_error",
            "model_used": model,
            "usage": {}
        }

    # 2. Handle empty retrieval (e.g. empty database or zero matches) -> genuine insufficient evidence
    if not retrieved_chunks:
        logger.info(f"No relevant chunks found in database for query: '{query}'")
        return {
            "query": query,
            "answer": config.INSUFFICIENT_EVIDENCE_PHRASE,
            "sources": [],
            "retrieved_chunks": [],
            "insufficient_evidence": True,
            "error": None,
            "error_type": None,
            "model_used": model,
            "usage": {}
        }

    # 3. Format context & construct prompt
    formatted_context = format_context_chunks(retrieved_chunks)
    user_prompt = build_user_prompt(query, formatted_context)

    # 4. Extract structured source list for UI citation cards
    sources = []
    seen_sources = set()
    for chunk in retrieved_chunks:
        source_key = f"{chunk['source']}_p{chunk['page']}"
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "similarity_score": chunk["similarity_score"],
                "snippet": chunk["chunk_text"][:250] + ("..." if len(chunk["chunk_text"]) > 250 else "")
            })

    # 5. Call OpenAI LLM with GPT-5 mini compatible parameters
    try:
        client = openai_client or get_openai_client()
        logger.info(f"Calling OpenAI LLM ({model}) with {len(retrieved_chunks)} context chunks...")

        # Note: GPT-5 mini uses max_completion_tokens (max_tokens is deprecated/unsupported)
        # Temperature is omitted to use model default
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=config.LLM_MAX_COMPLETION_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
        )

        answer_text = response.choices[0].message.content.strip()

        # 6. Evaluate if the answer returned the insufficient evidence fallback
        is_insufficient = (
            config.INSUFFICIENT_EVIDENCE_PHRASE.lower() in answer_text.lower() or
            "insufficient evidence" in answer_text.lower()
        )

        # If genuine insufficient evidence was declared by the LLM, suppress citation cards
        sources_to_return = [] if is_insufficient else sources

        usage_info = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0
        }

        return {
            "query": query,
            "answer": answer_text,
            "sources": sources_to_return,
            "retrieved_chunks": retrieved_chunks,
            "insufficient_evidence": is_insufficient,
            "error": None,
            "error_type": None,
            "model_used": model,
            "usage": usage_info
        }

    except OpenAIError as e:
        # Crucial: API errors are NOT insufficient evidence!
        err_msg = getattr(e, "message", str(e))
        logger.error(f"OpenAI API Error during generation: {type(e).__name__} - {err_msg}")
        return {
            "query": query,
            "answer": f"AI Service Communication Error ({type(e).__name__}): {err_msg}",
            "sources": [],
            "retrieved_chunks": retrieved_chunks,
            "insufficient_evidence": False,
            "error": f"OpenAI API error: {type(e).__name__} - {err_msg}",
            "error_type": "api_error",
            "model_used": model,
            "usage": {}
        }
    except Exception as e:
        logger.error(f"Unexpected error in RAG engine: {str(e)}")
        return {
            "query": query,
            "answer": f"An unexpected application error occurred: {str(e)}",
            "sources": [],
            "retrieved_chunks": retrieved_chunks,
            "insufficient_evidence": False,
            "error": f"Unexpected error: {str(e)}",
            "error_type": "application_error",
            "model_used": model,
            "usage": {}
        }
