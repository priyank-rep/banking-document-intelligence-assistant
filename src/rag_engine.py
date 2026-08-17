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
    get_adjacent_chunks,
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

STRICT GROUNDING & PARTIAL-ANSWERING RULES:
1. Answer ONLY using information explicitly stated in the provided Context Chunks (both directly retrieved Primary Chunks and Supporting Adjacent Pages from the same document).
2. Use Supporting Adjacent Pages to resolve entity definitions, preambles, or context referenced in primary clauses (e.g., connecting a defined 'Borrower' to the company name).
3. Do NOT use outside knowledge, prior assumptions, or extrapolate beyond the text.
4. Do NOT guess or invent numbers, percentages, fees, deadlines, names, or legal terms.
5. When answering supported facts, cite your sources inline using the format [Document: filename, Page: X] where the supporting facts appear.
6. HANDLING MULTI-PART QUESTIONS & PARTIAL EVIDENCE:
   - FULLY SUPPORTED: If all requested facts/metrics are supported in the context, answer them completely with inline citations.
   - PARTIALLY SUPPORTED: If some requested facts/metrics are supported but other requested items are missing or not found in the context:
     * Answer all supported facts/metrics and provide inline citations for each.
     * Explicitly state which specific requested fact(s) or metric(s) were NOT found in the retrieved context (e.g., "[Item X] was not found in the retrieved evidence for this question, so I have not inferred a value.").
     * Do NOT refuse the entire question when useful evidence is present.
   - COMPLETELY UNSUPPORTED: If NONE of the requested information is supported by the context, or if the question is entirely out-of-domain / unanswerable from the documents, you MUST respond EXACTLY with:
     "{config.INSUFFICIENT_EVIDENCE_PHRASE}"
7. Do NOT provide speculative answers, guesses, or conversational apologies when returning the insufficient evidence phrase.
"""


def format_context_chunks(
    primary_chunks: List[Dict[str, Any]],
    adjacent_chunks: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Format primary retrieved chunks and supporting adjacent chunks into a clean,
    structured context string for the LLM prompt.
    """
    if not primary_chunks and not adjacent_chunks:
        return "No context available."

    formatted_blocks = []

    # 1. Primary retrieved chunks (direct semantic matches)
    for idx, chunk in enumerate(primary_chunks or [], 1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page", 1)
        chunk_id = chunk.get("chunk_id", f"chunk_{idx}")
        similarity = chunk.get("similarity_score", 0.0)
        text = chunk.get("chunk_text", "").strip()

        block = (
            f"--- PRIMARY CONTEXT CHUNK {idx} (Direct Match) ---\n"
            f"Source Document: {source} (Page {page})\n"
            f"Chunk ID: {chunk_id} | Similarity: {similarity:.4f}\n"
            f"Content:\n{text}"
        )
        formatted_blocks.append(block)

    # 2. Supporting adjacent chunks (neighboring pages from the same document)
    if adjacent_chunks:
        for idx, chunk in enumerate(adjacent_chunks, 1):
            source = chunk.get("source", "unknown")
            page = chunk.get("page", 1)
            chunk_id = chunk.get("chunk_id", f"adj_{idx}")
            rel = chunk.get("relationship", f"Adjacent to Page {chunk.get('adjacent_to_page', 'N')}")
            text = chunk.get("chunk_text", "").strip()

            block = (
                f"--- SUPPORTING CONTEXT {idx} (Adjacent Supporting Page) ---\n"
                f"Source Document: {source} (Page {page})\n"
                f"Relationship: {rel}\n"
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
    model: str = config.LLM_MODEL,
    enable_adjacent_context: Optional[bool] = None,
    max_adjacent_chunks: int = config.MAX_ADJACENT_CHUNKS
) -> Dict[str, Any]:
    """
    Execute the end-to-end RAG pipeline for a user question.

    Distinguishes strictly between:
    1. Grounding fallback (insufficient evidence in context)
    2. Retrieval failure (database error)
    3. OpenAI API errors (e.g. invalid params, quota, connection)
    4. Programming / input validation errors
    """
    # 0. Input validation
    if not query or not query.strip():
        return {
            "query": query,
            "answer": "Please enter a valid question.",
            "sources": [],
            "retrieved_chunks": [],
            "adjacent_chunks": [],
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
            "adjacent_chunks": [],
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
            "adjacent_chunks": [],
            "insufficient_evidence": True,
            "error": None,
            "error_type": None,
            "model_used": model,
            "usage": {}
        }

    # 3. Optional: Adjacent-page context augmentation
    use_adjacent = (
        enable_adjacent_context
        if enable_adjacent_context is not None
        else config.ENABLE_ADJACENT_CONTEXT
    )
    adjacent_chunks = []
    if use_adjacent:
        try:
            adjacent_chunks = get_adjacent_chunks(
                primary_chunks=retrieved_chunks,
                collection=collection,
                max_adjacent_chunks=max_adjacent_chunks
            )
            if adjacent_chunks:
                logger.info(
                    f"Augmented context with {len(adjacent_chunks)} adjacent page chunk(s) "
                    f"from same source document(s)."
                )
        except Exception as e:
            logger.warning(f"Error fetching adjacent chunks (proceeding with primary only): {e}")

    # 4. Format context & construct prompt
    formatted_context = format_context_chunks(retrieved_chunks, adjacent_chunks)
    user_prompt = build_user_prompt(query, formatted_context)

    # 5. Extract structured source list for UI citation cards
    sources = []
    seen_sources = set()

    for chunk in retrieved_chunks + adjacent_chunks:
        source_key = f"{chunk['source']}_p{chunk['page']}_{chunk.get('retrieval_role', 'primary')}"
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "retrieval_role": chunk.get("retrieval_role", "primary"),
                "similarity_score": chunk.get("similarity_score"),
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
        clean_answer = answer_text.strip()
        is_insufficient = (
            clean_answer == config.INSUFFICIENT_EVIDENCE_PHRASE or
            clean_answer.startswith(config.INSUFFICIENT_EVIDENCE_PHRASE) or
            (
                "insufficient evidence" in clean_answer.lower()
                and not any(tag in clean_answer for tag in ["[Document:", "[Page:", "(Page ", "Page:"])
                and len(clean_answer) < 250
            )
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
            "adjacent_chunks": adjacent_chunks,
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
            "adjacent_chunks": adjacent_chunks,
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
            "adjacent_chunks": adjacent_chunks,
            "insufficient_evidence": False,
            "error": f"Unexpected error: {str(e)}",
            "error_type": "application_error",
            "model_used": model,
            "usage": {}
        }
