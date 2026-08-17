"""
Diagnostic script to investigate the HDFC Bank FY26 Financial Highlights Query Failure.

Question:
"What are HDFC Bank's key financial highlights for FY26? Include balance sheet size, deposits, advances, ROE, ROA, and net revenue."

Investigates:
1. Top-K Chroma retrieval results for the exact query.
2. Content audit: Checks presence of all 6 requested metrics in retrieved chunks.
3. Generation behavior under production RAG system prompt vs diagnostic extraction prompt.
4. Root cause classification: Retrieval Recall vs. Context Formatting vs. Prompt Refusal.
"""

import sys
import json
import logging
from pathlib import Path
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.pdf_loader import load_pdf_pages
from src.chunker import chunk_pages
from src.vector_store import (
    get_or_create_collection,
    index_chunks,
    query_vector_store,
    get_adjacent_chunks,
    get_collection_stats
)
from src.rag_engine import generate_grounded_answer, format_context_chunks, SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose():
    print("=" * 90)
    print("🔍 HDFC FY26 FINANCIAL HIGHLIGHTS QUERY FAILURE DIAGNOSTIC")
    print("=" * 90)

    user_query = (
        "What are HDFC Bank's key financial highlights for FY26? "
        "Include balance sheet size, deposits, advances, ROE, ROA, and net revenue."
    )
    print(f"\nUser Query:\n\"{user_query}\"\n")

    # 1. Ensure HDFC PDF is indexed in Chroma
    collection = get_or_create_collection()
    stats = get_collection_stats(collection)
    print(f"Current Chroma chunks in collection: {stats['total_chunks']}")

    hdfc_pdf = config.SAMPLE_DOCS_DIR / "HDFC_FY26.pdf"
    if not hdfc_pdf.exists():
        print(f"❌ Error: {hdfc_pdf} not found.")
        return

    # Check if HDFC is indexed
    check_query = collection.get(where={"source": {"$eq": "HDFC_FY26.pdf"}}, limit=5)
    if not check_query["ids"]:
        print("\n[Indexing] HDFC_FY26.pdf is not in Chroma collection. Ingesting and indexing now...")
        pages = load_pdf_pages(hdfc_pdf, filename="HDFC_FY26.pdf")
        chunks = chunk_pages(pages)
        index_chunks(chunks, collection=collection)
        stats = get_collection_stats(collection)
        print(f"Indexing complete. Total collection chunks: {stats['total_chunks']}")
    else:
        print("HDFC_FY26.pdf is already indexed in Chroma.")

    # 2. Run Top-K retrieval
    print(f"\n--- Step 1: Retrieving Top-K ({config.RETRIEVAL_TOP_K}) Chunks ---")
    retrieved_chunks = query_vector_store(
        query_text=user_query,
        collection=collection,
        top_k=config.RETRIEVAL_TOP_K
    )

    adjacent_chunks = []
    if config.ENABLE_ADJACENT_CONTEXT:
        adjacent_chunks = get_adjacent_chunks(
            collection=collection,
            primary_chunks=retrieved_chunks,
            max_adjacent=config.MAX_ADJACENT_CHUNKS
        )

    for idx, c in enumerate(retrieved_chunks, 1):
        print(f"\n[Retrieved Chunk #{idx}]")
        print(f"  • Source:     {c['source']}")
        print(f"  • Page:       {c['page']}")
        print(f"  • Chunk ID:   {c['chunk_id']}")
        print(f"  • Score:      {c['similarity_score']:.4f} (Cosine Dist: {c['distance']:.4f})")
        print(f"  • Token Count:{c['token_count']}")
        print(f"  • Text Snippet (First 250 chars):\n    {repr(c['chunk_text'][:250])}")

    # 3. Content Audit of Retrieved Chunks
    all_context_text = "\n\n".join([c["chunk_text"] for c in retrieved_chunks + adjacent_chunks])
    
    print("\n" + "=" * 90)
    print("--- Step 2: Content Audit for Requested Metrics in Retrieved Chunks ---")
    print("=" * 90)

    requested_metrics = {
        "balance sheet size": ["balance sheet", "43,64,886", "4364886", "balance sheet size"],
        "deposits": ["deposits", "31,05,250", "3105250"],
        "advances": ["advances", "29,37,166", "2937166"],
        "ROE (Return on Equity)": ["return on equity", "roe", "14.3", "14.3%"],
        "ROA (Return on Assets)": ["return on assets", "roa", "1.94", "1.94%"],
        "net revenue": ["net revenue", "net revenues", "1,91,218", "191218", "revenue"]
    }

    found_summary = {}
    for metric, terms in requested_metrics.items():
        found = False
        matching_term = None
        for term in terms:
            if term.lower() in all_context_text.lower():
                found = True
                matching_term = term
                break
        found_summary[metric] = (found, matching_term)
        status_icon = "✅ FOUND" if found else "❌ MISSING"
        term_detail = f"(matched term: '{matching_term}')" if found else "(no search term matched in top-K)"
        print(f"  • {metric:<25}: {status_icon} {term_detail}")

    # 4. Production RAG Generation Call
    print("\n" + "=" * 90)
    print("--- Step 3: Production RAG Execution with System Prompt ---")
    print("=" * 90)

    prod_result = generate_grounded_answer(query=user_query, collection=collection)
    print(f"Production Response:\n{prod_result['answer']}")
    print(f"\nInsufficient Evidence Flag: {prod_result.get('insufficient_evidence')}")
    print(f"Usage: {prod_result.get('usage')}")

    # 5. Diagnostic Direct LLM Prompt (Diagnostic Extraction)
    print("\n" + "=" * 90)
    print("--- Step 4: Diagnostic GPT-5 mini Extraction Test ---")
    print("=" * 90)

    client = OpenAI(api_key=config.get_openai_api_key())
    formatted_context = format_context_chunks(retrieved_chunks, adjacent_chunks)

    diagnostic_prompt = (
        "You are an expert banking analyst inspecting extracted document context.\n\n"
        "USER QUESTION:\n"
        f"{user_query}\n\n"
        "RETRIEVED CONTEXT CHUNKS:\n"
        f"{formatted_context}\n\n"
        "INSTRUCTIONS:\n"
        "1. For each requested metric (balance sheet size, deposits, advances, ROE, ROA, net revenue):\n"
        "   - If explicitly stated in the context for FY26, state the exact number, unit, and source page.\n"
        "   - If NOT stated or only partially stated, explicitly state: 'NOT FOUND in retrieved context'.\n"
        "2. Explain why a conservative grounding system prompt might refuse this question."
    )

    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "user", "content": diagnostic_prompt}
        ],
        max_completion_tokens=1000
    )
    diagnostic_answer = resp.choices[0].message.content
    print(f"Diagnostic LLM Output:\n{diagnostic_answer}")


if __name__ == "__main__":
    diagnose()
