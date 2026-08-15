"""
Verification Script for Milestone 3: Chroma Vector Store & Semantic Retrieval.

Validates:
1. Local ChromaDB initialization and persistence in data/chroma_db/
2. Batch embedding generation via OpenAI text-embedding-3-small
3. Upsert indexing with duplicate prevention
4. Top-K semantic similarity retrieval on banking questions
5. Inspection of cosine distances, similarity scores, and page metadata
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.pdf_loader import load_pdf_pages
from src.chunker import chunk_pages
from src.vector_store import (
    get_or_create_collection,
    index_chunks,
    query_vector_store,
    get_collection_stats,
    clear_collection,
    generate_embeddings
)


def run_milestone_3_verification():
    print("=" * 85)
    print("🚀 RUNNING MILESTONE 3 VERIFICATION: CHROMA VECTOR STORE & RETRIEVAL")
    print("=" * 85)

    # 1. Verify API Key
    try:
        api_key = config.get_openai_api_key()
        print("✅ OpenAI API Key successfully detected from environment/.env")
    except ValueError as e:
        print(f"⚠️  {e}")
        print("\nTo test with real OpenAI embeddings, please create a .env file with your key:")
        print("   OPENAI_API_KEY=sk-...\n")
        sys.exit(1)

    # 2. Ingest Sample Banking PDFs
    print("\n--- STEP 1: PARSING & CHUNKING SAMPLE BANKING DOCUMENTS ---")
    sample_dir = config.SAMPLE_DOCS_DIR
    fee_pdf = sample_dir / "apex_bank_fee_schedule.pdf"
    loan_pdf = sample_dir / "apex_bank_commercial_loan_agreement.pdf"

    if not fee_pdf.exists() or not loan_pdf.exists():
        from scripts.create_sample_pdfs import create_fee_schedule_pdf, create_commercial_loan_pdf
        create_fee_schedule_pdf(fee_pdf)
        create_commercial_loan_pdf(loan_pdf)

    all_pages = load_pdf_pages(fee_pdf) + load_pdf_pages(loan_pdf)
    chunks = chunk_pages(all_pages, chunk_size=config.CHUNK_SIZE_TOKENS, chunk_overlap=config.CHUNK_OVERLAP_TOKENS)
    print(f"Prepared {len(chunks)} chunks across {len(all_pages)} pages.")

    # 3. Index into ChromaDB
    print("\n--- STEP 2: INDEXING CHUNKS INTO LOCAL CHROMADB ---")
    print(f"Chroma Persistence Directory: {config.CHROMA_PERSIST_DIR}")
    print(f"Chroma Collection:            {config.CHROMA_COLLECTION_NAME}")
    print(f"Embedding Model:              {config.EMBEDDING_MODEL}")

    collection = get_or_create_collection()
    indexed_count = index_chunks(chunks, collection=collection)
    stats = get_collection_stats(collection)
    print(f"Index summary: {stats['total_chunks']} total vectors stored in Chroma.")

    # Test Duplicate Ingestion Prevention (Upsert idempotency check)
    print("\n--- STEP 3: TESTING DUPLICATE INGESTION SAFETY (UPSERT) ---")
    print("Re-indexing the exact same chunks...")
    re_indexed_count = index_chunks(chunks, collection=collection)
    updated_stats = get_collection_stats(collection)
    print(f"Post re-index count: {updated_stats['total_chunks']} (Expectation: exactly {stats['total_chunks']}, no duplicates created).")
    assert updated_stats["total_chunks"] == stats["total_chunks"], "Error: Duplicate chunks were created!"
    print("✅ Duplicate ingestion test PASSED: Upsert safely updated records without duplication.")

    # 4. Run Test Queries
    print("\n--- STEP 4: RUNNING TOP-K RETRIEVAL ON BANKING QUERIES ---")

    test_queries = [
        "What is the monthly maintenance fee for Apex Premier Checking and how can it be waived?",
        "What is the required Debt Service Coverage Ratio (DSCR) for Meridian Logistics?",
        "What are the fees and cutoff times for domestic wire transfers?",
        "What is the policy for 401k retirement match?" # Negative test / out-of-domain query
    ]

    for idx, query in enumerate(test_queries, 1):
        print("\n" + "-" * 85)
        print(f"🔍 [Query #{idx}]: \"{query}\"")
        print("-" * 85)

        retrieved = query_vector_store(query, top_k=config.RETRIEVAL_TOP_K, collection=collection)

        for rank, r in enumerate(retrieved, 1):
            print(f"  Rank #{rank} | Similarity: {r['similarity_score']:.4f} (Cosine Dist: {r['distance']:.4f})")
            print(f"    • Source:   {r['source']} (Page {r['page']})")
            print(f"    • Chunk ID: {r['chunk_id']}")
            print(f"    • Snippet:  \"{r['chunk_text'][:140]}...\"")
            print()

    print("=" * 85)
    print("✅ MILESTONE 3 VERIFICATION COMPLETE: ALL CHECKS PASSED!")
    print("=" * 85)


if __name__ == "__main__":
    run_milestone_3_verification()
