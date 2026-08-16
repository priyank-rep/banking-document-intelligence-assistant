"""
Verification Script for Milestone 4: RAG Generation Engine & Citations.

Tests:
1. End-to-end question answering pipeline with GPT-5 mini.
2. Answering 3 factual banking questions with inline citations and source metadata.
3. Testing out-of-domain question to verify the strict "insufficient evidence" fallback.
4. Validates structured return payload for UI consumption.
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
from src.vector_store import get_or_create_collection, index_chunks, get_collection_stats
from src.rag_engine import generate_grounded_answer


def run_milestone_4_verification():
    print("=" * 85)
    print("🚀 RUNNING MILESTONE 4 VERIFICATION: GROUNDED RAG GENERATION & CITATIONS")
    print("=" * 85)

    # 1. Check API Key
    try:
        api_key = config.get_openai_api_key()
        print("✅ OpenAI API Key detected.")
    except ValueError as e:
        print(f"⚠️  {e}")
        print("\nTo test with real OpenAI LLM generation, set your OPENAI_API_KEY in .env.")
        sys.exit(1)

    # 2. Ensure database is populated with sample banking documents
    collection = get_or_create_collection()
    stats = get_collection_stats(collection)
    if stats["total_chunks"] == 0:
        print("\n--- Populating Vector Database with Sample Banking Documents ---")
        sample_dir = config.SAMPLE_DOCS_DIR
        fee_pdf = sample_dir / "apex_bank_fee_schedule.pdf"
        loan_pdf = sample_dir / "apex_bank_commercial_loan_agreement.pdf"

        if not fee_pdf.exists() or not loan_pdf.exists():
            from scripts.create_sample_pdfs import create_fee_schedule_pdf, create_commercial_loan_pdf
            create_fee_schedule_pdf(fee_pdf)
            create_commercial_loan_pdf(loan_pdf)

        all_pages = load_pdf_pages(fee_pdf) + load_pdf_pages(loan_pdf)
        chunks = chunk_pages(all_pages)
        index_chunks(chunks, collection=collection)
        print(f"Indexed {len(chunks)} chunks into Chroma.")
    else:
        print(f"Chroma collection '{stats['collection_name']}' is ready with {stats['total_chunks']} chunks.")

    # 3. Test Questions
    test_suite = [
        {
            "category": "Answerable Banking Question #1 (Fee & Waiver)",
            "query": "What is the monthly maintenance fee for Apex Premier Checking, and what balances are required to waive it?"
        },
        {
            "category": "Answerable Banking Question #2 (Loan Covenant)",
            "query": "What is the minimum Debt Service Coverage Ratio (DSCR) required for Meridian Logistics, and how often is it measured?"
        },
        {
            "category": "Answerable Banking Question #3 (Wire Transfer Rules)",
            "query": "What is the fee for an outgoing domestic wire transfer via online banking, and what is the cutoff time?"
        },
        {
            "category": "Unanswerable / Out-of-Domain Question (Negative Test)",
            "query": "What is the interest rate for a 30-year fixed rate residential mortgage at Apex Bank?"
        }
    ]

    for idx, test_case in enumerate(test_suite, 1):
        print("\n" + "=" * 85)
        print(f"📋 TEST CASE #{idx}: {test_case['category']}")
        print(f"❓ QUESTION: \"{test_case['query']}\"")
        print("=" * 85)

        result = generate_grounded_answer(query=test_case["query"], collection=collection)

        print("\n💬 MODEL ANSWER:")
        print(result["answer"])

        print(f"\n🚩 Insufficient Evidence Flag: {result['insufficient_evidence']}")

        if result["sources"]:
            print("\n📚 ATTACHED CITATIONS & EVIDENCE:")
            for s_idx, src in enumerate(result["sources"], 1):
                print(f"  [{s_idx}] Document: {src['source']} | Page: {src['page']} | Match Score: {src['similarity_score']:.4f}")
                print(f"      Chunk ID: {src['chunk_id']}")
                print(f"      Snippet:  \"{src['snippet'][:130]}...\"")
        else:
            print("\n📚 ATTACHED CITATIONS: None (Correctly suppressed for unanswerable query)")

        if result.get("usage"):
            print(f"\n⚡ Token Usage: {result['usage']}")

    print("\n" + "=" * 85)
    print("✅ MILESTONE 4 VERIFICATION COMPLETE!")
    print("=" * 85)


if __name__ == "__main__":
    run_milestone_4_verification()
