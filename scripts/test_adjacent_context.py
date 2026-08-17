"""
Unit tests for Adjacent-Page Context Augmentation.

Verifies:
1. Page 2 retrieving Page 1 as adjacent context.
2. Page 1 retrieving Page 2 as adjacent context.
3. Strict document boundaries: no cross-document adjacent context.
4. Duplicate prevention: never duplicate chunks already in primary retrieval.
5. Metadata integrity: retrieval_role is explicitly 'primary' or 'adjacent'.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.pdf_loader import load_pdf_pages
from src.chunker import chunk_pages
from src.vector_store import (
    get_or_create_collection,
    index_chunks,
    clear_collection,
    get_adjacent_chunks,
    query_vector_store
)


def run_tests():
    print("=" * 75)
    print("🧪 RUNNING ADJACENT CONTEXT UNIT TESTS")
    print("=" * 75)

    # 1. Setup sample collection
    sample_dir = config.SAMPLE_DOCS_DIR
    fee_pdf = sample_dir / "apex_bank_fee_schedule.pdf"
    loan_pdf = sample_dir / "apex_bank_commercial_loan_agreement.pdf"

    collection = get_or_create_collection()
    clear_collection(collection)

    all_pages = load_pdf_pages(fee_pdf) + load_pdf_pages(loan_pdf)
    chunks = chunk_pages(all_pages, chunk_size=500, chunk_overlap=100)
    index_chunks(chunks, collection=collection)

    print(f"✅ Indexed {len(chunks)} chunks across fee schedule (2 pages) and loan agreement (3 pages).")

    # TEST 1: Page 2 retrieves Page 1 as adjacent context
    print("\n--- Test 1: Page 2 retrieves Page 1 as adjacent context ---")
    mock_primary_p2 = [{
        "chunk_id": "apex_bank_commercial_loan_agreement_p2_c0",
        "source": "apex_bank_commercial_loan_agreement.pdf",
        "page": 2,
        "chunk_text": "Article II covenants...",
        "token_count": 350,
        "similarity_score": 0.85,
        "retrieval_role": "primary"
    }]

    adj_p2 = get_adjacent_chunks(mock_primary_p2, collection=collection, max_adjacent_chunks=2)
    adj_pages_p2 = [c["page"] for c in adj_p2]
    adj_sources_p2 = [c["source"] for c in adj_p2]

    assert 1 in adj_pages_p2, f"Expected Page 1 in adjacent chunks for Page 2, got: {adj_pages_p2}"
    assert all(s == "apex_bank_commercial_loan_agreement.pdf" for s in adj_sources_p2)
    print(f"✅ Test 1 Passed: Page 2 successfully retrieved adjacent pages {adj_pages_p2} from the same document.")

    # TEST 2: Page 1 retrieves Page 2 as adjacent context
    print("\n--- Test 2: Page 1 retrieves Page 2 as adjacent context ---")
    mock_primary_p1 = [{
        "chunk_id": "apex_bank_fee_schedule_p1_c0",
        "source": "apex_bank_fee_schedule.pdf",
        "page": 1,
        "chunk_text": "Section 1 checking fees...",
        "token_count": 300,
        "similarity_score": 0.82,
        "retrieval_role": "primary"
    }]

    adj_p1 = get_adjacent_chunks(mock_primary_p1, collection=collection, max_adjacent_chunks=2)
    adj_pages_p1 = [c["page"] for c in adj_p1]
    assert 2 in adj_pages_p1, f"Expected Page 2 in adjacent chunks for Page 1, got: {adj_pages_p1}"
    assert all(c["source"] == "apex_bank_fee_schedule.pdf" for c in adj_p1)
    print(f"✅ Test 2 Passed: Page 1 successfully retrieved adjacent page {adj_pages_p1}.")

    # TEST 3: No cross-document adjacent context
    print("\n--- Test 3: No cross-document leakage ---")
    for chunk in adj_p1:
        assert chunk["source"] == "apex_bank_fee_schedule.pdf", f"Cross-document leak detected: {chunk['source']}"
    for chunk in adj_p2:
        assert chunk["source"] == "apex_bank_commercial_loan_agreement.pdf", f"Cross-document leak detected: {chunk['source']}"
    print("✅ Test 3 Passed: Zero cross-document leakage observed.")

    # TEST 4: No duplicate context entries
    print("\n--- Test 4: No duplicate context entries when adjacent is already in primary ---")
    mock_primary_both = [
        {
            "chunk_id": "apex_bank_commercial_loan_agreement_p1_c0",
            "source": "apex_bank_commercial_loan_agreement.pdf",
            "page": 1,
            "chunk_text": "Page 1 text",
            "retrieval_role": "primary"
        },
        {
            "chunk_id": "apex_bank_commercial_loan_agreement_p2_c0",
            "source": "apex_bank_commercial_loan_agreement.pdf",
            "page": 2,
            "chunk_text": "Page 2 text",
            "retrieval_role": "primary"
        }
    ]
    # Page 1's neighbor is Page 2 (already in primary), Page 2's neighbor is Page 1 (already in primary) and Page 3 (new)
    adj_both = get_adjacent_chunks(mock_primary_both, collection=collection, max_adjacent_chunks=2)
    adj_ids = [c["chunk_id"] for c in adj_both]
    assert "apex_bank_commercial_loan_agreement_p1_c0" not in adj_ids, "Duplicate Page 1 added to adjacent"
    assert "apex_bank_commercial_loan_agreement_p2_c0" not in adj_ids, "Duplicate Page 2 added to adjacent"
    assert "apex_bank_commercial_loan_agreement_p3_c0" in adj_ids, "Page 3 should be added as adjacent to Page 2"
    print(f"✅ Test 4 Passed: Deduplication works properly. Retrieved only non-primary adjacent: {adj_ids}")

    # TEST 5: Metadata correctness
    print("\n--- Test 5: Metadata role tagging ---")
    assert all(c["retrieval_role"] == "adjacent" for c in adj_both)
    assert all(c["similarity_score"] is None for c in adj_both)
    print("✅ Test 5 Passed: All adjacent chunks have retrieval_role='adjacent' and similarity_score=None.")

    print("\n" + "=" * 75)
    print("🎉 ALL 5 ADJACENT CONTEXT UNIT TESTS PASSED!")
    print("=" * 75)


if __name__ == "__main__":
    run_tests()
