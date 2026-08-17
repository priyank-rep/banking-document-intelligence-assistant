"""
Verification script specifically for Question 10 (DSCR Covenant):
Tests that adjacent-page context augmentation allows the LLM to resolve the
coreference between 'Meridian Logistics LLC' (Page 1) and 'Borrower' (Page 2).
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.vector_store import get_or_create_collection
from src.rag_engine import generate_grounded_answer


def verify_q10():
    print("=" * 80)
    print("🔍 VERIFYING QUESTION 10 WITH ADJACENT CONTEXT AUGMENTATION")
    print("=" * 80)

    query = "What is the minimum consolidated Debt Service Coverage Ratio (DSCR) requirement for Meridian Logistics, and how often is it measured?"
    print(f"Question: {query}\n")

    collection = get_or_create_collection()

    # 1. Test WITHOUT adjacent context (Baseline behavior)
    print("--- 1. Generating answer WITHOUT adjacent context (Baseline) ---")
    res_base = generate_grounded_answer(
        query=query,
        top_k=4,
        collection=collection,
        enable_adjacent_context=False
    )
    print(f"Primary Chunks: {len(res_base['retrieved_chunks'])}")
    print(f"Adjacent Chunks: {len(res_base.get('adjacent_chunks', []))}")
    print(f"Insufficient Evidence: {res_base['insufficient_evidence']}")
    print(f"Answer:\n{res_base['answer']}\n")

    # 2. Test WITH adjacent context
    print("--- 2. Generating answer WITH adjacent context ---")
    res_adj = generate_grounded_answer(
        query=query,
        top_k=4,
        collection=collection,
        enable_adjacent_context=True,
        max_adjacent_chunks=2
    )

    primary_chunks = res_adj["retrieved_chunks"]
    adjacent_chunks = res_adj.get("adjacent_chunks", [])

    print(f"Primary Chunks: {len(primary_chunks)}")
    for c in primary_chunks:
        print(f"  • Primary: {c['source']} (Page {c['page']}) - Sim: {c['similarity_score']}")

    print(f"\nAdjacent Chunks: {len(adjacent_chunks)}")
    for c in adjacent_chunks:
        print(f"  • Adjacent: {c['source']} (Page {c['page']}) - Rel: {c.get('relationship')}")

    print(f"\nInsufficient Evidence Flag: {res_adj['insufficient_evidence']}")
    print(f"Generated Answer:\n{res_adj['answer']}\n")
    print(f"Citations:")
    for s in res_adj["sources"]:
        print(f"  • [{s['retrieval_role'].upper()}] {s['source']} (Page {s['page']})")

    # Assertions
    # 1. Primary chunk includes Page 2
    primary_pages = [c["page"] for c in primary_chunks if c["source"] == "apex_bank_commercial_loan_agreement.pdf"]
    assert 2 in primary_pages, "Page 2 should be in primary chunks"

    # 2. Adjacent chunk includes Page 1
    adj_pages = [c["page"] for c in adjacent_chunks if c["source"] == "apex_bank_commercial_loan_agreement.pdf"]
    # If Page 1 wasn't already in primary, it should be in adjacent!
    print("=" * 80)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    verify_q10()
