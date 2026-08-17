"""
Run exact HDFC multi-metric query and report exact output, citations, and telemetry.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.vector_store import get_or_create_collection
from src.rag_engine import generate_grounded_answer

user_query = "What are HDFC Bank's key financial highlights for FY26? Include balance sheet size, deposits, advances, ROE, ROA, and net revenue."

collection = get_or_create_collection()
res = generate_grounded_answer(query=user_query, collection=collection)

print("=" * 90)
print("🏦 HDFC MULTI-METRIC QUERY RESULT (POST-FONT NORMALIZATION)")
print("=" * 90)
print(f"Query: \"{user_query}\"\n")
print(f"Answer:\n{res['answer']}\n")
print(f"Insufficient Evidence: {res['insufficient_evidence']}")
print(f"Number of Citations: {len(res['sources'])}")
print(f"Usage: {res['usage']}")
print("\n--- Attached Citations ---")
for idx, s in enumerate(res["sources"], 1):
    print(f"[{idx}] {s['source']} (Page {s['page']}) - Role: {s['retrieval_role']} - Match: {s['similarity_score']:.4f}")
    print(f"    Snippet: {repr(s['snippet'][:180])}")
