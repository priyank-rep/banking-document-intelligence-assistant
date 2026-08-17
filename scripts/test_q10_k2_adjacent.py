"""
Test q10 with K=2 and Adjacent Context enabled.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vector_store import get_or_create_collection
from src.rag_engine import generate_grounded_answer

collection = get_or_create_collection()
query = "What is the minimum consolidated Debt Service Coverage Ratio (DSCR) requirement for Meridian Logistics, and how often is it measured?"

res = generate_grounded_answer(
    query=query,
    top_k=2,
    collection=collection,
    enable_adjacent_context=True,
    max_adjacent_chunks=2
)

print("K=2 WITH ADJACENT CONTEXT ANSWER:")
print(res["answer"])
print("\nINSUFFICIENT EVIDENCE:", res["insufficient_evidence"])
print("\nSOURCES:")
for s in res["sources"]:
    print(f"  • [{s['retrieval_role'].upper()}] {s['source']} (Page {s['page']})")
