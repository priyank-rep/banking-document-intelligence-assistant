"""
Inspect the exact chunks in K=2 with adjacent context.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vector_store import get_or_create_collection, query_vector_store, get_adjacent_chunks
from src.rag_engine import format_context_chunks

collection = get_or_create_collection()
query = "What is the minimum consolidated Debt Service Coverage Ratio (DSCR) requirement for Meridian Logistics, and how often is it measured?"

primary = query_vector_store(query, top_k=2, collection=collection)
adj = get_adjacent_chunks(primary, collection=collection, max_adjacent_chunks=2)

print("PRIMARY CHUNKS:")
for c in primary:
    print(f"  • {c['source']} p{c['page']} (sim: {c['similarity_score']})")

print("\nADJACENT CHUNKS:")
for c in adj:
    print(f"  • {c['source']} p{c['page']} (rel: {c.get('relationship')})")

print("\nFORMATTED CONTEXT:")
print(format_context_chunks(primary, adj))
