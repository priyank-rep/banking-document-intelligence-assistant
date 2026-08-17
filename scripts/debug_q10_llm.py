"""
Debug script to inspect GPT-5 mini's exact reasoning/response on Question 10.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.vector_store import get_or_create_collection, query_vector_store, get_adjacent_chunks, get_openai_client
from src.rag_engine import format_context_chunks, build_user_prompt, SYSTEM_PROMPT

collection = get_or_create_collection()
query = "What is the minimum consolidated Debt Service Coverage Ratio (DSCR) requirement for Meridian Logistics, and how often is it measured?"

primary = query_vector_store(query, top_k=4, collection=collection)
adj = get_adjacent_chunks(primary, collection=collection, max_adjacent_chunks=2)

formatted = format_context_chunks(primary, adj)
user_prompt = build_user_prompt(query, formatted)

client = get_openai_client()
response = client.chat.completions.create(
    model="gpt-5-mini",
    max_completion_tokens=1000,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
)

print("RESPONSE CONTENT:")
print(response.choices[0].message.content)
print("\nUSAGE:")
print(response.usage)
