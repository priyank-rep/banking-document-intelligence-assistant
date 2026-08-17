"""
Test reasoning tokens on HDFC query with 2500 and 3000 max completion tokens.
"""

import sys
from pathlib import Path
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.vector_store import get_or_create_collection, query_vector_store
from src.rag_engine import format_context_chunks, SYSTEM_PROMPT, build_user_prompt

user_query = "What are HDFC Bank's key financial highlights for FY26? Include balance sheet size, deposits, advances, ROE, ROA, and net revenue."
collection = get_or_create_collection()
chunks = query_vector_store(query_text=user_query, collection=collection, top_k=4)
context = format_context_chunks(chunks)
prompt = build_user_prompt(user_query, context)

client = OpenAI(api_key=config.get_openai_api_key())

for budget in [2500, 3000]:
    print(f"\n--- Testing with max_completion_tokens = {budget} ---")
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        max_completion_tokens=budget,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    c = resp.choices[0]
    print(f"Finish Reason: {c.finish_reason}")
    print(f"Content:\n{c.message.content}")
    print(f"Usage: {resp.usage}")
