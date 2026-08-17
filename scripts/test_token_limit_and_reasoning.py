"""
Detailed diagnostic script for gpt-5-mini reasoning tokens and max_completion_tokens budget.
"""

import sys
from pathlib import Path
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.vector_store import get_or_create_collection, query_vector_store
from src.rag_engine import format_context_chunks, SYSTEM_PROMPT, build_user_prompt

user_query = (
    "What are HDFC Bank's key financial highlights for FY26? "
    "Include balance sheet size, deposits, advances, ROE, ROA, and net revenue."
)

collection = get_or_create_collection()
retrieved_chunks = query_vector_store(query_text=user_query, collection=collection, top_k=4)
formatted_context = format_context_chunks(retrieved_chunks)
user_prompt = build_user_prompt(user_query, formatted_context)

client = OpenAI(api_key=config.get_openai_api_key())

print("=" * 80)
print("1. Testing with max_completion_tokens = 1000 (Current config.py value)")
print("=" * 80)
resp_1000 = client.chat.completions.create(
    model="gpt-5-mini",
    max_completion_tokens=1000,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
)
choice_1000 = resp_1000.choices[0]
usage_1000 = resp_1000.usage
print(f"Finish Reason: {choice_1000.finish_reason}")
print(f"Content: {repr(choice_1000.message.content)}")
print(f"Usage Details: {usage_1000}")
if hasattr(usage_1000, "completion_tokens_details"):
    print(f"Completion Tokens Details: {usage_1000.completion_tokens_details}")

print("\n" + "=" * 80)
print("2. Testing with max_completion_tokens = 3000")
print("=" * 80)
resp_3000 = client.chat.completions.create(
    model="gpt-5-mini",
    max_completion_tokens=3000,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
)
choice_3000 = resp_3000.choices[0]
usage_3000 = resp_3000.usage
print(f"Finish Reason: {choice_3000.finish_reason}")
print(f"Content:\n{choice_3000.message.content}")
print(f"Usage Details: {usage_3000}")
if hasattr(usage_3000, "completion_tokens_details"):
    print(f"Completion Tokens Details: {usage_3000.completion_tokens_details}")
