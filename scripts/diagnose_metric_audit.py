"""
Diagnostic script to inspect metric-by-metric presence in retrieved chunks and reasoning.
"""

import sys
from pathlib import Path
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.vector_store import get_or_create_collection, query_vector_store
from src.rag_engine import format_context_chunks

user_query = (
    "What are HDFC Bank's key financial highlights for FY26? "
    "Include balance sheet size, deposits, advances, ROE, ROA, and net revenue."
)

collection = get_or_create_collection()
retrieved_chunks = query_vector_store(query_text=user_query, collection=collection, top_k=4)

print("=" * 80)
print("RETRIEVED CHUNKS CONTENT:")
print("=" * 80)
for idx, c in enumerate(retrieved_chunks, 1):
    print(f"\n--- Chunk {idx}: Page {c['page']} (ID: {c['chunk_id']}) ---")
    print(c["chunk_text"])

formatted_context = format_context_chunks(retrieved_chunks)

client = OpenAI(api_key=config.get_openai_api_key())

prompt = f"""You are analyzing retrieved document context for a banking assistant.

USER QUESTION:
{user_query}

RETRIEVED CONTEXT:
{formatted_context}

TASK:
1. For EACH of the 6 requested metrics, inspect the retrieved context and answer:
   a) Balance Sheet Size for FY26
   b) Deposits for FY26
   c) Advances for FY26
   d) ROE for FY26
   e) ROA for FY26
   f) Net Revenue for FY26

   For each metric, state:
   - Status: [EXPLICITLY FOUND / PARTIALLY FOUND / NOT FOUND]
   - Value & Unit (if found)
   - Source Page & Quote snippet (if found)
   - If NOT found, confirm it is absent.

2. Explain why the current system prompt (which enforces "If the context does not contain sufficient evidence to answer the question, or if key facts are missing, you must respond strictly with: I cannot find sufficient evidence in the uploaded documents to answer this question.") triggers a complete refusal when 1 of the 6 requested metrics is missing.
"""

resp = client.chat.completions.create(
    model="gpt-5-mini",
    max_completion_tokens=3000,
    messages=[{"role": "user", "content": prompt}]
)

print("\n" + "=" * 80)
print("DIAGNOSTIC METRIC-BY-METRIC EXTRACTION & REASONING ANALYSIS:")
print("=" * 80)
print(resp.choices[0].message.content)
