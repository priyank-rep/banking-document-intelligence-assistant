"""
Comprehensive test suite for Evidence-Aware Partial Answering in src/rag_engine.py.

Tests:
Case A: All requested facts supported (e.g. Premier Checking fee & waiver rules).
Case B: Some supported + some missing (e.g. HDFC 6-metric question where 5 are present and ROA is missing).
Case C: None supported (e.g. Out-of-domain 401(k) matching question).
Case D: API error handling (e.g. invalid API key or connection error).
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from openai import OpenAIError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.vector_store import get_or_create_collection
from src.rag_engine import generate_grounded_answer


def run_grounding_tests():
    print("=" * 90)
    print("🧪 RUNNING EVIDENCE-AWARE PARTIAL ANSWERING TEST SUITE")
    print("=" * 90)

    collection = get_or_create_collection()

    # --------------------------------------------------------------------------
    # Case A: All Requested Facts Supported
    # --------------------------------------------------------------------------
    print("\n--- CASE A: All Requested Facts Supported ---")
    query_a = "What is the monthly maintenance fee for Apex Premier Checking and what is the minimum daily balance to waive it?"
    print(f"Query: \"{query_a}\"")
    res_a = generate_grounded_answer(query=query_a, collection=collection)
    
    print(f"Answer:\n{res_a['answer']}\n")
    print(f"Insufficient Evidence: {res_a['insufficient_evidence']}")
    print(f"Citations Count: {len(res_a['sources'])}")
    print(f"Token Usage: {res_a['usage']}")

    assert res_a["insufficient_evidence"] is False, "Case A should NOT be marked insufficient."
    assert len(res_a["sources"]) > 0, "Case A must include source citations."
    assert "$25" in res_a["answer"] or "25.00" in res_a["answer"], "Case A must include the $25 fee."
    print("✅ Case A Passed: Fully supported question answered and cited accurately.")

    # --------------------------------------------------------------------------
    # Case B: Some Supported + Some Missing (The Core HDFC Multi-Metric Scenario)
    # --------------------------------------------------------------------------
    print("\n--- CASE B: Some Supported + Some Missing (HDFC Multi-Metric) ---")
    query_b = "What are HDFC Bank's key financial highlights for FY26? Include balance sheet size, deposits, advances, ROE, ROA, and net revenue."
    print(f"Query: \"{query_b}\"")
    res_b = generate_grounded_answer(query=query_b, collection=collection)

    print(f"Answer:\n{res_b['answer']}\n")
    print(f"Insufficient Evidence: {res_b['insufficient_evidence']}")
    print(f"Citations Count: {len(res_b['sources'])}")
    print(f"Token Usage: {res_b['usage']}")

    # Assertions for Case B:
    # 1. Supported facts are returned
    assert ("43,64,886" in res_b["answer"] or "4364886" in res_b["answer"] or "43,64" in res_b["answer"]), "Supported Balance Sheet Size must be in answer."
    assert ("31,05,250" in res_b["answer"] or "3105250" in res_b["answer"] or "31,05" in res_b["answer"]), "Supported Deposits must be in answer."
    assert ("14.3" in res_b["answer"]), "Supported ROE (14.3%) must be in answer."
    
    # 2. Missing fact is explicitly declared
    assert ("roa" in res_b["answer"].lower() or "return on assets" in res_b["answer"].lower()), "ROA must be addressed."
    assert ("not found" in res_b["answer"].lower() or "not available" in res_b["answer"].lower() or "not provided" in res_b["answer"].lower() or "not explicitly" in res_b["answer"].lower()), "ROA must be explicitly declared as missing/not found."
    
    # 3. Insufficient evidence is NOT set to True (since 5/6 facts are present)
    assert res_b["insufficient_evidence"] is False, "Case B must NOT trigger global insufficient_evidence=True."
    
    # 4. Citations are returned
    assert len(res_b["sources"]) > 0, "Case B must return supporting source citations."
    print("✅ Case B Passed: Partial evidence answered, missing metric declared, and citations preserved.")

    # --------------------------------------------------------------------------
    # Case C: None Supported (Out-of-Domain / Unanswerable)
    # --------------------------------------------------------------------------
    print("\n--- CASE C: None Supported (Out-of-Domain Unanswerable) ---")
    query_c = "What is the average cruising speed of a Boeing 747 aircraft in kilometers per hour?"
    print(f"Query: \"{query_c}\"")
    res_c = generate_grounded_answer(query=query_c, collection=collection)

    print(f"Answer:\n{res_c['answer']}\n")
    print(f"Insufficient Evidence: {res_c['insufficient_evidence']}")
    print(f"Citations Count: {len(res_c['sources'])}")
    print(f"Token Usage: {res_c['usage']}")

    assert res_c["insufficient_evidence"] is True, "Case C must be marked as insufficient evidence."
    assert config.INSUFFICIENT_EVIDENCE_PHRASE.lower() in res_c["answer"].lower(), "Case C must return exact refusal phrase."
    assert len(res_c["sources"]) == 0, "Case C must suppress citations on full refusal."
    print("✅ Case C Passed: Fully unanswerable query correctly refused.")

    # --------------------------------------------------------------------------
    # Case D: API / Service Error
    # --------------------------------------------------------------------------
    print("\n--- CASE D: API Error Handling ---")
    mock_client = MagicMock()
    # Provide valid 1536-dim embedding so vector retrieval reaches chat.completions
    mock_embed_resp = MagicMock()
    mock_embed_item = MagicMock()
    mock_embed_item.embedding = [0.01] * 1536
    mock_embed_resp.data = [mock_embed_item]
    mock_client.embeddings.create.return_value = mock_embed_resp

    # Make chat.completions raise OpenAIError
    mock_client.chat.completions.create.side_effect = OpenAIError("Simulated API rate limit or network timeout")

    res_d = generate_grounded_answer(
        query="What is the monthly maintenance fee for Apex Premier Checking?",
        collection=collection,
        openai_client=mock_client
    )

    print(f"Answer: {res_d['answer']}")
    print(f"Error Type: {res_d['error_type']}")
    print(f"Insufficient Evidence: {res_d['insufficient_evidence']}")

    assert res_d["error_type"] == "api_error", "Case D must classify as api_error."
    assert res_d["insufficient_evidence"] is False, "API error must NOT be classified as insufficient evidence."
    assert "OpenAIError" in res_d["answer"] or "API" in res_d["answer"]
    print("✅ Case D Passed: API errors gracefully handled and distinctly classified.")

    print("\n" + "=" * 90)
    print("🎉 ALL 4 GROUNDING & ERROR HANDLING TEST CASES PASSED!")
    print("=" * 90)


if __name__ == "__main__":
    run_grounding_tests()
