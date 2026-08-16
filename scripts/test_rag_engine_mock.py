"""
Unit Test for RAG Engine Logic (Offline / Mock Mode).

Tests:
1. Context chunk formatting and structured prompt generation.
2. Source deduplication and page metadata extraction.
3. Insufficient evidence detection logic.
4. Payload structure validation for UI consumption.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.rag_engine import (
    format_context_chunks,
    build_user_prompt,
    generate_grounded_answer,
    SYSTEM_PROMPT
)


def test_rag_engine_components():
    print("=" * 80)
    print("🧪 TESTING RAG ENGINE LOGIC & PROMPT CONSTRUCTION (OFFLINE)")
    print("=" * 80)

    # 1. Test Context Formatting
    sample_chunks = [
        {
            "chunk_id": "fee_schedule_p1_c0",
            "source": "fee_schedule.pdf",
            "page": 1,
            "chunk_text": "Apex Premier Checking monthly fee is $25.00, waived with $5,000 balance.",
            "token_count": 45,
            "similarity_score": 0.8850,
            "distance": 0.1150
        },
        {
            "chunk_id": "loan_p2_c0",
            "source": "loan_agreement.pdf",
            "page": 2,
            "chunk_text": "Borrower shall maintain DSCR not less than 1.25:1.00.",
            "token_count": 40,
            "similarity_score": 0.8200,
            "distance": 0.1800
        }
    ]

    formatted_context = format_context_chunks(sample_chunks)
    print("1. Formatted Context Output:")
    print("-" * 50)
    print(formatted_context)
    print("-" * 50)

    assert "--- CONTEXT CHUNK 1 ---" in formatted_context
    assert "fee_schedule.pdf (Page 1)" in formatted_context
    assert "loan_agreement.pdf (Page 2)" in formatted_context
    print("✅ Context formatting verified.")

    # 2. Test User Prompt Construction
    user_prompt = build_user_prompt("What is the fee for premier checking?", formatted_context)
    assert "CONTEXT DOCUMENTS:" in user_prompt
    assert "USER QUESTION:" in user_prompt
    assert "What is the fee for premier checking?" in user_prompt
    print("✅ User prompt construction verified.")

    # 3. Test Mock LLM Positive Answer Flow
    mock_openai_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "The monthly fee for Apex Premier Checking is $25.00 [Document: fee_schedule.pdf, Page: 1]."
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 150
    mock_response.usage.completion_tokens = 25
    mock_response.usage.total_tokens = 175
    mock_openai_client.chat.completions.create.return_value = mock_response

    mock_collection = MagicMock()
    # Mock vector query returning sample chunk
    mock_collection.query.return_value = {
        "ids": [["fee_schedule_p1_c0"]],
        "documents": [["Apex Premier Checking monthly fee is $25.00, waived with $5,000 balance."]],
        "metadatas": [[{"source": "fee_schedule.pdf", "page": 1, "chunk_id": "fee_schedule_p1_c0", "token_count": 45}]],
        "distances": [[0.1150]]
    }

    # Mock embeddings generation for query
    mock_emb_response = MagicMock()
    mock_emb_item = MagicMock()
    mock_emb_item.embedding = [0.1] * 1536
    mock_emb_response.data = [mock_emb_item]
    mock_openai_client.embeddings.create.return_value = mock_emb_response

    result_pos = generate_grounded_answer(
        query="What is the fee for premier checking?",
        collection=mock_collection,
        openai_client=mock_openai_client
    )

    print("\n2. Positive Answer Result:")
    print(f"   • Answer: {result_pos['answer']}")
    print(f"   • Insufficient Evidence: {result_pos['insufficient_evidence']}")
    print(f"   • Sources Count: {len(result_pos['sources'])}")
    assert not result_pos["insufficient_evidence"]
    assert len(result_pos["sources"]) == 1
    assert result_pos["sources"][0]["page"] == 1
    print("✅ Positive answer flow verified.")

    # 4. Test Mock LLM Insufficient Evidence Flow
    mock_choice_neg = MagicMock()
    mock_choice_neg.message.content = config.INSUFFICIENT_EVIDENCE_PHRASE
    mock_response_neg = MagicMock()
    mock_response_neg.choices = [mock_choice_neg]
    mock_response_neg.usage.prompt_tokens = 120
    mock_response_neg.usage.completion_tokens = 20
    mock_response_neg.usage.total_tokens = 140
    mock_openai_client.chat.completions.create.return_value = mock_response_neg

    result_neg = generate_grounded_answer(
        query="What is the crypto policy?",
        collection=mock_collection,
        openai_client=mock_openai_client
    )

    print("\n3. Insufficient Evidence Fallback Result:")
    print(f"   • Answer: {result_neg['answer']}")
    print(f"   • Insufficient Evidence: {result_neg['insufficient_evidence']}")
    print(f"   • Sources Count: {len(result_neg['sources'])} (Expected: 0)")
    assert result_neg["insufficient_evidence"]
    assert len(result_neg["sources"]) == 0
    print("✅ Insufficient evidence fallback flow verified.")

    print("\n" + "=" * 80)
    print("✅ ALL RAG ENGINE LOGIC TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    test_rag_engine_components()
