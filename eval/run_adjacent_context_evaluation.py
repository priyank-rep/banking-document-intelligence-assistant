"""
Evaluation runner for Adjacent-Page Context Augmentation.

Runs the 15-question benchmark with:
- chunk_size = 500
- overlap = 100
- top_k = 4
- enable_adjacent_context = True (max_adjacent_chunks = 2)

Compares directly against the baseline (enable_adjacent_context = False).
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.vector_store import get_or_create_collection
from src.rag_engine import generate_grounded_answer
from eval.run_evaluation import normalize_text, check_answer_match


def run_evaluation_with_adjacent_context():
    print("=" * 85)
    print("🚀 RUNNING BENCHMARK EVALUATION WITH ADJACENT CONTEXT AUGMENTATION")
    print("=" * 85)
    print(f"• LLM Model:           {config.LLM_MODEL}")
    print(f"• Embedding Model:     {config.EMBEDDING_MODEL}")
    print(f"• Retrieval Top-K:     {config.RETRIEVAL_TOP_K}")
    print(f"• Adjacent Context:    ENABLED (max {config.MAX_ADJACENT_CHUNKS} chunks)")
    print("=" * 85)

    eval_file = config.EVAL_DIR / "eval_questions.json"
    with open(eval_file, "r") as f:
        questions = json.load(f)

    collection = get_or_create_collection()

    total_q = len(questions)
    answerable_q = [q for q in questions if q["answerable"]]
    unanswerable_q = [q for q in questions if not q["answerable"]]

    source_hits = 0
    page_hits = 0
    correct_refusals = 0
    answer_matches = 0

    detailed_results = []
    start_time = time.time()

    for idx, item in enumerate(questions, 1):
        q_id = item["id"]
        query = item["question"]
        is_answerable = item["answerable"]
        exp_source = item.get("expected_source")
        exp_page = item.get("expected_page")
        required_phrases = item.get("required_phrases", [])

        # Execute with adjacent context enabled
        result = generate_grounded_answer(
            query=query,
            top_k=config.RETRIEVAL_TOP_K,
            collection=collection,
            enable_adjacent_context=True,
            max_adjacent_chunks=config.MAX_ADJACENT_CHUNKS
        )

        gen_answer = result["answer"]
        is_insufficient = result["insufficient_evidence"]
        retrieved_chunks = result.get("retrieved_chunks", [])
        adjacent_chunks = result.get("adjacent_chunks", [])

        # Primary retrieval analysis
        retrieved_sources = [c["source"] for c in retrieved_chunks]
        retrieved_page_pairs = [(c["source"], c["page"]) for c in retrieved_chunks]

        # Context pages (primary + adjacent)
        all_context_page_pairs = retrieved_page_pairs + [(c["source"], c["page"]) for c in adjacent_chunks]

        source_hit = False
        page_hit = False
        refusal_correct = False

        if is_answerable:
            source_hit = exp_source in retrieved_sources
            page_hit = (exp_source, exp_page) in retrieved_page_pairs
            if source_hit:
                source_hits += 1
            if page_hit:
                page_hits += 1
        else:
            refusal_correct = is_insufficient or ("cannot find sufficient evidence" in normalize_text(gen_answer))
            if refusal_correct:
                correct_refusals += 1

        answer_match = check_answer_match(gen_answer, required_phrases, is_answerable)
        if answer_match:
            answer_matches += 1

        status_str = f"Page Hit: {'✅' if page_hit else '❌'} | Answer: {'✅' if answer_match else '❌'}" if is_answerable else f"Refusal: {'✅' if refusal_correct else '❌'}"
        print(f"[{idx:02d}/{total_q:02d}] {q_id}: {query[:50]}... -> {status_str} (adj: {len(adjacent_chunks)})")

        detailed_results.append({
            "id": q_id,
            "question": query,
            "answerable": is_answerable,
            "expected_source": exp_source,
            "expected_page": exp_page,
            "generated_answer": gen_answer,
            "insufficient_evidence": is_insufficient,
            "source_hit": source_hit if is_answerable else None,
            "page_hit": page_hit if is_answerable else None,
            "refusal_correct": refusal_correct if not is_answerable else None,
            "answer_match": answer_match,
            "retrieved_sources": retrieved_sources,
            "retrieved_pages": [p[1] for p in retrieved_page_pairs],
            "adjacent_pages": [c["page"] for c in adjacent_chunks],
            "num_primary_chunks": len(retrieved_chunks),
            "num_adjacent_chunks": len(adjacent_chunks),
            "usage": result.get("usage", {})
        })

    elapsed_time = time.time() - start_time

    source_hit_rate = (source_hits / len(answerable_q) * 100) if answerable_q else 0.0
    page_hit_rate = (page_hits / len(answerable_q) * 100) if answerable_q else 0.0
    correct_refusal_rate = (correct_refusals / len(unanswerable_q) * 100) if unanswerable_q else 0.0
    overall_answer_match_rate = (answer_matches / total_q * 100) if total_q else 0.0

    print("\n" + "=" * 85)
    print("📊 ADJACENT CONTEXT AUGMENTATION RESULTS SUMMARY")
    print("=" * 85)
    print(f"Total Questions Evaluated:     {total_q}")
    print(f"  • Answerable Questions:      {len(answerable_q)}")
    print(f"  • Unanswerable Questions:    {len(unanswerable_q)}")
    print("-" * 85)
    print(f"1. Retrieval Source Hit Rate:  {source_hit_rate:6.2f}% ({source_hits}/{len(answerable_q)})")
    print(f"2. Page Hit Rate:              {page_hit_rate:6.2f}% ({page_hits}/{len(answerable_q)})")
    print(f"3. Correct Refusal Rate:       {correct_refusal_rate:6.2f}% ({correct_refusals}/{len(unanswerable_q)})")
    print(f"4. Overall Answer Match Rate:  {overall_answer_match_rate:6.2f}% ({answer_matches}/{total_q})")
    print("-" * 85)
    print(f"Total Execution Time:          {elapsed_time:.2f} seconds")
    print("=" * 85)

    # Save results to json
    results_payload = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "chunk_size": 500,
            "chunk_overlap": 100,
            "top_k": config.RETRIEVAL_TOP_K,
            "enable_adjacent_context": True,
            "max_adjacent_chunks": config.MAX_ADJACENT_CHUNKS,
            "llm_model": config.LLM_MODEL,
            "embedding_model": config.EMBEDDING_MODEL
        },
        "metrics": {
            "source_hit_rate_pct": round(source_hit_rate, 2),
            "page_hit_rate_pct": round(page_hit_rate, 2),
            "correct_refusal_rate_pct": round(correct_refusal_rate, 2),
            "answer_match_rate_pct": round(overall_answer_match_rate, 2),
            "elapsed_seconds": round(elapsed_time, 2)
        },
        "detailed_results": detailed_results
    }

    out_file = config.EVAL_DIR / "results_adjacent_context.json"
    with open(out_file, "w") as f:
        json.dump(results_payload, f, indent=2)


if __name__ == "__main__":
    run_evaluation_with_adjacent_context()
