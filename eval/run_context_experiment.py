"""
Retrieval Context Experiment Runner for Banking Document Intelligence Assistant.

Compares three top-K context configurations on the 15-question banking benchmark:
- Config A: top_k = 4 (Baseline)
- Config B: top_k = 2 (Lower context / higher precision)
- Config C: top_k = 6 (Higher context / higher recall)

Keeps all other variables strictly constant:
- Chunk size = 500 tokens, Overlap = 100 tokens
- Embedding Model: text-embedding-3-small
- LLM Model: gpt-5-mini
- System Prompt & Grounding Guardrails

Exports comprehensive comparative results to eval/experiment_k_results.json.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.pdf_loader import load_pdf_pages
from src.chunker import chunk_pages
from src.vector_store import (
    get_or_create_collection,
    index_chunks,
    clear_collection,
    get_collection_stats
)
from src.rag_engine import generate_grounded_answer
from eval.run_evaluation import normalize_text, check_answer_match


def run_experiment_for_k(k_value: int, questions: List[Dict[str, Any]], collection) -> Dict[str, Any]:
    print(f"\n{'=' * 85}")
    print(f"🔬 RUNNING BENCHMARK EVALUATION FOR TOP-K = {k_value}")
    print(f"{'=' * 85}")

    total_q = len(questions)
    answerable_q_count = sum(1 for q in questions if q["answerable"])
    unanswerable_q_count = total_q - answerable_q_count

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

        # Call RAG engine with specific top_k
        result = generate_grounded_answer(query=query, top_k=k_value, collection=collection)
        gen_answer = result["answer"]
        is_insufficient = result["insufficient_evidence"]
        retrieved_chunks = result.get("retrieved_chunks", [])

        retrieved_sources = [c["source"] for c in retrieved_chunks]
        retrieved_page_pairs = [(c["source"], c["page"]) for c in retrieved_chunks]

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
        print(f"[{idx:02d}/{total_q:02d}] {q_id}: {query[:50]}... -> {status_str}")

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
            "num_retrieved_chunks": len(retrieved_chunks),
            "top_similarity_score": retrieved_chunks[0]["similarity_score"] if retrieved_chunks else 0.0,
            "usage": result.get("usage", {})
        })

    elapsed_time = time.time() - start_time

    source_hit_rate = (source_hits / answerable_q_count * 100) if answerable_q_count > 0 else 0.0
    page_hit_rate = (page_hits / answerable_q_count * 100) if answerable_q_count > 0 else 0.0
    correct_refusal_rate = (correct_refusals / unanswerable_q_count * 100) if unanswerable_q_count > 0 else 0.0
    overall_answer_match_rate = (answer_matches / total_q * 100) if total_q > 0 else 0.0

    return {
        "top_k": k_value,
        "elapsed_seconds": round(elapsed_time, 2),
        "metrics": {
            "source_hit_rate_pct": round(source_hit_rate, 2),
            "source_hits": f"{source_hits}/{answerable_q_count}",
            "page_hit_rate_pct": round(page_hit_rate, 2),
            "page_hits": f"{page_hits}/{answerable_q_count}",
            "correct_refusal_rate_pct": round(correct_refusal_rate, 2),
            "correct_refusals": f"{correct_refusals}/{unanswerable_q_count}",
            "answer_match_rate_pct": round(overall_answer_match_rate, 2),
            "answer_matches": f"{answer_matches}/{total_q}"
        },
        "detailed_results": detailed_results
    }


def run_full_experiment():
    print("=" * 85)
    print("🚀 STARTING RETRIEVAL-CONTEXT EXPERIMENT (TOP-K = 2 vs 4 vs 6)")
    print("=" * 85)

    # 1. Setup sample data
    sample_dir = config.SAMPLE_DOCS_DIR
    fee_pdf = sample_dir / "apex_bank_fee_schedule.pdf"
    loan_pdf = sample_dir / "apex_bank_commercial_loan_agreement.pdf"

    if not fee_pdf.exists() or not loan_pdf.exists():
        from scripts.create_sample_pdfs import create_fee_schedule_pdf, create_commercial_loan_pdf
        create_fee_schedule_pdf(fee_pdf)
        create_commercial_loan_pdf(loan_pdf)

    # Ensure clean indexing of sample docs
    collection = get_or_create_collection()
    clear_collection(collection)
    all_pages = load_pdf_pages(fee_pdf) + load_pdf_pages(loan_pdf)
    chunks = chunk_pages(all_pages, chunk_size=500, chunk_overlap=100)
    index_chunks(chunks, collection=collection)

    # 2. Load questions
    eval_file = config.EVAL_DIR / "eval_questions.json"
    with open(eval_file, "r") as f:
        questions = json.load(f)

    k_configurations = [4, 2, 6]  # Baseline (4), Lower (2), Higher (6)
    experiment_results = {}

    for k in k_configurations:
        config_name = f"K_{k}"
        res = run_experiment_for_k(k, questions, collection)
        experiment_results[config_name] = res

    # 3. Save to file
    out_file = config.EVAL_DIR / "experiment_k_results.json"
    payload = {
        "timestamp": datetime.now().isoformat(),
        "constants": {
            "chunk_size": 500,
            "chunk_overlap": 100,
            "llm_model": config.LLM_MODEL,
            "embedding_model": config.EMBEDDING_MODEL
        },
        "experiments": experiment_results
    }
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)

    # 4. Print Comparison Table
    print("\n" + "=" * 95)
    print("📊 EXPERIMENTAL COMPARISON TABLE")
    print("=" * 95)
    print(f"{'Metric':<32} | {'Config B (K=2)':<18} | {'Config A (K=4 - Base)':<21} | {'Config C (K=6)':<18}")
    print("-" * 95)

    b_met = experiment_results["K_2"]["metrics"]
    a_met = experiment_results["K_4"]["metrics"]
    c_met = experiment_results["K_6"]["metrics"]

    print(f"{'Retrieval Source Hit Rate':<32} | {b_met['source_hit_rate_pct']:>6.2f}% ({b_met['source_hits']})    | {a_met['source_hit_rate_pct']:>6.2f}% ({a_met['source_hits']})       | {c_met['source_hit_rate_pct']:>6.2f}% ({c_met['source_hits']})")
    print(f"{'Retrieval Page Hit Rate':<32}   | {b_met['page_hit_rate_pct']:>6.2f}% ({b_met['page_hits']})    | {a_met['page_hit_rate_pct']:>6.2f}% ({a_met['page_hits']})       | {c_met['page_hit_rate_pct']:>6.2f}% ({c_met['page_hits']})")
    print(f"{'Correct Refusal Rate':<32}      | {b_met['correct_refusal_rate_pct']:>6.2f}% ({b_met['correct_refusals']})    | {a_met['correct_refusal_rate_pct']:>6.2f}% ({a_met['correct_refusals']})       | {c_met['correct_refusal_rate_pct']:>6.2f}% ({c_met['correct_refusals']})")
    print(f"{'Overall Answer Match Rate':<32} | {b_met['answer_match_rate_pct']:>6.2f}% ({b_met['answer_matches']})   | {a_met['answer_match_rate_pct']:>6.2f}% ({a_met['answer_matches']})      | {c_met['answer_match_rate_pct']:>6.2f}% ({c_met['answer_matches']})")
    print(f"{'Total Execution Time':<32}     | {experiment_results['K_2']['elapsed_seconds']:>6.2f}s           | {experiment_results['K_4']['elapsed_seconds']:>6.2f}s              | {experiment_results['K_6']['elapsed_seconds']:>6.2f}s")
    print("=" * 95)

    # 5. Question-Level Focus: q06 and q10
    print("\n🔍 DRILL-DOWN ON TARGET QUESTIONS: q06 (Regulation E) & q10 (DSCR Covenant)")
    print("-" * 95)
    for target_id in ["q06", "q10"]:
        print(f"\n📌 Question [{target_id}]: {next(q['question'] for q in questions if q['id'] == target_id)}")
        for k in [2, 4, 6]:
            q_res = next(r for r in experiment_results[f"K_{k}"]["detailed_results"] if r["id"] == target_id)
            print(f"  • K={k}: Page Hit = {'✅' if q_res['page_hit'] else '❌'} | Answer Match = {'✅' if q_res['answer_match'] else '❌'} | Insufficient Flag = {q_res['insufficient_evidence']}")
            print(f"    Answer: \"{q_res['generated_answer'][:130]}...\"")
    print("=" * 95)


if __name__ == "__main__":
    run_full_experiment()
