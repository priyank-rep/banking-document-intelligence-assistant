"""
Evaluation Runner for Banking Document Intelligence Assistant.

Evaluates the end-to-end RAG pipeline against a 15-question benchmark:
- Retrieval Source Hit Rate
- Retrieval Page Hit Rate
- Correct Refusal Rate (Insufficient Evidence on unanswerable questions)
- Answer Factual Match Rate

Exports structured metrics to eval/results.json.
"""

import sys
import json
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add project root to sys.path
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


def normalize_text(text: str) -> str:
    """Normalize text for deterministic case-insensitive comparison."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s\$\.\%\:\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def check_answer_match(generated_answer: str, required_phrases: List[str], is_answerable: bool) -> bool:
    """
    Deterministic factual check:
    - For answerable questions: Checks that all key factual terms/numbers appear in the response.
    - For unanswerable questions: Checks that the insufficient evidence fallback was triggered.
    """
    norm_answer = normalize_text(generated_answer)

    if not is_answerable:
        return (
            "cannot find sufficient evidence" in norm_answer or
            "insufficient evidence" in norm_answer
        )

    # For answerable questions, all required key phrases must be present in the normalized answer
    for phrase in required_phrases:
        norm_phrase = normalize_text(phrase)
        if norm_phrase not in norm_answer:
            return False
    return True


def run_benchmark(eval_file_path: Path = None, results_file_path: Path = None) -> Dict[str, Any]:
    eval_file = eval_file_path or (config.EVAL_DIR / "eval_questions.json")
    results_file = results_file_path or (config.EVAL_DIR / "results.json")

    print("=" * 85)
    print("🚀 RUNNING BANKING RAG EVALUATION BENCHMARK")
    print("=" * 85)
    print(f"• LLM Model:           {config.LLM_MODEL}")
    print(f"• Embedding Model:     {config.EMBEDDING_MODEL}")
    print(f"• Chunk Size (tokens): {config.CHUNK_SIZE_TOKENS}")
    print(f"• Chunk Overlap:       {config.CHUNK_OVERLAP_TOKENS}")
    print(f"• Retrieval Top-K:     {config.RETRIEVAL_TOP_K}")
    print(f"• Evaluation File:     {eval_file.name}")
    print("=" * 85)

    # 1. Ensure sample documents are indexed
    sample_dir = config.SAMPLE_DOCS_DIR
    fee_pdf = sample_dir / "apex_bank_fee_schedule.pdf"
    loan_pdf = sample_dir / "apex_bank_commercial_loan_agreement.pdf"

    if not fee_pdf.exists() or not loan_pdf.exists():
        from scripts.create_sample_pdfs import create_fee_schedule_pdf, create_commercial_loan_pdf
        create_fee_schedule_pdf(fee_pdf)
        create_commercial_loan_pdf(loan_pdf)

    collection = get_or_create_collection()
    stats = get_collection_stats(collection)
    if stats["total_chunks"] == 0:
        print("\nIndexing sample banking documents...")
        all_pages = load_pdf_pages(fee_pdf) + load_pdf_pages(loan_pdf)
        chunks = chunk_pages(all_pages, chunk_size=config.CHUNK_SIZE_TOKENS, chunk_overlap=config.CHUNK_OVERLAP_TOKENS)
        index_chunks(chunks, collection=collection)
        print(f"Indexed {len(chunks)} chunks.")

    # 2. Load Evaluation Questions
    with open(eval_file, "r") as f:
        questions = json.load(f)

    total_q = len(questions)
    answerable_q_count = sum(1 for q in questions if q["answerable"])
    unanswerable_q_count = total_q - answerable_q_count

    detailed_results = []
    source_hits = 0
    page_hits = 0
    correct_refusals = 0
    answer_matches = 0

    start_time = time.time()

    print(f"\nEvaluating {total_q} questions ({answerable_q_count} answerable, {unanswerable_q_count} unanswerable)...\n")

    for idx, item in enumerate(questions, 1):
        q_id = item["id"]
        query = item["question"]
        is_answerable = item["answerable"]
        exp_source = item.get("expected_source")
        exp_page = item.get("expected_page")
        required_phrases = item.get("required_phrases", [])

        # Execute RAG generation
        result = generate_grounded_answer(query=query, top_k=config.RETRIEVAL_TOP_K, collection=collection)
        gen_answer = result["answer"]
        is_insufficient = result["insufficient_evidence"]
        retrieved_chunks = result.get("retrieved_chunks", [])

        # Analyze retrieval
        retrieved_sources = [c["source"] for c in retrieved_chunks]
        retrieved_page_pairs = [(c["source"], c["page"]) for c in retrieved_chunks]

        # Calculate metrics per question
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
            # Unanswerable questions: success is declared when system flags insufficient evidence
            refusal_correct = is_insufficient or ("cannot find sufficient evidence" in normalize_text(gen_answer))
            if refusal_correct:
                correct_refusals += 1

        # Check factual answer match
        answer_match = check_answer_match(gen_answer, required_phrases, is_answerable)
        if answer_match:
            answer_matches += 1

        # Status badge for terminal output
        if is_answerable:
            status_str = f"Page Hit: {'✅' if page_hit else '❌'} | Answer: {'✅' if answer_match else '❌'}"
        else:
            status_str = f"Refusal: {'✅' if refusal_correct else '❌'}"

        print(f"[{idx:02d}/{total_q:02d}] {q_id}: {query[:55]}... -> {status_str}")

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
            "top_similarity_score": retrieved_chunks[0]["similarity_score"] if retrieved_chunks else 0.0,
            "notes": item.get("notes", "")
        })

    elapsed_time = time.time() - start_time

    # Calculate Percentage Rates
    source_hit_rate = (source_hits / answerable_q_count * 100) if answerable_q_count > 0 else 0.0
    page_hit_rate = (page_hits / answerable_q_count * 100) if answerable_q_count > 0 else 0.0
    correct_refusal_rate = (correct_refusals / unanswerable_q_count * 100) if unanswerable_q_count > 0 else 0.0
    overall_answer_match_rate = (answer_matches / total_q * 100) if total_q > 0 else 0.0

    summary_metrics = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "llm_model": config.LLM_MODEL,
            "embedding_model": config.EMBEDDING_MODEL,
            "chunk_size_tokens": config.CHUNK_SIZE_TOKENS,
            "chunk_overlap_tokens": config.CHUNK_OVERLAP_TOKENS,
            "retrieval_top_k": config.RETRIEVAL_TOP_K,
        },
        "metrics": {
            "total_questions": total_q,
            "answerable_questions": answerable_q_count,
            "unanswerable_questions": unanswerable_q_count,
            "retrieval_source_hit_rate_pct": round(source_hit_rate, 2),
            "page_hit_rate_pct": round(page_hit_rate, 2),
            "correct_refusal_rate_pct": round(correct_refusal_rate, 2),
            "overall_answer_match_rate_pct": round(overall_answer_match_rate, 2),
            "elapsed_seconds": round(elapsed_time, 2)
        }
    }

    # Export to results.json
    output_payload = {
        "summary": summary_metrics,
        "results": detailed_results
    }

    with open(results_file, "w") as f:
        json.dump(output_payload, f, indent=2)

    # 3. Print Final Terminal Summary
    print("\n" + "=" * 85)
    print("📊 EVALUATION RESULTS SUMMARY")
    print("=" * 85)
    print(f"Total Questions Evaluated:     {total_q}")
    print(f"  • Answerable Questions:      {answerable_q_count}")
    print(f"  • Unanswerable Questions:    {unanswerable_q_count}")
    print("-" * 85)
    print(f"1. Retrieval Source Hit Rate:  {source_hit_rate:6.2f}% ({source_hits}/{answerable_q_count})")
    print(f"2. Page Hit Rate:              {page_hit_rate:6.2f}% ({page_hits}/{answerable_q_count})")
    print(f"3. Correct Refusal Rate:       {correct_refusal_rate:6.2f}% ({correct_refusals}/{unanswerable_q_count})")
    print(f"4. Overall Answer Match Rate:  {overall_answer_match_rate:6.2f}% ({answer_matches}/{total_q})")
    print("-" * 85)
    print(f"Total Execution Time:          {elapsed_time:.2f} seconds")
    print(f"Detailed Results Saved To:     {results_file.relative_to(PROJECT_ROOT)}")
    print("=" * 85)

    return output_payload


if __name__ == "__main__":
    run_benchmark()
