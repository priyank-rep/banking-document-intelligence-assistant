"""
Diagnostic script for Real-World Banking Document Ingestion: HDFC_FY26.pdf.

Evaluates:
A. PDF Characteristics (file size, total pages)
B. Text Extraction Quality (character counts, empty/scanned pages, extraction rate)
C. Extraction & Layout Artifacts (tables, multi-column reading order, headers/footers, broken words)
D. Chunking Behavior (chunk distribution, token counts, sample slices)
E. Metadata Quality (source, page, chunk_id integrity)

Outputs full diagnostic analysis to data/hdfc_diagnostic_report.json and terminal.
"""

import sys
import json
import re
import statistics
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.pdf_loader import load_pdf_pages
from src.chunker import chunk_pages


def analyze_hdfc_document():
    pdf_path = config.SAMPLE_DOCS_DIR / "HDFC_FY26.pdf"
    if not pdf_path.exists():
        print(f"❌ Error: File not found at {pdf_path}")
        return

    print("=" * 85)
    print("🏦 REAL-WORLD BANKING PDF DIAGNOSTIC: HDFC_FY26.pdf")
    print("=" * 85)

    # --------------------------------------------------------------------------
    # A. PDF Characteristics
    # --------------------------------------------------------------------------
    file_size_bytes = pdf_path.stat().st_size
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

    # Ingestion using existing pdf_loader
    print("\n[Step 1/3] Ingesting PDF via src.pdf_loader...")
    pages = load_pdf_pages(pdf_path)
    total_pages = len(pages)

    print(f"  • Filename:    {pdf_path.name}")
    print(f"  • File Size:   {file_size_mb} MB ({file_size_bytes:,} bytes)")
    print(f"  • Total Pages: {total_pages}")

    # --------------------------------------------------------------------------
    # B. Text Extraction Quality
    # --------------------------------------------------------------------------
    print("\n[Step 2/3] Analyzing text extraction quality...")
    char_counts = [p["char_count"] for p in pages]
    total_characters = sum(char_counts)

    empty_pages = [p["page"] for p in pages if p["char_count"] == 0]
    low_text_pages = [p["page"] for p in pages if 0 < p["char_count"] < 100]
    moderate_text_pages = [p["page"] for p in pages if 100 <= p["char_count"] < 1000]
    dense_text_pages = [p["page"] for p in pages if p["char_count"] >= 1000]

    pct_extractable = round(((total_pages - len(empty_pages)) / total_pages) * 100, 2) if total_pages else 0.0

    mean_chars = round(statistics.mean(char_counts), 1) if char_counts else 0
    median_chars = round(statistics.median(char_counts), 1) if char_counts else 0
    min_chars = min(char_counts) if char_counts else 0
    max_chars = max(char_counts) if char_counts else 0

    print(f"  • Total Extracted Characters: {total_characters:,}")
    print(f"  • Characters Per Page (Mean / Median): {mean_chars} / {median_chars}")
    print(f"  • Characters Per Page (Min / Max):     {min_chars} / {max_chars}")
    print(f"  • Empty Pages (0 chars):               {len(empty_pages)} (Pages: {empty_pages[:10]}{'...' if len(empty_pages) > 10 else ''})")
    print(f"  • Low Text Pages (<100 chars):         {len(low_text_pages)} (Pages: {low_text_pages[:10]}{'...' if len(low_text_pages) > 10 else ''})")
    print(f"  • Dense Pages (>=1000 chars):          {len(dense_text_pages)}")
    print(f"  • Pages with Extractable Text:         {pct_extractable}%")

    # --------------------------------------------------------------------------
    # C. Extraction & Layout Artifacts Analysis
    # --------------------------------------------------------------------------
    # Inspect pages for common financial PDF extraction issues
    layout_issues = []

    # 1. Header / Footer repetition detection across pages
    first_lines = []
    last_lines = []
    for p in pages:
        lines = [line.strip() for line in p["text"].split("\n") if line.strip()]
        if lines:
            first_lines.append(lines[0])
            last_lines.append(lines[-1])

    # Count common repeating header/footer candidates
    from collections import Counter
    header_counts = Counter(first_lines)
    footer_counts = Counter(last_lines)

    common_headers = [(h, c) for h, c in header_counts.items() if c >= 5 and len(h) > 5]
    common_footers = [(f, c) for f, c in footer_counts.items() if c >= 5 and len(f) > 5]

    # 2. Table / Tabular content detection
    table_candidate_pages = []
    for p in pages:
        text = p["text"]
        # High density of numbers, percentage signs, or multiple consecutive spaces/tabs
        num_numbers = len(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))
        num_lines = len(text.splitlines())
        if num_lines > 0 and (num_numbers / num_lines) > 2.5:
            table_candidate_pages.append(p["page"])

    # 3. Multi-column / reading order inspection
    # Check for short jagged lines or fragmented sentences
    jagged_pages = []
    for p in pages:
        lines = [l.strip() for l in p["text"].split("\n") if l.strip()]
        if len(lines) > 10:
            short_line_ratio = sum(1 for l in lines if len(l) < 30) / len(lines)
            if short_line_ratio > 0.6:
                jagged_pages.append((p["page"], round(short_line_ratio, 2)))

    # 4. Broken hyphenated words across lines
    hyphenated_word_samples = []
    for p in pages:
        matches = re.findall(r"(\b[A-Za-z]+-\n[A-Za-z]+\b)", p["text"])
        if matches:
            hyphenated_word_samples.extend([(p["page"], m.replace("\n", "")) for m in matches[:3]])

    # --------------------------------------------------------------------------
    # D. Chunking Behavior
    # --------------------------------------------------------------------------
    print("\n[Step 3/3] Running token-aware chunking via src.chunker...")
    chunks = chunk_pages(pages, chunk_size=config.CHUNK_SIZE_TOKENS, chunk_overlap=config.CHUNK_OVERLAP_TOKENS)
    total_chunks = len(chunks)

    token_counts = [c["token_count"] for c in chunks]
    avg_tokens = round(statistics.mean(token_counts), 1) if token_counts else 0
    median_tokens = round(statistics.median(token_counts), 1) if token_counts else 0
    min_tokens = min(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0

    # Pages producing multiple chunks
    chunks_per_page = Counter([c["page"] for c in chunks])
    pages_with_multiple_chunks = {page: count for page, count in chunks_per_page.items() if count > 1}

    print(f"  • Total Chunks Generated:      {total_chunks}")
    print(f"  • Average Chunk Tokens:        {avg_tokens}")
    print(f"  • Chunk Tokens (Min / Max):    {min_tokens} / {max_tokens}")
    print(f"  • Pages with Multiple Chunks:  {len(pages_with_multiple_chunks)} pages")

    # Sample chunks for analysis:
    # 1. Narrative chunk
    narrative_sample = None
    for c in chunks:
        if c["token_count"] > 250 and len(re.findall(r"[A-Za-z]{4,}", c["chunk_text"])) > 40:
            # Low proportion of isolated numbers
            num_ratio = len(re.findall(r"\b\d+\b", c["chunk_text"])) / max(1, len(c["chunk_text"].split()))
            if num_ratio < 0.15:
                narrative_sample = c
                break

    # 2. Financial Table / Number-heavy chunk
    table_sample = None
    for c in chunks:
        if c["page"] in table_candidate_pages and c["token_count"] > 150:
            table_sample = c
            break

    # 3. Numeric / percentage dense chunk
    numeric_sample = None
    for c in chunks:
        pct_count = c["chunk_text"].count("%")
        curr_count = c["chunk_text"].count("₹") + c["chunk_text"].count("Rs") + c["chunk_text"].count("$") + c["chunk_text"].count("Cr")
        if pct_count >= 5 or curr_count >= 5:
            numeric_sample = c
            break

    # 4. Complex layout / high chunk count page sample
    complex_sample = None
    if pages_with_multiple_chunks:
        most_split_page = max(pages_with_multiple_chunks.items(), key=lambda x: x[1])[0]
        for c in chunks:
            if c["page"] == most_split_page:
                complex_sample = c
                break

    # --------------------------------------------------------------------------
    # E. Metadata Integrity Validation
    # --------------------------------------------------------------------------
    metadata_valid = True
    for c in chunks:
        if not (c["source"] == pdf_path.name and isinstance(c["page"], int) and c["chunk_id"] and c["token_count"] > 0):
            metadata_valid = False
            break

    # --------------------------------------------------------------------------
    # F. Classification of Issues
    # --------------------------------------------------------------------------
    critical_issues = []
    moderate_issues = []
    minor_issues = []

    # Check for empty pages
    if len(empty_pages) > 0:
        critical_issues.append({
            "category": "Unextracted / Scanned / Image-Only Pages",
            "impact": f"{len(empty_pages)} pages ({len(empty_pages)/total_pages*100:.1f}%) returned 0 extractable text characters.",
            "pages": empty_pages[:10],
            "severity": "CRITICAL",
            "explanation": "PyMuPDF text extraction cannot read scanned image pages, charts, infographics, or visual presentation slides without OCR."
        })

    # Check table structure loss
    if len(table_candidate_pages) > 0:
        moderate_issues.append({
            "category": "Tabular Structure Flattening",
            "impact": f"At least {len(table_candidate_pages)} pages contain dense tabular financial disclosures.",
            "pages": table_candidate_pages[:10],
            "severity": "MODERATE",
            "explanation": "Standard PyMuPDF text extraction extracts stream text without preserving 2D column-row grid structure, resulting in decoupled headers and numeric values."
        })

    # Check header/footer pollution
    if common_headers or common_footers:
        moderate_issues.append({
            "category": "Header & Footer Noise Contamination",
            "impact": f"Detected {len(common_headers)} repeating headers and {len(common_footers)} repeating footers embedded inside chunk texts.",
            "examples": [f"Header: '{h[0][:40]}' (seen on {h[1]} pages)" for h in common_headers[:3]],
            "severity": "MODERATE",
            "explanation": "Repeating headers and footers consume embedding space and can cause vector search to match on boilerplates rather than content."
        })

    # Check jagged / fragmented lines from multi-column layouts
    if jagged_pages:
        moderate_issues.append({
            "category": "Multi-Column Layout Interweaving",
            "impact": f"Detected {len(jagged_pages)} pages with high short-line ratios indicative of multi-column or callout box layouts.",
            "pages": [p[0] for p in jagged_pages[:10]],
            "severity": "MODERATE",
            "explanation": "Multi-column layouts can cause text from column A and column B to interleave horizontally depending on bounding box order."
        })

    # Hyphenated word breaks
    if hyphenated_word_samples:
        minor_issues.append({
            "category": "Hyphenated Line-Break Splitting",
            "impact": f"Observed {len(hyphenated_word_samples)} line-break hyphenations (e.g., {hyphenated_word_samples[0][1] if hyphenated_word_samples else ''}).",
            "severity": "MINOR",
            "explanation": "Words split across lines with hyphens (e.g. 'manage-\\nment') may not match exact search terms unless normalized."
        })

    # --------------------------------------------------------------------------
    # Save Report
    # --------------------------------------------------------------------------
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "pdf_characteristics": {
            "filename": pdf_path.name,
            "file_size_mb": file_size_mb,
            "file_size_bytes": file_size_bytes,
            "total_pages": total_pages
        },
        "text_extraction_quality": {
            "total_characters": total_characters,
            "mean_characters_per_page": mean_chars,
            "median_characters_per_page": median_chars,
            "min_characters_per_page": min_chars,
            "max_characters_per_page": max_chars,
            "extractable_percentage": pct_extractable,
            "empty_page_count": len(empty_pages),
            "empty_pages": empty_pages,
            "low_text_page_count": len(low_text_pages),
            "low_text_pages": low_text_pages,
            "dense_text_page_count": len(dense_text_pages)
        },
        "chunking_behavior": {
            "total_chunks": total_chunks,
            "avg_tokens_per_chunk": avg_tokens,
            "median_tokens_per_chunk": median_tokens,
            "min_tokens_per_chunk": min_tokens,
            "max_tokens_per_chunk": max_tokens,
            "pages_with_multiple_chunks_count": len(pages_with_multiple_chunks),
            "sample_chunks": {
                "narrative": {
                    "page": narrative_sample["page"] if narrative_sample else None,
                    "chunk_id": narrative_sample["chunk_id"] if narrative_sample else None,
                    "token_count": narrative_sample["token_count"] if narrative_sample else None,
                    "preview": narrative_sample["chunk_text"][:350] if narrative_sample else None
                },
                "table": {
                    "page": table_sample["page"] if table_sample else None,
                    "chunk_id": table_sample["chunk_id"] if table_sample else None,
                    "token_count": table_sample["token_count"] if table_sample else None,
                    "preview": table_sample["chunk_text"][:350] if table_sample else None
                },
                "numeric_percentage": {
                    "page": numeric_sample["page"] if numeric_sample else None,
                    "chunk_id": numeric_sample["chunk_id"] if numeric_sample else None,
                    "token_count": numeric_sample["token_count"] if numeric_sample else None,
                    "preview": numeric_sample["chunk_text"][:350] if numeric_sample else None
                },
                "complex_dense": {
                    "page": complex_sample["page"] if complex_sample else None,
                    "chunk_id": complex_sample["chunk_id"] if complex_sample else None,
                    "token_count": complex_sample["token_count"] if complex_sample else None,
                    "preview": complex_sample["chunk_text"][:350] if complex_sample else None
                }
            }
        },
        "metadata_quality": {
            "source_preserved": metadata_valid,
            "page_preserved": metadata_valid,
            "chunk_id_preserved": metadata_valid
        },
        "classified_findings": {
            "critical": critical_issues,
            "moderate": moderate_issues,
            "minor": minor_issues
        }
    }

    out_file = config.DATA_DIR / "hdfc_diagnostic_report.json"
    with open(out_file, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n✅ Diagnostic Report saved to: {out_file}")

    # Print summary table of classified findings
    print("\n" + "=" * 85)
    print("📋 INGESTION DIAGNOSTIC FINDINGS CLASSIFICATION")
    print("=" * 85)
    print("🔴 CRITICAL ISSUES (RAG Reliability Blockers):")
    if not critical_issues:
        print("  • None detected.")
    for issue in critical_issues:
        print(f"  • [{issue['category']}]: {issue['impact']}")
        print(f"    Why: {issue['explanation']}")

    print("\n🟡 MODERATE ISSUES (Retrieval / Citation Degradation):")
    if not moderate_issues:
        print("  • None detected.")
    for issue in moderate_issues:
        print(f"  • [{issue['category']}]: {issue['impact']}")
        print(f"    Why: {issue['explanation']}")

    print("\n🟢 MINOR ISSUES (Cosmetic / Low Impact):")
    if not minor_issues:
        print("  • None detected.")
    for issue in minor_issues:
        print(f"  • [{issue['category']}]: {issue['impact']}")
        print(f"    Why: {issue['explanation']}")
    print("=" * 85)


if __name__ == "__main__":
    analyze_hdfc_document()
