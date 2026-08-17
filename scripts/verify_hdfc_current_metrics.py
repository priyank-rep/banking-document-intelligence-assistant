"""
Directly verify HDFC_FY26.pdf ingestion & chunking metrics against current codebase.
"""

import sys
from pathlib import Path
import tiktoken

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.pdf_loader import load_pdf_pages
from src.chunker import chunk_pages

hdfc_pdf = config.SAMPLE_DOCS_DIR / "HDFC_FY26.pdf"

print("=" * 80)
print("🔍 RUNNING CURRENT PRODUCTION INGESTION ON HDFC_FY26.PDF")
print("=" * 80)

# 1. Load PDF pages with current production loader
pages = load_pdf_pages(hdfc_pdf, filename="HDFC_FY26.pdf")
total_pages = len(pages)
extractable_pages = sum(1 for p in pages if not p["is_empty"] and p["char_count"] > 0)
empty_pages = sum(1 for p in pages if p["is_empty"])
total_chars = sum(p["char_count"] for p in pages)

# 2. Chunk pages with current production chunker
chunks = chunk_pages(pages)
total_chunks = len(chunks)

# 3. Calculate token sizes
enc = tiktoken.get_encoding("cl100k_base")
token_counts = [c["token_count"] for c in chunks]
avg_tokens = sum(token_counts) / total_chunks if total_chunks > 0 else 0
avg_chars = sum(len(c["chunk_text"]) for c in chunks) / total_chunks if total_chunks > 0 else 0

# 4. Validate metadata integrity
required_keys = {"source", "page", "chunk_id", "chunk_text", "token_count"}
valid_metadata_count = 0
invalid_reasons = []

for idx, c in enumerate(chunks):
    missing = required_keys - set(c.keys())
    if missing:
        invalid_reasons.append(f"Chunk {idx} missing keys: {missing}")
        continue
    if c["source"] != "HDFC_FY26.pdf":
        invalid_reasons.append(f"Chunk {idx} source mismatch: {c['source']}")
        continue
    if not (1 <= c["page"] <= total_pages):
        invalid_reasons.append(f"Chunk {idx} invalid page: {c['page']}")
        continue
    if c["token_count"] <= 0 or c["token_count"] > config.CHUNK_SIZE_TOKENS:
        invalid_reasons.append(f"Chunk {idx} invalid token count: {c['token_count']}")
        continue
    if not c.get("chunk_text", "").strip():
        invalid_reasons.append(f"Chunk {idx} empty text")
        continue
    valid_metadata_count += 1

metadata_integrity_pct = (valid_metadata_count / total_chunks * 100.0) if total_chunks > 0 else 0.0

print(f"\n1. Total PDF Pages: {total_pages}")
print(f"2. Pages with Extractable Text: {extractable_pages} ({extractable_pages/total_pages*100:.2f}%) [Empty pages: {empty_pages}]")
print(f"3. Total Chunks Generated: {total_chunks}")
print(f"   (Config: chunk_size={config.CHUNK_SIZE_TOKENS}, chunk_overlap={config.CHUNK_OVERLAP_TOKENS}, filter_headers_footers={config.PDF_FILTER_HEADERS_FOOTERS})")
print(f"4. Average Chunk Size:")
print(f"   • Average Tokens per Chunk: {avg_tokens:.2f} tokens (Min: {min(token_counts)}, Max: {max(token_counts)})")
print(f"   • Average Characters per Chunk: {avg_chars:.2f} chars")
print(f"   • Total Extracted Characters: {total_chars:,} chars")
print(f"5. Metadata Integrity: {metadata_integrity_pct:.2f}% ({valid_metadata_count}/{total_chunks} chunks valid)")
if invalid_reasons:
    print(f"   ⚠️ Metadata Errors ({len(invalid_reasons)}):")
    for r in invalid_reasons[:5]:
        print(f"     - {r}")
else:
    print("   ✅ 100% of chunks have valid, complete, and schema-compliant metadata.")
print("=" * 80)
