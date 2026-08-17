"""
Reindex HDFC_FY26.pdf in ChromaDB with deterministic font-normalized text.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.pdf_loader import load_pdf_pages
from src.chunker import chunk_pages
from src.vector_store import get_or_create_collection, index_chunks, get_collection_stats

hdfc_pdf = config.SAMPLE_DOCS_DIR / "HDFC_FY26.pdf"
pages = load_pdf_pages(hdfc_pdf, filename="HDFC_FY26.pdf")
chunks = chunk_pages(pages)

collection = get_or_create_collection()

# Remove old HDFC chunks first to ensure fresh clean replacement
old_ids = collection.get(where={"source": {"$eq": "HDFC_FY26.pdf"}})["ids"]
if old_ids:
    print(f"Deleting {len(old_ids)} existing HDFC chunks from collection...")
    # Chroma delete supports batches
    batch_size = 500
    for i in range(0, len(old_ids), batch_size):
        collection.delete(ids=old_ids[i:i + batch_size])

print(f"Indexing {len(chunks)} fresh font-normalized HDFC chunks...")
indexed_count = index_chunks(chunks, collection=collection)
stats = get_collection_stats(collection)
print(f"✅ Reindexing complete! Total chunks in Chroma: {stats['total_chunks']}")
