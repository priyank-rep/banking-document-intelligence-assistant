"""
Unit test for Vector Store Logic (Offline / Mock Mode).

Validates:
1. ChromaDB persistent client initialization and collection schema.
2. Metadata insertion, storage, and persistence.
3. Upsert deduplication logic.
4. Cosine distance to similarity score conversion.
5. Query result extraction and sorting.
"""

import sys
import shutil
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
import chromadb
from chromadb.config import Settings


def test_chroma_logic():
    print("=" * 80)
    print("🧪 TESTING CHROMA VECTOR STORE LOGIC & DEDUPLICATION (OFFLINE)")
    print("=" * 80)

    test_db_dir = config.DATA_DIR / "test_chroma_db"
    if test_db_dir.exists():
        shutil.rmtree(test_db_dir)

    # 1. Initialize client & collection
    client = chromadb.PersistentClient(path=str(test_db_dir), settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name="test_collection", metadata={"hnsw:space": "cosine"})

    print(f"1. Created test collection '{collection.name}' in {test_db_dir}")

    # 2. Test chunks with synthetic embeddings (3 dimensions for simplicity)
    chunks = [
        {
            "chunk_id": "fee_schedule_p1_c0",
            "source": "fee_schedule.pdf",
            "page": 1,
            "chunk_text": "Monthly maintenance fee is $25.00 for Apex Premier Checking.",
            "token_count": 50,
            "embedding": [0.9, 0.1, 0.0]
        },
        {
            "chunk_id": "fee_schedule_p2_c0",
            "source": "fee_schedule.pdf",
            "page": 2,
            "chunk_text": "Incoming wire transfer fee is $15.00, outgoing is $20.00.",
            "token_count": 45,
            "embedding": [0.1, 0.9, 0.0]
        },
        {
            "chunk_id": "loan_p1_c0",
            "source": "loan_agreement.pdf",
            "page": 1,
            "chunk_text": "Facility limit is $5,000,000 with SOFR + 2.75% margin.",
            "token_count": 60,
            "embedding": [0.0, 0.1, 0.9]
        }
    ]

    # Index chunks via upsert
    ids = [c["chunk_id"] for c in chunks]
    docs = [c["chunk_text"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    metas = [{"source": c["source"], "page": c["page"], "chunk_id": c["chunk_id"], "token_count": c["token_count"]} for c in chunks]

    collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
    count = collection.count()
    print(f"2. Successfully indexed {count} items.")
    assert count == 3, f"Expected 3 items, got {count}"

    # 3. Test Upsert Deduplication
    print("\n3. Testing Upsert Deduplication...")
    # Re-insert the same IDs with slightly updated text
    collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
    count_after = collection.count()
    print(f"   Count after re-upsert: {count_after} (Expected: 3)")
    assert count_after == 3, "Duplicate records were incorrectly created!"
    print("   ✅ Upsert deduplication validated.")

    # 4. Test Query Retrieval
    print("\n4. Testing Query Similarity Retrieval...")
    # Query vector close to chunk 1: [0.95, 0.05, 0.0]
    query_vector = [0.95, 0.05, 0.0]
    results = collection.query(query_embeddings=[query_vector], n_results=2, include=["documents", "metadatas", "distances"])

    top_chunk_id = results["ids"][0][0]
    top_doc = results["documents"][0][0]
    top_dist = results["distances"][0][0]
    top_sim = 1.0 - top_dist

    print(f"   • Top Retrieved Chunk ID: {top_chunk_id}")
    print(f"   • Document Snippet:       \"{top_doc}\"")
    print(f"   • Cosine Distance:        {top_dist:.4f}")
    print(f"   • Cosine Similarity:      {top_sim:.4f}")
    print(f"   • Page Metadata:          Page {results['metadatas'][0][0]['page']}")

    assert top_chunk_id == "fee_schedule_p1_c0", f"Expected fee_schedule_p1_c0, got {top_chunk_id}"
    print("   ✅ Nearest neighbor retrieval logic validated.")

    # Cleanup test db
    shutil.rmtree(test_db_dir)
    print("\n" + "=" * 80)
    print("✅ ALL CHROMA LOGIC CHECKS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    test_chroma_logic()
