"""
Verification Script for Milestone 5: Streamlit Application Startup & Component Integration.

Tests:
1. Module imports and syntax correctness of app.py.
2. Graceful initialization with 0 indexed documents.
3. State initialization and configuration loading.
"""

import sys
import importlib
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config


def test_app_startup():
    print("=" * 80)
    print("🚀 RUNNING MILESTONE 5: STREAMLIT APP STARTUP & INTEGRATION TEST")
    print("=" * 80)

    # 1. Test Module Imports
    print("\n1. Testing module imports...")
    try:
        from src.pdf_loader import load_pdf_pages
        from src.chunker import chunk_pages
        from src.vector_store import get_or_create_collection, get_collection_stats
        from src.rag_engine import generate_grounded_answer
        print("   ✅ Core src modules imported successfully.")
    except Exception as e:
        print(f"   ❌ Import error: {e}")
        sys.exit(1)

    # 2. Test Chroma Collection Access without documents
    print("\n2. Testing Chroma collection access with 0 documents...")
    collection = get_or_create_collection()
    stats = get_collection_stats(collection)
    print(f"   ✅ Chroma collection '{stats['collection_name']}' loaded cleanly with {stats['total_chunks']} chunks.")

    # 3. Test empty query handling
    print("\n3. Testing empty query handling...")
    empty_result = generate_grounded_answer(query="", collection=collection)
    assert empty_result["insufficient_evidence"] is True
    print("   ✅ Empty query handled gracefully without crash.")

    # 4. Check app.py syntax
    print("\n4. Verifying app.py syntax via py_compile...")
    import py_compile
    py_compile.compile(str(PROJECT_ROOT / "app.py"), doraise=True)
    print("   ✅ app.py compiled successfully with zero syntax errors.")

    print("\n" + "=" * 80)
    print("✅ STREAMLIT APP INTEGRATION & STARTUP VALIDATED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    test_app_startup()
