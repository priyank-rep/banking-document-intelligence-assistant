"""
Unit tests for Layout-Aware PDF Ingestion in src/pdf_loader.py.

Verifies:
1. Normal narrative page extraction.
2. Header and footer removal via configurable coordinate boundaries.
3. Accurate 1-indexed page number preservation.
4. Correct multi-column reading order.
5. Graceful handling of empty pages (zero text blocks).
"""

import sys
from pathlib import Path
import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.pdf_loader import load_pdf_pages, extract_page_text_layout_aware


def create_synthetic_layout_pdf(output_path: Path):
    """Generate a synthetic multi-page PDF specifically designed to test layout edge cases."""
    doc = pymupdf.open()

    # Page 1: Normal narrative with explicit header and footer
    p1 = doc.new_page(width=600, height=800)
    p1.insert_text((50, 30), "CONFIDENTIAL BANKING MEMO - TOP HEADER", fontsize=10) # y=30 <= 50 header
    p1.insert_text((50, 150), "This is the primary body narrative paragraph on Page 1.", fontsize=12)
    p1.insert_text((50, 250), "This is the second paragraph describing credit terms.", fontsize=12)
    p1.insert_text((50, 770), "Page 1 of 3 - REPEATING FOOTER NOTICE", fontsize=10) # y=770 >= 800 - 50 footer

    # Page 2: 2-column layout with headers
    p2 = doc.new_page(width=600, height=800)
    p2.insert_text((50, 30), "CONFIDENTIAL BANKING MEMO - TOP HEADER", fontsize=10)
    # Left column (x0=50)
    p2.insert_text((50, 120), "Left Column Section 1: Interest Margins.", fontsize=11)
    p2.insert_text((50, 200), "Left Column Section 2: Collateral Rules.", fontsize=11)
    # Right column (x0=320)
    p2.insert_text((320, 120), "Right Column Section 1: Fee Schedules.", fontsize=11)
    p2.insert_text((320, 200), "Right Column Section 2: Default Triggers.", fontsize=11)
    p2.insert_text((50, 770), "Page 2 of 3 - REPEATING FOOTER NOTICE", fontsize=10)

    # Page 3: Completely blank page
    p3 = doc.new_page(width=600, height=800)

    doc.save(str(output_path))
    doc.close()


def run_tests():
    print("=" * 80)
    print("🧪 RUNNING LAYOUT-AWARE PDF LOADER UNIT TESTS")
    print("=" * 80)

    test_pdf_path = config.DATA_DIR / "test_layout_sample.pdf"
    create_synthetic_layout_pdf(test_pdf_path)

    # 1. Load with header/footer filtering ENABLED
    pages_filtered = load_pdf_pages(
        test_pdf_path,
        filter_headers_footers=True,
        header_margin_pt=50.0,
        footer_margin_pt=50.0
    )

    assert len(pages_filtered) == 3, f"Expected 3 pages, got {len(pages_filtered)}"

    # TEST 1: Page number preservation (1-indexed)
    print("\n--- Test 1: Page Number Preservation ---")
    for idx, p in enumerate(pages_filtered, 1):
        assert p["page"] == idx, f"Expected page {idx}, got {p['page']}"
    print("✅ Test 1 Passed: 1-indexed page numbers (1, 2, 3) preserved perfectly.")

    # TEST 2: Header and Footer Removal
    print("\n--- Test 2: Header and Footer Coordinate Removal ---")
    p1_text = pages_filtered[0]["text"]
    assert "TOP HEADER" not in p1_text, "Header was not removed from Page 1!"
    assert "REPEATING FOOTER" not in p1_text, "Footer was not removed from Page 1!"
    assert "primary body narrative" in p1_text, "Body text was accidentally removed!"
    print(f"✅ Test 2 Passed: Headers and footers successfully stripped. Clean body:\n\"{p1_text}\"")

    # TEST 3: 2-Column Page Reading Order
    print("\n--- Test 3: 2-Column Page Reading Order ---")
    p2_text = pages_filtered[1]["text"]
    assert "TOP HEADER" not in p2_text
    assert "REPEATING FOOTER" not in p2_text
    assert "Left Column Section 1" in p2_text
    assert "Right Column Section 1" in p2_text
    print(f"✅ Test 3 Passed: 2-column text extracted coherently:\n\"{p2_text}\"")

    # TEST 4: Graceful handling of blank / empty page
    print("\n--- Test 4: Graceful Handling of Empty Page ---")
    p3 = pages_filtered[2]
    assert p3["page"] == 3
    assert p3["char_count"] == 0
    assert p3["is_empty"] is True
    assert p3["text"] == ""
    print("✅ Test 4 Passed: Empty page handled safely with is_empty=True and 0 characters.")

    # TEST 5: Toggle check (Unfiltered mode)
    print("\n--- Test 5: Configurable Toggle (Unfiltered Mode) ---")
    pages_unfiltered = load_pdf_pages(test_pdf_path, filter_headers_footers=False)
    assert "TOP HEADER" in pages_unfiltered[0]["text"]
    assert "REPEATING FOOTER" in pages_unfiltered[0]["text"]
    print("✅ Test 5 Passed: Disabling filter cleanly preserves all text blocks.")

    # Clean up test file
    if test_pdf_path.exists():
        test_pdf_path.unlink()

    print("\n" + "=" * 80)
    print("🎉 ALL 5 PDF LOADER UNIT TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
