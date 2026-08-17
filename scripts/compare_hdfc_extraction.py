"""
Compare Plain Text Extraction vs. PyMuPDF Layout-Aware Block Extraction
on representative pages of HDFC_FY26.pdf.

Evaluates:
- Page 2: Normal narrative text ("Harnessing AI in Banking")
- Page 5: Visual Infographic & Bento KPI tiles ("Our Performance")
- Page 234: Complex 10-Year Financial Highlights Table
- Page 238: Dense statutory page with repeating headers & footers

Analyzes:
1. page.get_text("text") vs. page.get_text("blocks", sort=True)
2. Bounding box coordinates (x0, y0, x1, y1)
3. Header / Footer boundary detection via y-coordinates
4. Multi-column separation via x-coordinate clustering
5. Built-in PyMuPDF Table detection (page.find_tables())
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pymupdf
import config


def inspect_page_extraction(doc, page_number: int, page_category: str):
    page = doc[page_number - 1]
    rect = page.rect
    page_width, page_height = rect.width, rect.height

    print("\n" + "=" * 90)
    print(f"📄 PAGE {page_number}: {page_category.upper()} (Dimensions: {page_width:.1f} x {page_height:.1f} pt)")
    print("=" * 90)

    # 1. Plain text extraction
    plain_text = page.get_text("text")
    plain_lines = [l for l in plain_text.splitlines() if l.strip()]

    # 2. Block-based extraction with sorting
    blocks = page.get_text("blocks", sort=True)
    text_blocks = [b for b in blocks if b[6] == 0]  # block_type 0 = text

    print(f"• Plain Text Length: {len(plain_text)} chars ({len(plain_lines)} non-empty lines)")
    print(f"• Total Layout Blocks Detected: {len(blocks)} (Text Blocks: {len(text_blocks)})")

    # 3. Coordinate Analysis: Headers, Footers, Columns
    header_threshold_y = 55.0   # Top ~6.5% of page
    footer_threshold_y = page_height - 55.0  # Bottom ~6.5% of page

    header_blocks = []
    footer_blocks = []
    body_blocks = []

    for b in text_blocks:
        x0, y0, x1, y1, text, block_no, btype = b
        cleaned = text.strip()
        if not cleaned:
            continue

        if y1 <= header_threshold_y:
            header_blocks.append((b, cleaned))
        elif y0 >= footer_threshold_y:
            footer_blocks.append((b, cleaned))
        else:
            body_blocks.append((b, cleaned))

    print(f"\n--- 1. Structural Region Classification via Coordinates ---")
    print(f"  • Header Blocks (y1 <= {header_threshold_y:.1f}pt): {len(header_blocks)}")
    for (b, txt) in header_blocks:
        print(f"    [Header] (x0={b[0]:.1f}, y0={b[1]:.1f}, x1={b[2]:.1f}, y1={b[3]:.1f}): \"{txt[:60]}\"")

    print(f"  • Footer Blocks (y0 >= {footer_threshold_y:.1f}pt): {len(footer_blocks)}")
    for (b, txt) in footer_blocks:
        print(f"    [Footer] (x0={b[0]:.1f}, y0={b[1]:.1f}, x1={b[2]:.1f}, y1={b[3]:.1f}): \"{txt[:60]}\"")

    print(f"  • Body Content Blocks: {len(body_blocks)}")

    # 4. Multi-Column Analysis
    # Group body blocks by x-coordinate to detect columns
    x_positions = [b[0][0] for b in body_blocks]
    left_col = [b for b in body_blocks if b[0][0] < page_width / 2]
    right_col = [b for b in body_blocks if b[0][0] >= page_width / 2]
    print(f"\n--- 2. Column Geometry Analysis ---")
    print(f"  • Left Column Blocks (x0 < {page_width/2:.1f}pt):  {len(left_col)}")
    print(f"  • Right Column Blocks (x0 >= {page_width/2:.1f}pt): {len(right_col)}")

    # 5. Native PyMuPDF Table Detection (page.find_tables())
    has_find_tables = hasattr(page, "find_tables")
    tables_found = []
    if has_find_tables:
        try:
            tabs = page.find_tables()
            tables_found = tabs.tables
            print(f"\n--- 3. Native PyMuPDF Table Finder (page.find_tables()) ---")
            print(f"  • Tables Detected: {len(tables_found)}")
            for t_idx, tab in enumerate(tables_found, 1):
                df_shape = f"{len(tab.extract())} rows x {len(tab.header.names) if tab.header else '?'} cols"
                print(f"    Table {t_idx}: Bounding Box: {tab.bbox} | Shape: {df_shape}")
                # Print first 2 rows of extracted table
                extracted_rows = tab.extract()
                if extracted_rows:
                    print(f"    Sample Row 0: {extracted_rows[0][:5]}")
                    if len(extracted_rows) > 1:
                        print(f"    Sample Row 1: {extracted_rows[1][:5]}")
        except Exception as e:
            print(f"  • Table detection note: {e}")

    # 6. First 3 Body Blocks Preview (with bounding boxes)
    print(f"\n--- 4. Sample Body Blocks (First 3) ---")
    for idx, (b, txt) in enumerate(body_blocks[:3], 1):
        print(f"  [Block {idx}] bbox=({b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}) | Length: {len(txt)} chars")
        first_line = txt.splitlines()[0] if txt.splitlines() else txt
        print(f"    Snippet: \"{first_line[:80]}...\"")

    return {
        "page": page_number,
        "category": page_category,
        "plain_text_len": len(plain_text),
        "total_blocks": len(blocks),
        "header_blocks_count": len(header_blocks),
        "footer_blocks_count": len(footer_blocks),
        "body_blocks_count": len(body_blocks),
        "tables_detected": len(tables_found) if has_find_tables else 0
    }


def run_comparison():
    pdf_path = config.SAMPLE_DOCS_DIR / "HDFC_FY26.pdf"
    if not pdf_path.exists():
        print(f"❌ PDF not found at {pdf_path}")
        return

    doc = pymupdf.open(str(pdf_path))

    test_pages = [
        (2, "Normal Narrative Page ('Harnessing AI in Banking')"),
        (5, "Visual Infographic & Bento KPI Page ('Our Performance')"),
        (234, "Complex 10-Year Financial Highlights Table"),
        (238, "Dense Statutory Page with Repeating Header/Footer")
    ]

    results = []
    for page_num, cat in test_pages:
        res = inspect_page_extraction(doc, page_num, cat)
        results.append(res)

    doc.close()

    # Comparison Summary
    print("\n" + "=" * 90)
    print("📊 EXTRACTION CAPABILITY COMPARISON SUMMARY")
    print("=" * 90)
    print(f"{'Page':<6} | {'Category':<32} | {'Plain Chars':<12} | {'Blocks':<8} | {'Headers':<8} | {'Footers':<8} | {'Tables':<8}")
    print("-" * 90)
    for r in results:
        print(f"{r['page']:<6} | {r['category'][:32]:<32} | {r['plain_text_len']:<12} | {r['total_blocks']:<8} | {r['header_blocks_count']:<8} | {r['footer_blocks_count']:<8} | {r['tables_detected']:<8}")
    print("=" * 90)


if __name__ == "__main__":
    run_comparison()
