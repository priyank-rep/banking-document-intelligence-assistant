"""
Test deterministic font-aware Rupee symbol remapping on HDFC PDF pages.
"""

import sys
from pathlib import Path
import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

def extract_page_with_rupee_font_fix(page) -> str:
    rect = page.rect
    page_height = rect.height
    header_y_limit = config.PDF_HEADER_MARGIN_PT
    footer_y_limit = page_height - config.PDF_FOOTER_MARGIN_PT

    # Extract dict with font spans
    page_dict = page.get_text("dict")
    body_blocks = []

    # Sort blocks vertically, then horizontally
    blocks = sorted(page_dict["blocks"], key=lambda b: (b["bbox"][1], b["bbox"][0]))

    for b in blocks:
        if b.get("type") != 0:  # 0 is text
            continue

        bbox = b["bbox"]
        # Coordinate filter for headers/footers
        if config.PDF_FILTER_HEADERS_FOOTERS:
            if bbox[3] <= header_y_limit:
                continue
            if bbox[1] >= footer_y_limit:
                continue

        block_lines = []
        for line in b.get("lines", []):
            line_str = ""
            for span in line.get("spans", []):
                span_text = span["text"]
                font_name = span.get("font", "").lower()
                
                # Deterministic Rupee Font Normalization
                if "rupee" in font_name:
                    # In ITFRupee / Rupee fonts, any single glyph or symbol character is the ₹ symbol
                    span_text = "₹ " if span_text.endswith(" ") else "₹"

                line_str += span_text
            
            cleaned_line = line_str.strip()
            if cleaned_line:
                block_lines.append(cleaned_line)

        if block_lines:
            block_text = "\n".join(block_lines)
            body_blocks.append(block_text)

    return "\n\n".join(body_blocks)


doc = pymupdf.open(str(config.SAMPLE_DOCS_DIR / "HDFC_FY26.pdf"))

for p_num in [5, 44, 234, 247, 250]:
    page = doc[p_num - 1]
    fixed_text = extract_page_with_rupee_font_fix(page)
    print(f"\n{'='*70}\nPAGE {p_num} (WITH FONT FIX)\n{'='*70}")
    for line in fixed_text.splitlines():
        if any(k in line for k in ["₹", "Cr", "crore", "43,64,886", "2,937,166", "1,91,218"]):
            print(f"  -> {repr(line)}")

doc.close()
