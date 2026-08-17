"""
Inspect raw font encodings, glyph unicodes, and text on HDFC PDF pages 5, 44, 234, 247, and 250.
"""

import sys
import unicodedata
from pathlib import Path
import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.pdf_loader import extract_page_text_layout_aware

pdf_path = config.SAMPLE_DOCS_DIR / "HDFC_FY26.pdf"
doc = pymupdf.open(str(pdf_path))

pages_to_check = [5, 44, 234, 247, 250]

print("=" * 90)
print("🔍 HDFC PDF GLYPH & FONT ENCODING AUDIT")
print("=" * 90)

for p_num in pages_to_check:
    page = doc[p_num - 1]
    print(f"\n{'='*90}")
    print(f"📄 PAGE {p_num} (Dimensions: {page.rect.width} x {page.rect.height} pt)")
    print(f"{'='*90}")

    # Layout-aware extracted text
    text = extract_page_text_layout_aware(page)
    print("--- Extracted Text Preview (First 800 chars) ---")
    print(text[:800])

    # Check for specific artifacts in the text
    print("\n--- Detected Glyph & Symbol Oddities on Page ---")
    lines = text.splitlines()
    for l_idx, line in enumerate(lines, 1):
        if any(token in line for token in ["(J Cr)", "(C crore)", "(I Cr)", "H Crore", "`", "₹", "Cr", "crore", "J Cr", "C crore"]):
            print(f"  Line {l_idx}: {repr(line)}")
            # Print character-by-character codes for these tokens
            for ch in line:
                if not ch.isalnum() and ch not in " ,.-/()":
                    u_name = unicodedata.name(ch, "UNKNOWN")
                    print(f"    Char: {repr(ch)} | Unicode: U+{ord(ch):04X} ({u_name})")

    # Font inspection from PyMuPDF dict / spans
    print("\n--- PyMuPDF Text Spans & Font Names for Currency/Unit Lines ---")
    page_dict = page.get_text("dict")
    for block in page_dict["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    span_text = span["text"]
                    if any(t in span_text for t in ["Cr", "crore", "43,64", "2,937", "1,91", "Balance Sheet", "Advances", "Deposits"]):
                        print(f"  [Span Font: {span['font']}, Size: {span['size']:.1f}pt, Flags: {span['flags']}] Text: {repr(span_text)}")
                        for ch in span_text:
                            if ch in "JCIH`\uf0db₹" or ord(ch) > 127:
                                print(f"    -> Char: {repr(ch)} (U+{ord(ch):04X}, name: {unicodedata.name(ch, 'UNKNOWN')})")

doc.close()
