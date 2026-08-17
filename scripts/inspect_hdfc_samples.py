"""
Inspect representative pages of HDFC_FY26.pdf in detail.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.pdf_loader import load_pdf_pages

pages = load_pdf_pages(config.SAMPLE_DOCS_DIR / "HDFC_FY26.pdf")

def inspect_page(page_num):
    p = pages[page_num - 1]
    print(f"\n{'='*75}")
    print(f"📄 PAGE {page_num} (Characters: {p['char_count']}, Is Empty: {p['is_empty']})")
    print(f"{'='*75}")
    print(p["text"][:1000])
    if len(p["text"]) > 1000:
        print("\n... [TRUNCATED] ...\n")
        print(p["text"][-500:])

for p_num in [1, 5, 234, 300, 628]:
    inspect_page(p_num)
