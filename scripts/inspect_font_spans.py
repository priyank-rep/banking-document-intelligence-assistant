"""
Inspect font names for specific currency symbol spans on Pages 5, 44, 234, 247, 250.
"""

import sys
from pathlib import Path
import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

pdf_path = config.SAMPLE_DOCS_DIR / "HDFC_FY26.pdf"
doc = pymupdf.open(str(pdf_path))

for p_num in [5, 44, 234, 247, 250]:
    page = doc[p_num - 1]
    print(f"\n{'='*70}\nPAGE {p_num}\n{'='*70}")
    d = page.get_text("dict")
    for b in d["blocks"]:
        if "lines" in b:
            for l in b["lines"]:
                line_spans = l["spans"]
                line_str = "".join(s["text"] for s in line_spans)
                if any(k in line_str for k in ["Cr", "crore", "43,64,886", "2,937,166", "1,91,218", "74,671"]):
                    print(f"Line text: {repr(line_str)}")
                    for s in line_spans:
                        print(f"   Span text: {repr(s['text']):<20} | Font: {s['font']:<30} | Size: {s['size']:.1f}")

doc.close()
