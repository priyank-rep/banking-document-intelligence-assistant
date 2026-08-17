"""
Inspect extracted text of both PDFs.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.pdf_loader import load_pdf_pages

fee_pages = load_pdf_pages(config.SAMPLE_DOCS_DIR / "apex_bank_fee_schedule.pdf")
loan_pages = load_pdf_pages(config.SAMPLE_DOCS_DIR / "apex_bank_commercial_loan_agreement.pdf")

print("=== FEE SCHEDULE PAGE 2 EXTRACTED TEXT ===")
print(fee_pages[1]["text"])

print("\n=== LOAN AGREEMENT PAGE 2 EXTRACTED TEXT ===")
print(loan_pages[1]["text"])
