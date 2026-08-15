"""
Helper script to generate realistic sample banking PDFs for local development and testing.
Uses PyMuPDF to programmatically build multi-page PDFs in data/sample_docs/.
"""

import sys
from pathlib import Path
import pymupdf  # PyMuPDF

# Add parent directory to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config


def create_fee_schedule_pdf(output_path: Path):
    """Generate a 2-page sample Fee Schedule and Account Agreement PDF."""
    doc = pymupdf.open()

    # Page 1: Deposit Accounts & Overdraft Terms
    page1 = doc.new_page()
    page1_text = """APEX COMMERCIAL BANK - SCHEDULE OF FEES & ACCOUNT TERMS
Effective Date: January 1, 2025 | Document Ref: ACB-FEE-2025-v1

SECTION 1: CONSUMER CHECKING & SAVINGS ACCOUNTS

1.1 Apex Premier Checking
- Monthly Maintenance Fee: $25.00 (Waived with a $5,000 minimum daily balance or $10,000 combined deposit balance).
- Non-Apex ATM Fee: $3.00 per transaction plus any third-party surcharge.
- International Transaction Fee: 3.0% of the transaction amount converted to USD.

1.2 Apex Everyday Checking
- Monthly Maintenance Fee: $12.00 (Waived with at least one qualifying direct deposit of $500 or more per statement cycle).
- Paper Statement Fee: $3.00 per month (Electronic statements are free).

1.3 Overdraft & Insufficient Funds (NSF) Policy
- Overdraft Paid Fee: $34.00 per item covered, capped at a maximum of 3 overdraft fees per business day ($102.00 total daily cap).
- De Minimis Buffer: No overdraft fee is assessed if the ending end-of-day account balance is overdrawn by $15.00 or less.
- Continuous Overdraft Fee: An additional $15.00 extended overdraft fee applies if the account remains negative for 5 consecutive business days.
- Returned Item (NSF) Fee: $0.00 (Apex Bank does not charge fees for items returned unpaid).
"""
    page1.insert_text((50, 60), page1_text, fontsize=10, fontname="helv")

    # Page 2: Wire Transfers, Safe Deposit, & Regulatory Disclosures
    page2 = doc.new_page()
    page2_text = """APEX COMMERCIAL BANK - SCHEDULE OF FEES & ACCOUNT TERMS (CONTINUED)
Effective Date: January 1, 2025 | Document Ref: ACB-FEE-2025-v1 | Page 2

SECTION 2: WIRE TRANSFERS & TREASURY SERVICES

2.1 Domestic Wire Transfers
- Incoming Domestic Wire: $15.00 per transfer.
- Outgoing Domestic Wire (Online Banking): $20.00 per transfer.
- Outgoing Domestic Wire (Branch Assisted): $35.00 per transfer.
- Cutoff Time: 4:30 PM Eastern Time for same-day processing.

2.2 International Wire Transfers
- Incoming International Wire: $20.00 per transfer.
- Outgoing International Wire (USD): $45.00 per transfer.
- Outgoing International Wire (Foreign Currency): $30.00 per transfer (subject to FX spread).

SECTION 3: DISPUTE RESOLUTION & REGULATION E DISCLOSURES

3.1 Error Resolution Notice
In case of errors or questions about electronic fund transfers, call Customer Support at 1-800-555-APEX or write to Apex Bank Dispute Operations, PO Box 8800, New York, NY 10001. We must hear from you no later than 60 calendar days after we sent the FIRST statement on which the problem or error appeared.

3.2 Provisional Credit Timeline
If we take more than 10 business days (20 business days for new accounts) to investigate an alleged electronic funds transfer error, we will provisionally credit your account for the amount in dispute while our investigation proceeds.
"""
    page2.insert_text((50, 60), page2_text, fontsize=10, fontname="helv")

    doc.save(str(output_path))
    doc.close()
    print(f"Created: {output_path}")


def create_commercial_loan_pdf(output_path: Path):
    """Generate a 3-page sample Commercial Loan & Credit Facility Agreement."""
    doc = pymupdf.open()

    # Page 1: Facility Terms & Interest Calculations
    page1 = doc.new_page()
    page1_text = """COMMERCIAL REVOLVING CREDIT AGREEMENT
Borrower: Meridian Logistics LLC | Lender: Apex Commercial Bank
Facility Limit: $5,000,000 USD | Agreement Date: March 15, 2024

ARTICLE I: DEFINITIONS & FACILITY TERMS

1.01 Credit Facility: A revolving line of credit up to an aggregate principal amount of $5,000,000.00 (the "Commitment").
1.02 Maturity Date: March 15, 2027 (36 months from execution date).
1.03 Applicable Interest Rate:
(a) Base Rate Option: Daily SOFR (Secured Overnight Financing Rate) plus an Applicable Margin of 2.75% per annum.
(b) Fixed Rate Tranche Option: 3-Year Fixed Treasury Benchmark plus 3.10% per annum, upon 5 days prior written notice.
(c) Default Interest Rate: Upon an Event of Default, the interest rate increases automatically by 300 basis points (3.00% per annum) above the otherwise applicable rate.

1.04 Unused Commitment Fee:
A fee equal to 0.35% per annum on the average daily unborrowed portion of the Facility, payable quarterly in arrears on the last business day of each calendar quarter.
"""
    page1.insert_text((50, 60), page1_text, fontsize=10, fontname="helv")

    # Page 2: Financial Covenants & Reporting
    page2 = doc.new_page()
    page2_text = """COMMERCIAL REVOLVING CREDIT AGREEMENT (CONTINUED)
Meridian Logistics LLC | Agreement Date: March 15, 2024 | Page 2

ARTICLE II: FINANCIAL COVENANTS & REPORTING REQUIREMENTS

2.01 Debt Service Coverage Ratio (DSCR):
Borrower shall maintain a consolidated Debt Service Coverage Ratio of not less than 1.25:1.00, measured at the end of each fiscal quarter on a trailing twelve-month (TTM) basis.

2.02 Maximum Leverage Ratio:
Consolidated Total Debt to Adjusted EBITDA shall not exceed 3.50:1.00 at any fiscal quarter end.

2.03 Minimum Tangible Net Worth:
Borrower shall maintain a Tangible Net Worth of not less than $8,500,000.00, increasing by 50% of positive consolidated net income for each completed fiscal year.

2.04 Required Financial Deliverables:
(a) Quarterly Financial Statements: Certified by Chief Financial Officer within 45 days after the close of each of the first three fiscal quarters.
(b) Annual Audited Statements: Prepared in accordance with GAAP by an independent CPA firm within 120 days after fiscal year end.
(c) Compliance Certificate: Submitted concurrently with all quarterly and annual financial statements.
"""
    page2.insert_text((50, 60), page2_text, fontsize=10, fontname="helv")

    # Page 3: Prepayment, Collateral & Events of Default
    page3 = doc.new_page()
    page3_text = """COMMERCIAL REVOLVING CREDIT AGREEMENT (CONTINUED)
Meridian Logistics LLC | Agreement Date: March 15, 2024 | Page 3

ARTICLE III: PREPAYMENT, SECURITY & REMEDIES

3.01 Optional Prepayment & Penalties:
Borrower may prepay SOFR-based advances at any time without penalty or premium upon 2 business days written notice. Fixed-rate tranches are subject to standard Yield Maintenance Make-Whole breakage fees if prepaid prior to the tranche maturity.

3.02 Collateral & Security Interest:
The Facility is secured by a first-priority blanket security interest on all Accounts Receivable, Inventory, Equipment, General Intangibles, and deposit accounts held at Apex Commercial Bank.

3.03 Events of Default:
Each of the following constitutes an Event of Default:
(a) Failure to make any principal or interest payment within 5 calendar days of the due date.
(b) Breach of any financial covenant set forth in Section 2.01 through 2.03 that remains uncured for 15 days following written notification.
(c) Any material adverse change (MAC) in the financial condition or operational viability of the Borrower.
(d) Involuntary bankruptcy or insolvency proceedings filed against Borrower not dismissed within 60 days.
"""
    page3.insert_text((50, 60), page3_text, fontsize=10, fontname="helv")

    doc.save(str(output_path))
    doc.close()
    print(f"Created: {output_path}")


def create_empty_scanned_sample_pdf(output_path: Path):
    """Generate an empty/scanned dummy PDF with no extractable text to test scanned detection."""
    doc = pymupdf.open()
    # Create a blank page without insert_text
    doc.new_page()
    doc.save(str(output_path))
    doc.close()
    print(f"Created: {output_path}")


if __name__ == "__main__":
    sample_dir = config.SAMPLE_DOCS_DIR
    sample_dir.mkdir(parents=True, exist_ok=True)

    create_fee_schedule_pdf(sample_dir / "apex_bank_fee_schedule.pdf")
    create_commercial_loan_pdf(sample_dir / "apex_bank_commercial_loan_agreement.pdf")
    create_empty_scanned_sample_pdf(sample_dir / "blank_scanned_sample.pdf")
    print("Sample PDFs successfully generated.")
