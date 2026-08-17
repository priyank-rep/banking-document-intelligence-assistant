"""
Diagnostic script to understand GPT-5 mini's grounding constraints.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vector_store import get_openai_client
from src.rag_engine import SYSTEM_PROMPT

client = get_openai_client()

context = """--- PRIMARY CONTEXT CHUNK 1 (Direct Match) ---
Source Document: apex_bank_commercial_loan_agreement.pdf (Page 2)
Content:
COMMERCIAL REVOLVING CREDIT AGREEMENT (CONTINUED)
Meridian Logistics LLC | Agreement Date: March 15, 2024 | Page 2

ARTICLE II: FINANCIAL COVENANTS & REPORTING REQUIREMENTS

2.01 Debt Service Coverage Ratio (DSCR):
Borrower shall maintain a consolidated Debt Service Coverage Ratio of not less than 1.25:1.00, measured at the end of each fiscal quarter on a trailing twelve-month (TTM) basis.

2.02 Maximum Leverage Ratio:
Consolidated Total Debt to Adjusted EBITDA shall not exceed 3.50:1.00 at any fiscal quarter end.

--- SUPPORTING CONTEXT 1 (Adjacent Supporting Page) ---
Source Document: apex_bank_commercial_loan_agreement.pdf (Page 1)
Content:
COMMERCIAL REVOLVING CREDIT AGREEMENT
Borrower: Meridian Logistics LLC | Lender: Apex Commercial Bank
Facility Limit: $5,000,000 USD | Agreement Date: March 15, 2024

ARTICLE I: DEFINITIONS & FACILITY TERMS
1.01 Credit Facility: A revolving line of credit up to an aggregate principal amount of $5,000,000.00 (the "Commitment").
"""

user_prompt = f"""CONTEXT DOCUMENTS:
{context}

USER QUESTION:
What is the minimum consolidated Debt Service Coverage Ratio (DSCR) requirement for Meridian Logistics, and how often is it measured?

Please provide a grounded answer based strictly on the context above. Include inline citations [Document: filename, Page: X]. If the context is insufficient, reply with the exact required phrase."""

resp = client.chat.completions.create(
    model="gpt-5-mini",
    max_completion_tokens=1000,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
)

print("ISOLATED PROMPT RESPONSE:")
print(resp.choices[0].message.content)
