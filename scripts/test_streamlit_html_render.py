"""
Test Streamlit markdown rendering of answer cards with bullets and inline tags.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test sample with markdown lists and citations
sample_answer = """Supported facts from the uploaded documents:

- **Balance sheet size:** `43,64,886 (J Cr)` [Document: HDFC_FY26.pdf, Page: 44]
- **Deposits:** `31,05,250 (J Cr)` [Document: HDFC_FY26.pdf, Page: 44]
- **Advances:** `2,937,166.3 (C crore)` [Document: HDFC_FY26.pdf, Page: 250]
- **Return on Equity (ROE):** `14.3%` [Document: HDFC_FY26.pdf, Page: 44]
- **Net revenue:** `1,91,218.6 crore` [Document: HDFC_FY26.pdf, Page: 247]

*ROA was not found in the retrieved evidence for this question, so I have not inferred a value.*"""

print("Sample Answer:")
print(sample_answer)
