# 🏦 Banking Document Intelligence Assistant

A lightweight, production-grade **Retrieval-Augmented Generation (RAG)** assistant for banking and financial documents built natively with **Python**, **ChromaDB**, **PyMuPDF**, and **OpenAI**.

---

## 🎯 Key Features

- **📄 Document Ingestion**: Extracts clean, layout-aware text from multi-page banking PDFs (agreements, fee schedules, disclosures) using PyMuPDF.
- **✂️ Token-Aware Chunking**: Uses `tiktoken` to construct overlapping chunks while strictly preserving page boundaries for unambiguous page-level citations.
- **⚡ Local Vector Search**: Stores dense vector embeddings (`text-embedding-3-small`) in a persistent local ChromaDB instance with zero external database dependencies.
- **🛡️ Strict Grounding & Guardrails**: Enforces context-only answers with `gpt-5-mini`, returning an explicit *"insufficient evidence"* response when context is missing to eliminate hallucinations.
- **📚 Interactive Source Citations**: Presents exact document name, 1-indexed page number, chunk ID, similarity match percentage, and supporting text snippets.
- **🖥️ Streamlit Interface**: Clean, analyst-focused web UI with multi-file upload, live indexing status, query inputs, and source drawers.

---

## 🏗️ System Architecture

```text
PDF Upload -> PyMuPDF Page Extraction -> Token Chunking (tiktoken) -> OpenAI Embeddings -> ChromaDB
                                                                                              │
User Query -> Query Embedding ────────────────────────────────────────────────────────────────┘
                   │
                   ▼ Top-K Retrieval
     [Retrieved Context Chunks + Page Metadata]
                   │
                   ▼ Grounded System Prompt
        OpenAI LLM (GPT-5 mini)
                   │
                   ▼
     [Grounded Answer + Page-Level Citations]
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.11+
- OpenAI API Key

### 2. Installation & Setup

```bash
# Clone the repository
git clone https://github.com/your-username/banking-document-intelligence-assistant.git
cd banking-document-intelligence-assistant

# Create virtual environment & activate
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key

Create a `.env` file from the provided template:

```bash
cp .env.example .env
```

Add your OpenAI API key to `.env`:

```ini
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 4. Run the Streamlit Application

```bash
.venv/bin/streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## 🧪 Development & Verification Scripts

- **Milestone 2 (PDF Extraction & Chunking)**:
  ```bash
  .venv/bin/python scripts/verify_milestone_2.py
  ```
- **Milestone 3 (Chroma Vector Store & Retrieval)**:
  ```bash
  .venv/bin/python scripts/test_vector_store_mock.py
  ```
- **Milestone 4 (RAG Generation & Guardrails)**:
  ```bash
  .venv/bin/python scripts/test_rag_engine_mock.py
  ```
- **Milestone 5 (App Startup Verification)**:
  ```bash
  .venv/bin/python scripts/verify_streamlit_startup.py
  ```

---

## 📂 Project Structure

```text
banking-document-intelligence-assistant/
├── app.py                      # Streamlit web application
├── config.py                   # Centralized configuration (models, chunk size, paths)
├── requirements.txt            # Pinned dependencies
├── .env.example                # Template for environment variables
├── src/                        # Core application modules
│   ├── pdf_loader.py           # PyMuPDF text extraction & scanned doc detection
│   ├── chunker.py              # Token-aware sliding-window chunker
│   ├── vector_store.py         # ChromaDB persistence & OpenAI embeddings
│   └── rag_engine.py           # Grounded prompt synthesis & citation extractor
├── data/                       # Local data storage
│   ├── sample_docs/            # Synthetic sample banking PDFs for testing
│   └── chroma_db/              # Persistent Chroma database (gitignored)
├── eval/                       # Evaluation dataset & benchmarks
└── scripts/                    # Verification & utility scripts
```
