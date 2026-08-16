"""
Streamlit Application for Banking Document Intelligence Assistant.

Provides an intuitive web interface for:
- Uploading and indexing banking PDF documents.
- Querying documents with grounded AI responses.
- Inspecting source citations, page numbers, and similarity metrics.
"""

import streamlit as st
from typing import List

import config
from src.pdf_loader import load_pdf_pages
from src.chunker import chunk_pages
from src.vector_store import (
    get_or_create_collection,
    index_chunks,
    get_collection_stats,
    clear_collection
)
from src.rag_engine import generate_grounded_answer


# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="Banking Document Intelligence Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==============================================================================
# Session State Initialization
# ==============================================================================
if "indexed_files" not in st.session_state:
    st.session_state["indexed_files"] = []

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

if "query_input" not in st.session_state:
    st.session_state["query_input"] = ""


# ==============================================================================
# Helper Functions
# ==============================================================================
def check_api_key_configured() -> bool:
    """Check whether a valid OpenAI API key is present in environment."""
    try:
        config.get_openai_api_key()
        return True
    except ValueError:
        return False


def handle_document_indexing(uploaded_files):
    """Process and index uploaded PDF documents into ChromaDB."""
    if not uploaded_files:
        st.sidebar.warning("Please select at least one PDF file to index.")
        return

    progress_bar = st.sidebar.progress(0, text="Starting document ingestion...")
    total_files = len(uploaded_files)
    all_chunks = []
    scanned_warnings = []

    collection = get_or_create_collection()

    for idx, uploaded_file in enumerate(uploaded_files):
        progress_val = int(((idx + 1) / total_files) * 50)
        progress_bar.progress(progress_val, text=f"Parsing {uploaded_file.name}...")

        try:
            pages = load_pdf_pages(uploaded_file, filename=uploaded_file.name)
            
            # Check for scanned/empty document
            total_chars = sum(p["char_count"] for p in pages)
            if len(pages) > 0 and total_chars < 50:
                scanned_warnings.append(uploaded_file.name)

            chunks = chunk_pages(pages)
            all_chunks.extend(chunks)

            if uploaded_file.name not in st.session_state["indexed_files"]:
                st.session_state["indexed_files"].append(uploaded_file.name)

        except Exception as e:
            st.sidebar.error(f"Error processing {uploaded_file.name}: {str(e)}")
            progress_bar.empty()
            return

    if all_chunks:
        progress_bar.progress(75, text="Generating embeddings and indexing into ChromaDB...")
        try:
            indexed_count = index_chunks(all_chunks, collection=collection)
            progress_bar.progress(100, text="Indexing complete!")
            st.sidebar.success(f"Successfully indexed {len(uploaded_files)} file(s) ({indexed_count} chunks)!")
        except Exception as e:
            st.sidebar.error(f"Failed to generate embeddings or index into Chroma: {str(e)}")
    else:
        st.sidebar.warning("No extractable text found in the uploaded documents.")

    if scanned_warnings:
        st.sidebar.warning(
            f"⚠️ Scanned document alert: The following files contain little/no extractable text: "
            f"{', '.join(scanned_warnings)}. OCR may be required."
        )

    progress_bar.empty()


# ==============================================================================
# Sidebar: Document Management & System Info
# ==============================================================================
with st.sidebar:
    st.title("📁 Document Management")

    # API Key Health Check Banner
    has_api_key = check_api_key_configured()
    if not has_api_key:
        st.error(
            "⚠️ **API Key Missing**: `OPENAI_API_KEY` is not set. "
            "Please add your key to `.env` to enable embedding and LLM generation."
        )

    # Multi-file PDF Uploader
    uploaded_files = st.file_uploader(
        "Upload Banking PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload credit agreements, fee schedules, disclosures, or audit reports."
    )

    if st.button("📥 Index Documents", type="primary", use_container_width=True):
        if not has_api_key:
            st.error("Cannot index without OpenAI API key.")
        else:
            with st.spinner("Processing and indexing documents..."):
                handle_document_indexing(uploaded_files)

    st.divider()

    # Indexed Collection Status
    st.subheader("📊 System Information")
    collection = get_or_create_collection()
    stats = get_collection_stats(collection)

    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Indexed Chunks", stats["total_chunks"])
    col_s2.metric("Top-K Retrieval", config.RETRIEVAL_TOP_K)

    st.caption(f"**LLM Model:** `{config.LLM_MODEL}`")
    st.caption(f"**Embedding Model:** `{config.EMBEDDING_MODEL}`")
    st.caption(f"**Vector Store:** ChromaDB (Local SQLite)")

    if st.session_state["indexed_files"]:
        st.markdown("**Indexed Documents:**")
        for f in st.session_state["indexed_files"]:
            st.markdown(f"- 📄 `{f}`")

    # Database Reset Option
    if st.button("🗑️ Reset Vector Database", use_container_width=True):
        clear_collection(collection)
        st.session_state["indexed_files"] = []
        st.session_state["last_result"] = None
        st.success("Vector database reset successfully.")
        st.rerun()


# ==============================================================================
# Main Area: Header & Search Interface
# ==============================================================================
st.title("🏦 Banking Document Intelligence Assistant")
st.markdown("Ask questions about your uploaded banking documents and receive source-grounded answers.")

# Helpful sample question chips for quick testing
if stats["total_chunks"] > 0:
    st.markdown("**Sample Banking Questions:**")
    q_col1, q_col2, q_col3 = st.columns(3)
    if q_col1.button("💳 Premier Checking Fee & Waivers"):
        st.session_state["query_input"] = "What is the monthly maintenance fee for Apex Premier Checking and how can it be waived?"
    if q_col2.button("📑 Commercial Loan DSCR Covenant"):
        st.session_state["query_input"] = "What is the required Debt Service Coverage Ratio (DSCR) and how often is it tested?"
    if q_col3.button("💸 Domestic Wire Transfer Rules"):
        st.session_state["query_input"] = "What are the fees and cutoff times for domestic wire transfers?"

# User Question Form
with st.form("query_form", clear_on_submit=False):
    user_query = st.text_input(
        "Enter your banking question:",
        value=st.session_state.get("query_input", ""),
        placeholder="e.g., What are the terms for prepayment penalties or overdraft caps?",
        key="user_question_input"
    )
    submit_button = st.form_submit_button("🔍 Ask Question", type="primary", use_container_width=False)

if submit_button:
    if not user_query.strip():
        st.warning("Please enter a question before submitting.")
    elif stats["total_chunks"] == 0:
        st.warning("⚠️ No documents indexed yet. Please upload and index banking PDFs in the sidebar first.")
    elif not has_api_key:
        st.error("OpenAI API key is missing. Please add `OPENAI_API_KEY` to `.env`.")
    else:
        with st.spinner(f"Retrieving evidence and generating grounded response using {config.LLM_MODEL}..."):
            result = generate_grounded_answer(
                query=user_query,
                collection=collection
            )
            st.session_state["last_result"] = result


# ==============================================================================
# Results & Source Citations Display
# ==============================================================================
if st.session_state["last_result"]:
    res = st.session_state["last_result"]

    st.divider()

    # Case 1: Application / API / Retrieval Error
    if res.get("error") or res.get("error_type"):
        st.error(
            f"❌ **Application / Service Error**\n\n"
            f"{res['answer']}"
        )
        if res.get("error"):
            st.caption(f"**Technical Details:** `{res['error']}`")

    # Case 2: Insufficient Evidence Fallback (Strict Grounding)
    elif res.get("insufficient_evidence"):
        st.warning(
            f"⚠️ **Insufficient Evidence in Document Context**\n\n"
            f"{res['answer']}"
        )
        st.info(
            "💡 **Analyst Note:** The assistant searched the indexed documents, but the context did not contain "
            "sufficient evidence to answer this question factually. This guardrail prevents hallucinated responses."
        )

    # Case 3: Grounded Factual Answer
    else:
        st.subheader("💡 Grounded Answer")
        st.markdown(res["answer"])

        # Display Source Citations
        if res.get("sources"):
            st.subheader("📚 Source Citations & Supporting Evidence")
            st.caption("Expand any source below to inspect the exact clause, page number, and semantic relevance score.")

            for idx, source in enumerate(res["sources"], 1):
                header_title = (
                    f"Source #{idx}: 📄 {source['source']} — Page {source['page']} "
                    f"(Match Score: {source['similarity_score']:.2%})"
                )
                with st.expander(header_title, expanded=(idx == 1)):
                    st.markdown(f"**Chunk ID:** `{source['chunk_id']}`")
                    st.markdown(f"**Document File:** `{source['source']}`")
                    st.markdown(f"**PDF Page:** `{source['page']}`")
                    st.markdown(f"**Semantic Relevance Score:** `{source['similarity_score']:.4f}`")
                    st.markdown("**Supporting Text Snippet:**")
                    st.info(f"\"{source['snippet']}\"")

        # Technical Metadata Drawer
        if res.get("usage"):
            with st.expander("⚙️ Execution & Token Metadata", expanded=False):
                st.write(f"- **Model Used:** `{res.get('model_used', config.LLM_MODEL)}`")
                st.write(f"- **Prompt Tokens:** `{res['usage'].get('prompt_tokens', 0)}`")
                st.write(f"- **Completion Tokens:** `{res['usage'].get('completion_tokens', 0)}`")
                st.write(f"- **Total Tokens:** `{res['usage'].get('total_tokens', 0)}`")
                st.write(f"- **Retrieved Chunks Evaluated:** `{len(res.get('retrieved_chunks', []))}`")
