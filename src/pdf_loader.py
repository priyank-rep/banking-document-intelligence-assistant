"""
PDF Loader Module for Banking Document Intelligence Assistant.

Responsible for extracting raw text page-by-page from banking PDF files using PyMuPDF (fitz).
Preserves exact 1-indexed page numbers and flags potential scanned/image-only PDFs.
"""

import logging
from pathlib import Path
from typing import Union, BinaryIO
import pymupdf  # PyMuPDF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Minimum characters across a document to consider it text-readable (vs scanned/image)
MIN_TEXT_THRESHOLD_CHARS = 50


def load_pdf_pages(
    file_source: Union[str, Path, bytes, BinaryIO],
    filename: str = "document.pdf"
) -> list[dict]:
    """
    Extract text page-by-page from a PDF file using PyMuPDF.

    Args:
        file_source: File path (str/Path), raw bytes, or a file-like stream (e.g., Streamlit UploadedFile).
        filename: Name of the file used for source tracking in citations.

    Returns:
        A list of dictionaries, where each dictionary represents one page:
        [
            {
                "source": "loan_agreement.pdf",
                "page": 1,                     # 1-indexed page number
                "text": "Extracted text...",
                "char_count": 1240,
                "is_empty": False
            },
            ...
        ]

    Raises:
        ValueError: If the file cannot be opened or is corrupted.
    """
    pages_data = []
    doc = None

    try:
        # 1. Open the PDF from either a file path or in-memory bytes/stream
        if isinstance(file_source, (str, Path)):
            path_obj = Path(file_source)
            filename = path_obj.name
            doc = pymupdf.open(str(path_obj))
        elif isinstance(file_source, bytes):
            doc = pymupdf.open(stream=file_source, filetype="pdf")
        elif hasattr(file_source, "read"):
            # File-like object (e.g., Streamlit UploadedFile)
            content = file_source.read()
            # If the object has a 'name' attribute, use it as filename
            if hasattr(file_source, "name") and file_source.name:
                filename = file_source.name
            doc = pymupdf.open(stream=content, filetype="pdf")
        else:
            raise ValueError(f"Unsupported file source type: {type(file_source)}")

        total_pages = len(doc)
        total_extracted_chars = 0

        # 2. Iterate page-by-page (PyMuPDF uses 0-based indexing internally)
        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_number = page_idx + 1  # Convert to standard 1-based page number

            # Extract clean layout-aware text
            raw_text = page.get_text("text")
            cleaned_text = raw_text.strip()
            char_count = len(cleaned_text)
            total_extracted_chars += char_count

            pages_data.append({
                "source": filename,
                "page": page_number,
                "text": cleaned_text,
                "char_count": char_count,
                "is_empty": char_count == 0
            })

        # 3. Detect scanned / image-only PDFs
        if total_pages > 0 and total_extracted_chars < MIN_TEXT_THRESHOLD_CHARS:
            logger.warning(
                f"⚠️ Warning: '{filename}' has {total_pages} page(s) but only {total_extracted_chars} "
                f"characters of extracted text. This document may be a scanned image or contain non-extractable text."
            )

        logger.info(
            f"Successfully loaded '{filename}': {total_pages} pages, {total_extracted_chars} total characters."
        )
        return pages_data

    except Exception as e:
        logger.error(f"Error extracting text from PDF '{filename}': {str(e)}")
        raise ValueError(f"Failed to extract text from '{filename}': {str(e)}") from e
    finally:
        if doc is not None:
            doc.close()
