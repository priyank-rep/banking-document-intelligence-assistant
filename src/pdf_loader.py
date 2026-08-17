"""
PDF Loader Module for Banking Document Intelligence Assistant.

Responsible for extracting raw text page-by-page from banking PDF files using PyMuPDF (fitz).
Preserves exact 1-indexed page numbers and flags potential scanned/image-only PDFs.
"""

import logging
from pathlib import Path
from typing import Union, BinaryIO, Optional
import pymupdf  # PyMuPDF

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Minimum characters across a document to consider it text-readable (vs scanned/image)
MIN_TEXT_THRESHOLD_CHARS = 50


def extract_page_text_layout_aware(
    page: pymupdf.Page,
    filter_headers_footers: bool = config.PDF_FILTER_HEADERS_FOOTERS,
    header_margin_pt: float = config.PDF_HEADER_MARGIN_PT,
    footer_margin_pt: float = config.PDF_FOOTER_MARGIN_PT
) -> str:
    """
    Extract text from a PyMuPDF page using layout-aware block sorting, coordinate filtering,
    and deterministic font-aware symbol glyph remapping (e.g., legacy ITFRupee / Rupee font symbols).

    1. Uses page.get_text("dict") to inspect bounding boxes and font families.
    2. Filters out repeating header blocks (y1 <= header_margin_pt) and footer blocks (y0 >= page_height - footer_margin_pt).
    3. Remaps legacy custom symbol fonts (e.g., ITFRupee 'J', 'I', 'C', 'H' and Rupee '`') to standard Indian Rupee '₹'.
    4. Combines remaining body blocks in natural reading order with clean paragraph breaks (\\n\\n).
    5. Falls back gracefully to standard text if dict extraction returns empty on a non-empty page.
    """
    rect = page.rect
    page_height = rect.height
    header_y_limit = header_margin_pt
    footer_y_limit = page_height - footer_margin_pt

    page_dict = page.get_text("dict")
    raw_blocks = page_dict.get("blocks", [])

    if not raw_blocks:
        return page.get_text("text").strip()

    # Sort blocks vertically, then horizontally for stable reading order
    sorted_blocks = sorted(
        raw_blocks,
        key=lambda b: (b.get("bbox", (0, 0, 0, 0))[1], b.get("bbox", (0, 0, 0, 0))[0])
    )

    body_blocks = []
    for b in sorted_blocks:
        # type 0 = text, 1 = image
        if b.get("type") != 0:
            continue

        bbox = b.get("bbox", (0, 0, 0, 0))
        # Coordinate filter for headers and footers
        if filter_headers_footers:
            if bbox[3] <= header_y_limit:
                continue
            if bbox[1] >= footer_y_limit:
                continue

        block_lines = []
        for line in b.get("lines", []):
            line_parts = []
            for span in line.get("spans", []):
                span_text = span.get("text", "")
                font_name = span.get("font", "").lower()

                # Deterministic Rupee Font Normalization:
                # Custom legacy fonts (ITFRupee, Rupee) placed the ₹ symbol at ASCII J, I, C, H, or `
                if "rupee" in font_name:
                    span_text = "₹ " if span_text.endswith(" ") else "₹"

                line_parts.append(span_text)

            line_str = "".join(line_parts).strip()
            if line_str:
                block_lines.append(line_str)

        if block_lines:
            body_blocks.append("\n".join(block_lines))

    if not body_blocks:
        # Fallback: if coordinate filtering pruned all blocks, preserve unfiltered text
        unfiltered = []
        for b in sorted_blocks:
            if b.get("type") == 0:
                for line in b.get("lines", []):
                    line_str = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
                    if line_str:
                        unfiltered.append(line_str)
        if unfiltered:
            return "\n".join(unfiltered)
        return ""

    return "\n\n".join(body_blocks)


def load_pdf_pages(
    file_source: Union[str, Path, bytes, BinaryIO],
    filename: str = "document.pdf",
    filter_headers_footers: Optional[bool] = None,
    header_margin_pt: Optional[float] = None,
    footer_margin_pt: Optional[float] = None
) -> list[dict]:
    """
    Extract text page-by-page from a PDF file using PyMuPDF layout-aware block extraction.

    Args:
        file_source: File path (str/Path), raw bytes, or a file-like stream (e.g., Streamlit UploadedFile).
        filename: Name of the file used for source tracking in citations.
        filter_headers_footers: Optional override for header/footer coordinate filtering.
        header_margin_pt: Optional override for top header margin threshold.
        footer_margin_pt: Optional override for bottom footer margin threshold.

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
    use_filter = (
        filter_headers_footers
        if filter_headers_footers is not None
        else config.PDF_FILTER_HEADERS_FOOTERS
    )
    h_margin = (
        header_margin_pt
        if header_margin_pt is not None
        else config.PDF_HEADER_MARGIN_PT
    )
    f_margin = (
        footer_margin_pt
        if footer_margin_pt is not None
        else config.PDF_FOOTER_MARGIN_PT
    )

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

            # Extract clean layout-aware text with coordinate filtering
            cleaned_text = extract_page_text_layout_aware(
                page=page,
                filter_headers_footers=use_filter,
                header_margin_pt=h_margin,
                footer_margin_pt=f_margin
            )
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
