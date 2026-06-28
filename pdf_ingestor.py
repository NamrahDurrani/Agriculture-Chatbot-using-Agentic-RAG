"""
PDF Ingestion Module
====================
Handles text extraction from PDF files with two strategies:
  1. Direct text layer extraction via PyMuPDF (fast, for text-based PDFs)
  2. OCR fallback via pytesseract (for scanned/image-based PDFs like PARC report)

Then chunks the text and returns (chunk_text, source_file, page_num) tuples
ready for embedding.

Why PyMuPDF + pytesseract?
- PARC Annual Report has 0 embedded fonts → it's a scanned PDF → OCR required
- Punjab Agri Dept rules PDF is also scanned (0 fonts)
- FAO Guidelines PDF has 16 fonts → direct text extraction works fine
"""

import os
import io
import re
import fitz          # PyMuPDF
import pytesseract
from PIL import Image
from typing import List, Tuple


# ── Chunking parameters ───────────────────────────────────────────────────────
CHUNK_SIZE   = 400   # target characters per chunk
CHUNK_OVERLAP = 80   # overlap to preserve context at boundaries


def _extract_text_direct(page: fitz.Page) -> str:
    """Extract text directly from PDF text layer (works for non-scanned PDFs)."""
    return page.get_text("text").strip()


def _extract_text_ocr(page: fitz.Page, dpi_scale: float = 2.5) -> str:
    """
    Rasterize a PDF page and run Tesseract OCR on it.
    dpi_scale=2.5 gives ~180 DPI — good balance of accuracy vs speed.
    """
    mat = fitz.Matrix(dpi_scale, dpi_scale)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    # --psm 6 = assume a uniform block of text
    text = pytesseract.image_to_string(img, config="--psm 6")
    return text.strip()


def _clean_text(text: str) -> str:
    """Remove excessive whitespace and noise from extracted text."""
    # Replace multiple spaces / tabs with single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are just punctuation/noise (OCR artifacts)
    lines = [ln for ln in text.split("\n")
             if len(ln.strip()) > 3 or ln.strip() == ""]
    return "\n".join(lines).strip()


def _chunk_text(
    text: str,
    source_file: str,
    page_num: int,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Tuple[str, str, int]]:
    """
    Split text into overlapping chunks.
    Returns list of (chunk_text, source_file, page_num).
    Tries to split at sentence/paragraph boundaries.
    """
    if not text:
        return []

    # Split on paragraph or sentence boundaries first
    # Prefer splitting at double-newlines, then single newlines, then periods
    sentences = re.split(r"(\n\n|\.\s+|\n)", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) + 1 <= chunk_size:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append((current, source_file, page_num))
            # Start new chunk with overlap from previous
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = (overlap_text + " " + sent).strip()

    if current:
        chunks.append((current, source_file, page_num))

    return chunks


def ingest_pdf(
    pdf_path: str,
    ocr_threshold: int = 50,
    max_pages: int = None,
    verbose: bool = True,
) -> List[Tuple[str, str, int]]:
    """
    Extract and chunk all text from a PDF file.

    Args:
        pdf_path:       Path to PDF file.
        ocr_threshold:  If direct extraction yields fewer than this many
                        characters per page, fall back to OCR.
        max_pages:      If set, only process first N pages (useful for testing).
        verbose:        Print progress.

    Returns:
        List of (chunk_text, source_filename, page_number) tuples.
    """
    source_name = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    if max_pages:
        total_pages = min(total_pages, max_pages)

    if verbose:
        print(f"\n[INGEST] {source_name} — {total_pages} pages")

    all_chunks = []
    ocr_pages = 0
    direct_pages = 0

    for page_num in range(total_pages):
        page = doc[page_num]

        # Try direct extraction first
        direct_text = _extract_text_direct(page)

        if len(direct_text) >= ocr_threshold:
            text = _clean_text(direct_text)
            direct_pages += 1
        else:
            # Fall back to OCR (scanned page)
            ocr_text = _extract_text_ocr(page)
            text = _clean_text(ocr_text)
            ocr_pages += 1

        page_chunks = _chunk_text(text, source_name, page_num + 1)
        all_chunks.extend(page_chunks)

        if verbose and (page_num + 1) % 10 == 0:
            print(f"  [INGEST] Processed page {page_num+1}/{total_pages} "
                  f"| chunks so far: {len(all_chunks)}")

    doc.close()

    if verbose:
        print(f"  [INGEST] Done — {len(all_chunks)} chunks total "
              f"(direct:{direct_pages} pages, ocr:{ocr_pages} pages)")

    # Filter out very short or empty chunks (OCR noise)
    all_chunks = [(c, s, p) for c, s, p in all_chunks if len(c.strip()) > 40]
    if verbose:
        print(f"  [INGEST] After filtering: {len(all_chunks)} quality chunks")

    return all_chunks


def ingest_all_pdfs(
    pdf_dir: str,
    pdf_files: List[str] = None,
    max_pages_per_pdf: int = None,
    verbose: bool = True,
) -> List[Tuple[str, str, int]]:
    """
    Ingest multiple PDFs from a directory.

    Args:
        pdf_dir:          Directory containing PDFs.
        pdf_files:        Optional explicit list of filenames. If None, all PDFs.
        max_pages_per_pdf: Limit pages per PDF (for testing).

    Returns:
        Combined list of all chunks from all PDFs.
    """
    if pdf_files is None:
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]

    all_chunks = []
    for fname in pdf_files:
        path = os.path.join(pdf_dir, fname)
        if not os.path.exists(path):
            print(f"[INGEST] WARNING: {path} not found, skipping.")
            continue
        chunks = ingest_pdf(path, max_pages=max_pages_per_pdf, verbose=verbose)
        all_chunks.extend(chunks)

    print(f"\n[INGEST] Total chunks across all PDFs: {len(all_chunks)}")
    return all_chunks


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    default_pdf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdfs", "i5550e.pdf")
    pdf = sys.argv[1] if len(sys.argv) > 1 else default_pdf
    chunks = ingest_pdf(pdf, max_pages=5, verbose=True)
    print(f"\nSample chunks:")
    for i, (text, src, pg) in enumerate(chunks[:3]):
        print(f"\n[Chunk {i+1}] src={src} page={pg}")
        print(text[:300])
