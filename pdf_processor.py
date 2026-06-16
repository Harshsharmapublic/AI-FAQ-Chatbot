"""
pdf_processor.py - PDF Text Extraction and Chunking
=====================================================
Author: AI-Powered FAQ Chatbot System
Description:
    Extracts text from uploaded PDF files using pypdf.
    Splits extracted text into overlapping semantic chunks for TF-IDF indexing.
    Handles multi-page PDFs, encoding issues, and empty pages gracefully.

Usage:
    from pdf_processor import PDFProcessor
    processor = PDFProcessor()
    chunks = processor.process("path/to/document.pdf")
"""

import re
import logging
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 500       # characters per chunk
DEFAULT_CHUNK_OVERLAP = 100    # overlap between consecutive chunks
MIN_CHUNK_LENGTH = 80          # discard very short / noisy chunks


# ---------------------------------------------------------------------------
# PDF Processor
# ---------------------------------------------------------------------------

class PDFProcessor:
    """
    Extracts text from PDF files and splits them into searchable chunks.

    Supports:
        - Multi-page PDF extraction using pypdf
        - Paragraph-aware text splitting
        - Sliding-window chunking with configurable overlap
        - Graceful handling of encrypted or corrupt PDFs

    Example:
        >>> proc = PDFProcessor(chunk_size=400, chunk_overlap=80)
        >>> chunks = proc.process("brochure.pdf")
        >>> print(f"Extracted {len(chunks)} chunks")
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        """
        Args:
            chunk_size:     Target character length of each text chunk.
            chunk_overlap:  Number of characters to overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, file_path: str) -> List[str]:
        """
        Extract and chunk text from a PDF file.

        Args:
            file_path:  Absolute or relative path to the PDF file.

        Returns:
            List of text chunks. Empty list if extraction fails.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error("PDF file not found: %s", file_path)
            return []

        raw_text, page_count = self._extract_text(path)
        if not raw_text.strip():
            logger.warning("No text extracted from PDF: %s", file_path)
            return []

        logger.info("Extracted %d characters from %d pages: %s", len(raw_text), page_count, path.name)

        cleaned = self._clean_text(raw_text)
        chunks = self._chunk_text(cleaned)
        logger.info("Created %d chunks from PDF: %s", len(chunks), path.name)
        return chunks

    def process_bytes(self, file_bytes: bytes, filename: str = "document.pdf") -> List[str]:
        """
        Extract and chunk text from PDF bytes (e.g., from Streamlit file_uploader).

        Args:
            file_bytes:  Raw bytes of the PDF.
            filename:    Display name for logging.

        Returns:
            List of text chunks.
        """
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(file_bytes))

            if reader.is_encrypted:
                logger.warning("PDF '%s' is encrypted — cannot extract text.", filename)
                return []

            raw_text = self._read_pages(reader)
            page_count = len(reader.pages)
        except Exception as exc:
            logger.error("Failed to process PDF bytes (%s): %s", filename, exc)
            return []

        if not raw_text.strip():
            logger.warning("No text extracted from PDF bytes: %s", filename)
            return []

        logger.info("Extracted %d chars from %d pages: %s", len(raw_text), page_count, filename)
        cleaned = self._clean_text(raw_text)
        chunks = self._chunk_text(cleaned)
        logger.info("Created %d chunks from: %s", len(chunks), filename)
        return chunks

    def get_page_texts(self, file_path: str) -> List[Tuple[int, str]]:
        """
        Return text per page as a list of (page_number, text) tuples.

        Useful for debugging or page-level processing.
        """
        path = Path(file_path)
        if not path.exists():
            return []

        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            result = []
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                result.append((i, self._clean_text(text)))
            return result
        except Exception as exc:
            logger.error("Failed to get page texts: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_text(self, path: Path) -> Tuple[str, int]:
        """Extract full text from all pages. Returns (text, page_count)."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))

            if reader.is_encrypted:
                logger.warning("PDF is encrypted: %s", path.name)
                return "", 0

            text = self._read_pages(reader)
            return text, len(reader.pages)

        except ImportError:
            logger.error(
                "pypdf is not installed. Run: pip install pypdf"
            )
            return "", 0
        except Exception as exc:
            logger.error("PDF extraction failed (%s): %s", path.name, exc)
            return "", 0

    @staticmethod
    def _read_pages(reader) -> str:
        """Read and concatenate text from all pages of a PdfReader object."""
        pages_text = []
        for page in reader.pages:
            try:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            except Exception:
                pass  # Skip unreadable pages silently
        return "\n\n".join(pages_text)

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean extracted PDF text by removing artifacts and normalizing whitespace.
        """
        # Normalize unicode hyphens / dashes
        text = text.replace("\u2013", "-").replace("\u2014", "-")
        # Remove form feeds and page breaks
        text = text.replace("\x0c", "\n")
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove header/footer-like lines (very short isolated lines)
        lines = text.split("\n")
        lines = [l for l in lines if len(l.strip()) > 3 or l.strip() == ""]
        text = "\n".join(lines)
        # Normalize spaces
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks for semantic indexing.

        Strategy:
            1. Try to split on paragraph boundaries first.
            2. Fall back to sliding-window character chunking.
        """
        # Try paragraph-level split first
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        chunks: List[str] = []
        buffer = ""

        for para in paragraphs:
            if len(buffer) + len(para) <= self.chunk_size:
                buffer += " " + para if buffer else para
            else:
                if buffer and len(buffer) >= MIN_CHUNK_LENGTH:
                    chunks.append(buffer.strip())
                # If the paragraph itself is very long, sub-split it
                if len(para) > self.chunk_size:
                    sub_chunks = self._sliding_window(para)
                    chunks.extend(sub_chunks)
                    buffer = ""
                else:
                    buffer = para

        if buffer and len(buffer) >= MIN_CHUNK_LENGTH:
            chunks.append(buffer.strip())

        return chunks

    def _sliding_window(self, text: str) -> List[str]:
        """Split long text using a sliding window approach."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()
            if len(chunk) >= MIN_CHUNK_LENGTH:
                chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks
