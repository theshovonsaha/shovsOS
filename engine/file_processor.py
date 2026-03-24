"""
File Processor
--------------
Handles uploaded files for multimodal input.

Protocol:
  Images  → base64 encoded, passed directly to vision LLM
  PDFs    → text extracted, injected into user message as context
  Text/Code/CSV → read as text, injected into user message
  Others  → unsupported, friendly error returned

The universal principle: everything becomes language before reaching the LLM.
Images become base64 tokens. Documents become text tokens. 
The adapter handles the translation to whatever the model expects.

Dependencies:
  pip install python-multipart pymupdf
  (pymupdf for PDF — optional; falls back to plain text extraction)
"""

import base64
import io
import mimetypes
from dataclasses import dataclass
from typing import Optional


# ── File type classification ──────────────────────────────────────────────

IMAGE_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/gif",
    "image/webp", "image/bmp"
}

TEXT_TYPES = {
    "text/plain", "text/markdown", "text/csv", "text/html",
    "text/css", "text/javascript", "application/json",
    "application/xml", "application/x-yaml", "text/yaml",
}

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp",
    ".h", ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt",
    ".sh", ".bash", ".zsh", ".sql", ".r", ".m", ".scala",
    ".html", ".css", ".json", ".yaml", ".yml", ".toml", ".xml",
    ".md", ".txt", ".csv", ".env", ".dockerfile",
}

MAX_TEXT_CHARS = 12_000   # ~3k tokens — cap injected text to avoid context explosion
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB


@dataclass
class ProcessedFile:
    filename:     str
    mime_type:    str
    is_image:     bool
    # For images:
    base64_data:  Optional[str] = None
    # For text/docs:
    text_content: Optional[str] = None
    # Metadata
    size_bytes:   int = 0
    error:        Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class FileProcessor:

    def process(self, filename: str, content: bytes, mime_type: Optional[str] = None) -> ProcessedFile:
        """
        Process a raw uploaded file into a form the LLM can consume.
        Returns ProcessedFile with either base64_data (image) or text_content (doc).
        """
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(filename)
            mime_type = mime_type or "application/octet-stream"

        size = len(content)

        if mime_type in IMAGE_TYPES:
            return self._process_image(filename, content, mime_type, size)

        if mime_type == "application/pdf":
            return self._process_pdf(filename, content, mime_type, size)

        if mime_type in TEXT_TYPES or self._is_code_file(filename):
            return self._process_text(filename, content, mime_type, size)

        return ProcessedFile(
            filename=filename,
            mime_type=mime_type,
            is_image=False,
            size_bytes=size,
            error=f"Unsupported file type: {mime_type}. Supported: images, PDF, text, code files."
        )

    def _process_image(self, filename, content, mime_type, size) -> ProcessedFile:
        if size > MAX_IMAGE_SIZE:
            return ProcessedFile(
                filename=filename, mime_type=mime_type, is_image=True, size_bytes=size,
                error=f"Image too large ({size // 1024 // 1024}MB). Max 10MB."
            )
        b64 = base64.b64encode(content).decode("utf-8")
        return ProcessedFile(
            filename=filename,
            mime_type=mime_type,
            is_image=True,
            base64_data=b64,
            size_bytes=size,
        )

    def _process_pdf(self, filename, content, mime_type, size) -> ProcessedFile:
        # Try pymupdf (fitz) first — best PDF extraction
        try:
            import fitz  # pymupdf
            doc = fitz.open(stream=content, filetype="pdf")
            pages = []
            for i, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    pages.append(f"[Page {i+1}]\n{text.strip()}")
            full_text = "\n\n".join(pages)
            doc.close()
            return self._make_text_file(filename, mime_type, size, full_text, source="PDF")
        except ImportError:
            pass

        # Fallback: try pdfminer
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(io.BytesIO(content))
            return self._make_text_file(filename, mime_type, size, text, source="PDF")
        except ImportError:
            pass

        return ProcessedFile(
            filename=filename, mime_type=mime_type, is_image=False, size_bytes=size,
            error="PDF extraction requires: pip install pymupdf  (or pdfminer.six)"
        )

    def _process_text(self, filename, content, mime_type, size) -> ProcessedFile:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("latin-1")
            except Exception:
                return ProcessedFile(
                    filename=filename, mime_type=mime_type, is_image=False, size_bytes=size,
                    error="Could not decode file as text."
                )
        return self._make_text_file(filename, mime_type, size, text, source="File")

    def _make_text_file(self, filename, mime_type, size, text, source) -> ProcessedFile:
        truncated = False
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS]
            truncated = True

        if truncated:
            text += f"\n\n[... truncated at {MAX_TEXT_CHARS} characters ...]"

        return ProcessedFile(
            filename=filename,
            mime_type=mime_type,
            is_image=False,
            text_content=text,
            size_bytes=size,
        )

    def _is_code_file(self, filename: str) -> bool:
        import os
        _, ext = os.path.splitext(filename.lower())
        return ext in CODE_EXTENSIONS

    def build_text_injection(self, files: list[ProcessedFile]) -> str:
        """
        Build text to prepend to the user's message for non-image attachments.
        Images are handled separately via base64 in the LLM adapter.
        """
        parts = []
        for f in files:
            if f.is_image or not f.ok or not f.text_content:
                continue
            ext = f.filename.rsplit(".", 1)[-1].upper() if "." in f.filename else "FILE"
            parts.append(
                f"--- Attached: {f.filename} ({f.size_bytes // 1024}KB) ---\n"
                f"```{ext.lower()}\n{f.text_content}\n```"
            )
        return "\n\n".join(parts)
