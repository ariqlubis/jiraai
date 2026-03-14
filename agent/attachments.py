import os
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple


MAX_ATTACHMENT_CHARS = int(os.getenv("JIRAAI_ATTACHMENT_MAX_CHARS", "8000"))

_PDF_REF_RE = re.compile(
    r"@(?P<path>[^\s#]+?\.pdf)(?:#pages=(?P<pages>[0-9,\-\s]+|all))?",
    re.IGNORECASE,
)


def _sanitize_pdf_path(path: str) -> Tuple[str, str]:
    suffix = ""
    while path and path[-1] in {".", ",", ";", ":", ")", "]", "}"}:
        candidate = path[:-1]
        if candidate.lower().endswith(".pdf"):
            suffix = path[-1] + suffix
            path = candidate
        else:
            break
    return path, suffix


def _resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def parse_page_spec(spec: str, max_pages: Optional[int] = None) -> list[int]:
    cleaned = (spec or "").strip().lower()
    if cleaned in {"", "all"}:
        if max_pages is None:
            return []
        return list(range(max_pages))

    pages: set[int] = set()
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    for part in parts:
        if "-" in part:
            start_text, end_text = [p.strip() for p in part.split("-", 1)]
            if not start_text or not end_text:
                raise ValueError(f"Invalid page range '{part}'")
            start = int(start_text)
            end = int(end_text)
            if start <= 0 or end <= 0:
                raise ValueError("Page numbers must be >= 1")
            if end < start:
                raise ValueError(f"Invalid page range '{part}'")
            for page in range(start, end + 1):
                pages.add(page - 1)
        else:
            page = int(part)
            if page <= 0:
                raise ValueError("Page numbers must be >= 1")
            pages.add(page - 1)

    if max_pages is not None:
        for page in pages:
            if page >= max_pages:
                raise ValueError(f"Page {page + 1} is out of range")

    return sorted(pages)


def _extract_pdf_text(path: Path, pages_spec: Optional[str]) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError("pypdf is required to read PDF attachments") from exc

    reader = PdfReader(str(path))
    total_pages = len(reader.pages)
    pages = parse_page_spec(pages_spec or "", max_pages=total_pages)
    if not pages:
        pages = list(range(total_pages))

    chunks: list[str] = []
    for idx in pages:
        page = reader.pages[idx]
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            chunks.append(f"--- Page {idx + 1} ---\n{text}")
    return "\n\n".join(chunks)


def expand_attachments(text: str) -> Tuple[str, list[str]]:
    if not text:
        return text, []

    matches = list(_PDF_REF_RE.finditer(text))
    if not matches:
        return text, []

    attachment_blocks: list[str] = []
    errors: list[str] = []

    for match in matches:
        raw_path = match.group("path")
        pages_spec = match.group("pages")

        clean_path, _suffix = _sanitize_pdf_path(raw_path)
        resolved = _resolve_path(clean_path)

        if not resolved.exists():
            message = f"Attachment not found: {resolved}"
            errors.append(message)
            attachment_blocks.append(f"[Attachment error] {message}")
            continue

        if resolved.suffix.lower() != ".pdf":
            message = f"Attachment is not a PDF: {resolved}"
            errors.append(message)
            attachment_blocks.append(f"[Attachment error] {message}")
            continue

        try:
            content = _extract_pdf_text(resolved, pages_spec)
            if not content:
                content = "[No extractable text found]"
            if len(content) > MAX_ATTACHMENT_CHARS:
                content = content[:MAX_ATTACHMENT_CHARS].rstrip() + "\n[Truncated]"
            pages_label = f" pages={pages_spec}" if pages_spec else " all pages"
            attachment_blocks.append(
                f"[Attachment] {resolved}{pages_label}\n{content}"
            )
        except Exception as exc:
            message = f"Failed to read {resolved}: {exc}"
            errors.append(message)
            attachment_blocks.append(f"[Attachment error] {message}")

    if not attachment_blocks:
        return text, errors

    appended = "\n\n".join(attachment_blocks)
    expanded = f"{text}\n\n---\nAttachments:\n{appended}"
    return expanded, errors


__all__ = [
    "expand_attachments",
    "parse_page_spec",
    "_sanitize_pdf_path",
]
