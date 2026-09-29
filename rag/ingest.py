"""PDF → pages → clean text → chunks that remember their page.

Chunks never cross a page boundary, so every citation can point to one page.
Tables are kept row by row ("E07 | Battery over temperature | Move ..."),
because a table flattened into one cell per line cannot be retrieved or cited.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

FOOTER_RE = re.compile(r"^.{0,120}(?:—|-)\s*page\s+\d+\s*$", re.I)
COLUMN_GAP = re.compile(r"\S {3,}\S")
NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z][^.!?:;]{0,70}$")


def _is_heading_line(line: str, next_line: str) -> bool:
    """A numbered or Title Case line that starts a block right before a new sentence."""
    if not next_line[:1].isupper() or len(line) >= 60 or line[-1:] in ".:;,!?":
        return False
    if NUMBERED_HEADING.match(line):
        return True
    words = [w for w in re.findall(r"[A-Za-z][\w'-]*", line)]
    return len(words) >= 2 and sum(w[0].isupper() for w in words) / len(words) >= 0.6


@dataclass
class Chunk:
    doc_id: str
    doc_title: str
    source: str          # file name
    page: int            # 1-based
    index: int           # position of the chunk within the page
    text: str
    meta: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}:p{self.page}:c{self.index}"


def _blocks(layout_text: str) -> list[str]:
    """Turn pypdf layout text into paragraphs and table rows."""
    lines = [ln.rstrip() for ln in layout_text.splitlines()]
    lines = [ln for ln in lines if not FOOTER_RE.match(ln.strip())]
    blocks, cur = [], []
    for ln in lines + [""]:
        if ln.strip():
            cur.append(ln)
        elif cur:
            blocks.append(cur)
            cur = []
    out = []
    for b in blocks:
        tabular = sum(bool(COLUMN_GAP.search(ln.strip())) for ln in b) >= max(2, len(b) // 2 + 1)
        if tabular:
            rows = [re.sub(r" {3,}", " | ", ln.strip()) for ln in b]
            out.append("\n".join(rows))
        else:
            lines = [ln.strip() for ln in b]
            # a heading printed right above its paragraph, with no blank line in between
            if len(lines) >= 2 and _is_heading_line(lines[0], lines[1]):
                out.append(lines[0])
                lines = lines[1:]
            out.append(re.sub(r"\s+", " ", " ".join(lines)).strip())
    return [b for b in out if b]


def read_pdf(path: str | Path) -> tuple[str, list[str]]:
    """Return (title, list of cleaned page texts)."""
    reader = PdfReader(str(path))
    meta_title = (reader.metadata.title or "").strip() if reader.metadata else ""
    if not meta_title or meta_title.lower() in {"untitled", "microsoft word - document1", "document"}:
        meta_title = Path(path).stem.replace("-", " ").replace("_", " ").title()
    title = meta_title
    pages = []
    for page in reader.pages:
        try:
            raw = page.extract_text(extraction_mode="layout")
        except Exception:  # some PDFs break layout mode; plain mode is the fallback
            raw = page.extract_text() or ""
        pages.append("\n\n".join(_blocks(raw or "")))
    return title, pages


def doc_id_for(path: str | Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:10]


def chunk_pages(pages: list[str], *, doc_id: str, doc_title: str, source: str,
                max_chars: int = 900, overlap_blocks: int = 1) -> list[Chunk]:
    """Greedy packing of paragraphs/table blocks into chunks of ~max_chars.

    Headings stay attached to the paragraph that follows them, and the last
    block of a chunk is repeated at the start of the next one (overlap) so an
    answer that straddles two chunks is still retrievable.
    """
    chunks: list[Chunk] = []
    for pno, text in enumerate(pages, 1):
        blocks = [b for b in text.split("\n\n") if b.strip()]
        # glue short headings ("3. Error codes") to the next block
        merged: list[str] = []
        carry = ""
        for b in blocks:
            if len(b) < 60 and not b.endswith(".") and "|" not in b:
                carry = (carry + "\n" + b).strip()  # one heading per line, so answers can skip them
                continue
            merged.append((carry + "\n" + b).strip() if carry else b)
            carry = ""
        if carry:
            merged.append(carry)

        cur: list[str] = []
        idx = 0
        for b in merged:
            if cur and sum(len(x) for x in cur) + len(b) > max_chars:
                chunks.append(Chunk(doc_id, doc_title, source, pno, idx, "\n\n".join(cur)))
                idx += 1
                cur = cur[-overlap_blocks:] if overlap_blocks else []
            cur.append(b)
        if cur:
            chunks.append(Chunk(doc_id, doc_title, source, pno, idx, "\n\n".join(cur)))
    return chunks


def ingest_pdf(path: str | Path, **kw) -> list[Chunk]:
    title, pages = read_pdf(path)
    return chunk_pages(pages, doc_id=doc_id_for(path), doc_title=title, source=Path(path).name, **kw)
