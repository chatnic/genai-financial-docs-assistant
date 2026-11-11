"""Section-aware chunking with metadata.

Annual reports have a natural structure, so chunks follow section boundaries instead of
fixed character windows. Long sections are split on paragraph boundaries with overlap.
Every chunk carries metadata (company, ticker, fiscal year, section) that retrieval can
filter on. The same chunks feed the Databricks Vector Search index in notebooks/.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_CHARS = 1200
OVERLAP_PARAS = 1


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    company: str
    ticker: str
    fiscal_year: int
    section: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def _split_long(text: str) -> list[str]:
    if len(text) <= MAX_CHARS:
        return [text]
    paras = [p for p in text.split("\n\n") if p.strip()]
    heading, body = paras[0], paras[1:]
    out, cur = [], []
    for p in body:
        if cur and len("\n\n".join([heading] + cur + [p])) > MAX_CHARS:
            out.append("\n\n".join([heading] + cur))
            cur = cur[-OVERLAP_PARAS:] if OVERLAP_PARAS else []
        cur.append(p)
    if cur:
        out.append("\n\n".join([heading] + cur))
    return out


def chunk_document(path: Path) -> list[Chunk]:
    raw = path.read_text()
    doc_id = path.stem                                   # e.g. NVPY_FY2025
    ticker, fy = doc_id.split("_FY")
    company = re.match(r"# (.+?) — Annual Report", raw).group(1)
    chunks = []
    for m in re.finditer(r"<!-- section: (\w+) -->\n(.*?)(?=\n<!-- section:|\Z)", raw, re.S):
        section, text = m.group(1), m.group(2).strip()
        for i, part in enumerate(_split_long(text)):
            # Prefix each chunk with its source so the text is self-describing for both
            # keyword and embedding retrieval (a cheap, effective form of contextual chunking).
            context = f"{company} ({ticker}) annual report, fiscal {fy}. "
            chunks.append(Chunk(f"{doc_id}#{section}" + (f"-{i}" if i else ""), doc_id, company, ticker,
                                int(fy), section, context + part))
    return chunks


def chunk_corpus(corpus_dir: Path) -> list[Chunk]:
    return [c for p in sorted(corpus_dir.glob("*.md")) for c in chunk_document(p)]
