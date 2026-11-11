"""Retrievers: BM25, TF-IDF cosine, hybrid (reciprocal rank fusion), and metadata filtering.

Locally these are all keyword-based (no embedding model is installed). On Databricks the
dense retriever is Mosaic AI Vector Search with a managed embedding model. See notebooks/.
The metadata filter works the same way in both places.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from finrag.chunking import Chunk

STOP = set("""a an the of in on for to and or is was were be by with what which how much many did does do
its it that this as at from who whom whose company fiscal report year end say about""".split())


def _stem(t: str) -> str:
    """Very light stemming so 'factors' matches 'factor' and 'segments' matches 'segment'."""
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss") and not t[0].isdigit():
        return t[:-1]
    return t


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    return [_stem(t) for t in toks if t not in STOP and len(t) > 1]


class BM25:
    # k1 is lower than the usual 1.2-2.0: in short, templated report sections, a term repeated
    # several times (e.g. "revenue" in a segment table) carries little extra signal.
    def __init__(self, k1: float = 0.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, chunks: list[Chunk]) -> "BM25":
        self.chunks = chunks
        self.docs = [Counter(tokenize(c.text)) for c in chunks]
        self.lens = np.array([sum(d.values()) for d in self.docs])
        self.avg = self.lens.mean()
        df = Counter(t for d in self.docs for t in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        return self

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        s = np.zeros(len(self.docs))
        for i, d in enumerate(self.docs):
            for t in q:
                if t in d:
                    tf = d[t]
                    s[i] += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
        return s


class TfidfRetriever:
    def fit(self, chunks: list[Chunk]) -> "TfidfRetriever":
        self.chunks = chunks
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, ngram_range=(1, 2), sublinear_tf=True,
                                   token_pattern=None)
        self.X = self.vec.fit_transform([c.text for c in chunks])
        return self

    def scores(self, query: str) -> np.ndarray:
        return (self.X @ self.vec.transform([query]).T).toarray().ravel()


def rrf(score_lists: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal rank fusion: combine rankings without needing comparable score scales."""
    fused = np.zeros(len(score_lists[0]))
    for s in score_lists:
        ranks = np.empty(len(s), dtype=int)
        ranks[np.argsort(-s)] = np.arange(1, len(s) + 1)
        fused += 1 / (k + ranks)
    return fused


def extract_filters(question: str, chunks: list[Chunk]) -> dict:
    """Self-querying: pull company and fiscal-year constraints out of the question."""
    q = question.lower()
    companies = sorted({c.company for c in chunks if c.company.lower() in q or
                        c.ticker.lower() in re.findall(r"[a-z]+", q) or c.company.split()[0].lower() in q.split()})
    years = sorted({int(y) for y in re.findall(r"(?:fiscal|fy)\s*(20\d\d)", q)})
    return {"companies": companies, "years": years}


class Retriever:
    """mode: 'bm25' | 'tfidf' | 'hybrid'; use_filters applies metadata filtering first."""

    def __init__(self, chunks: list[Chunk], mode: str = "hybrid", use_filters: bool = True):
        self.chunks, self.mode, self.use_filters = chunks, mode, use_filters
        self.bm25 = BM25().fit(chunks)
        self.tfidf = TfidfRetriever().fit(chunks)
        self.known_years = {c.fiscal_year for c in chunks}

    def search(self, question: str, k: int = 5) -> list[tuple[Chunk, float]]:
        if self.mode == "bm25":
            s = self.bm25.scores(question)
        elif self.mode == "tfidf":
            s = self.tfidf.scores(question)
        else:
            s = rrf([self.bm25.scores(question), self.tfidf.scores(question)])
        mask = np.ones(len(self.chunks), dtype=bool)
        if self.use_filters:
            f = extract_filters(question, self.chunks)
            if f["companies"]:
                mask &= np.array([c.company in f["companies"] for c in self.chunks])
            # Comparison questions mention two years; keep both. The FY+1 guidance lives in FY's report.
            years = set(f["years"]) | {y - 1 for y in f["years"] if "guide" in question.lower()}
            years &= self.known_years
            if years:
                mask &= np.array([c.fiscal_year in years for c in self.chunks])
        s = np.where(mask, s, -np.inf)
        top = np.argsort(-s)[:k]
        return [(self.chunks[i], float(s[i])) for i in top if np.isfinite(s[i])]
