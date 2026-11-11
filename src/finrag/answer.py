"""Answer generation.

Two answerers share one interface:

* ExtractiveAnswerer runs locally with no model: it returns the single retrieved sentence
  that best matches the question, or abstains. It is a floor for comparison, and it makes
  the whole pipeline testable offline.
* LLMAnswerer builds a grounded prompt (prompts/answer_with_citations.md) and calls a chat
  model: a Databricks Foundation Model API endpoint or any OpenAI-compatible endpoint. It was
  not run to produce the README numbers. See notebooks/03_rag_chain_and_evaluation.py.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from finrag.chunking import Chunk
from finrag.retrieval import extract_filters, tokenize

ABSTAIN = "I can't find that in the available reports."
PROMPT = (Path(__file__).resolve().parents[2] / "prompts" / "answer_with_citations.md").read_text()


def _sentences(text: str) -> list[str]:
    text = re.sub(r"^.*?annual report, fiscal \d{4}\. ", "", text)      # drop the chunk context prefix
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip(" -*#") for p in parts if len(p.strip()) > 20 and not p.startswith("##")]


class ExtractiveAnswerer:
    """Pick the best-matching sentence from the top chunks. Abstain when the question asks for
    a year the corpus doesn't cover, or when no sentence shares a content word with the question
    beyond the company name and year."""

    name = "extractive"

    def __init__(self, known_years: set[int], top_chunks: int = 3):
        self.known_years, self.top_chunks = known_years, top_chunks

    def answer(self, question: str, hits: list[tuple[Chunk, float]]) -> dict:
        asked_years = {int(y) for y in re.findall(r"(?:fiscal|fy)\s*(20\d\d)", question.lower())}
        guide = "guide" in question.lower()
        report_years = {y - 1 for y in asked_years} if guide else asked_years
        if report_years and not report_years & self.known_years:
            return {"answer": ABSTAIN, "abstained": True, "citations": []}
        entity_terms = set()
        for c in {h[0].company for h in hits}:
            entity_terms |= set(tokenize(c))
        q_terms = set(tokenize(question)) - entity_terms - {str(y) for y in asked_years}
        best, best_score, best_src = None, 0.0, None
        for rank, (chunk, _) in enumerate(hits[: self.top_chunks]):
            for s in _sentences(chunk.text):
                overlap = len(q_terms & set(tokenize(s)))
                if overlap == 0:
                    continue
                # Tie-breakers: the sentence mentions the asked year, and the chunk ranked higher.
                if any(str(y) in s for y in asked_years):
                    overlap += 0.5
                overlap += 0.1 * (self.top_chunks - rank)
                # Prefer sentences with figures when the question asks for a quantity.
                if re.search(r"how much|how many|what was|range", question.lower()) and re.search(r"\d", s):
                    overlap += 0.5
                if overlap > best_score:
                    best, best_score, best_src = s, overlap, chunk.chunk_id
        if best is None or best_score < 1:
            return {"answer": ABSTAIN, "abstained": True, "citations": []}
        return {"answer": best, "abstained": False, "citations": [best_src]}


class LLMAnswerer:
    """Grounded generation with citations via an OpenAI-compatible chat endpoint.

    Databricks:   LLM_BASE_URL=https://<workspace>/serving-endpoints
                  LLM_MODEL=databricks-meta-llama-3-3-70b-instruct   LLM_API_KEY=<PAT>
    Any other OpenAI-compatible provider works the same way.
    """

    name = "llm"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("LLM_MODEL", "databricks-meta-llama-3-3-70b-instruct")
        self.base_url = os.environ.get("LLM_BASE_URL", "")
        self.api_key = os.environ.get("LLM_API_KEY", "")

    @staticmethod
    def build_prompt(question: str, hits: list[tuple[Chunk, float]]) -> str:
        context = "\n\n".join(f"[{c.chunk_id}]\n{c.text}" for c, _ in hits)
        return PROMPT.replace("{context}", context).replace("{question}", question)

    def answer(self, question: str, hits: list[tuple[Chunk, float]]) -> dict:
        import requests

        r = requests.post(f"{self.base_url}/chat/completions",
                          headers={"Authorization": f"Bearer {self.api_key}"},
                          json={"model": self.model, "temperature": 0, "max_tokens": 300,
                                "messages": [{"role": "user", "content": self.build_prompt(question, hits)}]},
                          timeout=60)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        cites = re.findall(r"\[([A-Z]+_FY\d{4}#\w+(?:-\d+)?)\]", text)
        return {"answer": text, "abstained": ABSTAIN.lower()[:20] in text.lower(), "citations": cites}


__all__ = ["ABSTAIN", "ExtractiveAnswerer", "LLMAnswerer", "extract_filters"]
