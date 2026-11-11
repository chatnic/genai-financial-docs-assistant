import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finrag.answer import ABSTAIN, ExtractiveAnswerer, LLMAnswerer  # noqa: E402
from finrag.chunking import chunk_corpus  # noqa: E402
from finrag.evaluate import answer_correct, grounded, retrieval_metrics  # noqa: E402
from finrag.retrieval import Retriever, extract_filters, rrf, tokenize  # noqa: E402

CORPUS = ROOT / "data" / "corpus"


@pytest.fixture(scope="module")
def chunks():
    return chunk_corpus(CORPUS)


def test_chunks_have_metadata(chunks):
    assert len(chunks) >= 72
    c = chunks[0]
    assert c.company and c.ticker and c.fiscal_year in (2024, 2025)
    assert c.text.startswith(c.company)


def test_every_gold_source_exists(chunks):
    ids = {c.chunk_id.split("-")[0] for c in chunks}
    qs = json.loads((ROOT / "data" / "eval_questions.json").read_text())
    for q in qs:
        for s in q["gold_sources"]:
            assert s in ids, s


def test_filter_extraction(chunks):
    f = extract_filters("Did NovaPay Technologies' margin improve from fiscal 2024 to fiscal 2025?", chunks)
    assert f == {"companies": ["NovaPay Technologies"], "years": [2024, 2025]}


def test_filtered_search_stays_in_scope(chunks):
    r = Retriever(chunks, "bm25", True)
    hits = r.search("How much cash did Harborline Financial have at the end of fiscal 2024?", k=5)
    assert all(c.company == "Harborline Financial" and c.fiscal_year == 2024 for c, _ in hits)
    assert hits[0][0].section == "liquidity"


def test_abstains_on_missing_year(chunks):
    r = Retriever(chunks, "bm25", True)
    q = "What was Harborline Financial's total revenue in fiscal 2019?"
    a = ExtractiveAnswerer({2024, 2025}).answer(q, r.search(q))
    assert a["abstained"] and a["answer"] == ABSTAIN


def test_metrics():
    m = retrieval_metrics(["A#x", "B#y"], ["C#z", "A#x", "B#y-1"], k=2)
    assert m["recall@2"] == 0.5 and m["rr"] == 0.5
    assert answer_correct("Revenue was $1.95 billion.", ["$1.95 billion"])


def test_grounded_catches_invented_numbers(chunks):
    assert not grounded("Revenue was $9,999.9 billion", chunks[:3])


def test_rrf_prefers_consensus():
    import numpy as np
    fused = rrf([np.array([3.0, 2.0, 1.0]), np.array([1.0, 3.0, 2.0])])
    assert fused.argmax() == 1


def test_tokenize_stems_plurals():
    assert tokenize("Risk Factors segments") == ["risk", "factor", "segment"]


def test_llm_prompt_contains_context_and_question(chunks):
    p = LLMAnswerer.build_prompt("What is X?", [(chunks[0], 1.0)])
    assert chunks[0].chunk_id in p and "What is X?" in p
