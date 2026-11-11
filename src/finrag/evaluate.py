"""Evaluation: retrieval quality, answer correctness, groundedness and abstention."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from finrag.chunking import Chunk


def retrieval_metrics(gold: list[str], retrieved: list[str], k: int) -> dict:
    """recall@k = share of gold chunks found in the top k (all must be found for multi-hop);
    reciprocal rank of the first gold chunk."""
    top = retrieved[:k]
    found = [g for g in gold if any(r == g or r.startswith(g + "-") for r in top)]
    rr = 0.0
    for i, r in enumerate(retrieved):
        if any(r == g or r.startswith(g + "-") for g in gold):
            rr = 1 / (i + 1)
            break
    return {f"recall@{k}": len(found) / len(gold), f"all_found@{k}": float(len(found) == len(gold)), "rr": rr}


def answer_correct(answer: str, keywords: list[str]) -> bool:
    a = answer.lower()
    return all(k.lower() in a for k in keywords)


def grounded(answer: str, context: list[Chunk]) -> bool:
    """Every number in the answer must appear in the retrieved context (a cheap hallucination check)."""
    nums = re.findall(r"\d[\d,]*\.?\d*%?", answer)
    ctx = " ".join(c.text for c in context)
    return all(n in ctx for n in nums)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    answerable = rows[rows.type != "unanswerable"]
    unans = rows[rows.type == "unanswerable"]
    out = answerable.groupby("type").agg(questions=("qid", "size"), recall_at_5=("recall@5", "mean"),
                                         all_found_at_5=("all_found@5", "mean"), mrr=("rr", "mean"),
                                         answer_accuracy=("correct", "mean"), grounded=("grounded", "mean"))
    out.loc["ALL answerable"] = [len(answerable), answerable["recall@5"].mean(), answerable["all_found@5"].mean(),
                                 answerable["rr"].mean(), answerable["correct"].mean(), answerable["grounded"].mean()]
    out.loc["unanswerable (abstained)"] = [len(unans), np.nan, np.nan, np.nan, unans["abstained"].mean(), np.nan]
    return out
