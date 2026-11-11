"""Run the retrieval ablation and the end-to-end extractive evaluation locally.

    PYTHONPATH=src python -m finrag.pipeline --data data --out reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from finrag.answer import ExtractiveAnswerer  # noqa: E402
from finrag.chunking import chunk_corpus  # noqa: E402
from finrag.evaluate import answer_correct, grounded, retrieval_metrics, summarize  # noqa: E402
from finrag.retrieval import Retriever  # noqa: E402

CONFIGS = [("bm25", False), ("tfidf", False), ("hybrid", False), ("bm25", True), ("tfidf", True), ("hybrid", True)]


def run(data: Path, out: Path) -> dict:
    chunks = chunk_corpus(data / "corpus")
    questions = json.loads((data / "eval_questions.json").read_text())
    answerable = [q for q in questions if q["type"] != "unanswerable"]
    print(f"{len(chunks)} chunks from {len({c.doc_id for c in chunks})} documents; {len(questions)} questions")

    # 1. Retrieval ablation
    abl = []
    for mode, filt in CONFIGS:
        r = Retriever(chunks, mode, filt)
        ms = []
        for q in answerable:
            ids = [c.chunk_id for c, _ in r.search(q["question"], k=10)]
            m = retrieval_metrics(q["gold_sources"], ids, k=5)
            m |= {"recall@1": retrieval_metrics(q["gold_sources"], ids, k=1)["recall@1"], "type": q["type"]}
            ms.append(m)
        ms = pd.DataFrame(ms)
        abl.append({"retriever": mode, "metadata_filter": filt, "recall@1": ms["recall@1"].mean(),
                    "recall@5": ms["recall@5"].mean(), "all_found@5": ms["all_found@5"].mean(), "mrr": ms["rr"].mean(),
                    "multi_hop_all_found@5": ms.loc[ms.type != "lookup", "all_found@5"].mean()})
        print(f"  {mode:<7} filter={filt!s:<5} recall@1 {abl[-1]['recall@1']:.2f}  recall@5 {abl[-1]['recall@5']:.2f}  "
              f"MRR {abl[-1]['mrr']:.2f}")
    abl = pd.DataFrame(abl)

    # 2. End-to-end with the best retrieval configuration and the extractive answerer
    best_cfg = abl.sort_values("mrr", ascending=False).iloc[0]
    r = Retriever(chunks, best_cfg.retriever, bool(best_cfg.metadata_filter))
    ans = ExtractiveAnswerer({c.fiscal_year for c in chunks})
    rows = []
    for q in questions:
        hits = r.search(q["question"], k=5)
        ids = [c.chunk_id for c, _ in hits]
        a = ans.answer(q["question"], hits)
        row = {"qid": q["qid"], "type": q["type"], "question": q["question"], "expected": q["answer"],
               "answer": a["answer"], "abstained": a["abstained"], "citations": ";".join(a["citations"])}
        if q["type"] == "unanswerable":
            row |= {"recall@5": np.nan, "all_found@5": np.nan, "rr": np.nan, "correct": a["abstained"],
                    "grounded": np.nan}
        else:
            row |= retrieval_metrics(q["gold_sources"], ids, k=5)
            row["correct"] = (not a["abstained"]) and answer_correct(a["answer"], q["answer_keywords"])
            row["grounded"] = grounded(a["answer"], [c for c, _ in hits]) if not a["abstained"] else True
        rows.append(row)
    rows = pd.DataFrame(rows)
    summary = summarize(rows)
    print(summary.round(3).to_string())

    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    abl.round(3).to_csv(out / "tables" / "retrieval_ablation.csv", index=False)
    summary.round(3).to_csv(out / "tables" / "end_to_end_extractive.csv")
    rows.to_csv(out / "tables" / "question_level_results.csv", index=False)
    best = abl.sort_values("mrr", ascending=False).iloc[0]
    metrics = {
        "chunks": len(chunks), "questions": len(questions),
        "best_retriever": {k: (v.item() if hasattr(v, "item") else v) for k, v in best.to_dict().items()},
        "no_filter_bm25": abl[(abl.retriever == "bm25") & (~abl.metadata_filter)].iloc[0].to_dict(),
        "extractive_answer_accuracy": summary.loc["ALL answerable", "answer_accuracy"],
        "extractive_lookup_accuracy": summary.loc["lookup", "answer_accuracy"],
        "abstention_rate_on_unanswerable": summary.loc["unanswerable (abstained)", "answer_accuracy"],
        "false_abstention_rate_on_answerable": float(rows.loc[rows.type != "unanswerable", "abstained"].mean()),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    figure(abl, out / "figures")
    return metrics


def figure(abl: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    fig, ax = plt.subplots(figsize=(8, 4))
    modes = ["bm25", "tfidf", "hybrid"]
    x = np.arange(len(modes))
    for i, (filt, col, lab) in enumerate([(False, "#9aa5b1", "no metadata filter"),
                                          (True, "#1f4e8c", "with company/year filter")]):
        vals = [abl[(abl.retriever == m) & (abl.metadata_filter == filt)]["all_found@5"].iloc[0] for m in modes]
        ax.bar(x + (i - 0.5) * 0.36, vals, 0.36, color=col, label=lab)
        for xi, v in zip(x + (i - 0.5) * 0.36, vals):
            ax.text(xi, v + 0.01, f"{v:.0%}", ha="center", fontsize=8.5)
    ax.set_xticks(x, ["BM25", "TF-IDF", "Hybrid (RRF)"])
    ax.set_ylabel("Questions with every needed\nsection in the top 5")
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Metadata filtering (company + fiscal year) helps every retriever")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(fig_dir / "retrieval_ablation.png", dpi=150); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    a = p.parse_args()
    run(a.data, a.out)


if __name__ == "__main__":
    main()
