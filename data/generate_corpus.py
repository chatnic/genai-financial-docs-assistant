"""Generate a corpus of fictional annual reports plus a gold-standard evaluation set.

Six fictional financial-services companies x two fiscal years = 12 annual-report style
documents (Markdown), built from a table of facts. Because every number comes from the fact
table, each evaluation question has a known answer and a known source section, so the
evaluation is exact rather than a matter of opinion.

All company names, people and figures are invented.

Usage:
    python data/generate_corpus.py --seed 42 --out data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COMPANIES = [
    {"name": "Harborline Financial", "ticker": "HBLF", "industry": "regional bank",
     "segments": ["Consumer Banking", "Commercial Banking", "Wealth Management"], "base_rev": 2400, "hq": "Portland, Maine"},
    {"name": "NovaPay Technologies", "ticker": "NVPY", "industry": "payments processor",
     "segments": ["Merchant Services", "Issuer Processing", "Cross-Border Payments"], "base_rev": 1850, "hq": "Austin, Texas"},
    {"name": "Crestview Insurance Group", "ticker": "CRVI", "industry": "property and casualty insurer",
     "segments": ["Personal Lines", "Commercial Lines"], "base_rev": 3100, "hq": "Hartford, Connecticut"},
    {"name": "Summit Ledger Software", "ticker": "SMLG", "industry": "core banking software provider",
     "segments": ["Subscription", "Professional Services"], "base_rev": 640, "hq": "Columbus, Ohio"},
    {"name": "Atlas Consumer Lending", "ticker": "ATCL", "industry": "consumer lender",
     "segments": ["Personal Loans", "Auto Finance", "Credit Cards"], "base_rev": 1220, "hq": "Phoenix, Arizona"},
    {"name": "Pinecrest Wealth Partners", "ticker": "PCWP", "industry": "wealth and asset manager",
     "segments": ["Advisory", "Asset Management"], "base_rev": 880, "hq": "Charlotte, North Carolina"},
]

RISKS = {
    "interest rates": "Changes in interest rates could compress our net interest margin and reduce demand for {product}.",
    "cybersecurity": "A cybersecurity incident could disrupt our systems, expose customer data and result in regulatory penalties.",
    "credit quality": "Deterioration in consumer or commercial credit quality could increase charge-offs and require higher loss reserves.",
    "regulation": "New regulations, including changes to capital and consumer protection rules, could increase compliance costs.",
    "competition": "Competition from fintech companies and large banks could pressure pricing and customer acquisition.",
    "third-party vendors": "We depend on third-party technology vendors, and an outage at a key provider could interrupt service.",
    "catastrophe losses": "Severe weather and natural catastrophes could produce claims losses well above our expectations.",
    "market volatility": "Declines in financial markets would reduce assets under management and the fees we earn on them.",
    "talent": "Our growth depends on attracting and retaining engineering and risk management talent.",
}
PRODUCT = {"regional bank": "loans", "payments processor": "transaction processing", "property and casualty insurer":
           "insurance products", "core banking software provider": "our software", "consumer lender": "consumer credit",
           "wealth and asset manager": "advisory services"}
FIRST = ["Elena", "Marcus", "Priya", "Daniel", "Grace", "Tomás", "Hannah", "Wei", "Olivia", "Samuel", "Nadia", "Robert"]
LAST = ["Whitaker", "Okafor", "Lindqvist", "Ramirez", "Chen", "Abernathy", "Kowalski", "Mehta", "Delacroix", "Brennan"]


def poss(name: str) -> str:
    return name + ("'" if name.endswith("s") else "'s")


def money(m: float) -> str:
    """Format millions consistently: $412.5 million or $2.41 billion."""
    return f"${m / 1000:.2f} billion" if m >= 1000 else f"${m:.1f} million"


def build_facts(rng: np.random.Generator) -> list[dict]:
    facts = []
    for c in COMPANIES:
        ceo = f"{FIRST[rng.integers(len(FIRST))]} {LAST[rng.integers(len(LAST))]}"
        cfo = f"{FIRST[rng.integers(len(FIRST))]} {LAST[rng.integers(len(LAST))]}"
        risks = list(rng.choice(list(RISKS), size=4, replace=False))
        rev_prev = c["base_rev"] * rng.uniform(0.9, 1.0)
        margin_prev = rng.uniform(0.12, 0.32)
        employees = int(c["base_rev"] * rng.uniform(2.5, 4.5))
        for year in (2024, 2025):
            growth = rng.uniform(-0.04, 0.16)
            rev = rev_prev * (1 + growth)
            margin = float(np.clip(margin_prev + rng.normal(0, 0.03), 0.05, 0.4))
            shares = rng.dirichlet(np.ones(len(c["segments"])) * 4)
            seg = {s: round(rev * w, 1) for s, w in zip(c["segments"], shares)}
            employees = int(employees * rng.uniform(0.96, 1.08))
            f = {
                "company": c["name"], "ticker": c["ticker"], "industry": c["industry"], "hq": c["hq"],
                "year": year, "ceo": ceo, "cfo": cfo,
                "revenue": round(rev, 1), "revenue_prev": round(rev_prev, 1), "growth_pct": round(growth * 100, 1),
                "operating_margin_pct": round(margin * 100, 1), "net_income": round(rev * margin * 0.76, 1),
                "segments": seg, "employees": employees,
                "cash": round(rev * rng.uniform(0.15, 0.6), 1), "debt": round(rev * rng.uniform(0.2, 1.2), 1),
                "buyback": round(rev * rng.uniform(0.0, 0.06), 1),
                "guidance_low": round(rng.uniform(1, 7), 0), "risks": risks if year == 2024 else
                list(dict.fromkeys(risks[1:] + [str(rng.choice([r for r in RISKS if r not in risks]))])),
                "customers_k": int(rng.uniform(50, 4000)),
            }
            f["guidance_high"] = f["guidance_low"] + rng.integers(2, 5)
            facts.append(f)
            rev_prev, margin_prev = rev, margin
    return facts


def render(f: dict) -> dict[str, str]:
    """Render one annual report as named sections (Markdown)."""
    seg_lines = "\n".join(f"- **{s}**: revenue of {money(v)}, or {v / f['revenue'] * 100:.0f}% of the total."
                          for s, v in f["segments"].items())
    top_seg = max(f["segments"], key=f["segments"].get)
    risk_text = "\n\n".join(f"**{r.capitalize()}.** " + RISKS[r].format(product=PRODUCT[f["industry"]])
                            for r in f["risks"])
    direction = "increased" if f["growth_pct"] >= 0 else "decreased"
    return {
        "business_overview": (
            f"## Business Overview\n\n{f['company']} ({f['ticker']}) is a {f['industry']} headquartered in {f['hq']}. "
            f"The company is led by Chief Executive Officer {f['ceo']} and Chief Financial Officer {f['cfo']}. "
            f"As of the end of fiscal {f['year']}, {f['company']} had {f['employees']:,} employees and served "
            f"approximately {f['customers_k']:,} thousand customers. The company reports results in "
            f"{len(f['segments'])} segments: {', '.join(f['segments'])}."),
        "segment_results": (
            f"## Segment Results\n\nFor fiscal {f['year']}, segment revenue was as follows:\n\n{seg_lines}\n\n"
            f"{top_seg} remained the largest segment by revenue."),
        "risk_factors": (
            f"## Risk Factors\n\nThe following are the principal risks management identified for fiscal {f['year']}.\n\n{risk_text}"),
        "management_discussion": (
            f"## Management's Discussion and Analysis\n\nTotal revenue for fiscal {f['year']} was {money(f['revenue'])}, "
            f"which {direction} {abs(f['growth_pct']):.1f}% from {money(f['revenue_prev'])} in fiscal {f['year'] - 1}. "
            f"Operating margin was {f['operating_margin_pct']:.1f}%. Net income was {money(f['net_income'])}. "
            f"Management attributed the year's performance primarily to results in the {top_seg} segment."),
        "liquidity": (
            f"## Liquidity and Capital Resources\n\nAt the end of fiscal {f['year']}, {f['company']} held cash and "
            f"cash equivalents of {money(f['cash'])} and total debt of {money(f['debt'])}. During the year the company "
            f"repurchased {money(f['buyback'])} of its common stock."),
        "outlook": (
            f"## Outlook\n\nFor fiscal {f['year'] + 1}, management expects total revenue growth of "
            f"{f['guidance_low']:.0f}% to {f['guidance_high']:.0f}%, driven by continued investment in "
            f"{' and '.join(list(f['segments'])[:2])}."),
    }


def build_questions(facts: list[dict], rng: np.random.Generator) -> list[dict]:
    qs = []
    by = {(f["company"], f["year"]): f for f in facts}

    def add(qtype, q, answer, sources, keywords=None):
        qs.append({"qid": f"Q{len(qs) + 1:03d}", "type": qtype, "question": q, "answer": answer,
                   "answer_keywords": keywords or [answer], "gold_sources": sources})

    for f in facts:
        doc = f"{f['ticker']}_FY{f['year']}"
        co, y = f["company"], f["year"]
        add("lookup", f"What was {poss(co)} total revenue in fiscal {y}?", money(f["revenue"]),
            [f"{doc}#management_discussion"])
        seg = list(f["segments"])[rng.integers(len(f["segments"]))]
        add("lookup", f"How much revenue did the {seg} segment of {co} generate in fiscal {y}?",
            money(f["segments"][seg]), [f"{doc}#segment_results"])
        add("lookup", f"How much cash and cash equivalents did {co} have at the end of fiscal {y}?",
            money(f["cash"]), [f"{doc}#liquidity"])
        add("lookup", f"What revenue growth range did {co} guide for fiscal {y + 1}?",
            f"{f['guidance_low']:.0f}% to {f['guidance_high']:.0f}%", [f"{doc}#outlook"])
        add("lookup", f"How many employees did {co} have at the end of fiscal {y}?", f"{f['employees']:,}",
            [f"{doc}#business_overview"])
        r = f["risks"][rng.integers(len(f["risks"]))]
        add("lookup", f"What did {co} say about {r} risk in its fiscal {y} report?", r,
            [f"{doc}#risk_factors"], keywords=[r])
        if y == 2025:
            prev = by[(co, 2024)]
            better = f["operating_margin_pct"] > prev["operating_margin_pct"]
            add("comparison", f"Did {poss(co)} operating margin improve from fiscal 2024 to fiscal 2025?",
                f"{'Yes' if better else 'No'}: {prev['operating_margin_pct']:.1f}% to {f['operating_margin_pct']:.1f}%",
                [f"{f['ticker']}_FY2024#management_discussion", f"{doc}#management_discussion"],
                keywords=[f"{prev['operating_margin_pct']:.1f}%", f"{f['operating_margin_pct']:.1f}%"])
            new_risks = [x for x in f["risks"] if x not in prev["risks"]]
            if new_risks:
                add("comparison", f"Which risk factor did {co} add in fiscal 2025 that was not in its fiscal 2024 report?",
                    new_risks[0], [f"{f['ticker']}_FY2024#risk_factors", f"{doc}#risk_factors"], keywords=[new_risks[0]])

    # Cross-company questions.
    for y in (2024, 2025):
        fy = [f for f in facts if f["year"] == y]
        top = max(fy, key=lambda f: f["operating_margin_pct"])
        add("cross_company", f"Which company had the highest operating margin in fiscal {y}?",
            top["company"], [f"{f['ticker']}_FY{y}#management_discussion" for f in fy], keywords=[top["company"]])

    # Unanswerable: the information is not in the corpus. The right behavior is to say so.
    for f in [x for x in facts if x["year"] == 2025]:
        co = f["company"]
        add("unanswerable", f"What was {poss(co)} total revenue in fiscal 2019?", "NOT_IN_CORPUS", [], [])
        add("unanswerable", f"Who is {poss(co)} independent auditor?", "NOT_IN_CORPUS", [], [])
    return qs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("data"))
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    facts = build_facts(rng)
    docs_dir = a.out / "corpus"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for f in facts:
        sections = render(f)
        header = (f"# {f['company']} — Annual Report, Fiscal {f['year']}\n\n"
                  f"*Fictional company. All figures are synthetic and for demonstration only.*\n\n")
        body = "\n\n".join(f"<!-- section: {k} -->\n{v}" for k, v in sections.items())
        (docs_dir / f"{f['ticker']}_FY{f['year']}.md").write_text(header + body + "\n")
    qs = build_questions(facts, rng)
    (a.out / "eval_questions.json").write_text(json.dumps(qs, indent=2))
    (a.out / "facts.json").write_text(json.dumps(facts, indent=2))
    types = {t: sum(q["type"] == t for q in qs) for t in dict.fromkeys(q["type"] for q in qs)}
    print(f"Wrote {len(facts)} documents to {docs_dir}/ and {len(qs)} evaluation questions {types}")


if __name__ == "__main__":
    main()
