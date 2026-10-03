"""Eval harness: deterministic metrics first; datasets are JSONL; baselines run with zero quota."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..rag.corpus import Corpus
from ..schemas import ClassificationResult


def digits(code: str | None) -> str:
    return "".join(c for c in (code or "") if c.isdigit())


def load_dataset(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


@dataclass
class CaseResult:
    id: str
    expected: str | None
    predicted: str | None
    status: str
    confidence: float
    steps: int
    abstain_expected: bool
    hallucinated: bool
    cite_valid: int
    cite_total: int


def evaluate(dataset: list[dict], predict: Callable[[str], ClassificationResult], corpus: Corpus) -> tuple[dict, list[CaseResult]]:
    rows: list[CaseResult] = []
    for c in dataset:
        r = predict(c["text"])
        pred = r.hts_code
        ct = len(r.citations)
        cv = sum(1 for x in r.citations if (x.type == "hts" and x.id in corpus.nodes)
                 or (x.type == "ruling" and x.id in corpus.rulings) or (x.type == "precedent" and x.id in corpus.precedents))
        rows.append(CaseResult(c["id"], c.get("expected_code"), pred, r.status, r.confidence, r.steps,
                               bool(c.get("expected_abstain")), bool(pred and pred not in corpus.nodes), cv, ct))
    return summarize(rows), rows


def summarize(rows: list[CaseResult]) -> dict:
    answerable = [r for r in rows if not r.abstain_expected]
    unanswerable = [r for r in rows if r.abstain_expected]

    def acc(n: int, subset: list[CaseResult]) -> float:
        return round(sum(1 for r in subset if r.predicted and digits(r.predicted)[:n] == digits(r.expected)[:n])
                     / len(subset), 4) if subset else 0.0

    auto = [r for r in answerable if r.status == "classified"]
    abstained = lambda r: r.status in ("abstained", "needs_info", "blocked")  # noqa: E731
    flagged = sum(1 for r in rows if abstained(r))
    correct_abst = sum(1 for r in unanswerable if abstained(r))
    ct = sum(r.cite_total for r in rows)
    return {
        "n": len(rows),
        "accuracy_2": acc(2, answerable), "accuracy_4": acc(4, answerable),
        "accuracy_6": acc(6, answerable), "accuracy_8": acc(8, answerable),
        "auto_accept_precision": round(sum(1 for r in auto if digits(r.predicted) == digits(r.expected)) / len(auto), 4) if auto else 0.0,
        "coverage": round(len(auto) / len(answerable), 4) if answerable else 0.0,
        "hallucination_rate": round(sum(r.hallucinated for r in rows) / len(rows), 4) if rows else 0.0,
        "citation_faithfulness": round(sum(r.cite_valid for r in rows) / ct, 4) if ct else 1.0,
        "abstention_recall": round(correct_abst / len(unanswerable), 4) if unanswerable else 1.0,
        "abstention_precision": round(correct_abst / flagged, 4) if flagged else 1.0,
        "mean_steps": round(sum(r.steps for r in rows) / len(rows), 2) if rows else 0.0,
    }


def retrieval_baseline(corpus: Corpus) -> Callable[[str], ClassificationResult]:
    """No-LLM baseline: top-1 leaf from hybrid search. Free, deterministic, a floor for the agent."""
    def predict(text: str) -> ClassificationResult:
        hits = [h for h in corpus.hts_index.search(text, k=10) if h.doc.meta["leaf"]]
        if not hits:
            return ClassificationResult("abstained", abstain_reason="no candidates")
        return ClassificationResult("classified", hits[0].doc.id, 0.5, steps=0)
    return predict


def check_gate(metrics: dict, gate: dict) -> list[str]:
    fails = []
    if metrics["accuracy_6"] < gate.get("accuracy_6_min", 0):
        fails.append(f"accuracy_6 {metrics['accuracy_6']} < {gate['accuracy_6_min']}")
    if metrics["hallucination_rate"] > gate.get("hallucination_max", 0):
        fails.append(f"hallucination_rate {metrics['hallucination_rate']} > {gate['hallucination_max']}")
    if metrics["citation_faithfulness"] < gate.get("citation_faithfulness_min", 0):
        fails.append(f"citation_faithfulness {metrics['citation_faithfulness']} < {gate['citation_faithfulness_min']}")
    return fails


def to_markdown(title: str, results: dict[str, dict]) -> str:
    cols = ["accuracy_2", "accuracy_4", "accuracy_6", "accuracy_8", "auto_accept_precision", "coverage",
            "hallucination_rate", "citation_faithfulness", "abstention_recall", "mean_steps"]
    out = [f"### {title}", "", "| system | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, m in results.items():
        out.append(f"| {name} | " + " | ".join(str(m[c]) for c in cols) + " |")
    return "\n".join(out) + "\n"
