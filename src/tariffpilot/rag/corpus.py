"""Corpus: HTS tree (USITC export format), rulings (JSONL), learned precedents."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .index import Doc, Embedder, HashEmbedder, HybridIndex


@dataclass
class Node:
    code: str
    description: str
    path: list[str]
    rate: str = ""
    units: list[str] = field(default_factory=list)
    parent: str | None = None
    children: list[str] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def text(self) -> str:
        return f"{self.code} " + " > ".join(self.path)


def parse_usitc(rows: list[dict]) -> dict[str, Node]:
    """Rebuild the hierarchy from USITC 'indent' levels. Rows without htsno are context only."""
    nodes: dict[str, Node] = {}
    stack: list[tuple[int, str, str]] = []  # (indent, description, code)
    for r in rows:
        indent, desc = int(r.get("indent", 0)), str(r.get("description", "")).strip().rstrip(":")
        code = str(r.get("htsno") or "").strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = [s[1] for s in stack] + [desc]
        if code:
            parent = next((s[2] for s in reversed(stack) if s[2]), None)
            nodes[code] = Node(code, desc, path, str(r.get("general") or ""), list(r.get("units") or []), parent)
            if parent in nodes:
                nodes[parent].children.append(code)
        stack.append((indent, desc, code))
    return nodes


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def split_by_date(rows: list[dict], train: float = 0.6, dev: float = 0.2) -> dict[str, list[dict]]:
    """Chronological split so test rulings are always newer than training ones (no leakage)."""
    rows = sorted(rows, key=lambda r: r["date"])
    a, b = int(len(rows) * train), int(len(rows) * (train + dev))
    return {"train": rows[:a], "dev": rows[a:b], "test": rows[b:]}


class Corpus:
    def __init__(self, nodes: dict[str, Node], rulings: list[dict], embedder: Embedder | None = None):
        self.embedder = embedder or HashEmbedder()
        self.nodes = nodes
        self.hts_index = HybridIndex([Doc(n.code, n.text, {"leaf": n.is_leaf}) for n in nodes.values()], self.embedder)
        self.rulings = {r["id"]: r for r in rulings}
        self.rulings_index = HybridIndex(
            [Doc(r["id"], r["text"], {"hts_code": r["hts_code"], "date": r["date"]}) for r in rulings], self.embedder)
        self.precedents: dict[str, dict] = {}
        self.precedent_index = HybridIndex([], self.embedder)

    def set_precedents(self, precedents: list[dict]) -> None:
        self.precedents = {p["id"]: p for p in precedents}
        self.precedent_index = HybridIndex(
            [Doc(p["id"], p["text"], {"hts_code": p["code"]}) for p in precedents], self.embedder)

    @classmethod
    def load(cls, data_dir: Path, embedder: Embedder | None = None) -> "Corpus":
        raw, sample = Path(data_dir) / "raw", Path(data_dir) / "sample"
        hts = raw / "hts.json" if (raw / "hts.json").exists() else sample / "hts_sample.json"
        rul = raw / "rulings.jsonl" if (raw / "rulings.jsonl").exists() else sample / "rulings_sample.jsonl"
        return cls(parse_usitc(json.loads(hts.read_text())), load_jsonl(rul), embedder)
