"""Typed, allowlisted, traced, time-limited tools for the agent."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from dataclasses import dataclass
from typing import Any, Callable

from ..governance.policy import Policy
from ..obs.tracing import Tracer
from ..rag.corpus import Corpus
from ..schemas import DutyEstimate
from ..skills import SkillLoader

_POOL = ThreadPoolExecutor(max_workers=4)


@dataclass
class Tool:
    name: str
    description: str
    args: dict[str, type]  # required args
    fn: Callable[..., Any]
    optional: tuple[str, ...] = ()
    timeout: float = 5.0


def duty_for(corpus: Corpus, code: str, declared_value: float | None = None) -> DutyEstimate:
    node = corpus.nodes.get(code)
    if node is None:
        raise ValueError(f"unknown code {code}")
    rate = node.rate.strip()
    notes = ("General (column 1) rate only. Excludes additional duties (e.g. Section 232/301), "
             "trade-remedy duties, fees and taxes. Verify before filing.")
    m = re.match(r"^(\d+(?:\.\d+)?)%$", rate)
    if rate.lower() == "free":
        pct = 0.0
    elif m:
        pct = float(m.group(1))
    else:
        return DutyEstimate(rate, None, None, "Specific or compound rate; cannot be computed automatically. " + notes)
    est = round(declared_value * pct / 100, 2) if declared_value is not None else None
    return DutyEstimate(rate, pct, est, notes)


class ToolRegistry:
    def __init__(self, tools: list[Tool], tracer: Tracer, max_calls: int = 12):
        self.tools = {t.name: t for t in tools}
        self.tracer, self.max_calls, self.calls = tracer, max_calls, 0

    def describe(self) -> str:
        lines = []
        for t in self.tools.values():
            sig = ", ".join([f"{k}: {v.__name__}" for k, v in t.args.items()] + [f"{o}?" for o in t.optional])
            lines.append(f"- {t.name}({sig}): {t.description}")
        return "\n".join(lines)

    def call(self, name: str, args: dict) -> str:
        """Always returns a string observation; errors become observations, never exceptions."""
        with self.tracer.span("tool.call", tool=name) as sp:
            if name not in self.tools:
                sp.status = "denied"
                return json.dumps({"error": f"tool '{name}' is not allowed"})
            if self.calls >= self.max_calls:
                return json.dumps({"error": "tool call cap reached"})
            self.calls += 1
            t = self.tools[name]
            for k, typ in t.args.items():
                if k not in args or not isinstance(args[k], typ):
                    return json.dumps({"error": f"argument '{k}' must be {typ.__name__}"})
            clean = {k: v for k, v in args.items() if k in t.args or k in t.optional}
            fut = _POOL.submit(t.fn, **clean)
            try:
                return json.dumps(fut.result(timeout=t.timeout), ensure_ascii=False)
            except FutTimeout:
                sp.status = "timeout"
                return json.dumps({"error": "tool timed out"})
            except Exception as e:  # tool bugs become observations the agent can react to
                sp.status = "error"
                return json.dumps({"error": f"{type(e).__name__}: {e}"[:200]})


def build_tools(corpus: Corpus, skills: SkillLoader, policy: Policy, tracer: Tracer) -> ToolRegistry:
    def search_hts(query: str, k: int = 5) -> list[dict]:
        out = []
        for h in corpus.hts_index.search(query, k=max(1, min(int(k), 10))):
            n = corpus.nodes[h.doc.id]
            out.append({"code": n.code, "leaf": n.is_leaf, "path": " > ".join(n.path)[:240],
                        "rate": n.rate, "score": h.score, "bm25": h.bm25, "cos": h.cos})
        return out

    def get_node(code: str) -> dict:
        n = corpus.nodes.get(code)
        if n is None:
            raise ValueError(f"unknown code {code}")
        return {"code": n.code, "leaf": n.is_leaf, "path": " > ".join(n.path), "rate": n.rate, "units": n.units,
                "parent": n.parent, "children": [{"code": c, "description": corpus.nodes[c].description[:100],
                                                  "leaf": corpus.nodes[c].is_leaf} for c in n.children[:15]]}

    def search_rulings(query: str, k: int = 3) -> list[dict]:
        return [{"id": h.doc.id, "date": h.doc.meta["date"], "hts_code": h.doc.meta["hts_code"],
                 "snippet": h.doc.text[:220]} for h in corpus.rulings_index.search(query, k=max(1, min(int(k), 5)))]

    def search_precedents(query: str, k: int = 3) -> list[dict]:
        return [{"id": h.doc.id, "hts_code": h.doc.meta["hts_code"], "snippet": h.doc.text[:220]}
                for h in corpus.precedent_index.search(query, k=max(1, min(int(k), 5)))]

    def duty_calc(code: str, declared_value: float | None = None) -> dict:
        from dataclasses import asdict
        return asdict(duty_for(corpus, code, declared_value))

    def restrictions_check(description: str) -> list[dict]:
        return [{"id": r["id"], "message": r["message"]} for r in policy.restrictions_for(description)]

    def load_skill(name: str) -> dict:
        s = skills.get(name)
        return {"name": s.name, "procedure": s.body}

    T = Tool
    return ToolRegistry([
        T("search_hts", "Hybrid search over HTS lines; returns codes, paths, rates.", {"query": str}, search_hts, ("k",)),
        T("get_node", "Fetch one HTS node with its children (to navigate the tree).", {"code": str}, get_node),
        T("search_rulings", "Find similar past rulings and their final codes.", {"query": str}, search_rulings, ("k",)),
        T("search_precedents", "Find human-approved precedents for similar products.", {"query": str}, search_precedents, ("k",)),
        T("duty_calc", "General duty rate for a code.", {"code": str}, duty_calc, ("declared_value",)),
        T("restrictions_check", "Flag dual-use, hazardous or sanctioned wording.", {"description": str}, restrictions_check),
        T("load_skill", "Load the full procedure of a skill from the catalog.", {"name": str}, load_skill),
    ], tracer)
