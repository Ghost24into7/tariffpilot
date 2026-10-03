"""Offline clients for tests and keyless demos. HeuristicAgentLLM is NOT an LLM: a deterministic policy."""
from __future__ import annotations

import json
from typing import Any, Callable

from .base import LLMError, Usage


class ScriptedLLM:
    """Returns queued responses (dict -> JSON, or str as-is, or Exception to raise)."""
    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, *, model, system, user, schema, meta) -> tuple[str, Usage]:
        self.calls.append({"model": model, "user": user, "meta": meta})
        if not self.responses:
            raise LLMError("ScriptedLLM exhausted")
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return (r if isinstance(r, str) else json.dumps(r)), Usage(len(user) // 4, 50)


class FunctionLLM:
    def __init__(self, fn: Callable[[dict[str, Any]], Any]):
        self.fn = fn

    def complete(self, *, model, system, user, schema, meta) -> tuple[str, Usage]:
        r = self.fn(meta)
        return (r if isinstance(r, str) else json.dumps(r)), Usage(0, 0)


class HeuristicAgentLLM:
    """Deterministic ReAct policy over the retrieval tool: search -> finalize with the top leaf."""
    def complete(self, *, model, system, user, schema, meta) -> tuple[str, Usage]:
        if meta.get("prompt_id") != "react_step":
            raise LLMError("offline mode only supports react_step")
        product, obs = meta["product"], meta.get("observations", [])
        words = [w for w in product.lower().split() if len(w) > 2]
        if len(words) < 3 and not meta.get("clarifications"):
            out: dict = {"kind": "ask_user", "thought": "too little detail",
                         "question": "What is it made of, and what is it used for?"}
        elif not obs:
            out = {"kind": "tool", "thought": "search the tariff", "tool": "search_hts",
                   "args_json": json.dumps({"query": product, "k": 5})}
        else:
            out = self._final(obs)
        return json.dumps(out), Usage(0, 0)

    @staticmethod
    def _final(obs: list[dict]) -> dict:
        hits = []
        for o in obs:
            if o["tool"] == "search_hts":
                try:
                    hits = [h for h in json.loads(o["obs"]) if h.get("leaf")]
                except json.JSONDecodeError:
                    pass
        if not hits:
            return {"kind": "final", "thought": "no leaf candidates", "final": {
                "hts_code": None, "confidence": 0.0, "abstain_reason": "no matching tariff line found"}}
        top = hits[0]
        second = hits[1]["bm25"] if len(hits) > 1 else 0.0
        margin = (top["bm25"] - second) / top["bm25"] if top["bm25"] > 0 else 0.0
        conf = round(min(0.95, 0.3 + 0.9 * margin + (0.1 if top["cos"] > 0.3 else 0.0)), 2)
        alts = [{"code": h["code"], "why_not": "lower retrieval score"} for h in hits[1:3]]
        return {"kind": "final", "thought": "top candidate", "final": {
            "hts_code": top["code"], "confidence": conf,
            "gri_path": [f"GRI 1: terms of heading {top['code'][:4]} match the product description"],
            "citations": [{"type": "hts", "id": top["code"], "quote": top["path"].split(" > ")[-1][:60]}],
            "alternatives": alts, "abstain_reason": None}}
