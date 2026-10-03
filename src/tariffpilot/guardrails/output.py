"""Output guardrails: schema-level grounding checks against the corpus. No hallucinated codes."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..rag.corpus import Corpus
from ..schemas import FinalAnswer, GuardEvent


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


@dataclass
class Verification:
    ok: bool
    errors: list[str] = field(default_factory=list)
    events: list[GuardEvent] = field(default_factory=list)
    valid_citations: int = 0
    total_citations: int = 0


def verify_final(final: FinalAnswer, corpus: Corpus) -> Verification:
    v = Verification(True)
    if not final.hts_code:
        return v  # abstention is always allowed
    node = corpus.nodes.get(final.hts_code)
    if node is None:
        v.errors.append(f"code {final.hts_code} does not exist in the tariff")
        v.events.append(GuardEvent("out-hallucinated-code", "block", "high", final.hts_code))
    elif not node.is_leaf:
        v.errors.append(f"code {final.hts_code} is not a final tariff line; choose a leaf from get_node")
        v.events.append(GuardEvent("out-non-leaf", "block", "medium", final.hts_code))
    if not final.citations:
        v.errors.append("at least one citation is required")
    for c in final.citations:
        v.total_citations += 1
        if c.type == "hts":
            src = corpus.nodes.get(c.id)
            text = src.text if src else None
        elif c.type == "ruling":
            r = corpus.rulings.get(c.id)
            text = r["text"] if r else None
        else:
            p = corpus.precedents.get(c.id)
            text = p["text"] if p else None
        if text is None:
            v.errors.append(f"citation {c.type}:{c.id} not found")
            v.events.append(GuardEvent("out-bad-citation", "block", "high", f"{c.type}:{c.id}"))
        elif c.quote and _norm(c.quote) not in _norm(text):
            v.errors.append(f"quote for {c.type}:{c.id} does not appear in the source")
            v.events.append(GuardEvent("out-bad-quote", "block", "medium", f"{c.type}:{c.id}"))
        else:
            v.valid_citations += 1
    v.ok = not v.errors
    return v
