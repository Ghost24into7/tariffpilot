"""Input guardrails: normalization, injection heuristics, PII redaction, scope limits."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from ..schemas import GuardEvent

MAX_CHARS = 2000
_INJECTION = [
    ("inj-override", r"ignore (all |any )?(previous|prior|above) (instructions|rules|prompts)"),
    ("inj-disregard", r"disregard (the |all |your )?(system|previous|prior|above)"),
    ("inj-reveal", r"(reveal|print|show|repeat) (me )?(your |the )?(system )?(prompt|instructions)"),
    ("inj-persona", r"you are now|act as (an? )?(unrestricted|dan|developer mode)|jailbreak"),
    ("inj-tag", r"</?\s*(system|assistant|product_description|scratchpad|tools)\s*>"),
    ("inj-force", r"(always|just) (answer|respond|output|return) (with )?(the )?(code|hts)"),
]
_PII = [
    ("pii-email", r"[\w.+-]+@[\w-]+\.[\w.-]+", "[EMAIL]"),
    ("pii-pan", r"\b[A-Z]{5}\d{4}[A-Z]\b", "[PAN]"),
    ("pii-ssn", r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
    ("pii-phone", r"(?<!\w)(\+?\d[\d\s().-]{8,}\d)(?!\w)", "[PHONE]"),
]


def _luhn(digits: str) -> bool:
    s, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        s, alt = s + d, not alt
    return s % 10 == 0


@dataclass
class InputVerdict:
    allowed: bool
    text: str
    events: list[GuardEvent] = field(default_factory=list)


def check_input(raw: str) -> InputVerdict:
    events: list[GuardEvent] = []
    if not raw or not raw.strip():
        return InputVerdict(False, "", [GuardEvent("scope-empty", "block", "low", "empty input")])
    text = unicodedata.normalize("NFKC", raw)
    text = "".join(c for c in text if c.isprintable() or c in "\n\t").strip()
    if len(text) > MAX_CHARS:
        events.append(GuardEvent("len-trunc", "warn", "low", f"truncated to {MAX_CHARS} chars"))
        text = text[:MAX_CHARS]
    low = text.lower()
    for rid, pat in _INJECTION:
        if re.search(pat, low):
            events.append(GuardEvent(rid, "block", "high", "possible prompt injection"))
            return InputVerdict(False, "", events)

    def _card(m: re.Match) -> str:
        d = re.sub(r"\D", "", m.group(0))
        if 13 <= len(d) <= 19 and _luhn(d):
            events.append(GuardEvent("pii-card", "redact", "high", "card number redacted"))
            return "[CARD]"
        return m.group(0)

    text = re.sub(r"\b(?:\d[ -]?){13,19}\b", _card, text)
    for rid, pat, repl in _PII:
        text, n = re.subn(pat, repl, text)
        if n:
            events.append(GuardEvent(rid, "redact", "medium", f"{n} item(s) redacted"))
    return InputVerdict(True, text, events)
