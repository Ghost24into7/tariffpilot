"""Structured outputs: strict dataclasses with validation and JSON schemas for Gemini."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

DISCLAIMER = ("Decision support only, not legal or customs-broker advice. "
              "Verify against the current HTS and consult a licensed customs broker before filing.")
STATUSES = ("classified", "needs_review", "needs_info", "abstained", "blocked")


def _str(d: dict, key: str, *, required: bool = True, maxlen: int = 2000) -> str:
    v = d.get(key)
    if v is None:
        if required:
            raise ValueError(f"missing field '{key}'")
        return ""
    if not isinstance(v, str):
        raise ValueError(f"field '{key}' must be a string")
    return v[:maxlen]


@dataclass
class Citation:
    type: str
    id: str
    quote: str = ""

    @classmethod
    def from_dict(cls, d: Any) -> "Citation":
        if not isinstance(d, dict):
            raise ValueError("citation must be an object")
        t = _str(d, "type")
        if t not in ("hts", "ruling", "precedent"):
            raise ValueError("citation.type must be hts|ruling|precedent")
        return cls(t, _str(d, "id"), _str(d, "quote", required=False, maxlen=400))


@dataclass
class Alternative:
    code: str
    why_not: str = ""

    @classmethod
    def from_dict(cls, d: Any) -> "Alternative":
        if not isinstance(d, dict):
            raise ValueError("alternative must be an object")
        return cls(_str(d, "code"), _str(d, "why_not", required=False, maxlen=400))


@dataclass
class FinalAnswer:
    hts_code: str | None
    confidence: float
    gri_path: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    alternatives: list[Alternative] = field(default_factory=list)
    abstain_reason: str | None = None

    @classmethod
    def from_dict(cls, d: Any) -> "FinalAnswer":
        if not isinstance(d, dict):
            raise ValueError("final must be an object")
        code = d.get("hts_code")
        if code is not None and not isinstance(code, str):
            raise ValueError("hts_code must be a string or null")
        conf = d.get("confidence")
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
            raise ValueError("confidence must be a number in [0,1]")
        gri = d.get("gri_path") or []
        if not isinstance(gri, list) or not all(isinstance(x, str) for x in gri):
            raise ValueError("gri_path must be a list of strings")
        return cls(
            hts_code=(code or None),
            confidence=float(conf),
            gri_path=[x[:400] for x in gri[:8]],
            citations=[Citation.from_dict(c) for c in (d.get("citations") or [])][:8],
            alternatives=[Alternative.from_dict(a) for a in (d.get("alternatives") or [])][:5],
            abstain_reason=d.get("abstain_reason") or None,
        )


@dataclass
class Decision:
    kind: str  # tool | final | ask_user
    thought: str = ""
    tool: str = ""
    args: dict = field(default_factory=dict)
    question: str = ""
    final: FinalAnswer | None = None

    @classmethod
    def from_dict(cls, d: Any) -> "Decision":
        if not isinstance(d, dict):
            raise ValueError("decision must be a JSON object")
        kind = d.get("kind")
        if kind not in ("tool", "final", "ask_user"):
            raise ValueError("kind must be tool|final|ask_user")
        thought = _str(d, "thought", required=False, maxlen=300)
        if kind == "tool":
            tool = _str(d, "tool")
            raw = d.get("args_json") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError as e:
                raise ValueError(f"args_json is not valid JSON: {e}") from e
            if not isinstance(args, dict):
                raise ValueError("args_json must encode an object")
            return cls("tool", thought, tool=tool, args=args)
        if kind == "ask_user":
            return cls("ask_user", thought, question=_str(d, "question", maxlen=300))
        return cls("final", thought, final=FinalAnswer.from_dict(d.get("final")))


@dataclass
class DutyEstimate:
    rate_text: str = ""
    ad_valorem_pct: float | None = None
    estimated_duty: float | None = None
    notes: str = ""


@dataclass
class GuardEvent:
    rule_id: str
    action: str  # allow | redact | block | review | warn
    severity: str = "info"
    detail: str = ""


@dataclass
class ClassificationResult:
    status: str
    hts_code: str | None = None
    confidence: float = 0.0
    gri_path: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    alternatives: list[Alternative] = field(default_factory=list)
    duty_estimate: DutyEstimate | None = None
    restrictions: list[str] = field(default_factory=list)
    needs_human_review: bool = False
    abstain_reason: str | None = None
    question: str | None = None
    trace_id: str = ""
    steps: int = 0
    guard_events: list[GuardEvent] = field(default_factory=list)
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return asdict(self)


_STR = {"type": "string"}
FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "hts_code": {"type": "string", "nullable": True},
        "confidence": {"type": "number"},
        "gri_path": {"type": "array", "items": _STR},
        "citations": {"type": "array", "items": {"type": "object", "properties": {
            "type": _STR, "id": _STR, "quote": _STR}, "required": ["type", "id"]}},
        "alternatives": {"type": "array", "items": {"type": "object", "properties": {
            "code": _STR, "why_not": _STR}, "required": ["code"]}},
        "abstain_reason": {"type": "string", "nullable": True},
    },
    "required": ["confidence"],
}
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["tool", "final", "ask_user"]},
        "thought": _STR, "tool": _STR, "args_json": _STR, "question": _STR,
        "final": FINAL_SCHEMA,
    },
    "required": ["kind"],
}
