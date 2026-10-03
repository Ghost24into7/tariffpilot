"""Reflection: turn a human correction into a structured lesson (stored, never auto-applied)."""
from __future__ import annotations

from dataclasses import dataclass

from ..llm.gateway import Gateway


@dataclass
class Lesson:
    failure_type: str
    missed_rule: str
    suggested_skill_edit: str

    @classmethod
    def from_dict(cls, d) -> "Lesson":
        if not isinstance(d, dict) or not all(isinstance(d.get(k), str) for k in
                                              ("failure_type", "missed_rule", "suggested_skill_edit")):
            raise ValueError("lesson needs string fields failure_type, missed_rule, suggested_skill_edit")
        return cls(d["failure_type"][:200], d["missed_rule"][:400], d["suggested_skill_edit"][:600])


_SCHEMA = {"type": "object", "properties": {k: {"type": "string"} for k in
           ("failure_type", "missed_rule", "suggested_skill_edit")},
           "required": ["failure_type", "missed_rule", "suggested_skill_edit"]}


def reflect(gw: Gateway, product: str, model_code: str, correct_code: str, rationale: str) -> Lesson:
    return gw.generate("reflect_lesson", {"product": product, "model_code": model_code,
                                          "correct_code": correct_code, "rationale": rationale},
                       schema=_SCHEMA, validator=Lesson.from_dict)
