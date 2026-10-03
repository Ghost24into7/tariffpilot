from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Policy:
    version: int
    confidence_floor: float
    max_steps: int
    max_obs_chars: int
    llm_calls_per_request: int
    allowed_models: tuple[str, ...]
    authorized_reviewers: tuple[str, ...]
    feedback_per_reviewer_per_day: int
    retention_days: int
    restricted_keywords: tuple[dict, ...]
    eval_gate: dict = field(default_factory=dict)
    data_policy: str = "public-data-only"

    @classmethod
    def load(cls, path: Path) -> "Policy":
        d = yaml.safe_load(Path(path).read_text())
        if not 0 < d["confidence_floor"] < 1:
            raise ValueError("confidence_floor must be in (0,1)")
        if d["max_steps"] < 1 or d["llm_calls_per_request"] < 1:
            raise ValueError("max_steps and llm_calls_per_request must be >= 1")
        for r in d["restricted_keywords"]:
            if not {"id", "patterns", "action", "message"} <= set(r):
                raise ValueError(f"restricted rule malformed: {r}")
        return cls(
            version=int(d["version"]), confidence_floor=float(d["confidence_floor"]),
            max_steps=int(d["max_steps"]), max_obs_chars=int(d["max_obs_chars"]),
            llm_calls_per_request=int(d["llm_calls_per_request"]),
            allowed_models=tuple(d["allowed_models"]), authorized_reviewers=tuple(d["authorized_reviewers"]),
            feedback_per_reviewer_per_day=int(d["feedback_per_reviewer_per_day"]),
            retention_days=int(d["retention_days"]), restricted_keywords=tuple(d["restricted_keywords"]),
            eval_gate=dict(d.get("eval_gate", {})), data_policy=d.get("data_policy", "public-data-only"))

    def restrictions_for(self, text: str) -> list[dict]:
        low = text.lower()
        return [r for r in self.restricted_keywords if any(p in low for p in r["patterns"])]
