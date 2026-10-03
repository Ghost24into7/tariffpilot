"""Human-approved precedents with a quarantine -> active promotion gate (anti-poisoning)."""
from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..governance.policy import Policy


@dataclass
class GateResult:
    passed: bool
    detail: str = ""


class PrecedentError(Exception): ...


def _norm_hash(text: str) -> str:
    return hashlib.sha256(re.sub(r"\W+", " ", text.lower()).strip().encode()).hexdigest()[:16]


class PrecedentStore:
    def __init__(self, path: Path | str, policy: Policy):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.policy, self._lock = policy, threading.Lock()
        self.db.execute("""CREATE TABLE IF NOT EXISTS precedents(
            id TEXT PRIMARY KEY, text TEXT, code TEXT, rationale TEXT, reviewer TEXT, state TEXT,
            text_hash TEXT, created REAL, gate_detail TEXT)""")
        self.db.commit()

    def add(self, text: str, code: str, rationale: str, reviewer: str, now: float | None = None) -> str:
        if reviewer not in self.policy.authorized_reviewers:
            raise PermissionError(f"reviewer '{reviewer}' is not authorized")
        now = time.time() if now is None else now
        with self._lock:
            n = self.db.execute("SELECT COUNT(*) FROM precedents WHERE reviewer=? AND created>?",
                                (reviewer, now - 86400)).fetchone()[0]
            if n >= self.policy.feedback_per_reviewer_per_day:
                raise PrecedentError("reviewer daily feedback limit reached")
            th = _norm_hash(text + "|" + code)
            row = self.db.execute("SELECT id FROM precedents WHERE text_hash=?", (th,)).fetchone()
            if row:
                return row[0]  # duplicate detection
            pid = f"PREC-{th}"
            self.db.execute("INSERT INTO precedents VALUES(?,?,?,?,?,?,?,?,?)",
                            (pid, text[:1000], code, rationale[:500], reviewer, "quarantine", th, now, ""))
            self.db.commit()
            return pid

    def promote(self, pid: str, reviewer: str, gate: Callable[[dict], GateResult]) -> GateResult:
        if reviewer not in self.policy.authorized_reviewers:
            raise PermissionError(f"reviewer '{reviewer}' is not authorized")
        p = self.get(pid)
        if p["state"] != "quarantine":
            raise PrecedentError(f"{pid} is {p['state']}, not quarantine")
        res = gate(p)
        with self._lock:
            self.db.execute("UPDATE precedents SET state=?, gate_detail=? WHERE id=?",
                            ("active" if res.passed else "rejected", res.detail[:300], pid))
            self.db.commit()
        return res

    def retire(self, pid: str) -> None:
        self.db.execute("UPDATE precedents SET state='retired' WHERE id=?", (pid,))
        self.db.commit()

    def get(self, pid: str) -> dict:
        r = self.db.execute("SELECT id,text,code,rationale,reviewer,state,created,gate_detail FROM precedents WHERE id=?",
                            (pid,)).fetchone()
        if not r:
            raise KeyError(pid)
        return dict(zip(("id", "text", "code", "rationale", "reviewer", "state", "created", "gate_detail"), r))

    def list(self, state: str | None = None) -> list[dict]:
        q = "SELECT id FROM precedents" + (" WHERE state=?" if state else "") + " ORDER BY created"
        ids = [r[0] for r in self.db.execute(q, (state,) if state else ())]
        return [self.get(i) for i in ids]


def regression_gate(run_dev_eval: Callable[[list[dict]], dict], baseline: dict, store: PrecedentStore,
                    tolerance: float = 0.0) -> Callable[[dict], GateResult]:
    """Candidate passes only if dev accuracy does not regress and nothing hallucinated."""
    def gate(candidate: dict) -> GateResult:
        m = run_dev_eval(store.list("active") + [candidate])
        if m["hallucination_rate"] > 0:
            return GateResult(False, "introduced hallucinated codes")
        if m["accuracy_6"] + tolerance < baseline["accuracy_6"]:
            return GateResult(False, f"dev accuracy_6 {m['accuracy_6']:.3f} < baseline {baseline['accuracy_6']:.3f}")
        return GateResult(True, f"dev accuracy_6 {m['accuracy_6']:.3f} >= baseline {baseline['accuracy_6']:.3f}")
    return gate
